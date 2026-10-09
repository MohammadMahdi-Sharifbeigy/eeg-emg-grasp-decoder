"""Tests for main.ablation — EEG vs kinematics input-ablation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.ablation import (  # noqa: E402
    ABLATION_CONDITIONS,
    AblationResult,
    apply_condition,
    input_ablation,
    pearson_per_channel,
    rest_threshold_from_loader,
    shuffle_time,
)

B, T, F_EEG, F_KIN, C = 6, 40, 4, 3, 2


class _KinModel(nn.Module):
    """Linear map of EEG and kinematics → uses both inputs."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.eeg = nn.Linear(F_EEG, C)
        self.kin = nn.Linear(F_KIN, C)

    def forward(self, eeg, kin=None):
        return self.eeg(eeg) + self.kin(kin)


class _EEGOnlyModel(nn.Module):
    """Ignores kinematics entirely."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.eeg = nn.Linear(F_EEG, C)

    def forward(self, eeg, kin=None):
        return self.eeg(eeg)


def _loader():
    g = torch.Generator().manual_seed(123)
    eeg = torch.randn(B, T, F_EEG, generator=g)
    kin = torch.randn(B, T, F_KIN, generator=g) * 2.0 + 5.0   # raw, not z-scored
    emg = torch.rand(B, T, C, generator=g)
    return DataLoader(TensorDataset(eeg, kin, emg), batch_size=4, shuffle=False)


def _prepare_batch():
    kin_mean = torch.full((F_KIN,), 5.0)
    kin_std = torch.full((F_KIN,), 2.0)

    def pb(eeg, kin, emg):
        return {"eeg": eeg, "kin": (kin - kin_mean) / kin_std}, emg

    return pb


# ── shuffle_time ────────────────────────────────────────────────────────────

def test_shuffle_time_keeps_shape_and_values_per_window():
    x = torch.randn(B, T, F_KIN)
    y = shuffle_time(x, seed=7)
    assert y.shape == x.shape
    # a permutation of time steps: sorted values along time are unchanged
    assert torch.allclose(torch.sort(y, dim=1).values, torch.sort(x, dim=1).values)
    assert not torch.allclose(y, x)


def test_shuffle_time_is_deterministic_for_a_seed():
    x = torch.randn(B, T, F_KIN)
    assert torch.equal(shuffle_time(x, seed=3), shuffle_time(x, seed=3))
    assert not torch.equal(shuffle_time(x, seed=3), shuffle_time(x, seed=4))


def test_apply_condition_shapes_and_targets():
    pb = _prepare_batch()
    eeg, kin, emg = next(iter(_loader()))
    inputs, _ = pb(eeg, kin, emg)
    for cond in ABLATION_CONDITIONS:
        out = apply_condition(inputs, cond, seed=0)
        assert out["eeg"].shape == inputs["eeg"].shape
        assert out["kin"].shape == inputs["kin"].shape
    z = apply_condition(inputs, "kin_zero", seed=0)
    assert torch.count_nonzero(z["kin"]) == 0
    assert torch.equal(z["eeg"], inputs["eeg"])
    e = apply_condition(inputs, "eeg_shuffle", seed=0)
    assert torch.equal(e["kin"], inputs["kin"])
    assert not torch.equal(e["eeg"], inputs["eeg"])
    with pytest.raises(ValueError):
        apply_condition(inputs, "bogus", seed=0)


# ── pearson ─────────────────────────────────────────────────────────────────

def test_pearson_matches_np_corrcoef():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(500, 3))
    p = 0.6 * y + rng.normal(size=(500, 3))
    r = pearson_per_channel(p, y)
    for c in range(3):
        assert r[c] == pytest.approx(np.corrcoef(p[:, c], y[:, c])[0, 1], abs=1e-10)


def test_pearson_constant_channel_is_zero():
    y = np.random.default_rng(1).normal(size=(50, 2))
    p = np.ones_like(y)
    assert np.allclose(pearson_per_channel(p, y), 0.0)


# ── input_ablation ──────────────────────────────────────────────────────────

def test_kin_zero_changes_kin_model_output():
    res = input_ablation(_KinModel(), _loader(), _prepare_batch(), torch.device("cpu"),
                         use_amp=False, seed=0)
    assert isinstance(res, AblationResult)
    assert not np.allclose(res.predictions["full"], res.predictions["kin_zero"])
    assert res.r["full"].shape == (C,)


def test_kin_zero_leaves_kin_free_model_unchanged():
    res = input_ablation(_EEGOnlyModel(), _loader(), _prepare_batch(), torch.device("cpu"),
                         use_amp=False, seed=0)
    assert np.allclose(res.predictions["full"], res.predictions["kin_zero"])
    assert np.allclose(res.predictions["full"], res.predictions["kin_shuffle"])
    assert np.allclose(res.r["full"], res.r["kin_zero"])
    assert not np.allclose(res.predictions["full"], res.predictions["eeg_shuffle"])


def test_input_ablation_full_r_matches_corrcoef_and_is_deterministic():
    loader, pb, dev = _loader(), _prepare_batch(), torch.device("cpu")
    model = _KinModel()
    a = input_ablation(model, loader, pb, dev, use_amp=False, seed=5)
    b = input_ablation(model, loader, pb, dev, use_amp=False, seed=5)
    for cond in ABLATION_CONDITIONS:
        assert np.array_equal(a.predictions[cond], b.predictions[cond])
        assert a.r_mean[cond] == pytest.approx(float(np.mean(a.r[cond])))
    P, Y = a.predictions["full"], a.target
    assert P.shape == Y.shape == (B * T, C)
    for c in range(C):
        assert a.r["full"][c] == pytest.approx(np.corrcoef(P[:, c], Y[:, c])[0, 1], abs=1e-6)


def test_rest_level_uses_threshold_mask():
    loader, pb, dev = _loader(), _prepare_batch(), torch.device("cpu")
    thr = rest_threshold_from_loader(loader, pb, q=10.0)
    assert thr.shape == (C,)
    res = input_ablation(_KinModel(), loader, pb, dev, use_amp=False, seed=0, rest_threshold=thr)
    Y = res.target
    for c in range(C):
        mask = Y[:, c] < thr[c]
        assert mask.any()
        assert res.rest_pred["full"][c] == pytest.approx(res.predictions["full"][mask, c].mean())
        assert res.rest_pred["kin_zero"][c] == pytest.approx(res.predictions["kin_zero"][mask, c].mean())
        assert res.rest_target[c] == pytest.approx(Y[mask, c].mean())
    table = res.as_table(["a", "b"])
    assert "kin_zero" in table and "rest" in table.lower()
