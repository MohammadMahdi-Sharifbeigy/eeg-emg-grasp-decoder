import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import wilcoxon

from v2.data import fit_norm
from v2.tokens import to_samples
from v3.config import CONFIG_V3
from v3.evaluate import evaluate_student, group_tests, subject_means
from v3.infer import normalise_eeg

MUSCLES = ["AD", "BR", "ED", "FD", "FDI"]
SUBJECTS = [f"P{i}" for i in range(1, 13)]


def _write_csvs(tmp_path, e_saved, k, phase, cue):
    rows_v2, rows_ph = [], []
    for i, s in enumerate(SUBJECTS):
        for m in MUSCLES:
            for out, r in (("E", e_saved[i]), ("K", k[i]), ("F", k[i])):
                rows_v2.append(dict(muscle=m, r=r, ccc=0.0, rmse=0.0, vaf=r * 100, output=out, subject=s))
            for var, r in (("phase", phase[i]), ("cue", cue[i]), ("coarse", 0.0)):
                rows_ph.append(dict(muscle=m, r=r, ccc=0.0, rmse=0.0, vaf=r * 100, subject=s, variant=var,
                                    alpha=1e-5, n_features=1, n_test_samples=1))
    a, b = tmp_path / "standard.csv", tmp_path / "phase_metrics.csv"
    pd.DataFrame(rows_v2).to_csv(a, index=False)
    pd.DataFrame(rows_ph).to_csv(b, index=False)
    return a, b


def _v3_rows(values_by_scale):
    rows = []
    for scale, vals in values_by_scale.items():
        for s, v in zip(SUBJECTS, vals):
            for m in MUSCLES:
                rows.append(dict(muscle=m, r=v, ccc=0.0, rmse=0.0, vaf=v * 100, subject=s, scale=scale))
    return pd.DataFrame(rows)


def _world(tmp_path, guided_mean):
    jitter = np.linspace(-0.01, 0.01, 12)
    control = 0.60 + jitter
    guided = guided_mean + jitter * 0.5
    e_saved = 0.645 + jitter
    a, b = _write_csvs(tmp_path, e_saved, np.full(12, 0.87), np.full(12, 0.81) + jitter, np.full(12, 0.58))
    return _v3_rows({0.0: control, 0.1: guided}), a, b


def test_subject_means_columns_and_values(tmp_path):
    v3, a, b = _world(tmp_path, 0.77)
    m = subject_means(v3, a, b)
    assert len(m) == 24 and {"subject", "scale", "E_guided_r", "E_saved_r", "K_r", "phase_r", "cue_r"} <= set(m.columns)
    row = m[(m.subject == "P1") & (m.scale == 0.1)].iloc[0]
    assert row.E_guided_r == pytest.approx(0.77 - 0.005) and row.K_r == pytest.approx(0.87)
    assert row.cue_r == pytest.approx(0.58) and row.E_saved_vaf == pytest.approx((0.645 - 0.01) * 100)


def test_group_tests_flags(tmp_path):
    v3, a, b = _world(tmp_path, 0.77)
    g = group_tests(subject_means(v3, a, b)).set_index("scale").loc[0.1]
    assert bool(g.beats_E) and bool(g.close_to_timing) and not bool(g.beats_timing)
    assert g.n_better_than_control == 12 and g.n_subjects == 12
    v3, a, b = _world(tmp_path, 0.74)
    g = group_tests(subject_means(v3, a, b)).set_index("scale").loc[0.1]
    assert bool(g.beats_E) and not bool(g.close_to_timing)


def test_wilcoxon_p_matches_scipy(tmp_path):
    v3, a, b = _world(tmp_path, 0.77)
    m = subject_means(v3, a, b)
    g = group_tests(m).set_index("scale").loc[0.1]
    sub = m[m.scale == 0.1].set_index("subject")
    ctrl = m[m.scale == 0.0].set_index("subject")
    expected = wilcoxon(sub.E_guided_r.values, ctrl.E_guided_r.loc[sub.index].values, alternative="greater").pvalue
    assert g.p_vs_control == pytest.approx(expected)


class _Stub(torch.nn.Module):
    """E is a fixed function of the first electrode and band, so a planted EEG-EMG relation exists."""
    causal = False

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, eeg):
        u = torch.sigmoid(eeg[:, :, 0, 0]) / float(np.sqrt(self.scale))
        return {"E": u[..., None].repeat(1, 1, 5)}


def test_evaluate_student_nulls_drop():
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["evaluation"]["n_null"] = 3
    r = np.random.default_rng(0)
    toks = [{"eeg": r.normal(size=(900, 32, 4)).astype(np.float32), "n_samples": 9000} for _ in range(2)]
    norm0 = {"eeg_mean": np.zeros((32, 4), np.float32), "eeg_std": np.ones((32, 4), np.float32)}
    for t in toks:
        z = normalise_eeg(t, norm0, cfg["eeg"]["clip"])
        t["emg"] = np.repeat((1 / (1 + np.exp(-z[:, 0, 0])) ** 2)[:, None], 5, axis=1).astype(np.float32)
    norm = fit_norm([{**t, "kin": np.zeros((900, 24), np.float32)} for t in toks])
    norm.update(eeg_mean=norm0["eeg_mean"], eeg_std=norm0["eeg_std"])
    scale = float(norm["emg_scale"][0])
    for t in toks:
        t["emg_full"] = to_samples(t["emg"], t["n_samples"])
    out = evaluate_student(_Stub(scale), toks, norm, cfg, torch.device("cpu"), MUSCLES)
    nulls = out["nulls"]
    assert (nulls.E_r > 0.9).all()
    assert (nulls.E_r > nulls.E_r_shuffle_mean + 0.3).all() and (nulls.E_r > nulls.E_r_swap_mean + 0.3).all()
    assert list(out["standard"].columns[:2]) == ["muscle", "r"]
