"""Muscle specificity (v3.evaluate.muscle_specificity): EEG beyond the task clock and the activation all muscles share."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from v2.tokens import Q, to_samples
from v3.config import CONFIG_V3
from v3.evaluate import muscle_group_test, muscle_specificity, remove_shared
from v3.nulls import trial_bounds

MUSCLES = ["AD", "BR", "ED", "FD", "FDI"]


class _Stub(torch.nn.Module):
    """The final tokens are electrodes 0-4 of band 0 (whatever the world put there) and a constant."""
    causal = False

    def forward(self, eeg):
        x = eeg[:, :, :5, 0]
        return {"E": torch.zeros(x.shape), "fin": torch.cat([x, torch.ones_like(x[..., :1])], -1)}


def _world(specific=True, n_tok=2600, seed=0):
    """Per trial a shared amplitude (gains differ per muscle) and, if `specific`, one amplitude per muscle.

    The EEG carries the trial's residual of every muscle when `specific`, else only the shared part; the target also
    has per-token noise the EEG never sees.
    """
    r = np.random.default_rng(seed)
    gain = np.array([1.0, 0.8, 0.6, 1.2, 0.9])
    toks, lifts, timing = [], [], []
    for s in range(2):
        l = [{"led_on": (t + 0) * Q, "hand_start": (t + 20) * Q, "first_touch": (t + 45) * Q, "lift_off": (t + 70) * Q,
              "replace": (t + 150) * Q, "hand_stop": (t + 190) * Q} for t in range(30, n_tok - 250, 250)]
        tmpl, res, sig = np.zeros(n_tok), np.zeros((n_tok, 5)), np.zeros((n_tok, 5))
        for a, b, led in trial_bounds(l, n_tok):
            t = np.arange(a, b) - led
            tmpl[a:b] = np.exp(-0.5 * ((t - 90) / 30.0) ** 2)
            shared = r.uniform(-0.25, 0.25) * gain
            own = r.uniform(-0.2, 0.2, size=5) if specific else np.zeros(5)
            res[a:b] = tmpl[a:b, None] * (shared + own)
            sig[a:b] = tmpl[a:b, None] * (shared + own if specific else shared)
        base = np.sqrt(0.02 + tmpl)
        u = np.maximum(base[:, None] + res + 0.03 * r.normal(size=(n_tok, 5)), 0.0)
        eeg = np.zeros((n_tok, 32, 4), np.float32)
        eeg[:, :5, 0] = sig
        emg = (u ** 2).astype(np.float32)
        toks.append({"series": s, "eeg": eeg, "emg": emg, "emg_full": to_samples(emg, n_tok * Q), "n_samples": n_tok * Q})
        lifts.append(l)
        timing.append(np.repeat(base[:, None], 5, 1))
    norm = {"eeg_mean": np.zeros((32, 4), np.float32), "eeg_std": np.ones((32, 4), np.float32),
            "emg_scale": np.ones(5, np.float32)}
    return toks, lifts, timing, norm


def _cfg():
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["evaluation"].update(n_null=19, seed=0)
    return cfg


def test_remove_shared_is_uncorrelated_with_the_shared_signal():
    r = np.random.default_rng(0)
    s = r.normal(size=2000)
    res = s[:, None] * np.array([1.0, 0.5, 2.0, -0.3, 1.0]) + 0.2 * r.normal(size=(2000, 5)) + 3.0
    for how in ("pc1", "mean"):
        spec, sh = remove_shared(res, how)
        assert np.abs((spec - spec.mean(0)).T @ (sh - sh.mean())).max() < 1e-6 * len(sh)
        np.testing.assert_allclose(spec.mean(0), res.mean(0))          # the muscle means stay
        assert spec.var(0).sum() < 0.3 * res.var(0).sum()
    with pytest.raises(ValueError):
        remove_shared(res, "pc2")


def test_finds_muscle_specific_information():
    toks, lifts, timing, norm = _world(specific=True)
    df = muscle_specificity(_Stub(), toks, timing, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)
    assert list(df.muscle) == MUSCLES + ["MEAN"]
    assert (df.skill > 0.3).all() and (df.p <= 0.05).all() and (df.r > 0.5).all()
    assert (df.shared == "pc1").all() and df.var_shared.iloc[0] > 0.2 and df.n_trials.iloc[0] == 20


def test_shared_information_alone_is_not_muscle_specific():
    toks, lifts, timing, norm = _world(specific=False)
    for shared in ("pc1", "mean"):
        df = muscle_specificity(_Stub(), toks, timing, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES, shared=shared)
        row = df.set_index("muscle").loc["MEAN"]
        assert row.skill < 0.05, shared
    # The same EEG does explain the residual before the shared part is removed (residual_over_timing would see it).
    assert df.var_shared.iloc[0] > 0.5


def test_checks_timing_length():
    toks, lifts, timing, norm = _world()
    with pytest.raises(ValueError):
        muscle_specificity(_Stub(), toks, [t[:-2] for t in timing], lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)


def test_muscle_group_test_holm_over_muscles():
    rows = []
    j = np.linspace(-0.005, 0.005, 12)
    for scale, base in ((0.0, 0.02), (0.1, 0.05)):
        for m in MUSCLES + ["MEAN"]:
            for i in range(12):
                rows.append({"subject": f"P{i + 1}", "scale": scale, "shared": "pc1", "muscle": m,
                             "skill": base + j[i], "r": 0.2, "var_shared": 0.6, "seed": 0})
    g = muscle_group_test(pd.DataFrame(rows))
    row = g[(g.scale == 0.1) & (g.muscle == "AD")].iloc[0]
    assert row.mean_skill_gain == pytest.approx(0.03) and row.p_skill_gain < 0.01 and row.p_skill < 0.01
    m = g[(g.scale == 0.1) & (g.muscle != "MEAN")]
    assert m.p_holm_muscles.to_numpy() == pytest.approx(np.minimum(1, 5 * m.p_skill.to_numpy()))
    assert np.isnan(g[g.muscle == "MEAN"].p_holm_muscles).all()
