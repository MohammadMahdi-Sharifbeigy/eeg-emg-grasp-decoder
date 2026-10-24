import copy

import numpy as np
import pandas as pd
import pytest
import torch

from v2.tokens import Q, to_samples
from v3.config import CONFIG_V3
from v3.evaluate import (OUTCOMES, criteria, evaluate_student, holm, residual_group_test, residual_over_timing,
                         subject_means)
from v3.nulls import trial_bounds

MUSCLES = ["AD", "BR", "ED", "FD", "FDI"]
SUBJECTS = [f"P{i}" for i in range(1, 13)]
OFFS = {"led_on": 0, "hand_start": 20, "first_touch": 45, "both_touch": 50, "lift_off": 70, "replace": 150,
        "release": 165, "hand_stop": 190}


def test_holm_values():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    out = holm([0.02, np.nan, 0.5])
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.04, 0.5])
    assert np.isnan(holm([np.nan])).all()


class _Stub(torch.nn.Module):
    """E and the final tokens are fixed functions of electrode 0 (bands 0 and 1): template * amplitude and template."""
    causal = False

    def __init__(self, informative=True):
        super().__init__()
        self.informative = informative

    def forward(self, eeg):
        drive, tmpl = eeg[:, :, 0, 0], eeg[:, :, 0, 1]
        fin = torch.stack([drive, tmpl, drive * tmpl, torch.ones_like(drive)], -1)
        if not self.informative:
            fin = torch.randn(fin.shape)                               # new noise in every window
        return {"E": torch.relu(drive)[..., None].repeat(1, 1, 5).sqrt(), "fin": fin}


def _world(n_tok=2500, seed=0):
    r = np.random.default_rng(seed)
    toks, lifts, timing = [], [], []
    for s in range(2):
        l = [{"lift": k, "CurW": 1, "CurS": 1, **{e: (t + o) * Q for e, o in OFFS.items()}}
             for k, t in enumerate(range(30, n_tok - 250, 250))]
        tmpl, amp = np.zeros(n_tok), np.ones(n_tok)
        for a, b, led in trial_bounds(l, n_tok):
            t = np.arange(a, b) - led
            tmpl[a:b] = np.exp(-0.5 * ((t - 90) / 30.0) ** 2)
            amp[a:b] = r.uniform(0.3, 1.7)
        y = 0.02 + tmpl * amp
        eeg = np.zeros((n_tok, 32, 4), np.float32)
        eeg[:, 0, 0], eeg[:, 0, 1] = tmpl * amp, tmpl
        emg = np.repeat(y[:, None], 5, 1).astype(np.float32)
        toks.append({"series": s, "eeg": eeg, "emg": emg, "emg_full": to_samples(emg, n_tok * Q), "n_samples": n_tok * Q})
        lifts.append(l)
        timing.append(np.repeat(np.sqrt(0.02 + tmpl)[:, None], 5, 1))      # the task clock, no amplitude
    norm = {"eeg_mean": np.zeros((32, 4), np.float32), "eeg_std": np.ones((32, 4), np.float32),
            "emg_scale": np.ones(5, np.float32)}
    return toks, lifts, timing, norm


def _cfg():
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["evaluation"].update(n_null=19, seed=0)
    return cfg


def test_residual_over_timing_finds_trial_specific_information():
    toks, lifts, timing, norm = _world()
    df = residual_over_timing(_Stub(True), toks, timing, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)
    assert list(df.segment) == ["pre_move", "reach", "load", "hold", "release", "other", "all"]
    row = df.set_index("segment").loc["all"]
    assert row.skill > 0.3 and row.r_ridge > 0.5 and row.r_direct > 0.5 and row.p <= 0.05
    assert df.set_index("segment").loc["hold"].skill > 0.3


def test_residual_over_timing_is_null_for_an_uninformative_student():
    toks, lifts, timing, norm = _world()
    torch.manual_seed(0)
    row = residual_over_timing(_Stub(False), toks, timing, lifts, norm, _cfg(), torch.device("cpu"),
                               MUSCLES).set_index("segment").loc["all"]
    assert row.skill < 0.05 and row.r_ridge < 0.2          # held-out fold means make noise anti-correlated, never positive


def test_residual_over_timing_checks_lengths():
    toks, lifts, timing, norm = _world()
    with pytest.raises(ValueError):
        residual_over_timing(_Stub(), toks, [t[:-3] for t in timing], lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)


def test_evaluate_student_scores_at_both_rates():
    toks, _, _, norm = _world()
    cfg = _cfg()
    cfg["evaluation"]["n_null"] = 1
    std = evaluate_student(_Stub(), toks, norm, cfg, torch.device("cpu"), MUSCLES)["standard"]
    assert sorted(std.rate.unique()) == [50, 500] and len(std) == 10
    assert (std.r > 0.95).all()


def _v2_csvs(tmp_path):
    rows_v2, rows_ph = [], []
    for s in SUBJECTS:
        for m in MUSCLES:
            for out, r in (("E", 0.645), ("K", 0.87)):
                rows_v2.append(dict(muscle=m, r=r, ccc=0, rmse=0, vaf=r * 100, output=out, subject=s))
            for var, r in (("phase", 0.81), ("cue", 0.58)):
                rows_ph.append(dict(muscle=m, r=r, ccc=0, rmse=0, vaf=r * 100, subject=s, variant=var, alpha=1,
                                    n_features=1, n_test_samples=1))
    a, b = tmp_path / "standard.csv", tmp_path / "phase_metrics.csv"
    pd.DataFrame(rows_v2).to_csv(a, index=False)
    pd.DataFrame(rows_ph).to_csv(b, index=False)
    return a, b


def _std(values):
    """values[(scale, seed)] -> per-subject r; also 50 Hz rows that must be ignored."""
    rows = []
    for (scale, seed), vals in values.items():
        for s, v in zip(SUBJECTS, vals):
            for m in MUSCLES:
                for rate, rr in ((500, v), (50, -1.0)):
                    rows.append(dict(muscle=m, r=rr, vaf=rr * 100, subject=s, scale=scale, seed=seed, rate=rate))
    return pd.DataFrame(rows)


def test_subject_means_averages_seeds_and_uses_500_hz(tmp_path):
    a, b = _v2_csvs(tmp_path)
    j = np.linspace(-0.01, 0.01, 12)
    std = _std({(0.0, 0): 0.60 + j, (0.0, 1): 0.62 + j, (0.0, 2): 0.64 + j})
    m = subject_means(std, a, b)
    row = m[m.subject == "P1"].iloc[0]
    assert row.E_guided_r == pytest.approx(0.62 - 0.01) and row.E_guided_r_seed_sd == pytest.approx(0.02)
    assert row.n_seeds == 3 and row.phase_r == pytest.approx(0.81)


def _residual_tests(skill_by_scale, gain_by_scale):
    rows = []
    for scale, skill in skill_by_scale.items():
        for s_i, s in enumerate(SUBJECTS):
            rows.append(dict(timing="cue_causal", segment="all", scale=scale, subject=s, seed=0,
                             skill=skill + 0.001 * s_i + (gain_by_scale.get(scale, 0.0) if scale else 0.0),
                             r_ridge=0.1, r_direct=0.1))
            rows.append(dict(timing="cue_causal", segment="hold", scale=scale, subject=s, seed=0, skill=0.01,
                             r_ridge=0.1, r_direct=0.1))
    return residual_group_test(pd.DataFrame(rows))


def _swap_tests(delta_ctrl, delta_guided, n_sub=12):
    rows = [{"scale": 0.0, "mean_delta": delta_ctrl, "p_delta": 0.001, "mean_delta_gain": np.nan, "p_delta_gain": np.nan}]
    for scale, (d, p, g, pg) in delta_guided.items():
        rows.append({"scale": scale, "mean_delta": d, "p_delta": p, "mean_delta_gain": g, "p_delta_gain": pg})
    return pd.DataFrame(rows)


def test_residual_group_test_columns_and_holm_over_phases():
    g = _residual_tests({0.0: 0.02, 0.1: 0.02}, {0.1: 0.03})
    assert {"p_skill", "p_skill_gain", "p_holm_phases", "mean_skill_gain"} <= set(g.columns)
    row = g[(g.segment == "all") & (g.scale == 0.1)].iloc[0]
    assert row.mean_skill_gain == pytest.approx(0.03) and row.p_skill_gain < 0.05
    assert np.isnan(row.p_holm_phases)
    assert g[g.segment == "hold"].p_holm_phases.notna().all()


def test_criteria_outcomes(tmp_path):
    a, b = _v2_csvs(tmp_path)
    j = np.linspace(-0.01, 0.01, 12)
    vals = {(0.0, 0): 0.60 + j, (0.05, 0): 0.60 + j + 0.05, (0.1, 0): 0.60 + j + 0.05, (0.3, 0): 0.60 + j * 0.5 - 0.05}
    means = subject_means(_std(vals), a, b)
    res = _residual_tests({0.0: 0.02, 0.05: 0.02, 0.1: 0.02, 0.3: 0.02}, {0.05: 0.03, 0.1: 0.0})
    sw = _swap_tests(0.05, {0.05: (0.1, 0.001, 0.05, 0.001), 0.1: (0.05, 0.001, 0.0, 0.6), 0.3: (0.05, 0.001, 0.0, 0.6)})
    c = criteria(means, sw, res).set_index("scale")
    assert c.loc[0.05, "outcome"] == "a" and c.loc[0.05, "verdict"] == OUTCOMES["a"]
    assert c.loc[0.1, "outcome"] == "b" and bool(c.loc[0.1, "beats_control"]) and not bool(c.loc[0.1, "survives_swap"])
    assert c.loc[0.3, "outcome"] == "d" and bool(c.loc[0.3, "worse"])
    assert not c.timing_level.any()                                    # 0.65 is far below phase 0.81: information only
    assert {"p_gain_holm", "p_residual_holm", "p_delta_gain_holm", "mean_r_seed_sd"} <= set(c.columns)


def test_criteria_no_help(tmp_path):
    a, b = _v2_csvs(tmp_path)
    r = np.random.default_rng(0).normal(0, 0.01, 12)
    means = subject_means(_std({(0.0, 0): 0.6 + r, (0.1, 0): 0.6 + r[::-1]}), a, b)
    res = _residual_tests({0.0: 0.0, 0.1: 0.0}, {})
    sw = _swap_tests(0.0, {0.1: (0.0, 0.5, 0.0, 0.5)})
    assert criteria(means, sw, res).set_index("scale").loc[0.1, "outcome"] == "c"
