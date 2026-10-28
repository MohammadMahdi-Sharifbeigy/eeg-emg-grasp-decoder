"""B vs F (v3.contrast): EEG history over a causal motion + task-event model, and its controls."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_controls.phase_ridge import PHASES, _fit, fit_and_score
from v2.evaluate import standard_metrics
from v2.tokens import Q
from v3 import contrast as C
from v3.config import CONFIG_V3, PERIPHERAL, preset_config
from v3.nulls import swap_series

N_EV, N_EEG = 6, 4


def _split(r, n, eeg_weight, ev_weight=1.0):
    """K, target y, residual, event and EEG features of one split; the residual is linear in both feature sets."""
    t = np.arange(n)
    ev = np.stack([np.sin(2 * np.pi * t / p) for p in (37, 53, 71, 97, 131, 173)], 1)
    eeg = r.normal(size=(n, N_EEG))
    w = np.random.default_rng(99).normal(size=(N_EV, 5))             # the same weights in both splits
    res = ev_weight * ev @ w * 0.3 + eeg_weight * eeg[:, :1] @ np.ones((1, 5)) * 0.3
    res = res + 0.1 * r.normal(size=(n, 5))
    k = np.full((n, 5), 3.0)
    lab = np.array(PHASES)[(t // 50) % len(PHASES)].astype(object)
    return k, k + res, res, ev, eeg, lab


def _scores(eeg_weight, ev_weight=1.0, seed=0, n_null=39):
    r = np.random.default_rng(seed)
    kv, yv, rv, evv, xv, _ = _split(r, 3000, eeg_weight, ev_weight)
    r = np.random.default_rng(seed + 100)
    kt, yt, rt, evt, xt, lab = _split(r, 3000, eeg_weight, ev_weight)
    df = C.b_vs_f_scores(xv, evv, rv, xt, evt, rt, lab, kt, yt, n_null=n_null, seed=0, cue_val=evv[:, :2], y_val=yv,
                         cue_test=evt[:, :2])
    return df, (xv, evv, rv, yv, xt, evt, rt, lab, kt, yt)


def test_eeg_information_beyond_events_gives_a_positive_eeg_skill():
    df, _ = _scores(eeg_weight=1.0)
    assert list(df.columns) == C.COLUMNS and list(df.segment) == list(PHASES) + ["all"]
    row = df.set_index("segment").loc["all"]
    assert row.eeg_skill > 0.5 and row.p <= 0.05 and row.vaf_gain > 5 and row.ccc_F > row.ccc_B
    assert row.eeg_skill == pytest.approx(1 - row.mse_F / row.mse_B)
    assert row.mse_F < row.mse_B < row.mse_K and np.isfinite(row.mse_causal_cue)
    assert (df.dropna().eeg_skill > 0.3).all()


def test_uninformative_eeg_gives_no_eeg_skill_and_the_null_keeps_the_event_term():
    df, _ = _scores(eeg_weight=0.0)
    row = df.set_index("segment").loc["all"]
    assert row.mse_B < 0.2 * row.mse_K and abs(row.eeg_skill) < 0.02 and abs(row.vaf_gain) < 1
    # Only the EEG term is shifted: the null stays near 0 instead of dropping by the event term's skill.
    assert abs(row.eeg_skill_null_mean) < 0.02


def test_b_and_f_are_phase_ridges_events_and_both():
    df, (xv, evv, rv, yv, xt, evt, rt, lab, kt, yt) = _scores(eeg_weight=1.0)
    ev = fit_and_score(evv, rv, lab, evt, rt, lab, kt, yt, fit="pooled", n_null=0, seed=0).set_index("segment")
    both = fit_and_score(np.concatenate([xv, evv], 1), rv, lab, np.concatenate([xt, evt], 1), rt, lab, kt, yt,
                         fit="pooled", n_null=0, seed=0).set_index("segment")
    df = df.set_index("segment")
    for ph in ("all", "reach", "hold"):
        assert 1 - df.loc[ph, "mse_B"] / df.loc[ph, "mse_K"] == pytest.approx(ev.loc[ph, "skill"])
        assert 1 - df.loc[ph, "mse_F"] / df.loc[ph, "mse_K"] == pytest.approx(both.loc[ph, "skill"])


def test_secondary_metrics_match_standard_metrics():
    df, (xv, evv, rv, yv, xt, evt, rt, lab, kt, yt) = _scores(eeg_weight=1.0)
    pred_b = np.maximum(kt + _fit(evv, rv)(evt), 0)
    pred_f = np.maximum(kt + _fit(np.concatenate([xv, evv], 1), rv)(np.concatenate([xt, evt], 1)), 0)
    names = list("abcde")
    mb, mf = standard_metrics(pred_b, yt, names), standard_metrics(pred_f, yt, names)
    row = df.set_index("segment").loc["all"]
    assert row.ccc_B == pytest.approx(mb.ccc.mean()) and row.ccc_F == pytest.approx(mf.ccc.mean())
    assert row.vaf_gain == pytest.approx(mf.vaf.mean() - mb.vaf.mean())
    cue = np.maximum(_fit(evv[:, :2], yv)(evt[:, :2]), 0)
    assert row.mse_causal_cue == pytest.approx(((cue - yt) ** 2).mean())


def test_fit_terms_matches_phase_ridge_and_splits_the_eeg_term():
    r = np.random.default_rng(1)
    x, y = r.normal(size=(500, 7)), r.normal(size=(500, 5))
    x[:, 5] = 1.0                                                   # a constant column is dropped by both
    z = r.normal(size=(200, 7)) * 3                                 # outside the fit range: clipped by both
    pred, term = C._fit_terms(x, y, n_eeg=3)(z)
    np.testing.assert_allclose(pred, _fit(x, y)(z))
    _, none = C._fit_terms(x, y, n_eeg=0)(z)
    assert np.all(none == 0.0)
    full, every = C._fit_terms(x, y, n_eeg=7)(z)
    np.testing.assert_allclose(full - every, np.broadcast_to(full[0] - every[0], full.shape), atol=1e-10)  # intercept


def test_channel_columns_follow_the_channel_major_layout():
    cols = C.channel_columns(CONFIG_V3, PERIPHERAL)
    assert len(cols) == 4 * len(PERIPHERAL) and list(cols[:8]) == list(range(8))        # Fp1 and Fp2 come first
    assert list(C.channel_columns(CONFIG_V3, ["C3"])) == [12 * 4 + b for b in range(4)]
    with pytest.raises(ValueError):
        C.channel_columns(preset_config("central"), PERIPHERAL)


class _Stub(torch.nn.Module):
    """A causal EEG stream whose tokens are the four bands of electrode 0, token by token."""
    causal = True

    def eeg(self, x):
        return None, x[:, :, 0, :]


def _toks(r, n_series=2, n_tok=1200):
    toks, lifts = [], []
    for s in range(n_series):
        l = [{"led_on": t * Q, "CurW": 1 + i % 2, "CurS": 1} for i, t in enumerate(range(30, n_tok - 100, 150))]
        toks.append({"series": s, "eeg": r.normal(size=(n_tok, 32, 4)).astype(np.float32),
                     "kin": np.zeros((n_tok, 24), np.float32)})
        lifts.append(l)
    return toks, lifts


def _norm():
    return {"eeg_mean": np.zeros((32, 4), np.float32), "eeg_std": np.ones((32, 4), np.float32),
            "kin_mean": np.zeros(24, np.float32), "kin_std": np.ones(24, np.float32), "emg_scale": np.ones(5, np.float32)}


def test_eeg_sources_swap_and_peripheral():
    r = np.random.default_rng(0)
    (tv, lv), (tt, lt) = _toks(r), _toks(r)
    cfg = copy.deepcopy(CONFIG_V3)
    out = C.eeg_sources(_Stub(), tv, tt, lv, lt, _norm(), cfg, torch.device("cpu"), rng=np.random.default_rng(5))
    assert set(out) == set(C.SOURCES)
    real = np.concatenate([t["eeg"][:, 0, :] for t in tt])
    np.testing.assert_allclose(out["eeg"][1], real, atol=1e-6)
    rng = np.random.default_rng(5)                                  # the default swap matches weight and surface
    alt_v, _ = swap_series([t["eeg"] for t in tv], lv, rng, match_condition=True)
    alt_t, _ = swap_series([t["eeg"] for t in tt], lt, rng, match_condition=True)
    np.testing.assert_allclose(out["swap"][0], np.concatenate([a[:, 0, :] for a in alt_v]), atol=1e-6)
    np.testing.assert_allclose(out["swap"][1], np.concatenate([a[:, 0, :] for a in alt_t]), atol=1e-6)
    assert np.abs(out["swap"][1] - real).mean() > 0.5                # another trial's EEG
    assert out["bandpower"][1].shape == (len(real), 128)
    np.testing.assert_allclose(out["peripheral"][1], out["bandpower"][1][:, C.channel_columns(cfg, PERIPHERAL)])
    with pytest.raises(ValueError):
        C.eeg_sources(_Stub(), tv, tt, lv, lt, _norm(), cfg, torch.device("cpu"), sources=("eeg", "kin"))


def test_score_b_vs_f_glue(monkeypatch):
    r = np.random.default_rng(2)
    (tv, lv), (tt, lt) = _toks(r), _toks(r)

    def fake_collect(model, toks, norm, cfg, device, p=None):
        n = sum(len(t["eeg"]) for t in toks)
        x = np.concatenate([t["eeg"][:, 0, :] for t in toks]).astype(np.float64)
        k = np.full((n, 5), 2.0)
        lab = np.array(PHASES)[(np.arange(n) // 40) % len(PHASES)].astype(object)
        return {"K": k, "y": k + 0.3 * x[:, :1] + 0.05 * r.normal(size=(n, 5)), "x": x, "labels": lab,
                "lifts": lv if toks is tv else lt, "n_tok": [len(t["eeg"]) for t in toks]}

    monkeypatch.setattr(C, "collect", fake_collect)
    sd = {"tokens": {"val": tv, "test": tt}, "norm": _norm()}
    df = C.score_b_vs_f(_Stub(), sd, copy.deepcopy(CONFIG_V3), torch.device("cpu"), 3, 500, n_null=9)
    assert list(df.columns) == ["subject", "horizon_ms", "source"] + C.COLUMNS and set(df.source) == set(C.SOURCES)
    a = df[df.segment == "all"].set_index("source")
    assert a.loc["eeg", "eeg_skill"] > 0.5 and a.loc["swap", "eeg_skill"] < 0.05     # the swap loses the trial's EEG
    assert a.loc["phase_swap", "eeg_skill"] < 0.05                                      # so does the phase-matched swap
    assert a.mse_causal_cue.notna().all() and (a.mse_B == a.mse_B.iloc[0]).all()       # B does not depend on the source
    assert (df.subject == "P3").all() and (df.horizon_ms == 500).all()


def _group_frame(gap=0.02):
    rows = []
    j = np.linspace(-0.004, 0.004, 12)
    for h in (0, 500):
        for seg in ("reach", "hold", "all"):
            for src, d in (("eeg", 0.03), ("swap", 0.03 - gap)):
                for i in range(12):
                    row = dict.fromkeys(C.METRICS, 0.1)
                    row.update(subject=f"P{i + 1}", horizon_ms=h, segment=seg, source=src, eeg_skill=d + j[i], p=0.01)
                    rows.append(row)
    return pd.DataFrame(rows)


def test_summarise_b_vs_f_gap_and_holm():
    g = C.summarise_b_vs_f(_group_frame())
    for col in ("eeg_skill_mean", "eeg_skill_sem", "vaf_gain_mean", "ccc_B_mean", "mse_causal_cue_sem", "n_sig",
                "p_eeg_skill", "p_eeg_skill_holm_phases", "p_eeg_skill_holm_horizons", "gap_mean", "p_gap_holm_phases"):
        assert col in g.columns, col
    eeg = g[g.source == "eeg"]
    assert eeg.p_eeg_skill.lt(0.01).all() and eeg.gap_mean.isna().all() and (eeg.n_sig == 12).all()
    sw = g[(g.source == "swap") & (g.segment == "hold")]
    assert sw.gap_mean.to_numpy() == pytest.approx([0.02, 0.02]) and sw.p_gap.lt(0.01).all()
    # Holm over the 2 phases of one horizon (not "all"), and over the 2 horizons of one phase ("all" included).
    ph = eeg[eeg.segment != "all"]
    assert ph.p_eeg_skill_holm_phases.to_numpy() == pytest.approx(np.minimum(1, 2 * ph.p_eeg_skill.to_numpy()))
    assert eeg[eeg.segment == "all"].p_eeg_skill_holm_phases.isna().all()
    assert eeg.p_eeg_skill_holm_horizons.to_numpy() == pytest.approx(np.minimum(1, 2 * eeg.p_eeg_skill.to_numpy()))


def test_config_has_the_b_vs_f_block():
    cc = CONFIG_V3["b_vs_f"]
    assert cc["horizons_ms"] == [0, 200, 500, 1000] and tuple(cc["sources"]) == C.SOURCES
    assert cc["checkpoint_dir"].endswith("checkpoints_v2_horizon") and cc["results_dir"] == "results/v3"
