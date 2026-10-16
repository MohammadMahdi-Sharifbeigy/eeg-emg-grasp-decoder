"""Part R of the EEG controls, phase 1: causal past-event features and the per-phase ridge."""
import numpy as np
import pytest

from eeg_controls.common import EVENTS, wilcoxon_p
from eeg_controls.phase_ridge import LAGS, PHASES, fit_and_score, past_event_features
from v2.tokens import FS_IN, Q

FS_TOK = FS_IN / Q          # 50 tokens per second


def _lift(**samples):
    return {ev: samples.get(ev) for ev in EVENTS}


def test_past_events_are_causal():
    n = 400
    base = _lift(led_on=1000, hand_start=1500, first_touch=2000, lift_off=2600)
    x = past_event_features([base], n)
    assert x.shape == (n, len(EVENTS) * len(LAGS))

    # Every row equals the row computed with the events after it removed.
    for t in (0, 99, 100, 149, 150, 230, 399):
        cut = {ev: (s if s is not None and s <= t * Q else None) for ev, s in base.items()}
        np.testing.assert_array_equal(x[t], past_event_features([cut], n)[t])

    # Moving an event later by less than one bump width (0.12 s < 0.25 s) leaves earlier rows unchanged.
    j = EVENTS.index("hand_start")
    blk = slice(j * len(LAGS), (j + 1) * len(LAGS))
    moved = past_event_features([dict(base, hand_start=1500 + 60)], n)
    np.testing.assert_array_equal(x[:150], moved[:150])
    assert np.all(moved[:156, blk] == 0.0)            # nothing before the moved event, not even a tail
    assert moved[156, blk][0] == pytest.approx(1.0)   # the lag-0 bump peaks at the event token
    assert np.all(x[:150, blk] == 0.0) and x[150, blk][0] == pytest.approx(1.0)

    # A lift with a missing event gives zeros for that event's block (and does not raise).
    k = EVENTS.index("replace")
    assert np.all(x[:, k * len(LAGS):(k + 1) * len(LAGS)] == 0.0)
    assert np.all(past_event_features([{"CurW": 2}], n) == 0.0)


def test_past_events_shape_and_lags():
    n = 400
    x = past_event_features([_lift(lift_off=100 * Q)], n)
    assert x.shape == (n, 8 * 13)
    j = EVENTS.index("lift_off")
    blk = x[:, j * 13:(j + 1) * 13]
    assert np.all(np.delete(x, np.s_[j * 13:(j + 1) * 13], axis=1) == 0.0)
    for i, lag in enumerate(np.arange(0.0, 3.0 + 1e-9, 0.25)):
        assert abs(int(np.argmax(blk[:, i])) - (100 + lag * FS_TOK)) <= 1.0
    # offset_tok shifts the token axis: row t is series token t + offset
    shifted = past_event_features([_lift(lift_off=100 * Q)], n, offset_tok=25)
    np.testing.assert_array_equal(shifted[:n - 25], x[25:])


def _synthetic(rng, n, small_phase=None):
    phases = [ph for ph in PHASES]
    lab = np.array([phases[(i // 50) % len(phases)] for i in range(n)], dtype=object)
    if small_phase is not None:
        idx = np.flatnonzero(lab == small_phase)
        lab[idx[5:]] = "other"
    x = rng.standard_normal((n, 6))
    res = 0.05 * rng.standard_normal((n, 5))
    m = lab == "reach"
    res[m] += 0.5 * x[m, :1]
    k = np.full((n, 5), 3.0)
    return x, res, lab, k, k + res


def test_per_phase_fit_handles_small_phase():
    rng = np.random.default_rng(0)
    xv, rv, lv, _, _ = _synthetic(rng, 3000, small_phase="pre_move")
    xt, rt, lt, kt, yt = _synthetic(rng, 3000)
    assert (lv == "pre_move").sum() == 5
    df = fit_and_score(xv, rv, lv, xt, rt, lt, kt, yt, fit="per_phase", n_null=5, seed=1)
    assert list(df.columns) == ["segment", "fit", "n_fit", "n_test", "r", "skill", "skill_null_mean", "p"]
    row = df.set_index("segment")
    assert np.isnan(row.loc["pre_move", "skill"]) and np.isnan(row.loc["pre_move", "r"])
    assert row.loc["pre_move", "n_fit"] == 5
    others = row.drop(index="pre_move")
    assert np.isfinite(others[["r", "skill", "skill_null_mean", "p"]].to_numpy(float)).all()
    assert set(df.fit) == {"per_phase"}


def test_pooled_and_per_phase_on_planted_signal():
    rng = np.random.default_rng(1)
    xv, rv, lv, _, _ = _synthetic(rng, 3000)
    xt, rt, lt, kt, yt = _synthetic(rng, 3000)
    pooled = fit_and_score(xv, rv, lv, xt, rt, lt, kt, yt, fit="pooled", n_null=5, seed=1).set_index("segment")
    per = fit_and_score(xv, rv, lv, xt, rt, lt, kt, yt, fit="per_phase", n_null=5, seed=1).set_index("segment")
    assert per.loc["reach", "skill"] > 0.9
    assert per.loc["reach", "skill"] > pooled.loc["reach", "skill"]
    assert per.loc["reach", "skill"] > per.loc["reach", "skill_null_mean"]
    assert pooled.loc["all", "n_fit"] == 3000 and per.loc["reach", "n_fit"] == (lv == "reach").sum()
    with pytest.raises(ValueError):
        fit_and_score(xv, rv, lv, xt, rt, lt, kt, yt, fit="nope", n_null=5, seed=1)


def test_wilcoxon_p():
    assert np.isnan(wilcoxon_p([]))
    assert np.isnan(wilcoxon_p([0.3]))
    assert np.isnan(wilcoxon_p([0.0, 0.0, 0.2, np.nan]))             # one finite non-zero value left
    assert np.isfinite(wilcoxon_p([0.1, 0.2]))
    pos = np.arange(1, 13) / 10.0
    assert wilcoxon_p(pos) < 0.01
    assert wilcoxon_p(-pos) > 0.99
    assert wilcoxon_p(-pos, alternative="less") < 0.01


def test_per_phase_fit_does_not_extrapolate():
    # A bump column that barely moves in one phase's validation tokens but is large in its test tokens
    # (a past event seen far more often in the test series) must not blow the prediction up.
    rng = np.random.default_rng(2)
    xv, rv, lv, _, _ = _synthetic(rng, 3000)
    xt, rt, lt, kt, yt = _synthetic(rng, 3000)
    xv, xt = np.c_[xv, np.zeros(len(xv))], np.c_[xt, np.zeros(len(xt))]
    mv, mt = lv == "other", lt == "other"
    xv[np.flatnonzero(mv)[:3], -1] = 1e-4 * np.array([1.0, 2.0, 3.0])
    rv[np.flatnonzero(mv)[:3]] += 0.05
    xt[mt, -1] = 1.0
    df = fit_and_score(xv, rv, lv, xt, rt, lt, kt, yt, fit="per_phase", n_null=5, seed=1).set_index("segment")
    assert df.loc["other", "skill"] > -1.0
    assert df.loc["all", "skill"] > -1.0 and df.loc["reach", "skill_null_mean"] > -1.0


def test_none_feature_set_learns_each_phase_mean():
    """`none` has no columns: per phase it predicts the phase's mean residual (phase knowledge only)."""
    import eeg_controls.phase_ridge as PRm
    assert PRm.FEATURES[0] == "none"
    rng = np.random.default_rng(3)
    n = 600
    lab = np.array(["reach"] * 300 + ["hold"] * 300, dtype=object)
    res = np.where((lab == "reach")[:, None], 0.5, -0.2) + rng.normal(0, 0.05, (n, 2))
    k = np.full((n, 2), 1.0)
    y = k + res
    empty = np.zeros((n, 0))
    per = fit_and_score(empty, res, lab, empty, res, lab, k, y, fit="per_phase", n_null=3, seed=0,
                        window=4, hop=2).set_index("segment")
    pooled = fit_and_score(empty, res, lab, empty, res, lab, k, y, fit="pooled", n_null=3, seed=0,
                           window=4, hop=2).set_index("segment")
    assert per.loc["reach", "skill"] > 0.9 and per.loc["hold", "skill"] > 0.9    # the phase mean alone
    assert pooled.loc["all", "skill"] < per.loc["all", "skill"]


def test_summarise_gain_vs_none_for_per_phase():
    import pandas as pd
    from eeg_controls.phase_ridge import summarise
    rows = []
    for s in range(1, 6):
        for f, sk in (("none", 0.10), ("eeg", 0.15), ("events", 0.12), ("both", 0.20)):
            for fit in ("pooled", "per_phase"):
                rows.append(dict(subject=f"P{s}", horizon_ms=0, features=f, fit=fit, segment="reach",
                                 n_fit=50, n_test=50, r=0.1, skill=sk + 0.01 * s * (f != "none"),
                                 skill_null_mean=0.0, p=0.5))
    out = summarise(pd.DataFrame(rows)).set_index(["fit", "features"])
    assert out.loc[("per_phase", "eeg"), "gain_vs_none"] == pytest.approx(0.05 + 0.03)
    assert out.loc[("per_phase", "both"), "gain_vs_none"] == pytest.approx(0.10 + 0.03)
    assert 0 < out.loc[("per_phase", "both"), "p_gt_none"] < 0.1
    assert np.isnan(out.loc[("pooled", "both"), "gain_vs_none"])        # only for per_phase
    assert "skill_sem" in out.columns and out.loc[("per_phase", "eeg"), "skill_sem"] > 0
    assert out.loc[("per_phase", "both"), "gain_both_minus_events"] == pytest.approx(0.08)
