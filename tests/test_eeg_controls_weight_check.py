"""Part W of the EEG controls, phase 1: does the EEG-only output E give different levels per lifted weight?"""
import numpy as np
import pandas as pd
import pytest

from eeg_controls.weight_check import calibrate, group_tests, lift_levels, weight_stats

FS = 500.0


def _lift(i, w, t0, **kw):
    ev = dict(lift=i, CurW=w, first_touch=t0, lift_off=t0 + 500, replace=t0 + 2000)
    ev.update(kw)
    return ev


def _series(lifts, loads, holds, n=9000):
    """(n, 2) target with a plateau per lift: `loads[i]` in first_touch..lift_off, `holds[i]` in the hold window;
    everything else (including the guard bands next to the windows) is 99 so a wrong window edge shows."""
    y = np.full((n, 2), 99.0)
    for lf, a, b in zip(lifts, loads, holds):
        y[lf["first_touch"]:lf["lift_off"]] = a
        y[lf["lift_off"] + 250:lf["replace"] - 100] = b
    return y


def test_lift_levels_windows():
    lifts = [_lift(0, 1, 500), _lift(1, 4, 4000)]
    y = _series(lifts, [1.0, 3.0], [2.0, 6.0])
    pred = {"K": y * 0.5, "E": y * 0.25, "F": y * 2.0}
    df = lift_levels(pred, y, lifts)
    assert len(df) == 2 * 2 * 2                                   # lifts x muscles x windows
    assert set(df.columns) >= {"lift", "weight", "window", "muscle", "target", "K", "E", "F"}
    row = lambda i, w, m: df[(df.lift == i) & (df.window == w) & (df.muscle == m)].iloc[0]
    for m in (0, 1):
        assert row(0, "load", m).target == pytest.approx(1.0)
        assert row(0, "hold", m).target == pytest.approx(2.0)
        assert row(1, "load", m).target == pytest.approx(3.0)
        assert row(1, "hold", m).target == pytest.approx(6.0)
        assert row(1, "hold", m).E == pytest.approx(1.5)
        assert row(1, "hold", m).K == pytest.approx(3.0)
        assert row(1, "hold", m).F == pytest.approx(12.0)
        assert row(1, "hold", m).weight == 4


def test_missing_event_or_weight_skipped():
    lifts = [_lift(0, 1, 500), _lift(1, 2, 3000, lift_off=None), _lift(2, None, 5000), _lift(3, 4, 6500, replace=None),
             _lift(4, 2, 100, first_touch=None), _lift(5, 4, 8000, replace=8100)]    # last: hold window empty
    y = _series(lifts[:1], [1.0], [2.0])
    pred = {k: y for k in ("K", "E", "F")}
    df = lift_levels(pred, y, lifts)
    win = lambda i: set(df[df.lift == i].window)
    assert set(df.lift) == {0, 3, 4, 5}                         # 1 (no lift_off) and 2 (no weight) are skipped
    assert win(0) == {"load", "hold"}
    assert win(3) == {"load"}                                   # no replace: no hold window
    assert win(4) == {"hold"}                                   # no first_touch: no load window
    assert win(5) == {"load"}                                   # replace before the hold window starts: empty


def test_calibration_recovers_gain():
    rng = np.random.default_rng(0)
    def lv(n):
        t = rng.uniform(0.5, 3.0, (n, 2))
        rows = [dict(lift=i, weight=1, window="hold", muscle=m, target=t[i, m], K=t[i, m], E=t[i, m] / 3.0, F=t[i, m])
                for i in range(n) for m in range(2)]
        return pd.DataFrame(rows)
    out = calibrate(lv(12), lv(8))
    np.testing.assert_allclose(out.E_cal, out.target)
    np.testing.assert_allclose(out.E, out.target / 3.0)       # raw E kept


def test_calibration_gain_is_per_muscle():
    v = pd.DataFrame(dict(lift=[0, 1, 0, 1], weight=1, window="hold", muscle=[0, 0, 1, 1],
                          target=[2.0, 4.0, 2.0, 4.0], K=0.0, E=[1.0, 2.0, 0.5, 1.0], F=0.0))
    out = calibrate(v, v)
    np.testing.assert_allclose(out.E_cal, out.target)


def test_weight_stats_on_planted_weights():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(18):
        w = (1, 2, 4)[i % 3]
        for m in (0, 1):
            rows.append(dict(lift=i, weight=w, window="hold", muscle=m, target=w * 1.0 + rng.normal(0, 0.05),
                             K=w * 1.0, E=1.0 + rng.normal(0, 0.05), F=w * 1.0))
    st = weight_stats(pd.DataFrame(rows)).set_index(["window", "muscle", "output"])
    t, e = st.loc[("hold", 0, "target")], st.loc[("hold", 0, "E")]
    assert t.rho_weight > 0.9 and abs(e.rho_weight) < 0.5
    assert t.ratio_heavy_light == pytest.approx(4.0, rel=0.1)
    assert e.ratio_heavy_light == pytest.approx(1.0, rel=0.1)
    assert abs(e.slope) < 0.1 and t.slope == pytest.approx(1.0)
    assert e.size_ratio == pytest.approx(1.0 / 2.33, rel=0.1)
    assert {"level_w1", "level_w2", "level_w4"} <= set(st.columns)


def test_group_tests_output():
    rows = []
    for p in range(1, 9):
        for o, lo, hi, rho in (("target", 1.0, 4.0, 0.9), ("E", 1.0, 1.0 + 0.1 * p, 0.1 * p)):
            rows.append(dict(participant=p, window="hold", muscle=0, output=o, level_w1=lo, level_w2=2.0, level_w4=hi,
                             ratio_heavy_light=hi / lo, rho_weight=rho, slope=0.0, size_ratio=1.0))
    g = group_tests(pd.DataFrame(rows))
    r = g.set_index(["window", "test"])
    assert r.loc[("hold", "rho_E_gt_0"), "n"] == 8
    assert 0.0 < r.loc[("hold", "rho_E_gt_0"), "p"] < 0.05
    assert r.loc[("hold", "ratio_E_lt_target"), "p"] < 0.05
    assert group_tests(pd.DataFrame(rows[:2])).p.isna().all()          # one participant: nan, no raise


def test_calibration_gain_is_per_muscle_and_window():
    """hold and load get their own gain: E is 1/2 of the target in hold and 1/4 in load."""
    rows = []
    for win, f in (("hold", 0.5), ("load", 0.25)):
        for m in (0, 1):
            for i, t in enumerate((1.0, 2.0, 3.0)):
                rows.append(dict(lift=i, weight=1, window=win, muscle=m, target=t * (m + 1), K=0.0,
                                 E=t * (m + 1) * f, F=0.0))
    v = pd.DataFrame(rows)
    out = calibrate(v, v)
    np.testing.assert_allclose(out.E_cal, out.target)
    test = v[v.window == "load"].iloc[:1].assign(window="other")     # a window without validation lifts: nan
    assert np.isnan(calibrate(v, test).E_cal).all()
