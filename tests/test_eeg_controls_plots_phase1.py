"""Figures of phase 1: each returns a Figure with the expected panels and titles that fit in the figure."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from eeg_controls import plots_phase1 as F

MUS = ["AD", "BR", "FD"]
RNG = np.random.default_rng(0)


def _check(fig, n_axes):
    assert isinstance(fig, Figure)
    assert len([a for a in fig.axes if a.get_label() != "<colorbar>"]) == n_axes
    fig.canvas.draw()
    width = fig.bbox.width
    texts = [fig._suptitle] + [t for a in fig.axes for t in (a.title, a._left_title, a._right_title)]
    texts = [t for t in texts if t is not None and t.get_text()]
    assert texts
    for t in texts:
        bb = t.get_window_extent()
        assert bb.x0 >= -1 and bb.x1 <= width + 1, t.get_text()
    import matplotlib.pyplot as plt
    plt.close(fig)


def _levels(n_part=3):
    rows = []
    for p in range(1, n_part + 1):
        for lift in range(9):
            w = (1, 2, 4)[lift % 3]
            for m in MUS:
                t = 0.02 * w + 0.01 * RNG.random()
                rows.append(dict(participant=p, series=4, lift=lift, weight=w, window="hold", muscle=m, target=t,
                                 K=t * 0.8, E=0.03 + 0.003 * RNG.random(), F=t, E_cal=0.5 * t + 0.01))
    return pd.DataFrame(rows)


def _stats(n_part=4):
    rows = []
    for p in range(1, n_part + 1):
        for win in ("hold", "load"):
            for m in MUS:
                for o in ("target", "K", "E", "E_cal", "F"):
                    base = 1.0 if o == "target" else RNG.random()
                    rows.append(dict(window=win, muscle=m, output=o, level_w1=0.02, level_w2=0.03 if o == "target" else 0.025,
                                     level_w4=0.04 * (1 + 0.2 * base), ratio_heavy_light=1 + base,
                                     rho_weight=0.8 if o == "target" else RNG.normal(0.1, 0.2),
                                     slope=1.0 if o == "target" else RNG.normal(0.2, 0.1), size_ratio=1.0,
                                     participant=p))
    return pd.DataFrame(rows)


def _ridge_summary():
    rows = []
    for f in ("eeg", "events", "both"):
        for fit in ("pooled", "per_phase"):
            for seg in ("pre_move", "reach", "load", "hold", "release", "other", "all"):
                for h in (0, 200, 500):
                    rows.append(dict(horizon_ms=h, segment=seg, features=f, fit=fit, n=5, skill_mean=RNG.normal(0.05, 0.1),
                                     r_mean=0.2, n_sig=int(RNG.integers(0, 5)), gain_both_minus_events=0.0,
                                     p_both_gt_events=0.5))
    return pd.DataFrame(rows)


def _phase_ridge(n_part=4):
    rows = []
    for s in range(1, n_part + 1):
        for f in ("eeg", "events", "both"):
            for fit in ("pooled", "per_phase"):
                for seg in ("pre_move", "reach", "load", "hold"):
                    for h in (0, 200, 500):
                        rows.append(dict(subject=f"P{s}", horizon_ms=h, features=f, fit=fit, segment=seg,
                                         r=0.2, skill=RNG.normal(0.05, 0.1), skill_null_mean=0.0, p=0.5))
    return pd.DataFrame(rows)


def _conditions(n_part=5):
    rows = []
    for c in ("all", "central", "edge", "eye_clean"):
        for h in (0, 200, 500):
            for s in range(1, n_part + 1):
                rows.append(dict(condition=c, horizon_ms=h, subject=f"P{s}", E_r=RNG.normal(0.2, 0.05),
                                 F_minus_K_vaf=RNG.normal(0, 1), res_r_tokens=0.05, pre_move_r=RNG.normal(0.1, 0.05),
                                 reach_r=RNG.normal(0.1, 0.05), pre_move_skill=0.1, reach_skill=0.1))
    return pd.DataFrame(rows)


def test_weight_levels():
    _check(F.weight_levels(_stats()), len(MUS))


def test_weight_scatter():
    _check(F.weight_scatter(_levels()), 2)


def _example(n=30000):
    names = ["AD", "BR", "FD", "EC", "BB"]
    lifts = [dict(lift=i, CurW=(1, 2, 4)[i % 3], first_touch=2000 + 6000 * i, lift_off=2500 + 6000 * i,
                  replace=6500 + 6000 * i) for i in range(4)]
    lifts.append(dict(lift=9, CurW=None, first_touch=None, lift_off=None, replace=None))
    arr = lambda: np.abs(RNG.normal(0.03, 0.01, (n, 5)))
    return dict(target=arr(), K=arr(), E=arr(), F=arr(), lifts=lifts, names=names)


def test_weight_example_with_gain():
    ex = _example()
    fig = F.weight_example(ex, ex["names"], start_s=0, length_s=30, gain={m: 0.8 for m in ex["names"]})
    _check(fig, 5)
    assert len(fig.axes[0].lines) == 3          # target, E, E_cal


def test_weight_example_without_gain_and_auto_start():
    ex = _example()
    fig = F.weight_example(ex, ex["names"], length_s=30)
    assert len(fig.axes[0].lines) == 2
    _check(fig, 5)


def test_pick_stretch_finds_distinct_weights():
    ex = _example()
    s = F._pick_stretch(ex["lifts"], 40.0, 500.0)
    inside = [lf for lf in ex["lifts"] if lf["CurW"] and lf["lift_off"] / 500 >= s and lf["replace"] / 500 <= s + 40]
    assert len({lf["CurW"] for lf in inside}) == 3


def test_gain_from_levels():
    lv = _levels()
    lv["E_cal"] = lv.E * lv.muscle.map({"AD": 2.0, "BR": 3.0, "FD": 4.0})
    assert F.gain_from_levels(lv) == pytest.approx({"AD": 2.0, "BR": 3.0, "FD": 4.0})


def test_weight_participants():
    _check(F.weight_participants(_stats()), 2)


def test_ridge_heatmaps():
    _check(F.ridge_heatmaps(_ridge_summary()), 6)


def test_ridge_lines():
    _check(F.ridge_lines(_ridge_summary()), 2)


def test_ridge_gain():
    _check(F.ridge_gain(_phase_ridge()), 3)


def test_conditions_overview():
    _check(F.conditions_overview(_conditions()), 2)


def test_conditions_phases():
    _check(F.conditions_phases(_conditions()), 2)


def test_conditions_participants():
    fig = F.conditions_participants(_conditions())                  # no all_frozen yet: falls back to all, says so
    assert F.CONDITION_LABELS["all"] in fig._suptitle.get_text()
    _check(fig, 3)


def _ridge_summary_with_none():
    d = _ridge_summary()
    none = d[d.features == "eeg"].assign(features="none", skill_mean=0.02)
    d = pd.concat([none, d], ignore_index=True)
    return d.assign(skill_sem=0.01)


def _all_texts(fig):
    return [t for a in fig.axes for t in a.texts] + list(fig.texts)


def test_ridge_heatmaps_readable_and_labelled():
    s = _ridge_summary_with_none()
    s.loc[(s.features == "both") & (s.segment == "reach"), "skill_mean"] = 0.9      # a dark cell
    fig = F.ridge_heatmaps(s)
    axes = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(axes) == 8
    cells = [t for a in axes for t in a.texts]
    assert cells and min(t.get_fontsize() for t in cells) >= 7
    colours = {matplotlib.colors.to_hex(t.get_color()) for t in cells if t.get_text().startswith("+0.90")}
    assert colours == {"#ffffff"}
    cbar = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    label = cbar[0].get_ylabel() + cbar[0].get_xlabel()
    assert "MSE(K)" in label and "of the mean" not in label
    text = " ".join([fig._suptitle.get_text()] + [t.get_text() for t in _all_texts(fig)]
                    + [a.get_title(loc="left") for a in axes])
    assert "phase knowledge" in text
    _check(fig, 8)


def test_ridge_heatmaps_all_nan():
    s = _ridge_summary().assign(skill_mean=np.nan)
    _check(F.ridge_heatmaps(s), 6)


def test_ridge_lines_sem_and_black_fit_legend():
    fig = F.ridge_lines(_ridge_summary_with_none())
    leg = fig.legends[0]
    fit_lines = [h for h, t in zip(leg.legend_handles, leg.get_texts()) if t.get_text() in F.FIT_LABELS.values()]
    assert fit_lines and all(matplotlib.colors.to_hex(h.get_color()) == "#000000" for h in fit_lines)
    events = matplotlib.colors.to_hex(F.FEATURE_COLORS["events"])
    assert all(matplotlib.colors.to_hex(h.get_color()) != events for h in fit_lines)
    assert all(len(a.collections) > 0 for a in fig.axes)                  # SEM bands
    text = " ".join([fig._suptitle.get_text(), leg.get_title().get_text()] + [t.get_text() for t in _all_texts(fig)])
    assert "phase knowledge" in text
    _check(fig, 2)


def test_conditions_overview_shows_all_frozen():
    fig = F.conditions_overview(_conditions_with_frozen())
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert F.CONDITION_LABELS["all_frozen"] in labels and F.CONDITION_LABELS["all"] in labels
    _check(fig, 2)


def _conditions_with_frozen():
    d = _conditions()
    return pd.concat([d, d[d.condition == "all"].assign(condition="all_frozen")], ignore_index=True)


def test_conditions_participants_against_all_frozen():
    fig = F.conditions_participants(_conditions_with_frozen())
    assert F.CONDITION_LABELS["all_frozen"] in fig._suptitle.get_text()
    assert "all 32 (" not in fig._suptitle.get_text() or "frozen" in fig._suptitle.get_text()
    assert fig.axes[0].get_xlabel().endswith(F.CONDITION_LABELS["all_frozen"])
    _check(fig, 3)
    fig = F.conditions_participants(_conditions_with_frozen(), reference="all")
    assert F.CONDITION_LABELS["all"] in fig._suptitle.get_text()
    _check(fig, 4)                                                   # all_frozen is compared with all too


def test_conditions_figures_without_condition_runs():
    only_all = _conditions()
    only_all = only_all[only_all.condition == "all"]
    for fig in (F.conditions_participants(only_all), F.conditions_participants(only_all.iloc[:0])):
        assert any("no condition runs yet" in t.get_text() for t in _all_texts(fig))
        _check(fig, 1)
