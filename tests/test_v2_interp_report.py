import copy
import warnings

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.electrodes import EEG_NAMES
from v2.interp import plots as P
from v2.interp.report import TABLES, _pairs, interpret_subject
from v2.model import build_model

NAMES = ["AD", "BR", "FD", "ED", "FDI"]


def test_view_colours_have_a_fixed_order():
    assert list(P.VIEW_COLORS) == list(P.VIEW_LABELS) == ["rollout", "flow", "relevance", "saliency"]
    assert len(set(P.VIEW_COLORS.values())) == 4


def test_every_plot_returns_a_figure():
    rng = np.random.default_rng(0)
    t = np.linspace(-4, 4, 50)
    curves = {"rollout": (t, rng.random((5, 50)), 0.1 * rng.random((5, 50))),
              "flow": (t[::5], rng.random((5, 10)), 0.1 * rng.random((5, 10))),
              "relevance": (t, rng.random((5, 50)), 0.1 * rng.random((5, 50)))}
    fig = P.plot_time_views(curves, NAMES, "title", "lift_off")
    assert len(fig.axes) == 5 and len(fig.legends) == 1 and len(fig.axes[0].lines) == 3 + 1          # three views + the event line
    fig = P.plot_layer_heads(rng.random((3, 4, 50)), t, "title", "lift_off")
    assert len(fig.axes) == 3 and len(fig.axes[0].lines) == 4 + 1 + 1                                # heads + mean + event line
    fig = P.plot_matrices({"a": rng.random((5, 5)), "b": rng.random((5, 5))}, "title", NAMES, NAMES)
    assert len(fig.axes) == 2 + 1                                                                    # two maps + one colour bar
    fig = P.plot_scalp({"rollout": rng.random(32), "relevance: AD": rng.random(32)}, EEG_NAMES, "title")
    assert len(fig.axes) == 2 * 2                                                                    # each map has its colour bar
    fig = P.plot_scalp({"AD": rng.standard_normal(32), "BR": np.zeros(32)}, EEG_NAMES, "title", signed=True,
                       masks={"AD": rng.random(32) > 0.8})
    assert len(fig.axes) == 2 * 2
    views = {v: (np.array([0.1, 0.2]), rng.random(2), rng.random(2)) for v in P.VIEW_COLORS}
    fig = P.plot_deletion({"E": views, "K": views}, "title")
    assert len(fig.axes) == 2 and len(fig.axes[0].lines) == 4 + 1 and len(fig.legends) == 1          # four views + random
    fig = P.plot_shares({"K": ([f"f{i}" for i in range(8)], rng.random(8)), "E": (["a", "b"], rng.random(2))}, "title")
    assert len(fig.axes) == 2
    plt.close("all")


def _look(fig):
    fig.canvas.draw()                                                  # ticks are made when the figure is drawn
    ax = fig.axes[0]
    labels = ax.get_xticklabels() + ax.get_yticklabels() + [fig._suptitle, ax.xaxis.label, ax.title]
    grid = ax.xaxis.get_gridlines()[0]
    return {"sizes": [l.get_fontsize() for l in labels], "fonts": {l.get_fontname() for l in labels},
            "ticks": [l.get_text() for l in labels],
            "face": ax.get_facecolor(), "figsize": tuple(fig.get_size_inches()), "dpi": fig.dpi,
            "grid": (grid.get_color(), grid.get_linewidth(), grid.get_linestyle(), grid.get_alpha()),
            "line": ax.lines[0].get_linewidth(), "tick_colour": ax.get_xticklabels()[0].get_color()}


def test_figures_do_not_depend_on_the_callers_style():
    import main.plots                                                  # sets a seaborn theme for the whole process
    rng = np.random.default_rng(0)
    t = np.linspace(-4, 4, 50)
    curves = {"rollout": (t, rng.random((5, 50)), 0.1 * rng.random((5, 50)))}
    with plt.rc_context():
        matplotlib.rcdefaults()
        plain = _look(P.plot_time_views(curves, NAMES, "title", "lift_off"))
    with plt.rc_context({**main.plots.PAPER_STYLE, "xtick.labelsize": 20, "ytick.labelsize": 20, "axes.facecolor": "grey",
                         "figure.figsize": (3, 3), "figure.dpi": 50, "font.family": "serif", "lines.linewidth": 5,
                         "grid.color": "red", "grid.linewidth": 3, "xtick.color": "green"}):
        themed = _look(P.plot_time_views(curves, NAMES, "title", "lift_off"))
    plt.close("all")
    assert themed == plain
    sizes = plain["sizes"]
    assert max(sizes[:-3]) < sizes[-2] <= sizes[-1] <= sizes[-3]       # tick labels < axis label <= panel title <= title


def _tok(T=12000, seed=0):
    rng = np.random.default_rng(seed)
    kin = rng.standard_normal((T, 26)).astype(np.float32); kin[:, 10:12] = 0
    return D.build_tokens({"series": 3, "eeg": rng.standard_normal((T, 32)).astype(np.float32), "kin": kin,
                           "emg": np.abs(rng.standard_normal((T, 5))).astype(np.float32)}, CONFIG_V2)


def test_interpret_subject_writes_tables_and_figures(tmp_path, monkeypatch):
    lifts = [{"lift": i + 1, "CurW": 1 + i % 2, "CurS": 1, "hand_start": 1500 + 3000 * i, "first_touch": 2000 + 3000 * i,
              "lift_off": 2300 + 3000 * i, "hand_stop": 3200 + 3000 * i} for i in range(4)]             # 4 lifts, 3 rests -> 3 pairs
    monkeypatch.setattr("main.interp.lift_events", lambda *a, **k: lifts)
    cfg = copy.deepcopy(CONFIG_V2)
    cfg["tokens"]["window"] = 100                                                                      # keep the CPU test quick
    cfg["interp"].update(events=["first_touch", "lift_off"], max_events=1, n_perm=20, n_random=2,
                         deletion_fractions=[0.1, 0.3], flow_bin=10)
    tok = _tok(T=13000); norm = D.fit_norm([tok])
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    with torch.no_grad():
        m.fusion.gate.fill_(0.3)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="More than 20 figures")               # two event types: 25 figures
        res = interpret_subject(m, [tok], norm, cfg, torch.device("cpu"), 1, NAMES, out_dir=tmp_path, log=lambda *_: None)
    assert res["figures"] == {} and plt.get_fignums() == []                            # each one saved and closed at once
    t = {k: df[df["event"] == "lift_off"] if "event" in df else df for k, df in res["tables"].items()}
    assert set(res["tables"]["time"]["event"]) == {"first_touch", "lift_off"}
    assert set(t) == set(TABLES) and all(len(df) for df in t.values())
    assert set(t["time"]["view"]) == {"rollout", "flow", "flow_excess", "relevance"} and set(t["time"]["output"]) == {"K", "E", "F_eeg"}
    assert set(t["electrodes"]["output"]) == {"E", "F_eeg"} and len(t["rest_contrast"]) == 5 * 32
    rc = t["rest_contrast"]
    assert set(rc["n_pairs"]) == {3} and set(rc["muscle"]) == set(NAMES)             # lift 4 has no rest after it
    assert np.isfinite(rc["diff"]).all() and rc["p"].dropna().between(1 / 21, 1).all()   # n_perm = 20
    e = t["electrodes"]
    one = e[(e["output"] == "E") & (e["view"] == "rollout") & (e["muscle"] == "AD")]
    assert len(one) == 32 and one["mean"].sum() == pytest.approx(1.0, abs=1e-4)
    k = t["time"][(t["time"]["output"] == "K") & (t["time"]["view"] == "rollout") & (t["time"]["muscle"] == "AD")]
    assert len(k) == 100 and k["mean"].sum() == pytest.approx(1.0, abs=1e-4) and k["time_s"].iloc[50] == 0.0
    assert set(t["faithfulness"]["view"]) == set(t["sanity"]["view"]) == {"rollout", "flow", "relevance", "saliency"}
    san = t["sanity"]
    assert (san["n"] == 1).all() and (san["spearman"].isna() == (san["n_nan"] == san["n"])).all()   # max_events = 1
    assert {"K", "E", "F", "K_base", "gain", "F_eeg"} <= set(t["parts"].columns) and len(t["parts"]) == 5
    assert len(t["muscle_graph"]) == 2 * 2 * 5 * 5                                                     # K and E, two views
    assert all((tmp_path / f"{name}.csv").exists() for name in TABLES) and (tmp_path / "layers.npz").exists()
    pngs = {p.stem for p in tmp_path.glob("*.png")}
    assert {"time_K_lift_off", "time_E_lift_off", "time_F_eeg_lift_off", "layers_K_lift_off", "scalp_E_lift_off",
            "scalp_F_eeg_lift_off", "electrode_layers_lift_off", "muscle_graph_lift_off", "rest_contrast_lift_off",
            "inputs_lift_off", "deletion", "time_K_first_touch", "rest_contrast_first_touch"} <= pngs
    assert len(pngs) == 25 and sorted(Path(f).stem for f in res["files"]) == sorted(pngs)
    with np.load(tmp_path / "layers.npz") as z:
        assert z["E__lift_off__time_layers"].shape == (2, 4, 100) and z["E__lift_off__electrode_layers"].shape == (2, 4, 32, 32)


def test_rest_is_paired_with_the_lift_before_it():
    import pandas as pd
    lift = pd.DataFrame({"series": [3, 3, 3, 4, 4], "lift": [1, 2, 3, 1, 2], "target": 0.0})
    rest = pd.DataFrame({"series": [3, 4, 4, 4], "lift": [2, 2, 1, 5], "target": 0.0})
    i, j = _pairs(lift, rest)
    assert sorted(zip(i.tolist(), j.tolist())) == [(1, 0), (3, 2), (4, 1)]
    i, j = _pairs(lift, pd.DataFrame())
    assert len(i) == len(j) == 0


def test_interpret_subject_reports_when_no_window_fits(monkeypatch):
    monkeypatch.setattr("main.interp.lift_events", lambda *a, **k: [])
    cfg = copy.deepcopy(CONFIG_V2)
    cfg["tokens"]["window"] = 100
    tok = _tok(); norm = D.fit_norm([tok])
    said = []
    res = interpret_subject(build_model(cfg).eval(), [tok], norm, cfg, torch.device("cpu"), 1, NAMES, log=said.append)
    assert all(len(df) == 0 for df in res["tables"].values()) and res["figures"] == {}
    assert any("no event window" in s for s in said)


def test_max_events_counts_the_events_whose_window_fits(tmp_path, monkeypatch):
    early = {"lift": 1, "CurW": 1, "CurS": 1, "hand_start": 50, "first_touch": 150, "lift_off": 300, "hand_stop": 700}
    lifts = [early] + [{"lift": i + 2, "CurW": 1, "CurS": 1, "hand_start": 1500 + 3000 * i, "first_touch": 2000 + 3000 * i,
                        "lift_off": 2300 + 3000 * i, "hand_stop": 3200 + 3000 * i} for i in range(2)]
    monkeypatch.setattr("main.interp.lift_events", lambda *a, **k: lifts)
    cfg = copy.deepcopy(CONFIG_V2)
    cfg["tokens"]["window"] = 100                                      # token 30 of the first lift: its window starts before 0
    cfg["interp"].update(events=["lift_off"], max_events=1, n_perm=20, n_random=2, deletion_fractions=[0.1], flow_bin=10)
    tok = _tok(T=9000); norm = D.fit_norm([tok])
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    res = interpret_subject(m, [tok], norm, cfg, torch.device("cpu"), 1, NAMES, log=lambda *_: None)
    plt.close("all")
    t = res["tables"]
    assert "flow" in set(t["time"]["view"])                            # the one event of the limit is the first that fits
    assert len(t["faithfulness"]) and (t["sanity"]["n"] == 1).all()
    assert (t["parts"]["n"] == 2).all()                                # the views without a limit still use both lifts that fit


def test_time_figure_with_own_share_and_a_flow_row():
    rng = np.random.default_rng(0)
    t = np.linspace(-4, 4, 50)
    mean = rng.random((5, 50))
    mean[:, 24:27] = np.nan                                                  # the explained token is left out
    curves = {"rollout": (t, mean, 0.1 * rng.random((5, 50))), "relevance": (t, mean * 0.5, 0.1 * rng.random((5, 50)))}
    flow = {"t": t[::5], "mean": rng.random(10) - 0.5, "sem": 0.05 * rng.random(10)}
    fig = P.plot_time_views(curves, NAMES, "title", "lift_off", flow=flow,
                            own_share={"rollout": np.full(5, 0.25), "relevance": np.full(5, 0.4)})
    assert len(fig.axes) == 6 and [ax.get_title(loc="left") for ax in fig.axes[:5]] == NAMES   # short titles: no overlap
    assert fig.axes[0].texts[0].get_text() == "own token: rollout 25%, grad-weighted 40%"
    assert fig.axes[0].get_ylabel() == "share among the other tokens"
    assert len(fig.axes[5].lines) == 3 and fig.axes[5].get_ylabel() == "flow above uniform"   # zero line, flow, event line
    plt.close(fig)


def test_flow_comes_with_the_uniform_attention_baseline():
    from v2.interp.explain import explain
    torch.manual_seed(0)
    m = build_model(CONFIG_V2).eval()
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1, 48, 24, generator=g), torch.rand(1, 48, 8, generator=g), torch.randn(1, 48, 32, 4, generator=g))
    for output in ("K", "E", "F_eeg"):
        f = explain(m, *x, output=output, t=20, flow=True, bin_size=4)["flow"]
        assert f["time_uniform"].shape == f["time"].shape and (f["time_uniform"] <= 1 + 1e-5).all()
        assert (f["time_uniform"] > 0.4).all()                              # equal attention already routes a lot of flow
        if output != "K":
            assert f["electrode_uniform"].shape == f["electrode"].shape and f["input_uniform"].shape == f["input"].shape
