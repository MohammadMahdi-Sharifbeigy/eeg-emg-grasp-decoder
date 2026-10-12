import copy

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
    cfg["interp"].update(events=["lift_off"], max_events=1, n_perm=20, n_random=2, deletion_fractions=[0.1, 0.3], flow_bin=10)
    tok = _tok(T=13000); norm = D.fit_norm([tok])
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    with torch.no_grad():
        m.fusion.gate.fill_(0.3)
    res = interpret_subject(m, [tok], norm, cfg, torch.device("cpu"), 1, NAMES, out_dir=tmp_path, log=lambda *_: None)
    t = res["tables"]
    assert set(t) == set(TABLES) and all(len(df) for df in t.values())
    assert set(t["time"]["view"]) == {"rollout", "flow", "relevance"} and set(t["time"]["output"]) == {"K", "E", "F_eeg"}
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
    assert {"K", "E", "F", "K_base", "gain", "F_eeg"} <= set(t["parts"].columns) and len(t["parts"]) == 5
    assert len(t["muscle_graph"]) == 2 * 2 * 5 * 5                                                     # K and E, two views
    assert all((tmp_path / f"{name}.csv").exists() for name in TABLES) and (tmp_path / "layers.npz").exists()
    pngs = {p.stem for p in tmp_path.glob("*.png")}
    assert {"time_K_lift_off", "time_E_lift_off", "time_F_eeg_lift_off", "layers_K_lift_off", "scalp_E_lift_off",
            "scalp_F_eeg_lift_off", "electrode_layers_lift_off", "muscle_graph_lift_off", "rest_contrast_lift_off",
            "inputs_lift_off", "deletion"} <= pngs
    assert len(res["files"]) == len(pngs) and plt.get_fignums() == []
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
