"""Prediction-error pilot: labels, weight-matched contrast, decoder, partial correlation, cluster test, error size."""
import numpy as np
import pandas as pd
import pytest

from pe_pilot.analysis import error_size, learning
from pe_pilot.data import emg_envelope, label_lifts
from pe_pilot.stats import cluster_test, decode, matched_contrast, partial_spearman, time_adjacency_only


def _table(cur, prev, run=None, cur_s=None, prev_s=None):
    n = len(cur)
    return pd.DataFrame({"Run": run if run is not None else [1] * n, "Lift": np.arange(1, n + 1),
                         "CurW": cur, "PrevW": prev, "CurS": cur_s or [1] * n, "PrevS": prev_s or [1] * n,
                         "dGF_Max": np.arange(n, dtype=float), "rejected": False})


def test_label_lifts_change_and_next():
    t = label_lifts(_table([1, 4, 4, 1], [np.nan, 1, 4, 4], cur_s=[1, 1, 2, 1], prev_s=[np.nan, 1, 1, 2]))
    assert pd.isna(t["change"][0])                 # first lift of the series
    assert t["change"][1] == "heavier"
    assert pd.isna(t["change"][2])                 # the surface changed too
    assert pd.isna(t["change"][3])
    assert t["next_dGF"].tolist()[:3] == [1.0, 2.0, 3.0] and pd.isna(t["next_dGF"][3])


def test_label_lifts_next_stays_in_series():
    t = label_lifts(_table([1, 1, 4, 4], [np.nan, 1, np.nan, 4], run=[1, 1, 2, 2]))
    assert pd.isna(t["next_dGF"][1])               # last lift of series 1 has no next lift


def test_matched_contrast_cancels_weight():
    # the signal is 10 * weight + 1 on unexpected lifts: a pooled contrast would mix the weights,
    # the matched one must give exactly 1
    cur = [4] * 6 + [1] * 6
    prev = [1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 1]
    t = label_lifts(_table(cur, prev))
    t["rejected"] = False
    x = np.array([10.0 * c + (1.0 if ch in ("heavier", "lighter") else 0.0) for c, ch in zip(cur, t["change"])])
    d, n = matched_contrast(x[:, None], t, "heavier")
    assert n == 3 and d[0] == pytest.approx(1.0)
    d, n = matched_contrast(x[:, None], t, "lighter")
    assert n == 3 and d[0] == pytest.approx(1.0)


def test_matched_contrast_needs_both_sides():
    t = label_lifts(_table([4, 4, 4], [1, 1, 1]))
    t["rejected"] = False
    assert matched_contrast(np.ones((3, 1)), t, "heavier") == (None, 0)


def test_decode_finds_signal_and_not_noise():
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], 60)
    groups = np.repeat(np.arange(6), 20)
    X = rng.normal(size=(120, 5))
    good = decode(X + 1.5 * y[:, None], y, groups, n_shuffles=50)
    bad = decode(X, y, groups, n_shuffles=50)
    assert good["auc"] > 0.8 and good["p"] < 0.05
    assert bad["p"] > 0.05


def test_partial_spearman_removes_shared_cause():
    rng = np.random.default_rng(1)
    c = rng.normal(size=300)
    x, y = c + 0.1 * rng.normal(size=300), c + 0.1 * rng.normal(size=300)
    assert partial_spearman(x, y) > 0.9
    assert abs(partial_spearman(x, y, c[:, None])) < 0.2


def test_cluster_test_finds_planted_window():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 60, 2))
    X[:, 20:30, 0] += 2.0
    cl = cluster_test(X, time_adjacency_only(2), n_permutations=500)
    best = cl.iloc[0]
    assert best["p"] < 0.05 and best["channels"] == [0]
    assert 18 <= best["t_first"] <= 22 and 27 <= best["t_last"] <= 31


def test_emg_envelope_follows_burst():
    fs = 4000.0
    t = np.arange(0, 2, 1 / fs)
    emg = np.random.default_rng(3).normal(size=(len(t), 1)) * ((t > 1.0) & (t < 1.5))[:, None]
    eeg_t = np.arange(0, 2, 1 / 500)
    env = emg_envelope(emg, t, eeg_t)[:, 0]
    assert env[(eeg_t > 1.1) & (eeg_t < 1.4)].mean() > 10 * env[eeg_t < 0.8].mean()


def _epochs(n=40, seed=4):
    rng = np.random.default_rng(seed)
    names = ["C3", "Cz", "CP1", "CP5", "FC1", "FC5", "Pz"]
    times = np.round(np.arange(-0.5, 1.5, 0.01), 3)
    t = label_lifts(pd.DataFrame({"Run": 1, "Lift": np.arange(1, n + 1), "CurW": np.tile([4, 4], n // 2),
                                  "PrevW": np.tile([1, 4], n // 2), "CurS": 1, "PrevS": 1,
                                  "dGF_Max": rng.normal(50, 5, n), "GF_Max": rng.normal(5, 1, n),
                                  "Dur_LoadPhase": rng.normal(0.3, 0.05, n)}))
    t["rejected"] = False
    beta = rng.normal(size=(n, len(times), len(names)))
    return {"table": t, "beta": beta, "times": times, "ch_names": names,
            "emg": rng.random((n, len(times), 5))}


def test_error_size_uses_window_and_roi():
    ep = _epochs()
    ep["beta"][:] = 0.0
    tw = (ep["times"] >= 0.2) & (ep["times"] <= 0.6)
    ep["beta"][3][np.ix_(tw, [0, 1])] = 6.0        # two of six ROI channels
    ep["beta"][4][:, 6] = 100.0                    # Pz is not in the ROI
    s = error_size(ep)
    assert s[3] == pytest.approx(2.0) and s[4] == 0.0


def test_learning_recovers_planted_relation():
    eps = {}
    for p in range(1, 7):
        ep = _epochs(seed=p)
        t = ep["table"]
        unexpected = (t["change"] == "heavier").to_numpy()
        size = np.random.default_rng(10 + p).random(len(t)) * 5
        tw = (ep["times"] >= 0.2) & (ep["times"] <= 0.6)
        ep["beta"][:, tw, :6] = size[:, None, None]
        # next lift (same weight 4) changes its grip force rate by the error size
        dgf = t["dGF_Max"].to_numpy().copy()
        for i in np.flatnonzero(unexpected)[:-1]:
            dgf[i + 1] = dgf[i] + 3 * size[i]
        t["dGF_Max"] = dgf
        ep["table"] = label_lifts(t.drop(columns=["change", "next_dGF", "next_W"])).assign(rejected=False)
        eps[p] = ep
    per, group = learning(eps)
    assert (per["rho"] > 0.5).all()


def test_reject_outliers_is_relative():
    from pe_pilot.data import reject_outliers
    ptp = np.r_[np.linspace(280, 320, 50), 900.0]
    r = reject_outliers(ptp)
    assert r[-1] and r[:-1].sum() == 0


def test_notebook_cells_compile():
    import ast
    from tools.build_pe_pilot_notebook import build
    nb = build()
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 6
    for c in code:
        ast.parse("".join(c["source"]))
