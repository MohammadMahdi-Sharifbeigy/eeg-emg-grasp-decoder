import numpy as np
import pytest
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.interp.events import (REST, concat_events, contrast, event_locked, event_tokens, event_windows, mean_sem,
                              paired_contrast)
from v2.interp.faithfulness import deletion_curve, map_similarity, shuffled_weights
from v2.model import build_model


def _lifts():
    def lift(i, start):
        return {"lift": i, "CurW": 1 + i % 2, "CurS": 1, "hand_start": start, "first_touch": start + 500,
                "lift_off": start + 800, "hand_stop": start + 3000}
    return [lift(1, 1000), lift(2, 6000), lift(3, 9200), {**lift(4, 14000), "lift_off": None}]


def _tok(T=12000, seed=0):
    rng = np.random.default_rng(seed)
    kin = rng.standard_normal((T, 26)).astype(np.float32); kin[:, 10:12] = 0
    return D.build_tokens({"series": 3, "eeg": rng.standard_normal((T, 32)).astype(np.float32), "kin": kin,
                           "emg": np.abs(rng.standard_normal((T, 5))).astype(np.float32)}, CONFIG_V2)


def test_event_and_rest_tokens(monkeypatch):
    monkeypatch.setattr("main.interp.lift_events", lambda *a, **k: _lifts())
    ev = event_tokens(1, 3, 20000, "lift_off")
    assert [e["token"] for e in ev] == [180, 680, 1000] and [e["lift"] for e in ev] == [1, 2, 3]      # lift 4 has no lift-off
    assert set(ev[0]) == {"lift", "CurW", "CurS", "token"}
    rest = event_tokens(1, 3, 20000, REST)                     # gaps: 4000-6000 (kept), 9000-9200 (too short), 12200-14000 (kept)
    assert [e["token"] for e in rest] == [500, 1310] and [e["lift"] for e in rest] == [1, 3]
    with pytest.raises(KeyError, match="unknown event"):
        event_tokens(1, 3, 20000, "jump")


def test_event_windows_centre_the_event_and_skip_the_edges():
    tok = _tok(); norm = D.fit_norm([tok])                     # 1200 tokens
    events = [{"token": 30}, {"token": 300}, {"token": 1100}, {"token": 1190}]
    got = list(event_windows(tok, norm, events, torch.device("cpu"), window=200))
    assert [ev["token"] for ev, _ in got] == [300, 1100]
    kin, hist, eeg = got[0][1]
    assert kin.shape == (1, 200, 24) and hist.shape == (1, 200, 8) and eeg.shape == (1, 200, 32, 4)
    kin_z, _ = D.normalise(tok, norm)
    np.testing.assert_array_equal(kin[0, 100].numpy(), kin_z[300])


def test_event_locked_stacks_one_window_per_event():
    tok = _tok(); norm = D.fit_norm([tok]); m = build_model(CONFIG_V2).eval()
    events = [{"lift": 1, "CurW": 1, "CurS": 1, "token": 30}, {"lift": 2, "CurW": 2, "CurS": 1, "token": 300},
              {"lift": 3, "CurW": 4, "CurS": 1, "token": 900}]
    ev = event_locked(m, tok, norm, events, torch.device("cpu"), output="E", muscle=1, window=200, flow=True, bin_size=10)
    meta = ev["meta"]
    assert list(meta["lift"]) == [2, 3] and list(meta["series"]) == [3, 3]
    assert {"target", "K", "E", "F", "K_base", "gain", "F_eeg", "CurW", "token"} <= set(meta.columns)
    assert ev["rollout/time"].shape == (2, 200) and ev["relevance/input"].shape == (2, 200, 32)
    assert ev["saliency/eeg"].shape == (2, 200, 32, 4) and ev["layers/time_layers"].shape == (2, 2, 4, 200)
    assert ev["flow/time"].shape == (2, 20) and ev["flow/input"].shape == (2, 20, 32)
    np.testing.assert_allclose(meta["target"], meta["E"], rtol=1e-6)
    mean, sem = mean_sem(ev["rollout/time"])
    assert mean.shape == sem.shape == (200,) and (sem >= 0).all() and float(mean_sem(ev["rollout/time"][:1])[1].sum()) == 0
    empty = event_locked(m, tok, norm, [], torch.device("cpu"), window=200)
    both = concat_events([ev, empty, ev])
    assert len(both["meta"]) == 4 and both["rollout/time"].shape == (4, 200)
    assert len(concat_events([empty])["meta"]) == 0


def test_contrast_finds_a_planted_difference_only():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((40, 6, 2)), rng.standard_normal((40, 6, 2))
    a[:, 3, 1] += 1.5
    c = contrast(a, b, n_perm=300, seed=0)
    assert c["p"].shape == c["diff"].shape == c["t"].shape == (6, 2)
    assert c["p"][3, 1] < 0.01 and c["diff"][3, 1] > 0.8
    assert np.median(np.delete(c["p"].ravel(), 7)) > 0.3       # the other 11 features are noise
    with pytest.raises(ValueError, match="at least 2"):
        contrast(a[:1], b)


def test_paired_contrast_finds_a_planted_paired_difference_only():
    rng = np.random.default_rng(1)
    b = rng.standard_normal((40, 6, 2))
    a = b + 0.5 * rng.standard_normal((40, 6, 2))
    a[:, 2, 0] += 0.6
    c = paired_contrast(a, b, n_perm=300, seed=0)
    assert c["p"].shape == c["diff"].shape == c["t"].shape == (6, 2)
    assert c["p"][2, 0] < 0.01 and c["diff"][2, 0] > 0.3 and c["t"][2, 0] > 0
    assert np.median(np.delete(c["p"].ravel(), 4)) > 0.3       # the other 11 features are noise
    with pytest.raises(ValueError, match="at least 3"):
        paired_contrast(a[:2], b[:2])
    with pytest.raises(ValueError, match="same shape"):
        paired_contrast(a, b[:39])


def test_paired_contrast_sees_through_the_variance_between_pairs():
    rng = np.random.default_rng(2)
    base = 10.0 * rng.standard_normal((30, 5))                 # shared by both members of a pair
    a, b = base + 0.3 * rng.standard_normal((30, 5)), base + 0.3 * rng.standard_normal((30, 5))
    a[:, 1] += 0.5                                             # small next to the spread of 10 between pairs
    assert contrast(a, b, n_perm=300, seed=0)["p"][1] > 0.5
    c = paired_contrast(a, b, n_perm=300, seed=0)
    assert c["p"][1] < 0.01 and np.delete(c["p"], 1).min() > 0.05


def test_contrasts_reject_values_that_are_not_finite():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((10, 4)), rng.standard_normal((10, 4))
    for bad in (np.nan, np.inf):
        x = a.copy(); x[2, 1] = bad
        for fn in (contrast, paired_contrast):
            with pytest.raises(ValueError, match="finite"):
                fn(x, b)
            with pytest.raises(ValueError, match="finite"):
                fn(a, x)


def test_contrasts_on_constant_features():
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((30, 4)), rng.standard_normal((20, 4))
    a[:, 0] = b[:, 0] = 0.1                                    # the same constant on both sides: no difference
    a[:, 1], b[:, 1] = 0.3, 0.1                                # constant on each side, but different: cannot be tested
    a[:, 3] += 2.0
    c = contrast(a, b, n_perm=200, seed=0)
    assert c["t"][0] == 0 and c["p"][0] == 1
    assert np.isnan(c["t"][1]) and np.isnan(c["p"][1]) and c["diff"][1] == pytest.approx(0.2)
    assert c["p"][3] < 0.01 and c["p"][2] > 0.05               # the untestable feature does not swamp the others
    rng = np.random.default_rng(5)
    a2 = b + 0.5 * rng.standard_normal((20, 4))
    a2[:, 0] = b[:, 0]
    a2[:, 1] = b[:, 1] + 0.1
    a2[:, 3] += 1.0
    c = paired_contrast(a2, b, n_perm=200, seed=0)
    assert c["t"][0] == 0 and c["p"][0] == 1
    assert np.isnan(c["t"][1]) and np.isnan(c["p"][1]) and c["diff"][1] == pytest.approx(0.1)
    assert c["p"][3] < 0.01 and c["p"][2] > 0.05
    none = paired_contrast(b + 1.0, b, n_perm=50)              # nothing can be tested
    assert np.isnan(none["p"]).all()


class _Stub(torch.nn.Module):
    """E = sum of electrode 3 over the whole window; K = sum of the kinematics of token 5."""

    def forward(self, kin, hist, eeg):
        e = eeg[:, :, 3].sum((1, 2))[:, None, None].expand(-1, eeg.shape[1], 5)
        k = kin[:, 5].sum(-1)[:, None, None].expand(-1, kin.shape[1], 5)
        return {"E": e, "K": k, "F": k}


def test_deletion_curve_rewards_a_score_that_points_at_the_used_inputs():
    g = torch.Generator().manual_seed(0)
    kin, hist = torch.randn(1, 40, 24, generator=g), torch.rand(1, 40, 8, generator=g)
    eeg = torch.rand(1, 40, 32, 4, generator=g) + 0.5                              # positive: every deleted cell lowers E
    good = np.zeros((40, 32)); good[:, 3] = 1.0
    c = deletion_curve(_Stub(), kin, hist, eeg, good, output="E", fractions=(0.01, 0.03), n_random=8)
    assert c["top"].shape == c["random_mean"].shape == (2,) and (c["top"] > c["random_mean"]).all()
    assert c["aopc"] > 0 and c["top"][1] > c["top"][0] and c["full"] == pytest.approx(float(eeg[0, :, 3].sum()), rel=1e-5)
    bad = np.zeros((40, 32)); bad[:, 9] = 1.0
    assert np.allclose(deletion_curve(_Stub(), kin, hist, eeg, bad, output="E", fractions=(0.01, 0.03))["top"], 0, atol=1e-3)
    k = deletion_curve(_Stub(), kin, hist, eeg, np.eye(40)[5], output="K", which="kin", fractions=(0.025,))
    assert k["top"][0] == pytest.approx(abs(float(kin[0, 5].sum())), rel=1e-4, abs=1e-5)
    with pytest.raises(ValueError, match="expected"):
        deletion_curve(_Stub(), kin, hist, eeg, np.zeros(7))


def test_deletion_curve_in_chunks_equals_one_batch_and_rejects_an_unknown_input():
    g = torch.Generator().manual_seed(1)
    kin, hist = torch.randn(1, 40, 24, generator=g), torch.rand(1, 40, 8, generator=g)
    eeg = torch.randn(1, 40, 32, 4, generator=g)
    score = np.random.default_rng(0).random((40, 32))
    kw = dict(output="E", fractions=(0.05, 0.2, 0.4), n_random=4)                  # 15 deleted copies
    one = deletion_curve(_Stub(), kin, hist, eeg, score, chunk=1000, **kw)
    for chunk in (1, 4, 15):
        got = deletion_curve(_Stub(), kin, hist, eeg, score, chunk=chunk, **kw)
        for key in ("top", "random_mean", "random_sd"):
            np.testing.assert_allclose(got[key], one[key], rtol=1e-5, atol=1e-5)
        assert got["aopc"] == pytest.approx(one["aopc"], rel=1e-5, abs=1e-5)
    k1 = deletion_curve(_Stub(), kin, hist, eeg, np.arange(40.0), output="K", which="kin", chunk=3)
    k2 = deletion_curve(_Stub(), kin, hist, eeg, np.arange(40.0), output="K", which="kin", chunk=1000)
    np.testing.assert_allclose(k1["random_mean"], k2["random_mean"], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="which"):
        deletion_curve(_Stub(), kin, hist, eeg, np.zeros(40), which="hist")
    with pytest.raises(ValueError, match="chunk"):
        deletion_curve(_Stub(), kin, hist, eeg, score, chunk=0)


def test_shuffled_weights_and_map_similarity():
    m = build_model(CONFIG_V2)
    s = shuffled_weights(m, seed=1)
    w0, w1 = m.eeg.up.weight, s.eeg.up.weight
    assert s is not m and not torch.equal(w0, w1) and torch.equal(w0.flatten().sort().values, w1.flatten().sort().values)
    assert torch.equal(m.eeg.up.bias, s.eeg.up.bias)
    assert map_similarity(np.arange(10), np.arange(10) * 3.0) == pytest.approx(1.0)
    assert map_similarity(np.arange(10), -np.arange(10)) == pytest.approx(-1.0)
    assert np.isnan(map_similarity(np.zeros(5), np.arange(5)))
