import copy

import numpy as np
import pytest

from v2.tokens import Q, to_samples
from v3.timing import (LAGS, cache_path, cached_timing, causal_features, features, fit_timing, horizon_tokens,
                       score_timing, time_since_last)

MUSCLES = ["AD", "BR", "ED", "FD", "FDI"]


def _lifts(n_tok, period=300, first=40, seed=0):
    """Lifts every `period` tokens; events at fixed offsets (500 Hz samples), slightly jittered."""
    r = np.random.default_rng(seed)
    out, offs = [], {"led_on": 0, "hand_start": 20, "first_touch": 45, "both_touch": 50, "lift_off": 70,
                     "replace": 150, "release": 165, "hand_stop": 190}
    for k, t0 in enumerate(range(first, n_tok - 200, period)):
        j = int(r.integers(-3, 4))
        out.append({"lift": k, "CurW": 1, "CurS": 1, **{e: (t0 + o + j) * Q for e, o in offs.items()}})
    return out


def _series(seed, n_tok=1500):
    lifts = _lifts(n_tok, seed=seed)
    dt = np.nan_to_num(time_since_last([l["led_on"] for l in lifts], n_tok), nan=99.0)
    bump = np.exp(-0.5 * ((dt - 1.2) / 0.4) ** 2)
    emg = (0.05 + bump[:, None] * np.array([1.0, 0.8, 0.6, 0.4, 0.3]))
    emg = emg * (1 + 0.05 * np.random.default_rng(seed).normal(size=emg.shape))
    emg = np.maximum(emg, 0).astype(np.float32)
    return {"series": seed, "emg": emg, "emg_full": to_samples(emg, n_tok * Q), "n_samples": n_tok * Q, "lifts": lifts}


def _splits():
    return {"train": [_series(0), _series(1), _series(2)], "val": [_series(3)], "test": [_series(4), _series(5)]}


def test_time_since_last_values():
    dt = time_since_last([10 * Q, None, 30 * Q], 40)
    assert np.isnan(dt[:10]).all() and dt[10] == 0.0 and dt[29] == pytest.approx(19 / 50)
    assert dt[30] == 0.0 and dt[39] == pytest.approx(9 / 50)


def test_causal_features_shapes_and_zero_before_first_event():
    lifts = _lifts(1000)
    x = causal_features(lifts, 1000, "cue_causal")
    assert x.shape == (1000, len(LAGS))
    assert np.all(x[: lifts[0]["led_on"] // Q] == 0.0)
    assert causal_features(lifts, 1000, "events_causal").shape == (1000, 8 * len(LAGS))
    with pytest.raises(ValueError):
        features(lifts, 1000, "nope")


@pytest.mark.parametrize("variant", ["cue_causal", "events_causal"])
def test_moving_a_later_event_leaves_earlier_features_unchanged(variant):
    lifts = _lifts(1500)
    t = lifts[2]["led_on"] // Q - 5                                # a token just before the third lift's cue
    moved = copy.deepcopy(lifts)
    for lift in moved[2:]:
        for ev in ("led_on", "hand_start", "first_touch", "lift_off", "replace"):
            lift[ev] += 37 * Q
    a, b = causal_features(lifts, 1500, variant), causal_features(moved, 1500, variant)
    assert np.array_equal(a[:t + 1], b[:t + 1])


@pytest.mark.parametrize("h_ms", [0, 200, 500, 1000])
def test_fitted_prediction_ignores_future_events(h_ms):
    sp = _splits()
    fitted = fit_timing(sp, "events_causal", h_ms)
    h = horizon_tokens(h_ms)
    s = sp["test"][0]
    t = s["lifts"][2]["led_on"] // Q - 5
    moved = copy.deepcopy(s)
    for lift in moved["lifts"][2:]:
        for ev in ("led_on", "hand_start", "first_touch", "lift_off", "replace", "release", "hand_stop"):
            lift[ev] += 41 * Q
    sp2 = dict(sp, test=[moved] + sp["test"][1:])
    refit = fit_timing(sp2, "events_causal", h_ms)                # same train/val: the same model
    # target token j uses features at token j - h; every target up to t + h uses features up to t only
    assert np.allclose(fitted["test"][0]["u"][:t + 1], refit["test"][0]["u"][:t + 1])
    assert len(fitted["test"][0]["u"]) == len(s["emg"]) - h
    assert len(fitted["test"][0]["y_500"]) == s["n_samples"] - h * Q


def test_horizon_must_be_whole_tokens_and_only_causal():
    with pytest.raises(ValueError):
        horizon_tokens(30)
    with pytest.raises(ValueError):
        fit_timing(_splits(), "cue", 200)


def test_causal_cue_model_learns_the_planted_envelope():
    sp = _splits()
    m = score_timing(fit_timing(sp, "cue_causal"), sp["test"], MUSCLES)
    assert set(m.rate) == {500, 50}
    assert (m[m.rate == 500].r > 0.9).all() and (m[m.rate == 50].r > 0.9).all()


def test_non_causal_variants_run():
    sp = _splits()
    for v in ("phase", "cue"):
        m = score_timing(fit_timing(sp, v), sp["test"], MUSCLES)
        assert np.isfinite(m.r).all()


def test_cache_roundtrip(tmp_path):
    sp = _splits()
    a = cached_timing(tmp_path, 3, sp, "cue_causal", 200)
    assert cache_path(tmp_path, 3, "cue_causal", 200).exists()
    b = cached_timing(tmp_path, 3, sp, "cue_causal", 200)
    assert b["alpha"] == a["alpha"]
    assert all(np.allclose(x["u"], y["u"]) and np.allclose(x["y_500"], y["y_500"]) for x, y in zip(a["test"], b["test"]))


def test_cache_is_refitted_when_the_training_series_change(tmp_path, monkeypatch):
    import v3.timing as T
    sp = _splits()
    cached_timing(tmp_path, 3, sp, "cue_causal")
    calls = []
    real = T.fit_timing
    monkeypatch.setattr(T, "fit_timing", lambda *a, **k: calls.append(1) or real(*a, **k))
    cached_timing(tmp_path, 3, sp, "cue_causal")
    assert calls == []                                               # same data: read from the cache
    cached_timing(tmp_path, 3, dict(sp, train=sp["train"][:2]), "cue_causal")
    assert calls == [1]                                              # other training series: fitted again
