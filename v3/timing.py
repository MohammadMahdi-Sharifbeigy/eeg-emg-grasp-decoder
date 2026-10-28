"""Timing baselines: the EMG envelope predicted from the task events alone (no EEG, no kinematic signal).

These are separate models, never inputs to the student. They answer "how much of the envelope is the task clock?".
  phase, cue                 the eeg_controls.phase_baseline models (bumps at lags -2 .. +3 s around every event,
                             or only the LED cue). They read events from the whole recording, future ones included:
                             a ceiling, not a fair competitor.
  cue_causal, events_causal  only information available up to token t: Gaussian bumps (the cue model's step and
                             width) of the time since the LAST LED onset, or since the last occurrence of each of the
                             8 task events, at lags 0 .. +3 s. A horizon h pairs the features at token t with the
                             target at t + h.
Fit as in eeg_controls.phase_baseline: ridge in the model domain u = sqrt(y / s_c) at the token rate, alpha by the
validation series, refit on training + validation, scored on the test series.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_controls.common import EVENTS
from eeg_controls.phase_baseline import ALPHAS, _fit, design
from v2.evaluate import standard_metrics
from v2.tokens import FS_IN, Q, to_samples

FS_TOK = FS_IN / Q
TOKEN_MS = 1000.0 / FS_TOK                                         # 20 ms
CAUSAL = {"cue_causal": ("led_on",), "events_causal": EVENTS}
NON_CAUSAL = ("phase", "cue")
VARIANTS = NON_CAUSAL + tuple(CAUSAL)
LAGS = np.arange(0.0, 3.0 + 1e-9, 0.25)                            # s after the last event; step of the cue model
WIDTH = 0.25                                                       # s; width of the cue model


def horizon_tokens(h_ms: float) -> int:
    h = int(round(float(h_ms) / TOKEN_MS))
    if abs(h * TOKEN_MS - float(h_ms)) > 1e-6:
        raise ValueError(f"horizon {h_ms} ms is not a multiple of the {TOKEN_MS:g} ms token")
    return h


def time_since_last(event_samples, n_tok: int) -> np.ndarray:
    """(n_tok,) seconds from the last event at sample <= t * Q to token t; NaN before the first event."""
    tok = np.arange(int(n_tok))
    ev = np.sort(np.asarray([s for s in event_samples if s is not None], dtype=np.int64))
    out = np.full(int(n_tok), np.nan)
    if not len(ev):
        return out
    idx = np.searchsorted(ev, tok * Q, side="right") - 1         # last event with sample <= t * Q
    ok = idx >= 0
    out[ok] = tok[ok] / FS_TOK - ev[idx[ok]] / FS_IN
    return out


def causal_features(lift_list, n_tok: int, variant: str) -> np.ndarray:
    """(n_tok, n_events * 13) bumps of the time since the last occurrence of each event; 0 before the first one.

    Row t depends only on events at sample <= t * Q.
    """
    if variant not in CAUSAL:
        raise ValueError(f"unknown causal variant {variant!r}; choose from {tuple(CAUSAL)}")
    cols = []
    for ev in CAUSAL[variant]:
        dt = time_since_last([lift.get(ev) for lift in lift_list], n_tok)
        bump = np.exp(-0.5 * ((dt[:, None] - LAGS[None, :]) / WIDTH) ** 2)
        cols.append(np.nan_to_num(bump, nan=0.0))
    return np.concatenate(cols, axis=1)


def features(lift_list, n_tok: int, variant: str) -> np.ndarray:
    if variant in CAUSAL:
        return causal_features(lift_list, n_tok, variant)
    if variant in NON_CAUSAL:
        return design(lift_list, n_tok, variant)
    raise ValueError(f"unknown timing variant {variant!r}; choose from {VARIANTS}")


def _pairs(s: dict, variant: str, h: int, to_u):
    """Features at tokens 0 .. n - h - 1 and the target at tokens h .. n - 1 of one series."""
    n = len(s["emg"])
    if h >= n:
        raise ValueError(f"horizon {h} tokens is longer than series {s.get('series')} ({n} tokens)")
    return features(s["lifts"], n, variant)[:n - h], to_u(np.asarray(s["emg"], np.float64))[h:]


def fit_timing(splits: dict, variant: str, horizon_ms: float = 0.0) -> dict:
    """Fit one timing model; returns its alpha, s_c and per test series the prediction.

    `splits` maps "train", "val", "test" to lists of series dicts with "emg" (token-rate envelope), "emg_full",
    "n_samples", "lifts" and "series". Per test series: "u" (model domain, target tokens h .. n - 1),
    "y_tok" (original units at the token rate) and "y_500" (500 Hz, aligned with emg_full[h * Q:]).
    """
    h = horizon_tokens(horizon_ms)
    if h and variant in NON_CAUSAL:
        raise ValueError(f"{variant} reads future events; a horizon is only defined for {tuple(CAUSAL)}")
    scale = np.maximum(np.percentile(np.concatenate([s["emg"] for s in splits["train"]]), 95, axis=0), 1e-8)
    to_u = lambda y: np.sqrt(np.maximum(y, 0.0) / scale)
    xy = {k: [_pairs(s, variant, h, to_u) for s in splits[k]] for k in ("train", "val")}
    cat = lambda pairs, i: np.concatenate([p[i] for p in pairs])
    best = None
    for a in ALPHAS:                                               # alpha by validation error in the model domain
        err = float(((_fit(cat(xy["train"], 0), cat(xy["train"], 1), a)(cat(xy["val"], 0)) - cat(xy["val"], 1)) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, a)
    both = xy["train"] + xy["val"]
    model = _fit(cat(both, 0), cat(both, 1), best[1])
    test = []
    for s in splits["test"]:
        x, _ = _pairs(s, variant, h, to_u)
        u = np.maximum(model(x), 0.0)
        y_tok = scale * u ** 2
        n500 = int(s["n_samples"]) - h * Q
        test.append({"series": s.get("series"), "u": u.astype(np.float32), "y_tok": y_tok.astype(np.float32),
                     "y_500": np.maximum(to_samples(y_tok, n500), 0.0)})
    return {"variant": variant, "horizon_ms": float(horizon_ms), "alpha": float(best[1]), "scale": scale, "test": test}


def score_timing(fitted: dict, test_series, names) -> pd.DataFrame:
    """r / CCC / RMSE / VAF of a fitted timing model on the test series, at 500 Hz and at the token rate."""
    h = horizon_tokens(fitted["horizon_ms"])
    p500 = np.concatenate([t["y_500"] for t in fitted["test"]])
    y500 = np.concatenate([np.asarray(s["emg_full"])[h * Q:] for s in test_series])
    p50 = np.concatenate([t["y_tok"] for t in fitted["test"]])
    y50 = np.concatenate([np.asarray(s["emg"])[h:] for s in test_series])
    rows = [standard_metrics(p500, y500, names).assign(rate=500), standard_metrics(p50, y50, names).assign(rate=50)]
    return pd.concat(rows, ignore_index=True).assign(variant=fitted["variant"], horizon_ms=fitted["horizon_ms"],
                                                     alpha=fitted["alpha"])


def cache_path(cache_dir, participant: int, variant: str, horizon_ms: float) -> Path:
    return Path(cache_dir) / f"P{int(participant)}_{variant}_h{int(round(horizon_ms))}ms.npz"


def _cache_key(splits: dict, variant: str, horizon_ms: float) -> str:
    """Everything a fit depends on: the series of every split, their lengths, the variant, horizon and basis."""
    sid = lambda s: -1 if s.get("series") is None else int(s["series"])
    parts = [f"{k}:" + ",".join(f"{sid(s)}/{len(s['emg'])}" for s in splits[k]) for k in ("train", "val", "test")]
    return "|".join(parts + [variant, f"{float(horizon_ms):g}", f"lags {LAGS[0]:g}-{LAGS[-1]:g} step "
                             f"{LAGS[1] - LAGS[0]:g} width {WIDTH:g}", f"alphas {len(ALPHAS)}"])


def cached_timing(cache_dir, participant: int, splits: dict, variant: str, horizon_ms: float = 0.0) -> dict:
    """fit_timing, read from or written to cache_dir (one .npz per participant, variant and horizon).

    A cached file is used only when its key (_cache_key: series, lengths, variant, horizon, basis) matches; otherwise
    the model is fitted again and the file replaced.
    """
    path = cache_path(cache_dir, participant, variant, horizon_ms)
    series = [-1 if s.get("series") is None else int(s["series"]) for s in splits["test"]]
    key = _cache_key(splits, variant, horizon_ms)
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            if "key" in d and str(d["key"]) == key:
                test = [{"series": sid, "u": d[f"u_{i}"], "y_tok": d[f"y_tok_{i}"], "y_500": d[f"y_500_{i}"]}
                        for i, sid in enumerate(series)]
                return {"variant": variant, "horizon_ms": float(horizon_ms), "alpha": float(d["alpha"]),
                        "scale": d["scale"], "test": test}
    fitted = fit_timing(splits, variant, horizon_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"series": np.asarray(series), "alpha": fitted["alpha"], "scale": fitted["scale"], "key": np.asarray(key)}
    for i, t in enumerate(fitted["test"]):
        arrays.update({f"u_{i}": t["u"], f"y_tok_{i}": t["y_tok"], f"y_500_{i}": t["y_500"]})
    np.savez(path, **arrays)
    return fitted
