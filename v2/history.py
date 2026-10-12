"""Trial history from the gated load and grip force only (no EMG, nothing from the future).

A "hold" is a stretch where the hand-near-object gated load force stays above a threshold
for at least `min_s`; it covers the whole loaded interval of a lift. A hold becomes known
to the model at its END, so every feature at time t depends on samples <= t only.
"""
import numpy as np

N_HIST = 8
_CAP_S = 30.0


def detect_holds(load, fs: float, thr: float = 1.0, min_s: float = 0.5) -> list[tuple[int, int]]:
    on = np.asarray(load, dtype=np.float64) > thr
    d = np.diff(on.astype(np.int8), prepend=0, append=0)
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    m = int(round(min_s * fs))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= m]


def history_features(load, grip, fs: float, thr: float = 1.0, min_s: float = 0.5):
    load = np.asarray(load, dtype=np.float64)
    grip = np.asarray(grip, dtype=np.float64)
    T = len(load)
    out = np.zeros((T, N_HIST), dtype=np.float32)
    out[:, 2] = np.arange(T) / fs / 300.0
    holds = detect_holds(load, fs, thr, min_s)
    sum_l = sum_g = 0.0
    for k, (a, b) in enumerate(holds, start=1):
        ml, mg = float(load[a:b].mean()), float(grip[a:b].mean())
        sum_l += ml
        sum_g += mg
        if b >= T:
            break
        out[b:, 0] = 1.0
        out[b:, 1] = k / 40.0
        out[b:, 3] = np.minimum((np.arange(b, T) - b) / fs, _CAP_S) / _CAP_S
        out[b:, 4] = ml / 10.0
        out[b:, 5] = mg / 10.0
        out[b:, 6] = sum_l / k / 10.0
        out[b:, 7] = sum_g / k / 10.0
    return out, holds


def hold_ids(n: int, holds) -> np.ndarray:
    ids = np.zeros(int(n), dtype=np.int64)
    for k, (a, b) in enumerate(holds, start=1):
        ids[a:b] = k
    return ids
