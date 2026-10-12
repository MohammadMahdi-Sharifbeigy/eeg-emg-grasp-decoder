"""Event-locked interpretability: one model window centred on each lift event, the maps stacked over events."""
import numpy as np
import pandas as pd
import torch

from ..data import normalise
from ..tokens import FS_IN, Q
from .explain import PARTS, explain

REST = "rest"
VIEWS = ("rollout", "relevance", "saliency", "layers", "flow")


def event_tokens(participant, series, n_samples, event="lift_off", data_dir=None, min_rest_s: float = 1.0) -> list[dict]:
    """Token index of `event` for every lift of a series that has it.

    event: a key of main.interp.EVENT_COLUMNS ("first_touch", "lift_off", ...) or "rest": the middle of
    the gap between one lift's hand_stop and the next lift's hand_start, when it lasts min_rest_s.
    A rest row carries the lift number, CurW and CurS of the lift BEFORE the gap: the rest row with
    lift = k follows lift k, which is what pairs a rest with a lift for paired_contrast.
    """
    from main.interp import EVENT_COLUMNS, lift_events

    if event != REST and event not in EVENT_COLUMNS:
        raise KeyError(f"unknown event {event!r}; choose from {sorted(EVENT_COLUMNS)} or {REST!r}")
    lifts = lift_events(int(participant), int(series), fs=FS_IN, data_dir=data_dir, n_samples=int(n_samples))

    def row(lift, sample):
        return {"lift": lift["lift"], "CurW": lift["CurW"], "CurS": lift["CurS"], "token": int(sample) // Q}

    if event != REST:
        return [row(e, e[event]) for e in lifts if e.get(event) is not None]
    out = []
    for a, b in zip(lifts[:-1], lifts[1:]):
        stop, start = a.get("hand_stop"), b.get("hand_start")
        if stop is not None and start is not None and start - stop >= min_rest_s * FS_IN:
            out.append(row(a, (stop + start) // 2))
    return out


def event_windows(tok, norm, events, device, window: int = 400, clip: float = 5.0):
    """Yield (event, (kin, hist, eeg)) with the event's token at index window // 2. Events whose window
    does not fit inside the series are skipped."""
    kin_z, eeg_z = normalise(tok, norm, clip)
    for ev in events:
        start = int(ev["token"]) - window // 2
        if start < 0 or start + window > kin_z.shape[0]:
            continue
        sl = slice(start, start + window)
        yield ev, tuple(torch.from_numpy(np.ascontiguousarray(a[sl]))[None].to(device)
                        for a in (kin_z, tok["hist"], eeg_z))


def event_locked(model, tok, norm, events, device, output="E", muscle=0, window=400, clip=5.0,
                 residual=0.5, flow=False, bin_size=4) -> dict:
    """`explain` on a window centred on every event of one series, stacked over the events."""
    rows, maps = [], {}
    for ev, x in event_windows(tok, norm, events, device, window, clip):
        res = explain(model, *x, output=output, muscle=muscle, t=window // 2, residual=residual, flow=flow, bin_size=bin_size)
        rows.append({"series": tok["series"], **ev, "target": float(res["target"][0]),
                     **{k: float(res["parts"][k][0]) for k in PARTS}})
        for view in VIEWS:
            for k, v in res.get(view, {}).items():
                v = v[0].cpu().numpy() if torch.is_tensor(v) else np.asarray(v)       # flow maps are NumPy already
                maps.setdefault(f"{view}/{k}", []).append(v)
    out = {k: np.stack(v) for k, v in maps.items()}
    out["meta"] = pd.DataFrame(rows)
    return out


def concat_events(parts) -> dict:
    parts = [p for p in parts if len(p["meta"])]
    if not parts:
        return {"meta": pd.DataFrame()}
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "meta"}
    out["meta"] = pd.concat([p["meta"] for p in parts], ignore_index=True)
    return out


def mean_sem(x):
    """Mean and standard error over the first axis (events); the error is 0 for a single event."""
    x = np.asarray(x, dtype=np.float64)
    sem = x.std(0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.zeros(x.shape[1:])
    return x.mean(0), sem


_FLAT = 1e-12            # a standard error below this share of the feature's own scale counts as exactly 0


def _finite(name, *arrays):
    for x in arrays:
        if not np.isfinite(x).all():
            raise ValueError(f"{name} needs finite values, got {int((~np.isfinite(x)).sum())} NaN or inf entries")


def _t(diff, se, tol):
    """diff / se per feature. Where se is 0: t = 0 when the difference is 0 too, else +-inf (cannot be tested)."""
    flat = se <= tol
    t = diff / np.where(flat, 1.0, se)
    t[flat] = np.where(np.abs(diff[flat]) <= tol[flat], 0.0, np.copysign(np.inf, diff[flat]))
    return t


def _max_corrected(obs, null_t, n_perm):
    """(t, p) from the observed t and a function giving one t vector under the null per call.

    Features whose observed t is infinite (a difference with exactly no variance) cannot be tested: they
    are left out of the maximum statistic and get t = p = NaN."""
    ok = np.isfinite(obs)
    p = np.full(obs.shape, np.nan)
    if ok.any():
        null_max = np.array([np.abs(null_t()[ok]).max() for _ in range(n_perm)])
        p[ok] = (1 + (null_max[:, None] >= np.abs(obs[ok])[None]).sum(0)) / (1 + n_perm)
    return np.where(ok, obs, np.nan), p


def contrast(a, b, n_perm: int = 1000, seed: int = 0) -> dict:
    """Difference between two UNPAIRED sets of maps (events x features...) with a label-permutation test.

    Valid when the events of the two sets are independent of each other and of the events of their own
    set (for example lifts of two weights). Not valid for rest against the lift it follows: use
    paired_contrast there. Windows that overlap in time share input, so events are never fully
    independent and every p-value here is approximate.

    The statistic is Welch's t per feature. Every permutation keeps the largest |t| over all features,
    so each feature's p-value is corrected for testing all of them at once; the smallest possible p is
    1 / (1 + n_perm). Returns {"diff", "t", "p"}, each with the feature shape. A feature that is constant
    over all events has t = 0 and p = 1; one that is constant on each side with different values cannot
    be tested and has t = p = NaN (its diff is still given). NaN or inf in the input -> ValueError.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"contrast needs at least 2 events on each side, got {len(a)} and {len(b)}")
    _finite("contrast", a, b)
    shape, n_a = a.shape[1:], len(a)
    x = np.concatenate([a.reshape(n_a, -1), b.reshape(len(b), -1)])
    x = x - x[0]                                     # t does not change; a constant feature becomes exactly 0
    tol = _FLAT * np.abs(x).max(0)

    def t_stat(idx):
        u, v = x[idx[:n_a]], x[idx[n_a:]]
        se = np.sqrt(u.var(0, ddof=1) / len(u) + v.var(0, ddof=1) / len(v))
        return _t(u.mean(0) - v.mean(0), se, tol)

    rng = np.random.default_rng(seed)
    t, p = _max_corrected(t_stat(np.arange(len(x))), lambda: t_stat(rng.permutation(len(x))), n_perm)
    return {"diff": a.mean(0) - b.mean(0), "t": t.reshape(shape), "p": p.reshape(shape)}


def paired_contrast(a, b, n_perm: int = 1000, seed: int = 0) -> dict:
    """Difference between two PAIRED sets of maps: row i of `a` belongs with row i of `b`.

    Use it when each event of one set has a partner in the other, such as a lift and the rest that
    follows it (pair them on meta's `series` and `lift`). The statistic is the one-sample t of the
    differences a - b per feature; the null flips the sign of each pair's difference at random, so
    whatever both members of a pair share (the lift, the stretch of signal their windows have in common)
    cancels. It assumes the pairs are independent of each other; windows of neighbouring lifts overlap,
    so the p-values are still approximate.

    Each permutation keeps the largest |t| over all features (correction for testing all of them); the
    smallest possible p is 1 / (1 + n_perm), and with n pairs there are only 2**n sign patterns.
    Returns {"diff", "t", "p"}, each with the feature shape. A feature whose differences are all 0 has
    t = 0 and p = 1; one whose differences are all the same non-zero value cannot be tested and has
    t = p = NaN. Needs at least 3 pairs; NaN or inf in the input -> ValueError.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired_contrast needs two arrays of the same shape, got {a.shape} and {b.shape}")
    if len(a) < 3:
        raise ValueError(f"paired_contrast needs at least 3 pairs, got {len(a)}")
    _finite("paired_contrast", a, b)
    shape, n = a.shape[1:], len(a)
    d = (a - b).reshape(n, -1)
    tol = _FLAT * np.abs(d).max(0)

    def t_stat(sign):
        x = d * sign[:, None]
        return _t(x.mean(0), x.std(0, ddof=1) / np.sqrt(n), tol)

    rng = np.random.default_rng(seed)
    t, p = _max_corrected(t_stat(np.ones(n)), lambda: t_stat(rng.choice([-1.0, 1.0], size=n)), n_perm)
    return {"diff": d.mean(0).reshape(shape), "t": t.reshape(shape), "p": p.reshape(shape)}
