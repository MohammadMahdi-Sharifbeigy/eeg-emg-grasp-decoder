"""500 Hz series <-> 50 Hz tokens. Token k sits at sample k * Q."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import decimate

FS_IN = 500.0
Q = 10
FS_TOK = FS_IN / Q
_MIN_SAMPLES = 700          # zero-phase FIR decimation needs more than 3 * (20 * Q + 1) samples


def n_tokens(n_samples: int, q: int = Q) -> int:
    return (int(n_samples) + q - 1) // q


def to_tokens(x: np.ndarray, q: int = Q) -> np.ndarray:
    """Anti-aliased, zero-phase reduction along time (axis 0) by q."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < _MIN_SAMPLES:
        raise ValueError(f"series too short for the anti-alias filter: {x.shape[0]} < {_MIN_SAMPLES} samples")
    y = decimate(x, q, ftype="fir", zero_phase=True, axis=0)
    return np.ascontiguousarray(y, dtype=np.float32)


def hold_to_tokens(x: np.ndarray, q: int = Q) -> np.ndarray:
    """Take every q-th sample without filtering (piecewise-constant signals, integer labels)."""
    return np.ascontiguousarray(np.asarray(x)[::q])


def to_samples(tok: np.ndarray, n_samples: int, q: int = Q) -> np.ndarray:
    """Cubic-spline interpolation of (Tk, C) tokens back to n_samples at the input rate."""
    tok = np.asarray(tok, dtype=np.float64)
    t_tok = np.arange(tok.shape[0]) * q
    t = np.clip(np.arange(int(n_samples)), 0, t_tok[-1])     # hold the last token past the final knot
    return CubicSpline(t_tok, tok, axis=0)(t).astype(np.float32)
