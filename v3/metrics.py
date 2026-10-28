"""Per-muscle metrics shared by every comparison: (n, c) arrays in, one value per muscle out, float64 inside."""
import numpy as np


def _arrays(*xs):
    return [np.asarray(x, dtype=np.float64) for x in xs]


def mse(pred, y) -> np.ndarray:
    pred, y = _arrays(pred, y)
    return ((pred - y) ** 2).mean(0)


def vaf(pred, y) -> np.ndarray:
    """Percent of the variance of y explained, as v2.evaluate._vaf."""
    pred, y = _arrays(pred, y)
    return 100.0 * (1.0 - np.var(y - pred, axis=0) / np.maximum(np.var(y, axis=0), 1e-20))


def ccc(pred, y) -> np.ndarray:
    """Lin's concordance correlation, as v2.evaluate.standard_metrics."""
    pred, y = _arrays(pred, y)
    pm, ym = pred.mean(0), y.mean(0)
    cov = ((pred - pm) * (y - ym)).mean(0)
    return 2 * cov / (pred.var(0) + y.var(0) + (pm - ym) ** 2 + 1e-20)


def skill(pred, ref, y) -> np.ndarray:
    """1 - MSE(pred) / MSE(ref): the share of the reference's error removed (NaN where the reference is exact)."""
    m_ref = mse(ref, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m_ref > 0, 1.0 - mse(pred, y) / np.where(m_ref > 0, m_ref, 1.0), np.nan)


def vaf_gain(pred, ref, y) -> np.ndarray:
    """VAF(pred) - VAF(ref) in VAF points."""
    return vaf(pred, y) - vaf(ref, y)
