import numpy as np
import pytest

from v2.evaluate import _vaf, standard_metrics
from v3.metrics import ccc, mse, skill, vaf, vaf_gain


def _data(seed=0):
    r = np.random.default_rng(seed)
    y = r.normal(size=(500, 5))
    return y, y + r.normal(scale=0.5, size=y.shape), y + r.normal(scale=1.0, size=y.shape)


def test_match_v2_metrics():
    y, good, bad = _data()
    assert np.allclose(vaf(good, y), _vaf(good, y))
    assert np.allclose(ccc(good, y), standard_metrics(good, y, list("abcde")).ccc.values)
    assert np.allclose(mse(good, y), ((good - y) ** 2).mean(0))


def test_skill_and_gain():
    y, good, bad = _data()
    assert np.allclose(skill(y, bad, y), 1.0) and np.allclose(skill(bad, bad, y), 0.0)
    assert (skill(good, bad, y) > 0).all() and (vaf_gain(good, bad, y) > 0).all()
    assert np.allclose(vaf_gain(bad, bad, y), 0.0)
    assert np.allclose(skill(good, bad, y), 1 - mse(good, y) / mse(bad, y))


def test_exact_reference_gives_nan_not_inf():
    y = np.ones((10, 2))
    s = skill(y + 1, y, y)
    assert np.isnan(s).all()
