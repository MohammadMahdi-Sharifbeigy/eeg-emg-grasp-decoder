import numpy as np
import pandas as pd

from v2 import evaluate as E

NAMES = ["AD", "BR", "FD", "ED", "FDI"]


def _sig(T=20000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.zeros((T, 5), np.float32)
    for a in range(1000, T - 3000, 4000):
        t[a:a + 2000] = 1.0 + 0.2 * rng.random(5)
    return t + 0.02 * np.abs(rng.standard_normal((T, 5))).astype(np.float32)


def test_standard_metrics():
    y = _sig()
    df = E.standard_metrics(y, y, NAMES)
    assert list(df.muscle) == NAMES and np.allclose(df.r, 1) and np.allclose(df.vaf, 100) and np.allclose(df.rmse, 0, atol=1e-7)
    half = E.standard_metrics(0.5 * y, y, NAMES)
    assert np.allclose(half.r, 1, atol=1e-6) and (half.vaf < 90).all() and (half.ccc < 0.95).all()


def test_ceiling_and_modulation():
    y = _sig()
    c = E.smooth_ceiling(y, NAMES)
    assert (c.ceiling_vaf > 80).all() and (c.ceiling_vaf <= 100).all()
    noisy = y + np.random.default_rng(1).standard_normal(y.shape).astype(np.float32)      # fast noise lowers the ceiling
    assert (E.smooth_ceiling(noisy, NAMES).ceiling_vaf < c.ceiling_vaf).all()
    m = E.modulation_depth(y, NAMES)
    assert (m.modulation > 5).all()


def test_block_shuffle_and_swap():
    rng = np.random.default_rng(0)
    x = np.arange(1000 * 2).reshape(1000, 2)
    s = E.block_shuffle(x, 400, rng)
    assert s.shape == x.shape and not np.array_equal(s, x)
    assert sorted(s[:400, 0]) == sorted(x[:400, 0]) and sorted(s[800:, 0]) == sorted(x[800:, 0])
    w = E.block_swap(x, 400, np.random.default_rng(0))
    assert np.array_equal(w[800:], x[800:])                                   # trailing partial block stays
    assert np.array_equal(w[:400], x[400:800]) and np.array_equal(w[400:800], x[:400])
    one = E.block_swap(np.arange(300)[:, None], 400, rng)                     # fewer than two full blocks: unchanged
    assert np.array_equal(one, np.arange(300)[:, None])


def test_empirical_p():
    assert E.empirical_p(1.0, np.zeros(20)) == 1 / 21
    assert E.empirical_p(0.0, np.ones(20)) == 1.0


def test_baseline_comparison(tmp_path):
    p = tmp_path / "base.csv"
    pd.DataFrame({"subject": ["P1"] * 5, "group": ["all test"] * 5, "muscle": NAMES, "r": [0.9] * 5,
                  "vaf": [80.0] * 5, "height_ratio": [1.0] * 5}).to_csv(p, index=False)
    v2 = pd.DataFrame({"output": ["K"] * 5, "muscle": NAMES, "r": [0.92] * 5, "vaf": [84.0] * 5})
    out = E.baseline_comparison(v2, "P1", p)
    assert list(out.columns) == ["muscle", "baseline_r", "v2_K_r", "baseline_vaf", "v2_K_vaf", "d_vaf"]
    assert np.allclose(out.d_vaf, 4.0)
    assert E.baseline_comparison(v2, "P99", p) is None and E.baseline_comparison(v2, "P1", tmp_path / "none.csv") is None
