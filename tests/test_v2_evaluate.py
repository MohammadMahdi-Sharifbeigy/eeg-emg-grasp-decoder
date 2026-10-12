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
    assert np.isnan(E.empirical_p(float("nan"), np.zeros(20))) and np.isnan(E.empirical_p(float("inf"), np.zeros(20)))


def test_baseline_comparison(tmp_path):
    p = tmp_path / "base.csv"
    pd.DataFrame({"subject": ["P1"] * 5, "group": ["all test"] * 5, "muscle": NAMES, "r": [0.9] * 5,
                  "vaf": [80.0] * 5, "height_ratio": [1.0] * 5}).to_csv(p, index=False)
    v2 = pd.DataFrame({"output": ["K"] * 5, "muscle": NAMES, "r": [0.92] * 5, "vaf": [84.0] * 5})
    out = E.baseline_comparison(v2, "P1", p)
    assert list(out.columns) == ["muscle", "baseline_r", "v2_K_r", "baseline_vaf", "v2_K_vaf", "d_vaf"]
    assert np.allclose(out.d_vaf, 4.0)
    assert E.baseline_comparison(v2, "P99", p) is None and E.baseline_comparison(v2, "P1", tmp_path / "none.csv") is None


def test_float64_metrics_agree_with_standard():
    y = _sig(T=300000)                                                        # float32, long enough for float32 sums to drift
    p = (0.3 * np.roll(y, 50, axis=0)).astype(np.float32)
    std = E.standard_metrics(p, y, NAMES)
    assert np.array_equal(E._vaf(p, y), std.vaf.values) and np.array_equal(E._r(p, y), std.r.values)
    assert E._vaf(p, y).dtype == np.float64


def test_swap_verdict_planted():
    rng = np.random.default_rng(0)
    null_r = 0.01 * rng.standard_normal((20, 5))
    null_d = 0.1 * rng.standard_normal((20, 5))
    real = E.swap_verdict(np.full(5, 0.3), null_r, np.full(5, 2.0), null_d, n_blocks=81)   # planted effect in every muscle
    assert real["E_p_swap"] == 1 / 21 and real["dF_p_swap"] == 1 / 21
    assert np.isclose(real["E_r_mean"], 0.3) and np.isclose(real["dF_vaf_mean"], 2.0)
    assert real["n_null"] == 20 and real["n_blocks"] == 81
    one = np.array([0.05, 0.0, 0.0, 0.0, 0.0])                                # one muscle above its own null, the mean is not
    weak = E.swap_verdict(one - 0.02, null_r, np.zeros(5), null_d, n_blocks=81)
    assert E.swap_p((one - 0.02)[0], null_r[:, 0], 81) == 1 / 21 and weak["E_p_swap"] > 0.2
    closed = E.swap_verdict(np.zeros(5), null_r, np.zeros(5), np.zeros((20, 5)), n_blocks=81)   # closed gate: F == K
    assert closed["dF_vaf_mean"] == 0.0 and closed["dF_p_swap"] == 1.0


def test_swap_verdict_degenerate_is_nan_and_logged():
    said = []
    v = E.swap_verdict(np.full(5, 0.3), np.full((20, 5), 0.3), np.zeros(5), np.zeros((20, 5)), n_blocks=1, log=said.append)
    assert np.isnan(v["E_p_swap"]) and np.isnan(v["dF_p_swap"]) and v["n_blocks"] == 1
    assert len(said) == 1 and "NaN" in said[0]
    assert np.isnan(E.swap_p(0.3, np.zeros(20), 0)) and E.swap_p(0.3, np.zeros(20), 2) == 1 / 2
    E.swap_verdict(np.zeros(5), np.zeros((2, 5)), np.zeros(5), np.zeros((2, 5)), n_blocks=1)   # no log given: no error


def test_swap_p_respects_the_number_of_distinct_surrogates():
    null = np.zeros(20)                                                       # observed above every draw: raw p = 1/21
    assert E.swap_p(1.0, null, 2) == 1 / 2                                    # 2 blocks: one surrogate, 20 identical draws
    assert E.swap_p(1.0, null, 3) == 1 / 3
    assert E.swap_p(1.0, null, 21) == 1 / 21 and E.swap_p(1.0, null, 81) == 1 / 21      # many blocks: the draws set the limit
    assert E.swap_p(1.0, np.zeros(200), 81) == 1 / 81                         # more draws than surrogates: floor at 1 / n_blocks
    assert E.swap_p(0.0, np.ones(20), 2) == 1.0                               # the floor never lowers a p
    assert np.isnan(E.swap_p(float("nan"), null, 81))
    z = np.zeros((20, 5))
    v = lambda nb: E.swap_verdict(np.ones(5), z, np.ones(5), z, n_blocks=nb)
    assert v(2)["E_p_swap"] == v(2)["dF_p_swap"] == 1 / 2 and v(2)["n_distinct"] == 1
    assert v(3)["E_p_swap"] == 1 / 3 and v(3)["n_distinct"] == 2
    assert v(81)["E_p_swap"] == 1 / 21 and v(81)["n_distinct"] == 20
    assert v(1)["n_distinct"] == 0 and v(1)["n_null"] == 20
    nan = E.swap_verdict(np.full(5, np.nan), z, np.ones(5), z, n_blocks=81)   # diverged E: NaN, not the smallest p
    assert np.isnan(nan["E_p_swap"]) and nan["dF_p_swap"] == 1 / 21
