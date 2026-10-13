import numpy as np
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.model import build_model
from v2.residual import FEATURES, eeg_features, fit_ridge, residual_test, _apply

NAMES = ["AD", "BR", "FD", "ED", "FDI"]


def _tok(seed, T=9003):
    rng = np.random.default_rng(seed)
    return D.build_tokens({"series": seed, "eeg": rng.standard_normal((T, 32)).astype(np.float32),
                           "kin": rng.standard_normal((T, 26)).astype(np.float32),
                           "emg": np.abs(rng.standard_normal((T, 5))).astype(np.float32)}, CONFIG_V2)


def test_ridge_recovers_a_linear_map_and_picks_a_small_alpha():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 6))
    w = rng.standard_normal((6, 2))
    fit, alpha = fit_ridge(x, x @ w + 0.01 * rng.standard_normal((2000, 2)))
    assert alpha <= 0.1
    np.testing.assert_allclose(_apply(fit, x[:50]), x[:50] @ w, rtol=0.03, atol=0.05)   # the smallest alpha shrinks by about 1 %
    _, alpha = fit_ridge(x, rng.standard_normal((2000, 2)))                # nothing to learn: strong shrinkage
    assert alpha >= 1.0


def test_eeg_features_have_one_row_per_token():
    tok = _tok(1); norm = D.fit_norm([tok])
    torch.manual_seed(0)
    m = build_model(CONFIG_V2).eval()
    fe = eeg_features(m, tok, norm, torch.device("cpu"))
    n = tok["kin"].shape[0]
    assert fe["tokens"].shape == (n, CONFIG_V2["model"]["d_model"]) and fe["bandpower"].shape == (n, 32 * 4)
    assert np.isfinite(fe["tokens"]).all() and np.isfinite(fe["bandpower"]).all()


def test_residual_test_finds_nothing_in_noise_and_has_the_mean_row():
    toks = [_tok(1, 30003), _tok(2, 30003)]; norm = D.fit_norm(toks[:1])
    torch.manual_seed(0)
    m = build_model(CONFIG_V2).eval()
    out = residual_test(m, toks[:1], toks[1:], norm, CONFIG_V2, torch.device("cpu"), NAMES, n_null=19, seed=0)
    assert len(out) == len(FEATURES) * (len(NAMES) + 1) and set(out["features"]) == set(FEATURES)
    mean = out[out["muscle"] == "MEAN"]
    assert ((mean["r"] - mean["r_null_mean"]).abs() < 0.08).all() and (mean["p"] > 0.05).all()   # random EEG and EMG
    assert out["p"].between(1 / out["n_shifts"].iloc[0], 1).all() and np.isfinite(out[["d_vaf", "r_model"]].to_numpy()).all()
