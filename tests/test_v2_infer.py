import numpy as np
import pytest
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.infer import predict_series, predict_tokens, window_plan
from v2.model import build_model


@pytest.mark.parametrize("n", [1, 37, 399, 400, 401, 600, 601, 1000, 1234])
def test_window_plan_tiles_every_token_once(n):
    plan = window_plan(n, 400, 200)
    covered = np.zeros(n, int)
    for start, lo, hi in plan:
        assert 0 <= start <= lo < hi <= min(start + 400, n)
        covered[lo:hi] += 1
    assert (covered == 1).all()
    if n > 400:
        assert plan[0][1] == 0 and plan[-1][2] == n and plan[-1][0] == n - 400
        assert all(lo - start == 100 for start, lo, hi in plan[1:-1])        # interior windows keep their centre


def _tok(T):
    rng = np.random.default_rng(0)
    kin = rng.standard_normal((T, 26)).astype(np.float32); kin[:, 10:12] = 0
    return D.build_tokens({"series": 1, "eeg": rng.standard_normal((T, 32)).astype(np.float32), "kin": kin,
                           "emg": np.abs(rng.standard_normal((T, 5))).astype(np.float32)}, CONFIG_V2)


def test_predictions_cover_the_series_in_original_units():
    tok = _tok(9003); norm = D.fit_norm([tok]); m = build_model(CONFIG_V2).eval()
    pt = predict_tokens(m, tok, norm, torch.device("cpu"))
    assert all(pt[k].shape == (901, 5) for k in ("K", "E", "F"))
    ps = predict_series(m, tok, norm, torch.device("cpu"))
    assert all(ps[k].shape == (9003, 5) and np.isfinite(ps[k]).all() and (ps[k] >= -1e-6).all() for k in ("K", "E", "F"))


def test_short_series_and_single_window_equal_a_direct_forward():
    tok = _tok(3000); norm = D.fit_norm([tok]); m = build_model(CONFIG_V2).eval()      # 300 tokens < window
    pt = predict_tokens(m, tok, norm, torch.device("cpu"))
    kin_z, eeg_z = D.normalise(tok, norm)
    with torch.no_grad():
        direct = m(torch.from_numpy(kin_z)[None], torch.from_numpy(tok["hist"])[None], torch.from_numpy(eeg_z)[None])
    np.testing.assert_allclose(pt["K"], direct["K"][0].numpy(), atol=1e-5)


def test_overrides_change_only_the_expected_outputs():
    tok = _tok(6000); norm = D.fit_norm([tok]); m = build_model(CONFIG_V2).eval()
    base = predict_tokens(m, tok, norm, torch.device("cpu"))
    _, eeg_z = D.normalise(tok, norm)
    alt = predict_tokens(m, tok, norm, torch.device("cpu"), eeg=eeg_z[::-1].copy())
    np.testing.assert_array_equal(base["K"], alt["K"])
    assert not np.allclose(base["E"], alt["E"])
