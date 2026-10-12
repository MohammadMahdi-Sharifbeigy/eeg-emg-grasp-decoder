import numpy as np
import pytest
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.electrodes import EEG_NAMES, electrode_prior


def _raw(T=6000, seed=0):
    rng = np.random.default_rng(seed)
    kin = rng.standard_normal((T, 26)).astype(np.float32)
    kin[:, 10] = 0.0; kin[:, 11] = 0.0
    kin[1000:2500, 10] = 4.0; kin[1000:2500, 11] = 6.0           # one hold
    emg = np.abs(rng.standard_normal((T, 5))).astype(np.float32) * 0.01
    emg[1000:2500] += 0.05
    return {"series": 4, "eeg": rng.standard_normal((T, 32)).astype(np.float32), "kin": kin, "emg": emg}


def test_electrode_prior():
    p = electrode_prior()
    assert len(EEG_NAMES) == 32 and p.shape == (32, 32) and p.dtype == np.float32
    assert np.allclose(p, p.T) and np.all(np.diag(p) == 0) and (p[~np.eye(32, dtype=bool)] < 0).all()
    i, j, k = EEG_NAMES.index("C3"), EEG_NAMES.index("CP1"), EEG_NAMES.index("O2")
    assert p[i, j] > p[i, k]                                      # a neighbour is closer than a far electrode
    with pytest.raises(ValueError, match="unknown electrode"):
        electrode_prior(["C3", "NOPE"])


def test_build_tokens_shapes_and_content():
    tok = D.build_tokens(_raw(), CONFIG_V2)
    assert tok["kin"].shape == (600, 24) and tok["hist"].shape == (600, 8) and tok["eeg"].shape == (600, 32, 4)
    assert tok["emg"].shape == (600, 5) and tok["hold_id"].shape == (600,) and tok["n_samples"] == 6000
    assert tok["emg_full"].shape == (6000, 5) and (tok["emg"] >= 0).all()
    assert set(np.unique(tok["hold_id"])) == {0, 1} and tok["hold_id"][150] == 1 and tok["hold_id"][300] == 0
    assert all(np.isfinite(tok[k]).all() for k in ("kin", "hist", "eeg", "emg"))


def test_emg_is_not_an_input():
    a = D.build_tokens(_raw(), CONFIG_V2)
    r = _raw(); r["emg"] = r["emg"] * 3.0 + 1.0
    b = D.build_tokens(r, CONFIG_V2)
    for k in ("kin", "hist", "eeg", "hold_id"):
        np.testing.assert_array_equal(a[k], b[k])


def test_norm_is_fitted_on_the_given_series_and_domain_round_trip():
    toks = [D.build_tokens(_raw(seed=s), CONFIG_V2) for s in (0, 1)]
    norm = D.fit_norm(toks)
    assert norm["kin_mean"].shape == (24,) and norm["eeg_std"].shape == (32, 4) and norm["emg_scale"].shape == (5,)
    kin_z, eeg_z = D.normalise(toks[0], norm)
    assert abs(float(kin_z.mean())) < 0.2 and np.abs(eeg_z).max() <= 5.0
    y = np.abs(np.random.default_rng(0).standard_normal((50, 5))).astype(np.float32)
    u = D.to_model_domain(y, norm["emg_scale"])
    np.testing.assert_allclose(D.from_model_domain(u, norm["emg_scale"]), y, rtol=1e-4, atol=1e-7)
    assert D.to_model_domain(-y, norm["emg_scale"]).max() == 0.0     # negatives clamp to zero


def test_token_windows():
    toks = [D.build_tokens(_raw(T=12000, seed=s), CONFIG_V2) for s in (0, 1)]     # 1200 tokens each
    norm = D.fit_norm(toks)
    ds = D.TokenWindows(toks, norm, window=400, stride=100)
    assert len(ds) == 2 * 9
    item = ds[0]
    assert item["kin"].shape == (400, 24) and item["hist"].shape == (400, 8) and item["eeg"].shape == (400, 32, 4)
    assert item["target"].shape == (400, 5) and item["hold_id"].dtype == torch.int64 and (item["target"] >= 0).all()
    rnd = D.TokenWindows(toks, norm, window=400, stride=100, random_offset=True, seed=1)
    rnd.set_epoch(0); a = rnd[0]["kin"].clone(); rnd.set_epoch(1); b = rnd[0]["kin"]
    assert len(rnd) >= 2 * 8 and not torch.equal(a, b)
    short = D.build_tokens(_raw(T=3000), CONFIG_V2)                               # 300 tokens < window
    assert len(D.TokenWindows([short], norm, window=400, stride=100)) == 0


def test_missing_cache_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"P1_train_S\d+\.npz.*main_1stage"):
        D.load_split(tmp_path, 1, "train")
