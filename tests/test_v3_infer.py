import numpy as np
import pytest
import torch

from v2.config import CONFIG_V2
from v2.data import fit_norm
from v2.infer import predict_series
from v2.model import build_model
from v3.config import CONFIG_V3
from v3.infer import predict_eeg_series, predict_eeg_tokens
from v3.model import build_student

DEV = torch.device("cpu")


def _tok(n=900, seed=0):
    r = np.random.default_rng(seed)
    return {"kin": r.normal(size=(n, 24)).astype(np.float32), "hist": r.random((n, 8)).astype(np.float32),
            "eeg": r.normal(size=(n, 32, 4)).astype(np.float32), "emg": np.abs(r.normal(size=(n, 5))).astype(np.float32),
            "n_samples": n * 10}


def test_works_without_kinematics():
    tok = _tok()
    norm = fit_norm([tok])
    only_eeg = {"eeg": tok["eeg"], "n_samples": tok["n_samples"]}
    y = predict_eeg_series(build_student(CONFIG_V3).eval(), only_eeg, norm, DEV)
    assert y.shape == (tok["n_samples"], 5) and (y >= 0).all() and np.isfinite(y).all()


def test_matches_v2_E():
    torch.manual_seed(0)
    v2 = build_model(CONFIG_V2).eval()
    student = build_student(CONFIG_V3).eval()
    student.load_state_dict({k: v for k, v in v2.state_dict().items() if k.startswith("eeg.")}, strict=True)
    tok = _tok()
    norm = fit_norm([tok])
    ours = predict_eeg_series(student, tok, norm, DEV)
    theirs = predict_series(v2, tok, norm, DEV)["E"]
    assert np.allclose(ours, theirs, atol=1e-5)


def test_eeg_override_shape_checked():
    tok = _tok()
    norm = fit_norm([tok])
    with pytest.raises(ValueError):
        predict_eeg_tokens(build_student(CONFIG_V3).eval(), tok, norm, DEV, eeg=np.zeros((5, 32, 4), np.float32))
