"""Opt-in EEG channel subset and eye-channel regression in v2 (EEG control experiment A)."""
import copy

import numpy as np
import pytest
import torch

from v2.config import CONFIG_V2
from v2.data import build_tokens
from v2.electrodes import EEG_NAMES, channel_names, eye_regress, select_channels
from v2.model import build_model

CENTRAL = ["C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6"]
EYE = ["Fp1", "Fp2", "F7", "F8"]


def _raw(T=4000, seed=0):
    g = np.random.default_rng(seed)
    return {"series": 1, "kin": g.standard_normal((T, 26)).astype(np.float32),
            "eeg": g.standard_normal((T, 32)).astype(np.float32),
            "emg": np.abs(g.standard_normal((T, 5))).astype(np.float32)}


def _cfg(**eeg):
    cfg = copy.deepcopy(CONFIG_V2)
    cfg["eeg"].update(eeg)
    return cfg


def test_default_keeps_32_channels():
    cfg = copy.deepcopy(CONFIG_V2)
    tok = build_tokens(_raw(), cfg)
    assert tok["eeg"].shape == (400, 32, 4)
    assert channel_names(cfg) == EEG_NAMES
    assert build_model(cfg).eeg.identity.shape == (32, 32)


def test_channel_subset_shapes():
    cfg = _cfg(channels=CENTRAL)
    assert build_tokens(_raw(), cfg)["eeg"].shape == (400, 7, 4)
    m = build_model(cfg).eval()
    assert m.eeg.identity.shape[0] == 7
    with torch.no_grad():
        out = m(torch.randn(1, 400, 24), torch.rand(1, 400, 8), torch.randn(1, 400, 7, 4))
    assert out["E"].shape[:2] == (1, 400)


def test_unknown_channel_raises():
    with pytest.raises(ValueError, match="Cx"):
        select_channels(["C3", "Cx"])
    with pytest.raises(ValueError, match="Cx"):
        build_tokens(_raw(), _cfg(channels=["C3", "Cx"]))
    with pytest.raises(ValueError, match="Zz"):
        build_tokens(_raw(), _cfg(eye_regress=["Zz"]))


def test_eye_regress_removes_planted_component():
    g = np.random.default_rng(1)
    eeg = g.standard_normal((5000, 32))
    eeg[:, EEG_NAMES.index("C3")] += 3 * eeg[:, EEG_NAMES.index("Fp1")]
    assert abs(np.corrcoef(eeg[:, 12], eeg[:, 0])[0, 1]) > 0.9
    out, names = eye_regress(eeg, EYE)
    assert out.shape == (5000, 28) and names == [n for n in EEG_NAMES if n not in EYE]
    assert abs(np.corrcoef(out[:, names.index("C3")], eeg[:, 0])[0, 1]) < 0.05


def test_eye_regress_then_subset():
    cfg = _cfg(eye_regress=EYE, channels=CENTRAL)
    assert channel_names(cfg) == CENTRAL
    assert build_tokens(_raw(), cfg)["eeg"].shape == (400, 7, 4)
    assert build_model(_cfg(eye_regress=EYE)).eeg.identity.shape[0] == 28


def test_channel_removed_by_eye_regression_raises():
    with pytest.raises(ValueError, match="Fp1"):
        channel_names(_cfg(eye_regress=EYE, channels=["C3", "Fp1"]))
    with pytest.raises(ValueError, match="Fp1"):
        select_channels(["Fp1"], [n for n in EEG_NAMES if n not in EYE])
