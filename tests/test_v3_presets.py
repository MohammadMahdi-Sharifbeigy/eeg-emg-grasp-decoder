import numpy as np
import pytest
import torch

from v2.data import build_tokens
from v2.electrodes import EEG_NAMES, channel_names
from v3.config import CENTRAL, CONFIG_V3, PERIPHERAL, PRESETS, preset_config
from v3.model import build_student

EXPECTED = {"central": len(CENTRAL), "peripheral": len(PERIPHERAL), "eye_regressed": 28, "no_gamma": 32}


def test_preset_names_are_real_channels():
    assert set(CENTRAL) <= set(EEG_NAMES) and set(PERIPHERAL) <= set(EEG_NAMES)
    assert len(CENTRAL) == 11 and not set(CENTRAL) & set(PERIPHERAL)
    with pytest.raises(ValueError):
        preset_config("nope")


@pytest.mark.parametrize("name", list(PRESETS))
def test_each_preset_builds_a_student_with_the_right_electrode_count(name):
    cfg = preset_config(name)
    n_el, n_bands = len(channel_names(cfg)), len(cfg["eeg"]["bands"])
    assert n_el == EXPECTED[name]
    assert n_bands == (3 if name == "no_gamma" else 4)
    student = build_student(cfg).eval()
    out = student(torch.randn(2, 60, n_el, n_bands))
    assert out["E"].shape == (2, 60, 5) and torch.isfinite(out["E"]).all()
    assert student.eeg.identity.shape[0] == n_el
    assert "channels" not in CONFIG_V3["eeg"] and "eye_regress" not in CONFIG_V3["eeg"]     # presets are copies


@pytest.mark.parametrize("name", ["central", "no_gamma", "eye_regressed"])
def test_preset_reaches_the_token_pipeline(name):
    r = np.random.default_rng(0)
    n = 3000
    raw = {"series": 1, "kin": r.normal(size=(n, 26)).astype(np.float32), "eeg": r.normal(size=(n, 32)).astype(np.float32),
           "emg": np.abs(r.normal(size=(n, 5))).astype(np.float32)}
    cfg = preset_config(name)
    tok = build_tokens(raw, cfg)
    assert tok["eeg"].shape[1:] == (EXPECTED[name], len(cfg["eeg"]["bands"]))
