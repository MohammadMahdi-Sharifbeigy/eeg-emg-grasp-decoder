import inspect

import torch

from v2.attention import recording
from v2.config import CONFIG_V2
from v2.model import build_model
from v3.config import CONFIG_V3
from v3.model import EEGOnlyModel, build_student

B, T = 2, 60


def _eeg(seed=0):
    return torch.randn(B, T, 32, 4, generator=torch.Generator().manual_seed(seed))


def _flat(layers):
    return layers["electrode"] + [layers["pool"]] + layers["temporal"] + layers["muscle"]


def test_forward_takes_eeg_only():
    assert list(inspect.signature(EEGOnlyModel.forward).parameters) == ["self", "eeg"]


def test_e_equals_v2_eeg_stream():
    torch.manual_seed(0)
    v2 = build_model(CONFIG_V2).eval()
    student = build_student(CONFIG_V3).eval()
    own = {k: v for k, v in v2.state_dict().items() if k.startswith("eeg.")}
    student.load_state_dict(own, strict=True)
    eeg = _eeg()
    out = student(eeg)
    kin, hist = torch.randn(B, T, 24), torch.rand(B, T, 8)
    assert torch.allclose(out["E"], v2(kin, hist, eeg)["E"], atol=1e-6)
    assert torch.allclose(out["fin"], v2.eeg(eeg)[1], atol=1e-6)


def test_state_keys_are_eeg_only():
    assert all(k.startswith("eeg.") for k in build_student(CONFIG_V3).state_dict())


def test_shapes():
    out = build_student(CONFIG_V3).eval()(_eeg())
    assert out["E"].shape == (B, T, 5)
    assert out["mid"].shape == out["fin"].shape == (B, T, CONFIG_V3["model"]["d_model"])
    assert all(torch.isfinite(v).all() for v in out.values())


def test_recording_lists_every_attention_module():
    student = build_student(CONFIG_V3).eval()
    with recording(student) as tape:
        student(_eeg())
    names = _flat(student.attention_layers())
    assert sorted(tape.attn) == sorted(names)


def test_config_values():
    g = CONFIG_V3["guide"]
    assert g["scales"] == [0.0, 0.05, 0.1, 0.3]
    assert (g["w_align"], g["w_distill"], g["w_aux"]) == (1.0, 0.5, 0.5)
    assert CONFIG_V3["model"] == CONFIG_V2["model"]
    assert "guide" not in CONFIG_V2
    assert CONFIG_V3["outputs"]["checkpoint_dir"] == "outputs/checkpoints_v3"
