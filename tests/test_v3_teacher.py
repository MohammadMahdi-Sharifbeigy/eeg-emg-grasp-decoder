import copy
from pathlib import Path

import pytest
import torch

from v2.config import CONFIG_V2
from v2.model import build_model
from v3.config import CONFIG_V3
from v3.teacher import Teacher, load_teacher, teacher_path

B, T = 2, 60


def _save(tmp_path, cfg_model=None, name="best.pt"):
    torch.manual_seed(0)
    v2 = build_model(CONFIG_V2).eval()
    model_cfg = copy.deepcopy(CONFIG_V2["model"]) if cfg_model is None else cfg_model
    path = tmp_path / name
    torch.save({"best_state": v2.state_dict(), "config": {"model": model_cfg}}, path)
    return v2, path


def _kin(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 24, generator=g), torch.rand(B, T, 8, generator=g)


def test_loads_k_exactly(tmp_path):
    v2, path = _save(tmp_path)
    teacher = load_teacher(CONFIG_V3, None, path, torch.device("cpu"))
    kin, hist = _kin()
    eeg = torch.randn(B, T, 32, 4)
    out = teacher(kin, hist)
    assert torch.allclose(out["K"], v2(kin, hist, eeg)["K"], atol=1e-6)
    d = CONFIG_V3["model"]["d_model"]
    assert out["mid"].shape == out["fin"].shape == (B, T, d)


def test_no_parameter_requires_grad_and_no_graph(tmp_path):
    _, path = _save(tmp_path)
    teacher = load_teacher(CONFIG_V3, None, path, torch.device("cpu"))
    assert not any(p.requires_grad for p in teacher.parameters())
    with torch.enable_grad():
        out = teacher(*_kin())
    assert all(not v.requires_grad for v in out.values())


def test_stays_in_eval(tmp_path):
    _, path = _save(tmp_path)
    teacher = load_teacher(CONFIG_V3, None, path, torch.device("cpu"))
    teacher.train()
    assert teacher.training is False
    a, b = teacher(*_kin()), teacher(*_kin())
    assert torch.equal(a["K"], b["K"])


def test_missing_checkpoint_names_the_path(tmp_path):
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["teacher"]["checkpoint_dir"] = str(tmp_path)
    with pytest.raises(FileNotFoundError, match="P7"):
        load_teacher(cfg, None, teacher_path(cfg, 7), torch.device("cpu"))


def test_config_mismatch_names_key(tmp_path):
    bad = copy.deepcopy(CONFIG_V2["model"])
    bad["causal"] = True
    _, path = _save(tmp_path, bad)
    with pytest.raises(ValueError, match="causal"):
        load_teacher(CONFIG_V3, None, path, torch.device("cpu"))


def test_teacher_path():
    assert teacher_path(CONFIG_V3, 3) == Path("outputs/checkpoints_v2/P3/best.pt")


def test_teacher_is_a_module_with_kin_only():
    assert all(k.startswith("kin.") for k in Teacher(CONFIG_V3["model"], None).state_dict())
