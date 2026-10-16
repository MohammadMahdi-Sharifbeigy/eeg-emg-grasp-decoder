import copy

import pytest
import torch

from v2.losses import StreamLoss
from v3.config import CONFIG_V3
from v3.guide import (GuideHeads, align_loss, aux_loss, build_heads, distill_loss, guide_grad_share, guide_weight,
                      guided_loss)

B, T, D = 2, 40, CONFIG_V3["model"]["d_model"]


def _cfg(**guide):
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["training"]["warmup_epochs"] = 5
    cfg["guide"].update(guide)
    return cfg


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    hold = torch.zeros(B, T, dtype=torch.int64)
    hold[:, 10:25] = 1
    return {"kin": torch.randn(B, T, 24, generator=g), "target": torch.rand(B, T, 5, generator=g), "hold_id": hold}


def _student(seed=1, grad=True):
    g = torch.Generator().manual_seed(seed)
    return {"E": torch.rand(B, T, 5, generator=g).requires_grad_(grad),
            "mid": torch.randn(B, T, D, generator=g).requires_grad_(grad),
            "fin": torch.randn(B, T, D, generator=g).requires_grad_(grad)}


def _teacher(seed=2, grad=False):
    g = torch.Generator().manual_seed(seed)
    return {"K": torch.rand(B, T, 5, generator=g).requires_grad_(grad),
            "mid": torch.randn(B, T, D, generator=g).requires_grad_(grad),
            "fin": torch.randn(B, T, D, generator=g).requires_grad_(grad)}


def _crit(cfg):
    return StreamLoss(**cfg["loss"])


def test_schedule_fade_values():
    cfg = _cfg(scale=1.0, schedule="fade", hold_until=40, off_at=60)
    got = [guide_weight(e, cfg) for e in (0, 4, 39, 40, 50, 59, 60, 149)]
    assert got == pytest.approx([0.2, 1.0, 1.0, 1.0, 0.5, 0.05, 0.0, 0.0], abs=1e-9)


def test_schedule_constant_and_off_and_scale():
    assert guide_weight(100, _cfg(scale=0.3, schedule="constant")) == pytest.approx(0.3)
    assert guide_weight(10, _cfg(scale=0.3, schedule="off")) == 0.0


def test_align_zero_when_identical():
    proj = torch.nn.Linear(D, D, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(D))
    t = torch.randn(B, T, D)
    assert float(align_loss(t.clone(), t, proj)) < 1e-6


def test_no_gradient_into_teacher_tokens():
    cfg = _cfg(scale=1.0, schedule="constant", terms=["align", "distill", "kin_aux"])
    t = _teacher(grad=True)
    total, _ = guided_loss(_student(), t, _batch(), build_heads(cfg), _crit(cfg), cfg, epoch=10)
    total.backward()
    assert all(v.grad is None for v in t.values())


def test_scale_zero_is_the_v2_e_loss():
    cfg = _cfg(scale=0.0)
    s, b = _student(), _batch()
    total, parts = guided_loss(s, None, b, build_heads(cfg), _crit(cfg), cfg, epoch=3)
    expected, _ = _crit(cfg)(s["E"], b["target"], b["hold_id"])
    assert torch.equal(total, expected)
    assert parts["align"] == parts["distill"] == parts["aux"] == parts["w"] == 0.0


def test_terms_switch():
    cfg = _cfg(scale=1.0, schedule="constant", terms=["distill"])
    heads = build_heads(cfg)
    total, parts = guided_loss(_student(), _teacher(), _batch(), heads, _crit(cfg), cfg, epoch=10)
    total.backward()
    assert parts["align"] == 0.0 and parts["distill"] > 0.0
    assert heads.align.weight.grad is None


def test_align_at_mid_uses_mid_tokens():
    cfg = _cfg(scale=1.0, schedule="constant", terms=["align"], align_at="mid")
    heads = build_heads(cfg)
    base, t, b = _student(), _teacher(), _batch()
    a = guided_loss(base, t, b, heads, _crit(cfg), cfg, 10)[1]["align"]
    other = dict(base, fin=torch.randn(B, T, D))
    assert guided_loss(other, t, b, heads, _crit(cfg), cfg, 10)[1]["align"] == pytest.approx(a)
    other = dict(base, mid=torch.randn(B, T, D))
    assert guided_loss(other, t, b, heads, _crit(cfg), cfg, 10)[1]["align"] != pytest.approx(a)


def test_finite_under_autocast():
    cfg = _cfg(scale=1.0, schedule="constant", terms=["align", "distill", "kin_aux"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        total, _ = guided_loss(_student(), _teacher(), _batch(), build_heads(cfg), _crit(cfg), cfg, epoch=10)
    assert torch.isfinite(total)


def test_grad_share_bounds():
    w = torch.nn.Parameter(torch.randn(3))
    l_emg, l_guide = (w ** 2).sum(), (w - 1).abs().sum()
    assert 0.0 < guide_grad_share([w], l_emg, l_guide) < 1.0
    assert guide_grad_share([w], l_emg, torch.tensor(1.0)) == 0.0


def test_aux_loss_and_distill_shapes():
    head = torch.nn.Linear(D, 12)
    kin = torch.randn(B, T, 24)
    assert aux_loss(torch.randn(B, T, D), kin, list(range(12, 24)), head, 0.5).ndim == 0
    assert distill_loss(torch.rand(B, T, 5), torch.rand(B, T, 5), 0.5).ndim == 0
    assert isinstance(build_heads(_cfg()), GuideHeads)
