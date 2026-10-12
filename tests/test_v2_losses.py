import pytest
import torch

from v2.config import CONFIG_V2
from v2.losses import build_loss, ccc_loss, diff_loss, huber, lift_level_loss, v2_loss


def _batch():
    t = torch.zeros(2, 100, 5); t[:, 20:60] = 1.0
    hold = torch.zeros(2, 100, dtype=torch.int64); hold[:, 20:60] = 1
    return t, hold


def test_terms_are_zero_for_a_perfect_prediction():
    t, hold = _batch()
    assert huber(t, t, 0.5) == 0 and ccc_loss(t, t) < 1e-5 and lift_level_loss(t, t, hold) == 0 and diff_loss(t, t) == 0


def test_hand_computed_values():
    t, hold = _batch()
    p = t.clone(); p[:, 20:60] = 0.5                      # half-height burst, error 0.5 on 40 % of samples
    assert huber(p, t, 0.5).item() == pytest.approx(0.4 * 0.5 * 0.5 ** 2, rel=1e-5)      # quadratic branch at |e| = delta
    assert lift_level_loss(p, t, hold).item() == pytest.approx(0.5, rel=1e-5)
    assert diff_loss(p, t).item() == pytest.approx(2 * 0.5 / 99, rel=1e-5)
    big = t.clone(); big[:, 20:60] = 3.0                  # error 2 > delta: linear branch
    assert huber(big, t, 0.5).item() == pytest.approx(0.4 * 0.5 * (2.0 - 0.25), rel=1e-5)


def test_window_without_a_hold_gives_zero_lift_loss():
    t, _ = _batch()
    none = torch.zeros(2, 100, dtype=torch.int64)
    out = lift_level_loss(t * 0.3, t, none)
    assert out == 0 and torch.isfinite(out)


def test_ccc_skips_constant_windows_and_penalises_scale():
    flat = torch.zeros(2, 100, 5)
    assert ccc_loss(torch.rand(2, 100, 5), flat) == 0
    t, _ = _batch()
    assert ccc_loss(0.5 * t, t) > 0.1


def test_v2_loss_sums_three_streams_and_logs_components():
    t, hold = _batch()
    crit = build_loss(CONFIG_V2)
    out = {"K": t.clone(), "E": 0.5 * t, "F": t.clone()}
    total, per, comps = v2_loss(out, t, hold, crit)
    assert set(per) == {"K", "E", "F"} and per["K"].item() < 1e-4 and per["E"].item() > 0.1
    assert total.item() == pytest.approx(sum(v.item() for v in per.values()), rel=1e-5)
    assert {"K/main", "K/ccc", "K/lift", "K/diff", "E/main", "F/main"} <= set(comps)


def test_loss_is_finite_under_autocast_and_differentiable():
    t, hold = _batch()
    p = (t + 0.1 * torch.rand_like(t)).requires_grad_()
    crit = build_loss(CONFIG_V2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        total, _, _ = v2_loss({"K": p, "E": p, "F": p}, t, hold, crit)
    total.float().backward()
    assert torch.isfinite(total.float()) and torch.isfinite(p.grad).all()
