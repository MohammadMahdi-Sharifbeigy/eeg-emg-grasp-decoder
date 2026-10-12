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


def _lift_reference(pred, target, hold_id, min_tokens=5):
    """The per-hold loop that lift_level_loss must reproduce: mean over holds of |mean(pred) - mean(target)|."""
    p, t = pred.float(), target.float()
    total, n = p.sum() * 0.0, 0
    for b in range(p.shape[0]):
        for k in torch.unique(hold_id[b]):
            if int(k) <= 0:
                continue
            m = hold_id[b] == k
            if int(m.sum()) < min_tokens:
                continue
            total = total + (p[b][m].mean(0) - t[b][m].mean(0)).abs().mean()
            n += 1
    return total / max(n, 1)


def _random_holds(g, batch, t):
    """Windows with 0 to 4 holds of 1 to 60 tokens (some shorter than min_tokens), ids not consecutive, one id split in two."""
    hold = torch.zeros(batch, t, dtype=torch.int64)
    for b in range(batch):
        for k in range(int(torch.randint(0, 5, (1,), generator=g))):
            start = int(torch.randint(0, t - 60, (1,), generator=g))
            hold[b, start:start + int(torch.randint(1, 61, (1,), generator=g))] = 3 * k + 2      # later holds overwrite
    hold[0] = 0                                               # a window without holds
    hold[1, :3], hold[1, 3:] = 7, 0                           # only a hold that is too short
    hold[2] = 0; hold[2, 10:14] = 5; hold[2, 50:53] = 5       # one id in two pieces: 7 tokens together
    hold[3, 100:110] = -1                                     # ids <= 0 are not holds
    return hold


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lift_loss_equals_the_per_hold_loop(seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(8, 200, 5, generator=g).requires_grad_()
    t = torch.rand(8, 200, 5, generator=g)
    hold = _random_holds(g, 8, 200)
    for min_tokens in (5, 1, 20):
        new, ref = lift_level_loss(p, t, hold, min_tokens), _lift_reference(p, t, hold, min_tokens)
        assert ref > 0 and abs(new.item() - ref.item()) < 1e-6
        g_new, g_ref = torch.autograd.grad(new, p)[0], torch.autograd.grad(ref, p)[0]
        assert (g_new - g_ref).abs().max() < 1e-6
    for empty in (torch.zeros(8, 200, dtype=torch.int64), hold[1:2].repeat(8, 1)):      # no hold / only too-short holds
        out = lift_level_loss(p, t, empty)
        assert out == 0 and torch.autograd.grad(out, p)[0].abs().max() == 0
