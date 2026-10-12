"""Loss of KG-GT v2, in the square-root amplitude domain (see v2.data.to_model_domain)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def huber(pred, target, delta: float):
    return F.huber_loss(pred.float(), target.float(), delta=delta)


def ccc_loss(pred, target, eps: float = 1e-8):
    p, t = pred.float(), target.float()
    pm, tm = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
    pv, tv = ((p - pm) ** 2).mean(1), ((t - tm) ** 2).mean(1)
    cov = ((p - pm) * (t - tm)).mean(1)
    ccc = 2 * cov / (pv + tv + (pm - tm).squeeze(1) ** 2 + eps)
    valid = tv > eps
    if not bool(valid.any()):
        return p.sum() * 0.0
    return (1.0 - ccc)[valid].mean()


def lift_level_loss(pred, target, hold_id, min_tokens: int = 5):
    """Mean over holds of |mean(pred) - mean(target)| per muscle; a hold is one positive id inside one window.

    Holds with fewer than min_tokens tokens are left out; without any hold the result is 0.
    All holds of the batch are handled at once (no Python loop and no GPU sync per hold).
    """
    p, t = pred.float(), target.float()
    b, n_tok, c = p.shape
    inside = hold_id > 0                                                           # (B, T)
    key = hold_id.clamp(min=0) + torch.arange(b, device=hold_id.device)[:, None] * (hold_id.max().clamp(min=0) + 1)
    holds, group = torch.unique(key[inside], return_inverse=True)                  # one group per (window, id)
    n = torch.zeros(holds.numel(), device=p.device, dtype=p.dtype).index_add_(0, group, torch.ones_like(group, dtype=p.dtype))
    diff = torch.zeros(holds.numel(), c, device=p.device, dtype=p.dtype).index_add_(0, group, (p - t)[inside])
    keep = (n >= min_tokens).to(p.dtype)
    per_hold = (diff / n.clamp(min=1)[:, None]).abs().mean(-1)                     # mean(p) - mean(t) = mean(p - t)
    return p.sum() * 0.0 + (per_hold * keep).sum() / keep.sum().clamp(min=1)


def diff_loss(pred, target):
    p, t = pred.float(), target.float()
    return ((p[:, 1:] - p[:, :-1]) - (t[:, 1:] - t[:, :-1])).abs().mean()


class StreamLoss(nn.Module):
    def __init__(self, w_main=1.0, w_ccc=0.3, w_lift=0.5, w_diff=0.1, huber_delta=0.5):
        super().__init__()
        self.w = {"main": w_main, "ccc": w_ccc, "lift": w_lift, "diff": w_diff}
        self.delta = huber_delta

    def forward(self, pred, target, hold_id):
        parts = {"main": huber(pred, target, self.delta), "ccc": ccc_loss(pred, target),
                 "lift": lift_level_loss(pred, target, hold_id), "diff": diff_loss(pred, target)}
        total = sum(self.w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}


def v2_loss(out, target, hold_id, crit: StreamLoss):
    per, comps = {}, {}
    for name in ("K", "E", "F"):
        per[name], parts = crit(out[name], target, hold_id)
        comps.update({f"{name}/{k}": v for k, v in parts.items()})
    return per["K"] + per["E"] + per["F"], per, comps


def build_loss(cfg) -> StreamLoss:
    return StreamLoss(**cfg["loss"])
