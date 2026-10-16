"""Training-only guide: heads, loss terms and schedule. Nothing here exists in the deployed student.

All terms read the teacher through .detach(), so the teacher never receives a gradient. Cosine and Huber are
computed in float32. `guide_weight` is the single knob: scale * schedule(epoch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.losses import huber


_TOKEN_KEY = {"final": "fin", "mid": "mid"}      # guide.align_at -> key of the model outputs


class GuideHeads(nn.Module):
    def __init__(self, d: int, n_aux: int):
        super().__init__()
        self.align = nn.Linear(d, d)
        self.aux = nn.Linear(d, n_aux)


def build_heads(cfg) -> GuideHeads:
    return GuideHeads(cfg["model"]["d_model"], len(cfg["guide"]["aux_columns"]))


def guide_weight(epoch: int, cfg) -> float:
    """scale * g(epoch); epoch counts from 0. fade: ramp over warm-up, 1 until hold_until, linear to 0 at off_at."""
    g = cfg["guide"]
    if g["schedule"] == "off":
        return 0.0
    if g["schedule"] == "constant":
        return float(g["scale"])
    warm = cfg["training"]["warmup_epochs"]
    if epoch < warm:
        shape = (epoch + 1) / warm
    elif epoch < g["hold_until"]:
        shape = 1.0
    elif epoch < g["off_at"]:
        shape = 1.0 - (epoch - g["hold_until"]) / (g["off_at"] - g["hold_until"])
    else:
        shape = 0.0
    return float(g["scale"]) * shape


def align_loss(student_tok, teacher_tok, proj) -> torch.Tensor:
    s = proj(student_tok).float()
    return (1.0 - F.cosine_similarity(s, teacher_tok.detach().float(), dim=-1)).mean()


def distill_loss(e, k, delta: float) -> torch.Tensor:
    return huber(e, k.detach(), delta)


def aux_loss(student_tok, kin_z, columns, head, delta: float) -> torch.Tensor:
    return huber(head(student_tok), kin_z[..., columns].detach(), delta)


def guided_loss(student_out, teacher_out, batch, heads, crit, cfg, epoch):
    """(total, parts): crit(E) + w * (w_align * align + w_distill * distill + w_aux * aux) over the enabled terms."""
    emg, _ = crit(student_out["E"], batch["target"], batch["hold_id"])
    g, w = cfg["guide"], guide_weight(epoch, cfg)
    parts = {"emg": float(emg.detach()), "align": 0.0, "distill": 0.0, "aux": 0.0, "w": w}
    if w == 0.0:
        return emg, parts
    if teacher_out is None:
        raise ValueError("guide weight is above zero but no teacher output was given")
    at, delta, total = _TOKEN_KEY[g["align_at"]], cfg["loss"]["huber_delta"], emg
    if "align" in g["terms"]:
        l = align_loss(student_out[at], teacher_out[at], heads.align)
        total, parts["align"] = total + w * g["w_align"] * l, float(l.detach())
    if "distill" in g["terms"]:
        l = distill_loss(student_out["E"], teacher_out["K"], delta)
        total, parts["distill"] = total + w * g["w_distill"] * l, float(l.detach())
    if "kin_aux" in g["terms"]:
        l = aux_loss(student_out[at], batch["kin"], g["aux_columns"], heads.aux, delta)
        total, parts["aux"] = total + w * g["w_aux"] * l, float(l.detach())
    return total, parts


def guide_grad_share(params, l_emg, l_guide) -> float:
    """|g_guide| / (|g_emg| + |g_guide|) over `params`; 0.0 when the guide loss has no graph."""
    if not l_guide.requires_grad:
        return 0.0
    params = list(params)

    def norm(loss):
        gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))

    a, b = norm(l_emg), norm(l_guide)
    return b / (a + b) if a + b > 0 else 0.0
