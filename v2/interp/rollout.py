"""Attention rollout on a stack of layers. Model-agnostic: it only sees matrices.

Convention: row = receiving position, column = source position, layers in forward order.
Plain rollout (Abnar & Zuidema 2020) uses the attention itself. Gradient-weighted rollout uses, in
every layer, the attention re-weighted by its positive gradient x attention (Chefer et al. 2021) and
rescaled per row, so it is specific to the explained output and does not depend on the output's units.
"""
import torch


def head_mean(attn: torch.Tensor) -> torch.Tensor:
    """(..., heads, n_q, n_s) -> (..., n_q, n_s)."""
    return attn.detach().float().mean(dim=-3)


def with_residual(m: torch.Tensor, residual: float = 0.5) -> torch.Tensor:
    """(1 - residual) * M + residual * I: a layer whose output is 'what it attended to' plus 'itself'."""
    if m.shape[-1] != m.shape[-2]:
        raise ValueError(f"a residual layer needs a square map, got {tuple(m.shape)}")
    return (1.0 - residual) * m + residual * torch.eye(m.shape[-1], dtype=m.dtype, device=m.device)


def layer_matrix(attn: torch.Tensor, residual: float = 0.5) -> torch.Tensor:
    return with_residual(head_mean(attn), residual)


def relevance_matrix(attn: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Mean over heads of max(A * dy/dA, 0), rows rescaled to sum to 1; rows without a positive entry stay 0."""
    r = (attn.detach().float() * grad.detach().float()).clamp_min(0.0).mean(dim=-3)
    return r / r.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(r.dtype).tiny)


def relevance_layer(attn: torch.Tensor, grad: torch.Tensor, residual: float = 0.5) -> torch.Tensor:
    return with_residual(relevance_matrix(attn, grad), residual)


def rollout(mats) -> torch.Tensor:
    """Product M_L ... M_1 of forward-ordered layer matrices: (..., n_L, n_0)."""
    out = mats[0]
    for m in mats[1:]:
        out = m @ out
    return out
