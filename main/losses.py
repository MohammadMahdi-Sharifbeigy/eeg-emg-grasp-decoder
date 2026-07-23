"""
Loss functions for the KG-GT EMG regressor (nb04).

CombinedEMGLoss = lambda * MSE + (1 - lambda) * SoftDTW_norm

With lambda = 1.0 (default) this is plain MSE (no O(T^2) DTW grid).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Large finite stand-in for +inf on the DP boundary
_BIG = 1.0e9


def _soft_min(a: Tensor, b: Tensor, c: Tensor, gamma: float) -> Tensor:
    """Differentiable soft-min: -gamma * logsumexp(-x/gamma)."""
    stacked = torch.stack((a, b, c), dim=0) / -gamma     # (3, ...)
    return -gamma * torch.logsumexp(stacked, dim=0)


def _squared_euclidean(pred: Tensor, target: Tensor) -> Tensor:
    """Pairwise squared L2 cost matrix (B, T, T) via torch.cdist."""
    return torch.cdist(pred.float(), target.float(), p=2).pow(2)


def soft_dtw(pred: Tensor, target: Tensor, gamma: float = 0.1) -> Tensor:
    """Batched Soft-DTW distance between two equal-length sequences.

    Args:
        pred:   (B, T, C) predicted sequence.
        target: (B, T, C) target sequence.
        gamma:  Soft-min smoothing (smaller -> closer to hard DTW).

    Returns:
        (B,) Soft-DTW value per batch element (unnormalised).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    B, T, _ = pred.shape
    device, dtype = pred.device, pred.dtype

    D = _squared_euclidean(pred, target).float()   # (B, T, T) fp32
    R = torch.full((B, T + 1, T + 1), _BIG, device=device, dtype=torch.float32)
    R[:, 0, 0] = 0.0

    for d in range(2, 2 * T + 1):
        i_lo = max(1, d - T)
        i_hi = min(T, d - 1)
        i = torch.arange(i_lo, i_hi + 1, device=device)
        j = d - i

        r0 = R[:, i - 1, j - 1]
        r1 = R[:, i - 1, j]
        r2 = R[:, i, j - 1]
        cost = D[:, i - 1, j - 1]
        R[:, i, j] = cost + _soft_min(r0, r1, r2, gamma)

    return R[:, T, T].to(dtype)


class SoftDTWLoss(nn.Module):
    """Soft-DTW as a reduction-aware loss module."""

    def __init__(
        self,
        gamma: float = 0.1,
        normalize: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction: {reduction!r}")
        self.gamma = gamma
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        out = soft_dtw(pred, target, self.gamma)
        if self.normalize:
            out = out / pred.size(1)
        if self.reduction == "mean":
            return out.mean()
        if self.reduction == "sum":
            return out.sum()
        return out


class CombinedEMGLoss(nn.Module):
    """Convex mix of MSE and Soft-DTW.

        loss = lambda * MSE + (1 - lambda) * SoftDTW_norm

    With lambda = 1.0 this is plain MSE (Soft-DTW skipped).
    """

    def __init__(self, loss_lambda: float = 0.9, soft_dtw_gamma: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= loss_lambda <= 1.0:
            raise ValueError(f"loss_lambda must be in [0, 1], got {loss_lambda}")
        self.loss_lambda = loss_lambda
        self.mse = nn.MSELoss()
        self.sdtw = SoftDTWLoss(gamma=soft_dtw_gamma, normalize=True, reduction="mean")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mse = self.mse(pred, target)
        if self.loss_lambda >= 1.0:
            return mse
        return self.loss_lambda * mse + (1.0 - self.loss_lambda) * self.sdtw(pred, target)


def build_loss_from_config(cfg: dict) -> CombinedEMGLoss:
    """Build CombinedEMGLoss from the training config section."""
    return CombinedEMGLoss(
        loss_lambda=cfg.get("loss_lambda", 0.9),
        soft_dtw_gamma=cfg.get("soft_dtw_gamma", 0.1),
    )
