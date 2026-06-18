"""
Batched Soft-DTW loss (Cuturi & Blondel, 2017) for EMG sequence regression.

Soft-DTW replaces the hard min in DTW with a differentiable soft-min, giving a
loss that tolerates small temporal misalignments between predicted and target
EMG envelopes — useful when the decoder gets the shape right but is shifted a
few samples in time, which plain MSE penalises harshly.

Implementation notes (GPU):
  * The DP grid is filled along anti-diagonals. Cells on one anti-diagonal are
    independent, so each of the ~2T steps is a single vectorised op over the
    batch and the diagonal — O(T) Python-level steps instead of O(T^2).
  * Boundaries use a large finite constant (not +inf) so soft-min gradients
    stay finite.
  * Forward only; gradients flow through autograd (no custom backward kernel),
    which is enough for training on the GTX 1660 Ti at W=500.

CombinedEMGLoss mixes MSE and Soft-DTW per the config:
    loss = loss_lambda * MSE + (1 - loss_lambda) * SoftDTW_norm
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Large finite stand-in for +inf on the DP boundary (keeps softmin grad finite).
_BIG = 1.0e8


def _soft_min(a: Tensor, b: Tensor, c: Tensor, gamma: float) -> Tensor:
    """Differentiable soft-min: -gamma * logsumexp(-x/gamma)."""
    stacked = torch.stack((a, b, c), dim=0) / -gamma     # (3, ...)
    return -gamma * torch.logsumexp(stacked, dim=0)


def _squared_euclidean(pred: Tensor, target: Tensor) -> Tensor:
    """Pairwise squared L2 cost matrix over time.

    Args:
        pred:   (B, T, C)
        target: (B, T, C)

    Returns:
        (B, T, T) where D[b, i, j] = ||pred[b, i] - target[b, j]||^2.
    """
    # (B, T, 1, C) - (B, 1, T, C) -> (B, T, T, C) -> sum over C
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    return (diff * diff).sum(dim=-1)


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
    if pred.dim() != 3:
        raise ValueError(f"expected (B, T, C), got {tuple(pred.shape)}")

    B, T, _ = pred.shape
    device, dtype = pred.device, pred.dtype
    D = _squared_euclidean(pred, target)                  # (B, T, T)

    # R is 1-indexed with a padded border: shape (B, T+1, T+1).
    R = torch.full((B, T + 1, T + 1), _BIG, device=device, dtype=dtype)
    R[:, 0, 0] = 0.0

    # Fill along anti-diagonals d = i + j, for i, j in 1..T.
    for d in range(2, 2 * T + 1):
        i_lo = max(1, d - T)
        i_hi = min(T, d - 1)
        i = torch.arange(i_lo, i_hi + 1, device=device)
        j = d - i

        r0 = R[:, i - 1, j - 1]      # diagonal predecessor
        r1 = R[:, i - 1, j]          # up
        r2 = R[:, i, j - 1]          # left
        cost = D[:, i - 1, j - 1]    # D is 0-indexed
        R[:, i, j] = cost + _soft_min(r0, r1, r2, gamma)

    return R[:, T, T]


class SoftDTWLoss(nn.Module):
    """Soft-DTW as a reduction-aware loss module.

    Args:
        gamma:      Soft-min smoothing parameter.
        normalize:  Divide by sequence length T so the scale is comparable to
                    a per-step MSE term.
        reduction:  'mean' | 'sum' | 'none' over the batch.
    """

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
        out = soft_dtw(pred, target, self.gamma)          # (B,)
        if self.normalize:
            out = out / pred.size(1)
        if self.reduction == "mean":
            return out.mean()
        if self.reduction == "sum":
            return out.sum()
        return out


class CombinedEMGLoss(nn.Module):
    """Convex mix of MSE and Soft-DTW (method1.tex training objective).

        loss = lambda * MSE + (1 - lambda) * SoftDTW_norm

    With lambda = 1.0 this is plain MSE (Soft-DTW skipped entirely, so no
    O(T^2) grid is built), matching the notebook's current MSE-only behaviour.

    Args:
        loss_lambda:     Weight on the MSE term in [0, 1].
        soft_dtw_gamma:  Soft-min smoothing for the Soft-DTW term.
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
    """Build CombinedEMGLoss from the training config section.

    Args:
        cfg: training dict with loss_lambda and soft_dtw_gamma.
    """
    return CombinedEMGLoss(
        loss_lambda=cfg.get("loss_lambda", 0.9),
        soft_dtw_gamma=cfg.get("soft_dtw_gamma", 0.1),
    )
