"""
synergy_decoding/losses.py
===========================
Dual-objective synergy and muscle reconstruction loss functions for Pivot 1.

TOTAL LOSS:
    L_total = L_CCC(C_hat, C)
            + 0.2 * L_Pearson(C_hat, C)
            + 0.1 * L_diff(C_hat, C)
            + 0.5 * L_rec(C_hat @ W, M)

Where:
    C_hat : Predicted synergy activations (B, T, k) >= 0
    C     : Ground-truth synergy activations (B, T, k) >= 0
    W     : Subject-specific synergy weight matrix (k, n_muscles)
    M     : Ground-truth EMG envelope (B, T, n_muscles) >= 0

Re-uses CCCLoss, PearsonCorrelationLoss, and TemporalSmoothnessLoss from
main.losses for consistent mathematical formulations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main.losses import CCCLoss, PearsonCorrelationLoss, TemporalSmoothnessLoss


# ============================================================================
# Synergy-Level Loss
# ============================================================================

class SynergyActivationLoss(nn.Module):
    """Composite loss on synergy activation coefficients C(t).

    Combines:
        - CCCLoss: Scale-invariant concordance (penalizes Mean Collapse)
        - PearsonCorrelationLoss: Timing reward (scale-invariant)
        - TemporalSmoothnessLoss: Physiological rise/fall time fidelity

    Args:
        w_ccc: Weight for Lin's CCC component.
        w_pearson: Weight for Pearson r component.
        w_diff: Weight for temporal smoothness L1 component.
        eps: Numerical stability clamp.
    """

    def __init__(
        self,
        w_ccc: float = 1.0,
        w_pearson: float = 0.2,
        w_diff: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w_ccc = w_ccc
        self.w_pearson = w_pearson
        self.w_diff = w_diff

        self.ccc_loss = CCCLoss(eps=eps)
        self.pearson_loss = PearsonCorrelationLoss(eps=eps)
        self.diff_loss = TemporalSmoothnessLoss()

        # Diagnostics: populated each forward pass
        self.last_components: Dict[str, float] = {}

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (B, T, k) predicted synergy activations (>= 0)
            target: (B, T, k) ground-truth synergy activations (>= 0)

        Returns:
            Scalar loss tensor.
        """
        l_ccc = self.ccc_loss(pred, target)
        l_pearson = self.pearson_loss(pred, target)
        l_diff = self.diff_loss(pred, target)

        total = (
            self.w_ccc * l_ccc
            + self.w_pearson * l_pearson
            + self.w_diff * l_diff
        )

        self.last_components = {
            "ccc": l_ccc.item(),
            "pearson": l_pearson.item(),
            "diff": l_diff.item(),
            "synergy_total": total.item(),
        }
        return total


# ============================================================================
# Muscle Reconstruction Loss
# ============================================================================

class MuscleReconstructionLoss(nn.Module):
    """Evaluates EMG reconstruction quality via the fixed synergy mixing matrix W.

    Projects predicted synergy activations back into muscle space:
        M_hat = C_hat @ W   (B, T, n_muscles)
    Then measures quality of M_hat against ground-truth M using CCC.

    Args:
        eps: Numerical stability clamp for CCC denominator.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.ccc_loss = CCCLoss(eps=eps)
        self.last_rec_loss: float = 0.0

    def forward(self, C_hat: Tensor, W: Tensor, M: Tensor) -> Tensor:
        """
        Args:
            C_hat: (B, T, k) predicted synergy activations
            W:     (k, n_muscles) fixed synergy weight matrix (non-negative, on device)
            M:     (B, T, n_muscles) ground-truth EMG envelopes

        Returns:
            Scalar reconstruction CCC loss.
        """
        # Reconstruct predicted EMG: (B, T, k) @ (k, n_muscles) = (B, T, n_muscles)
        M_hat = torch.matmul(C_hat, W)
        loss = self.ccc_loss(M_hat, M)
        self.last_rec_loss = loss.item()
        return loss


# ============================================================================
# Unified Dual-Objective Loss
# ============================================================================

@dataclass
class SynergyLossConfig:
    """Configuration for SynergyDualObjectiveLoss."""
    w_ccc: float = 1.0
    w_pearson: float = 0.2
    w_diff: float = 0.1
    w_rec: float = 0.5
    eps: float = 1e-6


class SynergyDualObjectiveLoss(nn.Module):
    """Full dual-objective loss for Pivot 1 Muscle Synergy Decoding.

    Total objective:
        L_total = L_CCC(C_hat, C)
                + 0.2 * L_Pearson(C_hat, C)
                + 0.1 * L_diff(C_hat, C)
                + 0.5 * L_rec(C_hat @ W, M)

    Args:
        config: SynergyLossConfig with all hyperparameters.

    Call signature:
        loss = criterion(C_hat, C, W, M)
    """

    def __init__(self, config: Optional[SynergyLossConfig] = None) -> None:
        super().__init__()
        cfg = config or SynergyLossConfig()
        self.w_rec = cfg.w_rec

        self.synergy_loss = SynergyActivationLoss(
            w_ccc=cfg.w_ccc,
            w_pearson=cfg.w_pearson,
            w_diff=cfg.w_diff,
            eps=cfg.eps,
        )
        self.rec_loss = MuscleReconstructionLoss(eps=cfg.eps)

        # Diagnostics
        self.last_components: Dict[str, float] = {}

    def forward(
        self,
        C_hat: Tensor,
        C_gt: Tensor,
        W: Tensor,
        M: Tensor,
    ) -> Tensor:
        """
        Args:
            C_hat: (B, T, k) predicted synergy activations (>= 0 via Softplus)
            C_gt:  (B, T, k) ground-truth synergy activations (>= 0)
            W:     (k, n_muscles) subject synergy mixing matrix (on same device)
            M:     (B, T, n_muscles) ground-truth EMG envelopes

        Returns:
            Scalar total loss.
        """
        l_syn = self.synergy_loss(C_hat, C_gt)
        l_rec = self.rec_loss(C_hat, W, M)
        total = l_syn + self.w_rec * l_rec

        self.last_components = {
            **self.synergy_loss.last_components,
            "rec": self.rec_loss.last_rec_loss,
            "total": total.item(),
        }
        return total
