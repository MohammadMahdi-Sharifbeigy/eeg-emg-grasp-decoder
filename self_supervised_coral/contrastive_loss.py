"""Contrastive Losses for Bio-CLIP EEG-EMG Self-Supervised Learning.

Implements three loss variants:

1. NTXentLoss — Standard symmetric NT-Xent (SimCLR) as baseline.
2. SymmetricInfoNCELoss — Bidirectional InfoNCE (CLIP-style) with temperature.
3. PhaseAwareInfoNCELoss — Phase-aware symmetric InfoNCE that masks out
   same-phase/temporally-contiguous pairs from the negative denominator.
   This prevents the model from pushing semantically identical neural states
   (e.g., two 'GRASP' windows from the same subject) apart.

All losses operate on L2-normalized embeddings (unit sphere).
Temperature parameter is learnable (log-parameterized) for stability.

Reference:
    Radford et al. "Learning Transferable Visual Models From Natural Language
    Supervision" (CLIP), 2021.
    Chen et al. "A Simple Framework for Contrastive Learning of Visual
    Representations" (SimCLR), 2020.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NTXentLoss(nn.Module):
    """NT-Xent Loss: Normalized Temperature-scaled Cross-Entropy (SimCLR).

    Treats each (z_i, z_j) pair as positive, all other 2(N-1) samples as negatives.
    Symmetric: loss = 0.5 * (L_{eeg→emg} + L_{emg→eeg}).

    Args:
        temperature: Softmax temperature (default: 0.07).
        learnable_temp: If True, temperature is a learnable log-parameter.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = True,
    ) -> None:
        super().__init__()
        if learnable_temp:
            # log(temperature) is the learnable parameter — exponentiated at use time
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(temperature)))
        self.learnable_temp = learnable_temp

    @property
    def temperature(self) -> Tensor:
        """Current effective temperature (clamped to [0.01, 10.0] for stability)."""
        return self.log_temp.exp().clamp(min=0.01, max=10.0)

    def forward(self, z_eeg: Tensor, z_emg: Tensor) -> Tensor:
        """
        Args:
            z_eeg: (B, D) L2-normalized EEG embeddings
            z_emg: (B, D) L2-normalized EMG embeddings

        Returns:
            Scalar symmetric NT-Xent loss
        """
        B = z_eeg.shape[0]
        # Concatenate all embeddings: (2B, D)
        z = torch.cat([z_eeg, z_emg], dim=0)  # (2B, D)

        # Full pairwise similarity matrix: (2B, 2B)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask diagonal (self-similarity)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)

        # Positive pair indices: (i, i+B) for i in [0, B), (i+B, i) for i in [0, B)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])  # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


class SymmetricInfoNCELoss(nn.Module):
    """Symmetric bidirectional InfoNCE (CLIP-style).

    Computes cross-modal alignment in both directions:
    L = 0.5 * (L_{eeg→emg} + L_{emg→eeg})

    Uses the full within-batch cross-similarity matrix.

    Args:
        temperature: Initial softmax temperature (default: 0.07).
        learnable_temp: Whether temperature is a learned parameter (default: True).
        eps: Numerical stability epsilon (default: 1e-8).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.register_buffer("log_temp", torch.tensor(math.log(temperature)))
        self.learnable_temp = learnable_temp
        self.eps = eps

    @property
    def temperature(self) -> Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=10.0)

    def _cross_modal_loss(
        self,
        z_a: Tensor,
        z_b: Tensor,
        neg_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """One-directional InfoNCE: z_a → z_b.

        Args:
            z_a: (B, D) query embeddings (anchors)
            z_b: (B, D) key embeddings (positive targets)
            neg_mask: (B, B) boolean — True = this pair is a FALSE NEGATIVE
                      (same phase) and should be excluded from the denominator.

        Returns:
            Scalar InfoNCE loss
        """
        B = z_a.shape[0]
        # (B, B) pairwise cosine similarity (already L2-normalized → dot product = cosine)
        logits = torch.mm(z_a, z_b.T) / self.temperature  # (B, B)

        # Diagonal entries are the positive pairs
        labels = torch.arange(B, device=z_a.device)

        if neg_mask is not None:
            # Mask out false negatives from denominator: set to -inf except for positives
            # Keep diagonal (positive pairs) regardless of phase mask
            false_neg = neg_mask & (~torch.eye(B, dtype=torch.bool, device=z_a.device))
            logits = logits.masked_fill(false_neg, -1e9)

        return F.cross_entropy(logits, labels)

    def forward(
        self,
        z_eeg: Tensor,
        z_emg: Tensor,
        neg_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z_eeg: (B, D) L2-normalized EEG embeddings
            z_emg: (B, D) L2-normalized EMG embeddings
            neg_mask: (B, B) boolean — True = false negatives (excluded from denominator)

        Returns:
            Scalar symmetric InfoNCE loss
        """
        loss_eeg_emg = self._cross_modal_loss(z_eeg, z_emg, neg_mask)
        loss_emg_eeg = self._cross_modal_loss(z_emg, z_eeg, neg_mask)
        return 0.5 * (loss_eeg_emg + loss_emg_eeg)


class PhaseAwareInfoNCELoss(nn.Module):
    """Phase-Aware Symmetric InfoNCE Loss for Bio-CLIP.

    Extends SymmetricInfoNCELoss with automatic false-negative masking:
    pairs belonging to the *same movement phase* are masked out from the
    negative denominator, since they represent semantically identical neural
    states that should NOT be pushed apart.

    Masking strategy:
        M_{ij} = True  if  phase_i == phase_j  AND  i ≠ j
        Only REST and ACTIVE phases are masked — TRANSIT pairs remain as negatives
        since they are more ambiguous.

    Additionally supports quiescent baseline oversampling weight:
        - REST windows are rare but important (baseline calibration)
        - A soft weight λ_rest upweights REST-window losses

    Args:
        temperature: InfoNCE temperature (default: 0.07).
        learnable_temp: Whether temperature is learned (default: True).
        mask_phases: Phases to mask from negatives (default: [REST, ACTIVE]).
        rest_weight: Additional loss weight for REST windows (default: 2.0).
        eps: Numerical stability epsilon.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temp: bool = True,
        mask_phases: Optional[list] = None,
        rest_weight: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        # Core symmetric InfoNCE
        self.infonce = SymmetricInfoNCELoss(
            temperature=temperature,
            learnable_temp=learnable_temp,
            eps=eps,
        )
        # Default: mask REST=0 and ACTIVE=1
        self.mask_phases = mask_phases if mask_phases is not None else [0, 1]
        self.rest_weight = rest_weight

    @property
    def temperature(self) -> Tensor:
        return self.infonce.temperature

    def _build_negative_mask(self, phases: Tensor) -> Tensor:
        """Build false-negative mask from phase labels.

        Args:
            phases: (B,) int64 phase labels

        Returns:
            (B, B) boolean mask — True = this pair is a false negative
        """
        # Outer equality: same_phase[i,j] = (phases[i] == phases[j])
        same_phase = phases.unsqueeze(0) == phases.unsqueeze(1)  # (B, B)

        # Only mask specified phases (TRANSIT pairs remain as negatives)
        phase_tensor = torch.tensor(self.mask_phases, device=phases.device, dtype=phases.dtype)
        is_maskable = torch.isin(phases, phase_tensor)  # (B,) bool

        # Both i AND j must be in a maskable phase for the pair to be masked
        maskable_pair = is_maskable.unsqueeze(0) & is_maskable.unsqueeze(1)  # (B, B)

        # Final mask: same phase AND both in maskable set AND not diagonal
        eye = torch.eye(phases.shape[0], dtype=torch.bool, device=phases.device)
        neg_mask = same_phase & maskable_pair & ~eye

        return neg_mask

    def _rest_weights(self, phases: Tensor) -> Tensor:
        """Per-sample loss weights upweighting REST windows.

        Args:
            phases: (B,) int64 phase labels

        Returns:
            (B,) float weight tensor
        """
        weights = torch.ones(phases.shape[0], device=phases.device)
        weights[phases == 0] = self.rest_weight  # REST = 0
        return weights

    def forward(
        self,
        z_eeg: Tensor,
        z_emg: Tensor,
        phases: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z_eeg: (B, D) L2-normalized EEG embeddings
            z_emg: (B, D) L2-normalized EMG embeddings
            phases: (B,) int64 movement phase labels (REST=0, ACTIVE=1, TRANSIT=2)
                    If None, falls back to standard symmetric InfoNCE (no masking).

        Returns:
            Scalar phase-aware symmetric InfoNCE loss
        """
        if phases is None:
            return self.infonce(z_eeg, z_emg, neg_mask=None)

        # Build false-negative mask from phase labels
        neg_mask = self._build_negative_mask(phases)  # (B, B)

        # Compute symmetric InfoNCE with false-negative masking
        loss = self.infonce(z_eeg, z_emg, neg_mask=neg_mask)

        return loss
