"""
synergy_decoding/synergy_model.py
===================================
Cortical-to-Synergy Neural Decoder for Pivot 1.

Architecture:
    1. LearnableFilterBank   — Causal physiologically-initialized FIR bank (re-uses CORAL-Net)
    2. MambaEncoder          — 4-layer causal Mamba SSM (re-uses CORAL-Net)
    3. LearnableLagAlignment — Differentiable Fourier fractional lag (re-uses CORAL-Net)
    4. SynergyHead           — Linear (d_model → k) + Softplus → C_hat(t) >= 0

Output:
    C_hat: (B, T, k) non-negative synergy activation coefficients.

Design Principles:
    - Strictly causal throughout: no future samples accessed at any layer.
    - k is fixed to 3 for architectural consistency across subjects.
    - Softplus ensures C_hat >= 0 (smooth, everywhere-differentiable ReLU).
    - Modular: SynergyHead can be swapped for any non-negative activation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main.coral_net import LearnableFilterBank, MambaEncoder, LearnableLagAlignment


# ============================================================================
# Non-Negative Synergy Head
# ============================================================================

class SynergyHead(nn.Module):
    """Maps encoder latents to k non-negative synergy activation time-series.

    Linear projection d_model → k followed by Softplus activation to
    enforce strict non-negativity (C_hat >= 0) with smooth gradients everywhere.

    The Softplus β=10 approximates ReLU tightly while retaining finite gradient
    at zero — critical for activation-zero resting intervals.

    Args:
        d_model: Input feature dimension from encoder.
        k: Number of synergy components (fixed at 3).
        beta: Softplus sharpness parameter. Higher β → closer to ReLU.
        bias: Include bias in linear projection.
    """

    def __init__(
        self,
        d_model: int = 256,
        k: int = 3,
        beta: float = 10.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.beta = beta
        # Two-layer projection: d_model → d_model//2 → k
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=bias),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, k, bias=bias),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Encoder output of shape (B, T, d_model).

        Returns:
            C_hat: Non-negative synergy activations of shape (B, T, k).
        """
        logits = self.proj(h)  # (B, T, k)
        return F.softplus(logits, beta=self.beta)


# ============================================================================
# Full Cortical-to-Synergy Model
# ============================================================================

class CorticosynergyDecoder(nn.Module):
    """Causal EEG-to-Muscle-Synergy decoder for Pivot 1.

    Maps raw multi-channel scalp EEG to k=3 non-negative synergy activation
    time-series C_hat(t) >= 0.

    Architecture:
        EEG (B,T,32) → [LearnableFilterBank] → [MambaEncoder x n_layers]
                     → [LearnableLagAlignment] → [SynergyHead] → C_hat (B,T,k)

    Args:
        n_eeg: Number of EEG input channels (default 32).
        k: Number of synergy components (default 3, fixed per spec).
        d_model: Hidden dimension of encoder (default 256).
        n_layers: Number of Mamba SSM layers (default 4).
        d_state: SSM state dimension (default 16).
        kernel_size: FIR filterbank kernel size in samples (default 125 = 250ms at 500Hz).
        fs: Sampling frequency in Hz (default 500.0).
        max_lag_ms: Maximum learnable corticospinal lag in ms (default 100.0).
        per_channel_lag: If True, learn per-channel lags (experimental). Default False.
        softplus_beta: Beta parameter for SynergyHead Softplus.
    """

    def __init__(
        self,
        n_eeg: int = 32,
        k: int = 3,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        kernel_size: int = 125,
        fs: float = 500.0,
        max_lag_ms: float = 100.0,
        per_channel_lag: bool = False,
        softplus_beta: float = 10.0,
    ) -> None:
        super().__init__()
        self.n_eeg = n_eeg
        self.k = k
        self.d_model = d_model
        self.fs = fs

        # 1. Causal physiologically-initialized FIR filterbank
        self.filterbank = LearnableFilterBank(
            in_channels=n_eeg,
            kernel_size=kernel_size,
            fs=fs,
            d_model=d_model,
        )

        # 2. Causal Mamba SSM sequence encoder
        self.encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
        )

        # 3. Differentiable corticospinal conduction lag alignment
        self.lag_align = LearnableLagAlignment(
            d_model=d_model,
            max_lag_ms=max_lag_ms,
            fs=fs,
            per_channel=per_channel_lag,
        )

        # 4. Non-negative synergy output head
        self.synergy_head = SynergyHead(
            d_model=d_model,
            k=k,
            beta=softplus_beta,
        )

    def forward(self, eeg: Tensor) -> Tensor:
        """Forward pass.

        Args:
            eeg: (B, T, n_eeg) raw multi-channel EEG.

        Returns:
            C_hat: (B, T, k) non-negative synergy activations.
        """
        # (B, T, 32) → (B, T, d_model)
        h = self.filterbank(eeg)

        # (B, T, d_model) → (B, T, d_model) — 4-layer causal SSM
        h = self.encoder(h)

        # (B, T, d_model) → (B, T, d_model) — fractional lag alignment
        h = self.lag_align(h)

        # (B, T, d_model) → (B, T, k) — non-negative synergy activations
        C_hat = self.synergy_head(h)
        return C_hat

    def get_lag_ms(self) -> float:
        """Return the current learned corticospinal lag in milliseconds."""
        lag = self.lag_align.current_lag_ms
        return float(lag.mean().item())


    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
