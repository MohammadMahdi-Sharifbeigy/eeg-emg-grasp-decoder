"""EEGEncoder — Twin-tower EEG encoder for Bio-CLIP self-supervised learning.

Architecture:
    1. Learnable causal FIR filterbank (re-uses main.coral_net.LearnableFilterBank)
    2. Causal Mamba SSM encoder (re-uses main.coral_net.MambaEncoder)
    3. Dual projection head:
       - Global latent z_global ∈ S^127 (L2-normalized, 128-dim)
       - Dense latent H_dense ∈ R^{B × T × 256} for token-level contrastive learning

Design principles:
    - All operations strictly causal (no future lookahead)
    - Outputs normalized unit-sphere embeddings for cosine-similarity InfoNCE
    - Shares the same backbone blocks as CORAL-Net for potential fine-tuning
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Re-use CORAL-Net backbone components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main.coral_net import LearnableFilterBank, MambaEncoder


class MultiHeadPooling(nn.Module):
    """Attention-weighted temporal pooling to extract a global summary token.

    Learns K attention heads over the time axis, concatenates, then projects.
    This allows the pooling to focus on movement-salient time-steps rather than
    averaging over resting baseline frames.

    Args:
        d_model: Input/output feature dimension.
        n_heads: Number of parallel attention heads.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # Head-wise scoring: (B, T, d_model) -> (B, T, n_heads)
        self.score_proj = nn.Linear(d_model, n_heads, bias=False)
        # Merge n_heads pooled vectors into d_model
        self.merge = nn.Linear(d_model * n_heads, d_model, bias=True)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: (B, T, d_model) sequence latents
            mask: (B, T) boolean mask — True means *keep* this timestep.

        Returns:
            Pooled summary of shape (B, d_model)
        """
        B, T, D = x.shape
        scores = self.score_proj(x)  # (B, T, n_heads)

        if mask is not None:
            # Set masked-out positions to -inf before softmax
            inf_mask = (~mask).float().unsqueeze(-1) * -1e9
            scores = scores + inf_mask

        weights = torch.softmax(scores, dim=1)  # (B, T, n_heads)

        # Weighted sum per head: (B, d_model, n_heads)
        pooled = torch.einsum("btn,btd->bdn", weights, x)  # (B, D, n_heads)
        pooled = pooled.reshape(B, D * self.n_heads)  # (B, D * n_heads)
        return self.merge(pooled)  # (B, d_model)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head mapping d_model → proj_dim with L2-norm.

    Following SimCLR / CLIP best practice: the contrastive loss is applied
    in the projected space *z*, while downstream probes use the pre-projection
    encoder output *h* (denser semantic representation).

    Args:
        in_dim: Input dimension (d_model of encoder).
        hidden_dim: Hidden layer width.
        out_dim: Final embedding dimension (128 by default → ‖z‖=1 on S^127).
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 512,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_dim) or (B, T, in_dim)

        Returns:
            L2-normalized embedding (B, out_dim) or (B, T, out_dim)
        """
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class EEGEncoder(nn.Module):
    """Twin-tower EEG encoder for Bio-CLIP contrastive pre-training.

    Outputs:
        z_global: (B, proj_dim) — L2-normalized global embedding on S^{proj_dim-1}
        H_dense:  (B, T, d_model) — frame-level latents for token-level InfoNCE

    Args:
        n_eeg_channels: Number of scalp EEG channels (default: 32).
        d_model: Hidden state dimension (default: 256).
        n_layers: Number of MambaEncoder layers (default: 4).
        d_state: SSM state size (default: 16).
        fs: Sampling frequency in Hz (default: 500).
        proj_dim: Projection head output dim (default: 128 → S^127).
        pool_heads: Attention pooling heads for global latent (default: 4).
        kernel_size: FIR filterbank kernel size (default: 125).
    """

    def __init__(
        self,
        n_eeg_channels: int = 32,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        fs: float = 500.0,
        proj_dim: int = 128,
        pool_heads: int = 4,
        kernel_size: int = 125,
    ) -> None:
        super().__init__()
        self.n_eeg_channels = n_eeg_channels
        self.d_model = d_model
        self.proj_dim = proj_dim

        # [1] Learnable causal multi-band FIR filterbank
        self.filterbank = LearnableFilterBank(
            in_channels=n_eeg_channels,
            kernel_size=kernel_size,
            fs=fs,
            d_model=d_model,
        )

        # [2] Causal Mamba SSM encoder (strictly causal receptive field)
        self.encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
        )

        # [3a] Global latent: attention-weighted temporal pooling → projection head
        self.pool = MultiHeadPooling(d_model=d_model, n_heads=pool_heads)
        self.global_proj = ProjectionHead(
            in_dim=d_model,
            hidden_dim=d_model * 2,
            out_dim=proj_dim,
        )

        # [3b] Dense latent: per-frame projection head (used for token-level InfoNCE)
        self.dense_proj = ProjectionHead(
            in_dim=d_model,
            hidden_dim=d_model,
            out_dim=d_model,  # Dense tokens stay at d_model dimension
        )

    def forward(
        self,
        eeg: Tensor,
        mask: Optional[Tensor] = None,
        return_dense: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            eeg: (B, T, n_eeg_channels) raw or lightly preprocessed EEG
            mask: (B, T) boolean mask — True = active, used for attention pooling
            return_dense: if True, also returns H_dense frame-level latents

        Returns:
            z_global: (B, proj_dim) L2-normalized global embedding
            H_dense:  (B, T, d_model) frame-level latents (None if return_dense=False)
        """
        # [1] Causal multi-band filtering
        x = self.filterbank(eeg)      # (B, T, d_model)

        # [2] Causal SSM sequence modeling
        H = self.encoder(x)           # (B, T, d_model) — dense latent representation

        # [3a] Global latent via attention-weighted pooling
        h_global = self.pool(H, mask=mask)      # (B, d_model)
        z_global = self.global_proj(h_global)   # (B, proj_dim), unit-sphere

        # [3b] Dense per-frame latents
        H_dense = self.dense_proj(H) if return_dense else None  # (B, T, d_model)

        return z_global, H_dense

    def encode(self, eeg: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Convenience: returns only z_global for downstream evaluation."""
        z_global, _ = self.forward(eeg, mask=mask, return_dense=False)
        return z_global


def build_eeg_encoder_from_config(cfg: Dict[str, Any]) -> EEGEncoder:
    """Instantiates EEGEncoder from a configuration dictionary."""
    ssl_cfg = cfg.get("ssl", {})
    enc_cfg = ssl_cfg.get("eeg_encoder", {})
    data_cfg = cfg.get("data", {})

    return EEGEncoder(
        n_eeg_channels=enc_cfg.get("n_eeg_channels", 32),
        d_model=enc_cfg.get("d_model", 256),
        n_layers=enc_cfg.get("n_layers", 4),
        d_state=enc_cfg.get("d_state", 16),
        fs=float(data_cfg.get("fs_eeg", 500.0)),
        proj_dim=enc_cfg.get("proj_dim", 128),
        pool_heads=enc_cfg.get("pool_heads", 4),
        kernel_size=enc_cfg.get("kernel_size", 125),
    )
