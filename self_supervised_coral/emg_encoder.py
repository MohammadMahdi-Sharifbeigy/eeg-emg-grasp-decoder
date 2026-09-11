"""EMGEncoder — Twin-tower EMG encoder for Bio-CLIP self-supervised learning.

Architecture:
    1. Causal temporal convolution stack (multi-scale receptive field, 4 stages)
    2. Gated recurrent integration (GRU) for long-range sequence context
    3. Dual projection head:
       - Global latent z_global ∈ S^127 (L2-normalized, 128-dim)
       - Dense latent H_dense ∈ R^{B × T × 256} for token-level InfoNCE

Design rationale:
    EMG is inherently lower-SNR than EEG but has a richer, sharper temporal
    structure. We use a convolutional frontend with multi-scale kernels to capture
    both fast motor unit action potentials (~1–10 ms) and slower envelope
    dynamics (~50–500 ms), followed by a bidirectional-causal GRU for context.
    The EMG tower is *NOT* shared with the EEG tower — it operates in a different
    signal space and needs dedicated feature extractors.

    During inference/downstream evaluation: the EMG encoder is DISCARDED.
    Only the EEG encoder is used (frozen) with a lightweight linear probe.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalConvBlock(nn.Module):
    """Single causal convolution + normalization + activation block.

    Strictly causal: pads only on the left so output at t depends only on x[0..t].

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Temporal stride (default: 1).
        dilation: Dilation factor for multi-scale receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.pad_left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,  # Handled manually
            bias=False,
        )
        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, T) input feature map

        Returns:
            (B, out_channels, T') output — T' = T // stride
        """
        x_padded = F.pad(x, (self.pad_left, 0), mode="constant", value=0.0)
        return self.act(self.norm(self.conv(x_padded)))


class MultiScaleEMGFrontend(nn.Module):
    """Multi-scale causal convolutional frontend for EMG.

    Uses 4 parallel branches with different kernel sizes to capture:
    - Fast transients (k=3, ~6 ms at 500 Hz)
    - Motor unit action potentials (k=15, ~30 ms)
    - Burst dynamics (k=63, ~126 ms)
    - Slow envelope modulation (k=125, ~250 ms)

    Outputs are concatenated and projected to d_model.

    Args:
        n_emg_channels: Number of EMG input channels (default: 5).
        d_model: Output hidden dimension (default: 256).
    """

    KERNEL_SIZES = [3, 15, 63, 125]

    def __init__(
        self,
        n_emg_channels: int = 5,
        d_model: int = 256,
        mid_channels: int = 64,
    ) -> None:
        super().__init__()
        self.n_emg_channels = n_emg_channels
        self.d_model = d_model
        n_branches = len(self.KERNEL_SIZES)

        # Parallel causal convolution branches
        self.branches = nn.ModuleList([
            CausalConvBlock(n_emg_channels, mid_channels, kernel_size=k, dilation=1)
            for k in self.KERNEL_SIZES
        ])

        # Merge branch outputs: (mid_channels * n_branches) -> d_model
        self.merge = nn.Sequential(
            nn.Conv1d(mid_channels * n_branches, d_model, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
            nn.SiLU(),
        )

    def forward(self, emg: Tensor) -> Tensor:
        """
        Args:
            emg: (B, T, n_emg_channels) EMG envelope or raw EMG

        Returns:
            (B, T, d_model) multi-scale feature representation
        """
        B, T, C = emg.shape
        x = emg.transpose(1, 2)  # (B, C, T)

        # Parallel multi-scale feature extraction
        branch_outs = [branch(x) for branch in self.branches]  # each: (B, mid_ch, T)
        x_cat = torch.cat(branch_outs, dim=1)                   # (B, mid_ch * 4, T)

        x_merged = self.merge(x_cat)                            # (B, d_model, T)
        return x_merged.transpose(1, 2)                         # (B, T, d_model)


class CausalGRUEncoder(nn.Module):
    """Causal GRU for long-range temporal context in EMG sequences.

    Uses a standard GRU in forward-only mode (inherently causal) with
    skip connections and layer normalization between stacked layers.

    Args:
        d_model: Feature dimension.
        n_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (default: 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Stack of causal single-layer GRUs with residual connections
        self.grus = nn.ModuleList([
            nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                bidirectional=False,  # Strictly causal (forward only)
                dropout=0.0,
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) with causal GRU context
        """
        for gru, norm in zip(self.grus, self.norms):
            residual = x
            out, _ = gru(x)           # (B, T, d_model)
            x = norm(self.dropout(out) + residual)
        return x


class EMGProjectionHead(nn.Module):
    """2-layer MLP projection for EMG encoder → contrastive embedding space."""

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
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class EMGEncoder(nn.Module):
    """Twin-tower EMG encoder for Bio-CLIP contrastive pre-training.

    Outputs:
        z_global: (B, proj_dim) — L2-normalized global embedding on S^{proj_dim-1}
        H_dense:  (B, T, d_model) — frame-level latents for token-level InfoNCE

    This encoder is only used during pre-training as an auxiliary supervisory
    modality. After training, only the EEGEncoder is deployed for inference.

    Args:
        n_emg_channels: Number of EMG channels (default: 5, WAY-EEG-GAL).
        d_model: Hidden dimension (default: 256).
        n_gru_layers: Stacked causal GRU layers (default: 2).
        proj_dim: Projection head output dim (default: 128 → S^127).
        mid_channels: Intermediate conv channels in multi-scale frontend (default: 64).
        dropout: Dropout in GRU encoder (default: 0.1).
    """

    def __init__(
        self,
        n_emg_channels: int = 5,
        d_model: int = 256,
        n_gru_layers: int = 2,
        proj_dim: int = 128,
        mid_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_emg_channels = n_emg_channels
        self.d_model = d_model
        self.proj_dim = proj_dim

        # [1] Multi-scale causal convolutional frontend
        self.frontend = MultiScaleEMGFrontend(
            n_emg_channels=n_emg_channels,
            d_model=d_model,
            mid_channels=mid_channels,
        )

        # [2] Causal GRU for long-range sequence context
        self.gru_encoder = CausalGRUEncoder(
            d_model=d_model,
            n_layers=n_gru_layers,
            dropout=dropout,
        )

        # [3a] Global latent: mean pooling → projection head
        # (simpler pooling for EMG — less spatial ambiguity than EEG)
        self.global_proj = EMGProjectionHead(
            in_dim=d_model,
            hidden_dim=d_model * 2,
            out_dim=proj_dim,
        )

        # [3b] Dense per-frame projection (for token-level InfoNCE)
        self.dense_proj = EMGProjectionHead(
            in_dim=d_model,
            hidden_dim=d_model,
            out_dim=d_model,
        )

    def _pool_global(self, H: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Mean pooling over active timesteps.

        Args:
            H: (B, T, d_model)
            mask: (B, T) boolean — True = active

        Returns:
            (B, d_model) mean-pooled representation
        """
        if mask is not None:
            m = mask.float().unsqueeze(-1)  # (B, T, 1)
            h_global = (H * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            h_global = H.mean(dim=1)
        return h_global

    def forward(
        self,
        emg: Tensor,
        mask: Optional[Tensor] = None,
        return_dense: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            emg: (B, T, n_emg_channels) EMG envelope or filtered EMG
            mask: (B, T) boolean mask — True = active timestep
            return_dense: if True, also returns H_dense frame-level latents

        Returns:
            z_global: (B, proj_dim) L2-normalized global embedding
            H_dense:  (B, T, d_model) frame-level latents (None if return_dense=False)
        """
        # [1] Multi-scale convolutional feature extraction
        x = self.frontend(emg)       # (B, T, d_model)

        # [2] Causal GRU sequence context
        H = self.gru_encoder(x)      # (B, T, d_model)

        # [3a] Global latent
        h_global = self._pool_global(H, mask=mask)      # (B, d_model)
        z_global = self.global_proj(h_global)            # (B, proj_dim)

        # [3b] Dense frame-level latents
        H_dense = self.dense_proj(H) if return_dense else None  # (B, T, d_model)

        return z_global, H_dense

    def encode(self, emg: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Convenience: returns only z_global."""
        z_global, _ = self.forward(emg, mask=mask, return_dense=False)
        return z_global


def build_emg_encoder_from_config(cfg: Dict[str, Any]) -> EMGEncoder:
    """Instantiates EMGEncoder from a configuration dictionary."""
    ssl_cfg = cfg.get("ssl", {})
    enc_cfg = ssl_cfg.get("emg_encoder", {})

    return EMGEncoder(
        n_emg_channels=enc_cfg.get("n_emg_channels", 5),
        d_model=enc_cfg.get("d_model", 256),
        n_gru_layers=enc_cfg.get("n_gru_layers", 2),
        proj_dim=enc_cfg.get("proj_dim", 128),
        mid_channels=enc_cfg.get("mid_channels", 64),
        dropout=enc_cfg.get("dropout", 0.1),
    )
