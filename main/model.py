"""
KG-GT model architecture for Method 1 (nb04 — no CCA, raw EEG channels).

Architecture:
  1. TransformerEncoder: Temporal encoder over CCA-aligned or raw EEG.
  2. MuscleNodeProjection: Projects transformer states to 5 muscle-node embeddings.
  3. MuscleGATEncoder / KinematicGuidedMuscleGATEncoder: Graph attention over muscles.
  4. KGGTModel: Full end-to-end model combining the above.
  5. CNN1dAligner: Optional learnable 1D CNN spatial filter replacing CCA.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Positional encoding
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the input sequence.

        PE[p, 2i]   = sin(p / 10000^(2i/d))
        PE[p, 2i+1] = cos(p / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) → x + PE[:, :T]."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ============================================================================
# Multi-head self-attention
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product MHSA using torch.nn.functional.scaled_dot_product_attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k     = d_k
        self.d_v     = d_v
        self.dropout = dropout

        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        B, T, _ = x.shape
        H, d_k, d_v = self.n_heads, self.d_k, self.d_v

        q = self.W_Q(x).view(B, T, H, d_k).transpose(1, 2)  # (B,H,T,d_k)
        k = self.W_K(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, d_v).transpose(1, 2)

        ctx = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, d_v)

        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H * d_v)
        return self.W_O(ctx)


# ============================================================================
# Feed-forward and encoder layers
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2  (ReLU)."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Post-LayerNorm Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(d_model, n_heads, d_k, d_v, dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_dim, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln1(x + self.dropout1(self.mhsa(x)))
        x = self.ln2(x + self.dropout2(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of L post-LN encoder layers with input embedding + PE."""

    def __init__(
        self,
        input_dim: int = 13,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_k: int = 32,
        d_v: int = 32,
        ffn_dim: int = 1024,
        dropout: float = 0.2,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(d_model, n_heads, d_k, d_v, ffn_dim, dropout)
            for _ in range(n_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, input_dim) → H_temp (B, T, d_model)."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, input_dim), got {tuple(x.shape)}")
        z = self.embed(x)
        z = self.pos_enc(z)
        for layer in self.layers:
            z = layer(z)
        return z


def build_transformer_from_config(cfg: dict, input_dim: int) -> TransformerEncoder:
    """Build TransformerEncoder from the model.transformer config section."""
    return TransformerEncoder(
        input_dim=input_dim,
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 8),
        d_k=cfg.get("d_k", 32),
        d_v=cfg.get("d_v", cfg.get("d_k", 32)),
        ffn_dim=cfg.get("ffn_dim", 1024),
        dropout=cfg.get("dropout", 0.2),
    )


# ============================================================================
# Node projection
# ============================================================================

class MuscleNodeProjection(nn.Module):
    """Project (B, T, D) transformer states into (B, T, N, F) nodes."""

    def __init__(self, input_dim: int, n_nodes: int = 5, node_dim: int = 32) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.proj = nn.Linear(input_dim, n_nodes * node_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = x.shape
        out = self.proj(x)
        return out.view(batch_size, steps, self.n_nodes, self.node_dim)


# ============================================================================
# Graph attention layers
# ============================================================================

class MuscleGATLayer(nn.Module):
    """Timewise multi-head attention over the fixed 5-muscle graph."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = out_dim or node_dim
        self.query = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.key = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.value = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.out = nn.Linear(hidden_dim * num_heads, self.out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.out_dim)
        self.residual = nn.Linear(node_dim, self.out_dim) if node_dim != self.out_dim else nn.Identity()
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        batch_size, steps, n_nodes, _ = nodes.shape
        flat = nodes.reshape(batch_size * steps, n_nodes, -1)
        query = self.query(flat).view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)
        key = self.key(flat).view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)
        value = self.value(flat).view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, value).permute(0, 2, 1, 3).reshape(batch_size * steps, n_nodes, self.num_heads * self.hidden_dim)
        out = self.out(out).view(batch_size, steps, n_nodes, self.out_dim)
        residual = self.residual(nodes)
        return self.norm(residual + self.dropout(out))


class MuscleGATEncoder(nn.Module):
    """Stacked baseline graph-attention encoder over the 5 EMG nodes."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_heads: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = node_dim
        for layer_idx in range(n_layers):
            layer_out = out_dim if layer_idx == n_layers - 1 else node_dim
            layers.append(
                MuscleGATLayer(
                    node_dim=in_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    out_dim=layer_out,
                )
            )
            in_dim = layer_out
        self.layers = nn.ModuleList(layers)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        out = nodes
        for layer in self.layers:
            out = layer(out)
        return out


class KinematicGuidedMuscleGATEncoder(nn.Module):
    """Graph attention whose edge scores are conditioned on kinematics."""

    def __init__(
        self,
        node_dim: int,
        kin_dim: int,
        hidden_dim: int,
        num_heads: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.node_proj = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.kin_proj = nn.Linear(kin_dim, hidden_dim * num_heads, bias=False)
        self.value_proj = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.out = nn.Linear(hidden_dim * num_heads, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, nodes: torch.Tensor, kin: torch.Tensor) -> torch.Tensor:
        batch_size, steps, n_nodes, _ = nodes.shape
        flat_nodes = nodes.reshape(batch_size * steps, n_nodes, -1)
        flat_kin = kin.reshape(batch_size * steps, -1)

        node_proj = self.node_proj(flat_nodes).view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)
        kin_proj = self.kin_proj(flat_kin).view(batch_size * steps, self.num_heads, self.hidden_dim).unsqueeze(2)
        values = self.value_proj(flat_nodes).view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim).permute(0, 2, 1, 3)

        guided = node_proj + kin_proj
        scores = torch.matmul(guided, guided.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, values).permute(0, 2, 1, 3).reshape(batch_size * steps, n_nodes, self.num_heads * self.hidden_dim)
        out = self.out(out).view(batch_size, steps, n_nodes, -1)
        residual = self.residual(nodes)
        return self.norm(residual + self.dropout(out))


# ============================================================================
# Full KG-GT model
# ============================================================================

class KGGTModel(nn.Module):
    """Transformer temporal encoder followed by muscle-graph reasoning."""

    def __init__(
        self,
        transformer_cfg: dict,
        input_dim: int,
        node_dim: int = 64,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        out_channels: int = 5,
        kin_dim: int = 13,
        n_gat_layers: int = 2,
        gat_dropout: float = 0.1,
        use_kinematic_guidance: bool = False,
    ) -> None:
        super().__init__()
        self.use_kinematic_guidance = use_kinematic_guidance
        self.out_channels = out_channels
        self.encoder = build_transformer_from_config(transformer_cfg, input_dim)
        self.node_projection = MuscleNodeProjection(
            input_dim=self.encoder.d_model,
            n_nodes=out_channels,
            node_dim=node_dim,
        )
        if use_kinematic_guidance:
            self.gat = KinematicGuidedMuscleGATEncoder(
                node_dim=node_dim,
                kin_dim=kin_dim,
                hidden_dim=gat_hidden_dim,
                num_heads=gat_heads,
                out_dim=node_dim,
                dropout=gat_dropout,
            )
        else:
            self.gat = MuscleGATEncoder(
                node_dim=node_dim,
                hidden_dim=gat_hidden_dim,
                num_heads=gat_heads,
                out_dim=node_dim,
                n_layers=n_gat_layers,
                dropout=gat_dropout,
            )
        self.decoder = nn.Linear(node_dim, 1)

    def forward(self, eeg: torch.Tensor, kin: torch.Tensor | None = None) -> torch.Tensor:
        temporal = self.encoder(eeg)
        nodes = self.node_projection(temporal)
        if self.use_kinematic_guidance:
            if kin is None:
                raise ValueError("kin is required when use_kinematic_guidance=True")
            refined = self.gat(nodes, kin)
        else:
            refined = self.gat(nodes)
        return self.decoder(refined).squeeze(-1)


def build_kg_gt_from_config(
    cfg: dict,
    input_dim: int,
    kin_dim: int = 13,
) -> KGGTModel:
    """Build a KG-GT variant from the project config."""
    model_cfg = cfg["model"]
    gat_cfg = model_cfg.get("gat", {})
    model_type = model_cfg.get("type", "transformer_regressor")
    use_kinematic_guidance = model_type == "kg_gt_kinematic" or gat_cfg.get("use_kinematic_guidance", False)
    return KGGTModel(
        transformer_cfg=model_cfg["transformer"],
        input_dim=input_dim,
        node_dim=gat_cfg.get("node_dim", 64),
        gat_hidden_dim=gat_cfg.get("hidden_dim", 64),
        gat_heads=gat_cfg.get("heads", 4),
        out_channels=model_cfg.get("decoder", {}).get("out_channels", 5),
        kin_dim=kin_dim,
        n_gat_layers=gat_cfg.get("n_layers", 2),
        gat_dropout=gat_cfg.get("dropout", model_cfg["transformer"].get("dropout", 0.1)),
        use_kinematic_guidance=use_kinematic_guidance,
    )


# ============================================================================
# CNN1d Aligner (optional, replaces CCA)
# ============================================================================

class CNN1dAligner(nn.Module):
    """Lightweight 1-D CNN that reduces EEG channels to a target embedding dim.

    Replaces the static CCA projection with a learnable, GPU-resident module.

    Input:  (B, T, C_in)   -- raw or bandpass-filtered EEG
    Output: (B, T, C_out)  -- aligned embedding

    Architecture:
        [Conv1d 1x1, GELU] -> [Conv1d kernel=3 depthwise, GELU] -> [Conv1d 1x1]
        + residual skip
        LayerNorm on output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck: int | None = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        mid = bottleneck or max(in_channels, out_channels)

        self.pw1 = nn.Conv1d(in_channels, mid, kernel_size=1, bias=False)
        self.dw  = nn.Conv1d(
            mid, mid,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=mid,
            bias=False,
        )
        self.pw2 = nn.Conv1d(mid, out_channels, kernel_size=1, bias=False)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C_in) -> (B, T, C_out)."""
        x_t = x.transpose(1, 2)                         # (B, C_in, T)
        h   = self.act(self.pw1(x_t))
        h   = self.act(self.dw(h))
        h   = self.pw2(h)                                # (B, C_out, T)
        out = h + self.skip(x_t)
        return self.norm(out.transpose(1, 2))            # (B, T, C_out)
