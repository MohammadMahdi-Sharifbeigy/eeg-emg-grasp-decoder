"""
Transformer encoder for KG-GT (method1.tex §3.3.2).

Input is the CCA-aligned EEG sequence X̃ ∈ R^{B × T × d_in} (d_in = CCA
components, 13 by default). A linear embedding lifts it to d_model, sinusoidal
positional encoding is added (Eqs. 9-11), and L post-LN Transformer layers
model multi-lag cortico-muscular temporal coupling (Eqs. 12-17).

Output H_temp ∈ R^{B × T × d_model} feeds the kinematic-guided GAT.

Equation map (method1.tex):
  PE_sin / PE_cos          → SinusoidalPositionalEncoding   (Eqs. 9-10)
  Z^(0) = X̃ + PE           → TransformerEncoder.forward     (Eq. 11)
  Q,K,V / head / MHSA       → MultiHeadSelfAttention         (Eqs. 12-14)
  FFN                       → PositionwiseFeedForward        (Eq. 15)
  Z'^(l), Z^(l) (post-LN)   → TransformerEncoderLayer        (Eqs. 16-17)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional encoding — Eqs. 9-10
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the input sequence.

        PE[p, 2i]   = sin(p / 10000^(2i/d))
        PE[p, 2i+1] = cos(p / 10000^(2i/d))

    Precomputed up to max_len and registered as a (non-trainable) buffer so it
    moves with the module across devices and survives state_dict save/load.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                       # (max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # div_term[i] = 1 / 10000^(2i/d) = exp(-2i/d * ln(10000))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))              # (1, max_len, d)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) → x + PE[:, :T]."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-head self-attention — Eqs. 12-14
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

        head_i = softmax(Q_i K_iᵀ / √d_k) V_i
        MHSA   = Concat(head_1..head_H) W_O

    Separate per-head projection dims (d_k, d_v) are supported, as in the
    config (d_k = d_v = 32, H = 8 ⇒ H·d_k = d_model = 256).
    """

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
        self.d_k = d_k
        self.d_v = d_v

        # Stacked projections for all heads at once: (d_model → H·d_*)
        self.W_Q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.attn_dropout = nn.Dropout(p=dropout)
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) → (B, T, d_model)."""
        B, T, _ = x.shape
        H, d_k, d_v = self.n_heads, self.d_k, self.d_v

        # Project then split into heads: (B, H, T, d_*)
        q = self.W_Q(x).view(B, T, H, d_k).transpose(1, 2)
        k = self.W_K(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, d_v).transpose(1, 2)

        # Scaled dot-product attention (Eq. 13)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B,H,T,T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, v)                                  # (B,H,T,d_v)

        # Concat heads and project (Eq. 14)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H * d_v)
        return self.W_O(ctx)


# ---------------------------------------------------------------------------
# Position-wise feed-forward — Eq. 15
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# One encoder layer — post-LN, Eqs. 16-17
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Post-LayerNorm Transformer encoder layer.

        Z'^(l) = LN( Z^(l-1) + MHSA(Z^(l-1)) )     (Eq. 16)
        Z^(l)  = LN( Z'^(l)  + FFN(Z'^(l))  )      (Eq. 17)

    Dropout is applied to each sublayer output before the residual add
    (standard "Attention Is All You Need" placement).
    """

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
        # Sublayer 1: MHSA + residual + LN
        x = self.ln1(x + self.dropout1(self.mhsa(x)))
        # Sublayer 2: FFN + residual + LN
        x = self.ln2(x + self.dropout2(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# Full encoder stack
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Stack of L post-LN encoder layers with input embedding + PE.

    Pipeline (method1.tex Eq. 11 onward):
        X̃ (B,T,d_in)
          → Linear embed (d_in → d_model)
          → + sinusoidal PE                       = Z^(0)
          → L × TransformerEncoderLayer
          → H_temp (B, T, d_model)

    The CCA stage outputs d_in (= n_components, 13) channels; the figure adds
    PE at that width but H_temp is 256-dim, so the embedding lifts d_in →
    d_model before the layers. If d_in == d_model the embedding is still a
    learnable linear map (no-op shape-wise).
    """

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
        """x: CCA-aligned EEG (B, T, input_dim) → H_temp (B, T, d_model)."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, input_dim), got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"input feature dim {x.size(-1)} != input_dim {self.input_dim}"
            )

        z = self.embed(x)            # (B, T, d_model)
        z = self.pos_enc(z)          # + PE  → Z^(0)
        for layer in self.layers:
            z = layer(z)             # Z^(l)
        return z                     # H_temp


# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------

def build_transformer_from_config(cfg: dict, input_dim: int) -> TransformerEncoder:
    """Build TransformerEncoder from the model.transformer config section.

    Args:
        cfg: model.transformer dict (n_layers, n_heads, d_model, d_k, d_v,
             ffn_dim, dropout).
        input_dim: CCA output width (preprocessing.cca.n_components).
    """
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
