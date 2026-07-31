"""
KG-GT model architecture for Method 1 (nb04 — no CCA, raw EEG channels).

Architecture:
  1. TransformerEncoder: Temporal encoder over CCA-aligned or raw EEG.
  2. MuscleNodeProjection: Projects transformer states to 5 muscle-node embeddings.
  3. MuscleGATEncoder / KinematicGuidedMuscleGATEncoder: Graph attention over muscles.
  4. KGGTModel: Full end-to-end model combining the above.
  5. CNN1dAligner: Optional learnable 1D CNN spatial filter replacing CCA.

GAT changes (vs. original dense self-attention):
  - MuscleGATLayer now adds a learnable (num_heads, n_nodes, n_nodes) edge_bias
    to scores before softmax, making it a proper graph-attention layer.
    Optionally initialised from a correlation-matrix prior via edge_prior arg.
  - KinematicGuidedMuscleGATEncoder replaces the node-broadcast kin_proj with
    a small MLP that maps kin_dim → (num_heads, n_nodes, n_nodes) edge bias,
    conditioned per timestep. A static edge_bias is also added (same as above).
"""

from __future__ import annotations

import logging
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# One-shot flag: log SDPA backend selection only once per process
_SDPA_BACKEND_LOGGED = False


def _log_sdpa_backend_once(device_type: str) -> None:
    """Log which scaled_dot_product_attention backend is active (once per process).

    Flash attention requires CUDA + fp16/bf16. Memory-efficient attention is
    the fallback on older GPUs or fp32. Math backend is pure-PyTorch (slowest).
    """
    global _SDPA_BACKEND_LOGGED
    if _SDPA_BACKEND_LOGGED:
        return
    _SDPA_BACKEND_LOGGED = True

    if device_type != "cuda":
        logger.info("SDPA backend: math (CPU path)")
        return

    flash = torch.backends.cuda.flash_sdp_enabled()
    mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
    if flash:
        backend = "flash_attention"
    elif mem_eff:
        backend = "memory_efficient"
    else:
        backend = "math (slowest — consider upgrading PyTorch or using fp16)"

    msg = (
        f"SDPA backend active: {backend}  "
        f"(flash={flash}, mem_efficient={mem_eff}). "
        "For best performance with T=4000, ensure AMP (fp16) is enabled and "
        "PyTorch >= 2.0 is installed."
    )
    logger.info(msg)
    print(f"[SDPA] {msg}")


# ============================================================================
# Positional encoding
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the input sequence.

        PE[p, 2i]   = sin(p / 10000^(2i/d))
        PE[p, 2i+1] = cos(p / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0) -> None:
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
        """x: (B, T, d_model) -> (B, T, d_model).

        DUAL-MODE IMPLEMENTATION:
          • Training mode:  Uses torch.nn.functional.scaled_dot_product_attention
            (fused CUDA kernel — FlashAttention / memory-efficient backend).
            Fast, memory-efficient, but the attention matrix is NOT returned.

          • Eval mode:      Manually computes softmax(QK^T/√d_k)·V.
            The full (B, H, T, T) attention matrix is stored in
            self.last_attn_weights for EEG temporal interpretability.
            Cost: O(T²) memory — acceptable at batch_size=1 for visualization.

        Access after model.eval() + forward pass:
            attn = model.encoder.layers[-1].mhsa.last_attn_weights  # (B, H, T, T)
            # Mean over heads for a (B, T, T) temporal relevance map.
            # Sum over query axis → (B, T,) column attention score per timestep.
        """
        B, T, _ = x.shape
        H, d_k, d_v = self.n_heads, self.d_k, self.d_v

        q = self.W_Q(x).view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_K(x).view(B, T, H, d_k).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, d_v).transpose(1, 2)

        if not self.training:
            # ── INTERPRETABILITY PATH (eval mode) ─────────────────────────
            # Manual scaled dot-product attention exposes the full attention matrix.
            # No dropout applied during inference.
            scale  = 1.0 / math.sqrt(d_k)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
            attn   = torch.softmax(scores, dim=-1)                  # (B, H, T, T)

            # Store for temporal interpretability (EEG attention heatmap).
            # Detach to avoid accumulating in the computation graph.
            self.last_attn_weights = attn.detach()

            ctx = torch.matmul(attn, v)                             # (B, H, T, d_v)
        else:
            # ── FAST TRAINING PATH ────────────────────────────────────────
            # Uses the fused SDPA kernel (FlashAttention when available).
            # Attention matrix is NOT stored to save memory during training.
            _log_sdpa_backend_once(x.device.type)
            ctx = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout,
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
    """Stack of L post-LN encoder layers with input embedding + PE.

    Chunked attention: if chunk_size is set and T % chunk_size == 0, the
    sequence is split into (T // chunk_size) independent chunks before the
    transformer layers, reducing attention cost from O(T²) to O(chunk²).
    If T % chunk_size != 0, a warning is emitted and full O(T²) attention
    is used instead — this can be catastrophic at large T.
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
        chunk_size: int = 500,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.chunk_size = chunk_size

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

        # ── Chunking trick (TRAINING MODE ONLY) ────────────────────────────
        # Reduces attention from O(T²) to O(chunk_size²) per chunk during training.
        # REQUIREMENT: T must be exactly divisible by chunk_size.
        #
        # INTERPRETABILITY NOTE: Chunking is intentionally DISABLED during eval().
        # Rationale: each chunk independently produces a (B*chunks, H, cs, cs)
        # attention matrix. For the EEG temporal attention heatmap (money plot),
        # we need a single unfragmented (B, H, T, T) matrix from the last encoder
        # layer's mhsa.last_attn_weights. Disabling chunking in eval() ensures this
        # at the cost of O(T²) memory — acceptable for batch_size=1 inference.
        is_chunked = False
        B, T, C = z.shape
        cs = getattr(self, "chunk_size", None)
        if self.training and cs is not None and T > cs:
            if T % cs != 0:
                warnings.warn(
                    f"TransformerEncoder: T={T} is NOT divisible by chunk_size={cs}. "
                    f"Falling back to FULL O(T²) self-attention — this will be very slow "
                    f"and memory-heavy at T={T}. "
                    f"Fix: set chunk_size to a divisor of T (e.g. chunk_size={T} for no "
                    f"chunking, or chunk_size=500 requires T divisible by 500).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                num_chunks = T // cs
                z = z.view(B * num_chunks, cs, C)
                is_chunked = True

        for layer in self.layers:
            z = layer(z)

        if is_chunked:
            z = z.view(B, T, C)

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
        chunk_size=cfg.get("chunk_size", 500),
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
    """Timewise multi-head graph attention over the fixed 5-muscle graph.

    Key fix vs. original: a learnable (num_heads, n_nodes, n_nodes) edge_bias
    matrix is added to the raw QK scores before softmax. This makes it a
    genuine graph-attention layer — each head can learn different edge strengths
    between muscle pairs.

    Optionally, a correlation-matrix prior (edge_prior) can initialise the bias
    as log(prior + eps), broadcast across heads. The bias remains learnable so
    the model can adjust away from the prior.

    Args:
        node_dim:   Input node feature dimension.
        hidden_dim: Per-head key/query dimension.
        num_heads:  Number of attention heads.
        n_nodes:    Number of graph nodes (default 5, one per EMG channel).
        dropout:    Dropout on attention weights.
        out_dim:    Output node feature dimension (defaults to node_dim).
        edge_prior: Optional (n_nodes, n_nodes) float tensor. If provided,
                    edge_bias is initialised as log(edge_prior + 1e-6) broadcast
                    across heads, instead of zeros. Must be non-negative.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_heads: int,
        n_nodes: int = 5,
        dropout: float = 0.0,
        out_dim: int | None = None,
        edge_prior: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.n_nodes    = n_nodes
        self.out_dim    = out_dim or node_dim

        self.query    = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.key      = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.value    = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.out      = nn.Linear(hidden_dim * num_heads, self.out_dim)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(self.out_dim)
        self.residual = (
            nn.Linear(node_dim, self.out_dim) if node_dim != self.out_dim else nn.Identity()
        )
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # ── Learnable edge-bias (H, N, N) ──────────────────────────────────
        # Initialised from prior if supplied, else zeros.
        # Always learnable (requires_grad=True).
        if edge_prior is not None:
            if edge_prior.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"edge_prior must be ({n_nodes}, {n_nodes}), got {tuple(edge_prior.shape)}"
                )
            # log(prior + eps) broadcast across heads
            log_prior = torch.log(edge_prior.float().clamp(min=0.0) + 1e-6)
            init_bias = log_prior.unsqueeze(0).expand(num_heads, -1, -1).clone()
        else:
            init_bias = torch.zeros(num_heads, n_nodes, n_nodes)

        self.edge_bias = nn.Parameter(init_bias)   # (H, N, N), always learnable

        # Register the initial biological prior as a FROZEN buffer.
        # Used by EdgePriorKLDivLoss (losses.py) which computes:
        #   KL( softmax(edge_bias) || softmax(edge_prior_anchor) )
        # This anchors the learned muscle connectivity distribution to the
        # EMG correlation prior and only allows deviation when training gradients
        # strongly demand it. The buffer moves with the model (GPU/CPU) but
        # receives no gradients — it is a reference, not a trainable parameter.
        self.register_buffer("edge_prior_anchor", init_bias.clone())

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """nodes: (B, T, N, node_dim) → (B, T, N, out_dim)."""
        batch_size, steps, n_nodes, _ = nodes.shape
        flat = nodes.reshape(batch_size * steps, n_nodes, -1)

        q = (
            self.query(flat)
            .view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim)
            .permute(0, 2, 1, 3)
        )  # (B*T, H, N, d_k)
        k = (
            self.key(flat)
            .view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.value(flat)
            .view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim)
            .permute(0, 2, 1, 3)
        )

        # Raw attention scores + learnable graph edge bias
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*T, H, N, N)
        scores = scores + self.edge_bias                             # broadcast over B*T

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Store for explainability plotting
        self.last_attn_weights = attn.detach()

        out = (
            torch.matmul(attn, v)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * steps, n_nodes, self.num_heads * self.hidden_dim)
        )
        out = self.out(out).view(batch_size, steps, n_nodes, self.out_dim)
        residual = self.residual(nodes)
        return self.norm(residual + self.dropout(out))


class MuscleGATEncoder(nn.Module):
    """Stacked graph-attention encoder over the 5 EMG nodes.

    Each layer is a MuscleGATLayer with its own learnable edge_bias.
    edge_prior (if supplied) is used to initialise the bias in every layer.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_heads: int,
        out_dim: int,
        n_nodes: int = 5,
        n_layers: int = 2,
        dropout: float = 0.0,
        edge_prior: Tensor | None = None,
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
                    n_nodes=n_nodes,
                    dropout=dropout,
                    out_dim=layer_out,
                    edge_prior=edge_prior,
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
    """Graph attention whose edge scores are conditioned on kinematics.

    Fix vs. original: instead of broadcasting kin to nodes (which gives all
    edges the same kinematic influence), a small MLP maps the per-timestep
    kinematic vector to a (num_heads, n_nodes, n_nodes) edge-bias matrix.
    This allows different kinematic states to express different muscle-pair
    co-activation patterns.

    Additionally, a static learnable edge_bias (same as MuscleGATLayer) is
    added, optionally initialised from a correlation prior. The total score is:

        scores = QK^T / sqrt(d) + static_edge_bias + kin_edge_bias(kin_t)

    The MLP is deliberately small: kin_dim → kin_hidden → heads * n_nodes²
    (e.g. 12 → 64 → 100 parameters for 4 heads, 5 nodes).

    Args:
        node_dim:   Input node feature dimension.
        kin_dim:    Kinematic vector dimension (e.g. 12 after dropping rho_GL).
        hidden_dim: Per-head key/query dimension.
        num_heads:  Number of attention heads.
        out_dim:    Output node feature dimension.
        n_nodes:    Number of muscle nodes (default 5).
        kin_hidden: Hidden size in the kin→edge MLP (default 64).
        dropout:    Dropout on attention weights.
        edge_prior: Optional (n_nodes, n_nodes) tensor for static bias init.
    """

    def __init__(
        self,
        node_dim: int,
        kin_dim: int,
        hidden_dim: int,
        num_heads: int,
        out_dim: int,
        n_nodes: int = 5,
        kin_hidden: int = 64,  # deprecated: unused after kin_edge_mlp → kin_edge_linear
        dropout: float = 0.0,
        edge_prior: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.n_nodes    = n_nodes

        # Node Q/K/V projections (unchanged from original)
        self.node_proj  = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.value_proj = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.out        = nn.Linear(hidden_dim * num_heads, out_dim)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(out_dim)
        self.residual   = (
            nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()
        )
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # ── Static learnable edge-bias (H, N, N) ───────────────────────────
        # Same as MuscleGATLayer: prior-init or zeros.
        if edge_prior is not None:
            if edge_prior.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"edge_prior must be ({n_nodes}, {n_nodes}), got {tuple(edge_prior.shape)}"
                )
            log_prior = torch.log(edge_prior.float().clamp(min=0.0) + 1e-6)
            init_bias = log_prior.unsqueeze(0).expand(num_heads, -1, -1).clone()
        else:
            init_bias = torch.zeros(num_heads, n_nodes, n_nodes)
        self.edge_bias = nn.Parameter(init_bias)   # (H, N, N)

        # Register the initial biological prior as a FROZEN buffer for KL regularization.
        # EdgePriorKLDivLoss (losses.py) computes:
        #   KL( softmax(edge_bias) || softmax(edge_prior_anchor) ) summed over all GAT layers.
        # This anchors the dynamic kinematic muscle-graph to the known co-activation structure.
        self.register_buffer("edge_prior_anchor", init_bias.clone())

        # ── Transparent kinematic → edge LINEAR mapping ─────────────────────
        # DESIGN CHOICE (Q1 answer: full transparency):
        # A single linear layer with NO hidden layer and NO activation maps the
        # per-timestep kinematic state directly to (num_heads × n_nodes²) edge scalars.
        #
        # INTERPRETABILITY: After training, inspect:
        #   W = model.gat.kin_edge_linear.weight      # shape: (H*N², kin_dim)
        #   W_4d = W.view(num_heads, n_nodes, n_nodes, kin_dim)
        # W_4d[h, i, j, k] = direct linear contribution of kinematic feature k
        # to the attention edge i→j in head h.
        #
        # EXPECTED NEUROPHYSIOLOGICAL PATTERN:
        #   d_grip (col 9, grip aperture)  → high weights on FDI-APB pinch edges
        #   F_L / F_G (cols 10-11, forces)  → high weights on power-grasp muscle pairs
        #   p_wrist (cols 0-2, position)    → low/diffuse — less muscle-specific
        # Deviations from this pattern are scientifically interesting findings.
        self.kin_edge_linear = nn.Linear(kin_dim, num_heads * n_nodes * n_nodes, bias=True)

    def forward(self, nodes: torch.Tensor, kin: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Kinematic-Guided Muscle GAT.

        Args:
            nodes: (B, T, N, node_dim) — muscle node embeddings from MuscleNodeProjection.
            kin:   (B, T, kin_dim)     — kinematic state vectors (13-dim k_t or 26-dim with vel).

        Returns:
            (B, T, N, out_dim) — graph-attended muscle node features.

        EDGE SCORE DECOMPOSITION per timestep t:
            scores_t = QK^T / √d_k       ← content attention (from node features)
                     + self.edge_bias     ← static muscle synergy (H, N, N); biological prior
                     + kin_bias_t         ← kinematic modulation (transparent linear)

        INTERPRETABILITY (eval() mode only):
            self.last_attn_weights      → (B*T, H, N, N)  total softmax attention weights
            self.last_kin_edge_bias     → (B, T, H, N, N) kinematic dynamic edge contribution
            self.last_static_edge_bias  → (H, N, N)        static learned synergy bias

        After training, inspect the kinematic mapping:
            W = model.gat.kin_edge_linear.weight.view(H, N, N, kin_dim)
            W[h, i, j, :]  → how each kinematic feature contributes to the i→j edge in head h
        """
        batch_size, steps, n_nodes, _ = nodes.shape
        flat_nodes = nodes.reshape(batch_size * steps, n_nodes, -1)  # (B*T, N, node_dim)
        flat_kin   = kin.reshape(batch_size * steps, -1)              # (B*T, kin_dim)

        # Q and V from nodes
        q = (
            self.node_proj(flat_nodes)
            .view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim)
            .permute(0, 2, 1, 3)
        )  # (B*T, H, N, d_k)
        v = (
            self.value_proj(flat_nodes)
            .view(batch_size * steps, n_nodes, self.num_heads, self.hidden_dim)
            .permute(0, 2, 1, 3)
        )  # (B*T, H, N, d_v)

        # ── Score Component 1: content-based attention ─────────────────────
        scores = torch.matmul(q, q.transpose(-2, -1)) * self.scale  # (B*T, H, N, N)

        # ── Score Component 2: static biological synergy bias ──────────────
        # Encodes stable muscle co-activation patterns (e.g. FDI-APB during pinch).
        # Anchored to EMG correlation prior via KL regularization in losses.py.
        # Broadcast over B*T: every timestep starts from the same static prior.
        scores = scores + self.edge_bias  # (H, N, N) → (B*T, H, N, N)

        # ── Score Component 3: dynamic kinematic modulation ────────────────
        # Transparent single linear layer: kin_dim → H*N² (no hidden layer).
        # W[h*N*N + i*N + j, k] = contribution of kinematic feature k to edge i→j in head h.
        kin_bias = self.kin_edge_linear(flat_kin)                              # (B*T, H*N*N)
        kin_bias = kin_bias.view(batch_size * steps, self.num_heads, n_nodes, n_nodes)
        scores   = scores + kin_bias                                            # (B*T, H, N, N)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # ── Interpretability Storage ───────────────────────────────────────
        # Total attention is stored in both training and eval modes (lightweight: detached).
        # Decomposed edge components are stored ONLY in eval() mode to avoid
        # extra memory allocation during training.
        self.last_attn_weights = attn.detach()
        if not self.training:
            # Reshape kin_bias to (B, T, H, N, N) for time-resolved visualization.
            # Plot self.last_kin_edge_bias[:, :, :, i, j] to see how the i→j
            # muscle edge evolves over the inference window.
            self.last_kin_edge_bias = kin_bias.view(
                batch_size, steps, self.num_heads, n_nodes, n_nodes
            ).detach()
            # Static bias: same (H, N, N) tensor every call — just a reference.
            self.last_static_edge_bias = self.edge_bias.detach()

        out = (
            torch.matmul(attn, v)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * steps, n_nodes, self.num_heads * self.hidden_dim)
        )
        out      = self.out(out).view(batch_size, steps, n_nodes, -1)
        residual = self.residual(nodes)
        return self.norm(residual + self.dropout(out))    # ← BUG FIX: missing return added
# ============================================================================
# Transformer-Only Model (Phase 1 Pretraining)
# ============================================================================

class TransformerOnlyModel(nn.Module):
    """Phase 1 pretraining model: Transformer -> NodeProjection -> Decoder (no GAT)."""
    
    def __init__(
        self,
        transformer_cfg: dict,
        input_dim: int,
        node_dim: int = 64,
        out_channels: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = build_transformer_from_config(transformer_cfg, input_dim)
        self.node_projection = MuscleNodeProjection(
            input_dim=self.encoder.d_model,
            n_nodes=out_channels,
            node_dim=node_dim,
        )
        self.decoder = nn.Linear(node_dim, 1)
        
    def forward(self, eeg: torch.Tensor, kin: torch.Tensor | None = None) -> torch.Tensor:
        # kin is ignored, just matching the function signature
        temporal = self.encoder(eeg)
        nodes    = self.node_projection(temporal)
        return self.decoder(nodes).squeeze(-1)


# ============================================================================
# Full KG-GT model
# ============================================================================

class KGGTModel(nn.Module):
    """Transformer temporal encoder followed by muscle-graph reasoning.

    Args:
        edge_prior: Optional (n_nodes, n_nodes) float tensor used to initialise
                    the learnable edge-bias in MuscleGATLayer /
                    KinematicGuidedMuscleGATEncoder. Compute via
                    compute_muscle_edge_prior() in preprocessing_emg_kin.py.
    """

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
        edge_prior: Tensor | None = None,
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
                n_nodes=out_channels,
                dropout=gat_dropout,
                edge_prior=edge_prior,
            )
        else:
            self.gat = MuscleGATEncoder(
                node_dim=node_dim,
                hidden_dim=gat_hidden_dim,
                num_heads=gat_heads,
                out_dim=node_dim,
                n_nodes=out_channels,
                n_layers=n_gat_layers,
                dropout=gat_dropout,
                edge_prior=edge_prior,
            )

        self.decoder = nn.Linear(node_dim, 1)

    def forward(self, eeg: torch.Tensor, kin: torch.Tensor | None = None) -> torch.Tensor:
        temporal = self.encoder(eeg)
        nodes    = self.node_projection(temporal)
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
    edge_prior: Tensor | None = None,
) -> KGGTModel:
    """Build a KG-GT variant from the project config.

    Args:
        cfg:        Full project config dict.
        input_dim:  EEG input channels (e.g. 32).
        kin_dim:    Kinematic feature dimension after preprocessing.
        edge_prior: Optional (5, 5) tensor from compute_muscle_edge_prior().
                    If None, edge_bias is initialised to zeros.
                    Compute via:
                        from main.preprocessing_emg_kin import compute_muscle_edge_prior
                        import torch
                        prior_np = compute_muscle_edge_prior(train_emgs)
                        edge_prior = torch.from_numpy(prior_np)
    """
    model_cfg = cfg["model"]
    gat_cfg   = model_cfg.get("gat", {})
    model_type = model_cfg.get("type", "transformer_regressor")
    use_kinematic_guidance = (
        model_type == "kg_gt_kinematic"
        or gat_cfg.get("use_kinematic_guidance", False)
    )

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
        edge_prior=edge_prior,
    )


def build_transformer_only_from_config(cfg: dict, input_dim: int) -> TransformerOnlyModel:
    """Build the Phase 1 pretraining model (no GAT) from a config dictionary."""
    model_cfg = cfg.get("model", {})
    transformer_cfg = model_cfg.get("transformer", {})
    return TransformerOnlyModel(
        transformer_cfg=transformer_cfg,
        input_dim=input_dim,
        node_dim=model_cfg.get("gat", {}).get("node_dim", 64),
        out_channels=model_cfg.get("decoder", {}).get("out_channels", 5),
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
