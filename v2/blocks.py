"""Encoders of KG-GT v2, all built from v2.attention and all pre-norm residual: x + f(norm(x))."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention


class AttentionBlock(nn.Module):
    """x + attention(norm(x)), then with ffn_dim also x + feed_forward(norm(x)).

    With n_nodes it is a graph layer: the attention logits get a learned edge bias per head and a
    scaled fixed prior (for example scalp distance or muscle co-activation). The prior must be
    (n_nodes, n_nodes); every layer keeps its own copy of it.
    """

    def __init__(self, d, n_heads, dropout, ffn_dim=None, n_nodes=None, prior=None):
        super().__init__()
        self.n1, self.attn, self.drop = nn.LayerNorm(d), Attention(d, n_heads, dropout), nn.Dropout(dropout)
        self.ffn = None
        if ffn_dim:
            self.n2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d))
        self.edge = None
        if n_nodes:
            self.edge = nn.Parameter(torch.zeros(n_heads, n_nodes, n_nodes))
            self.prior_scale = nn.Parameter(torch.ones(()))
            if prior is None:
                p = torch.zeros(n_nodes, n_nodes)
            else:                                                      # an own copy: nothing shared with the caller
                p = torch.as_tensor(prior, dtype=torch.float32).detach().clone()
                if tuple(p.shape) != (n_nodes, n_nodes):
                    raise ValueError(f"prior has shape {tuple(p.shape)}, expected ({n_nodes}, {n_nodes})")
            self.register_buffer("prior", p)

    def forward(self, x, mask=None):
        """mask: optional additive (n, n) logit mask (0 = allowed, -inf = blocked), e.g. causal_mask(t)."""
        bias = None if self.edge is None else self.edge + self.prior_scale * self.prior
        if mask is not None:
            bias = mask if bias is None else bias + mask
        x = x + self.drop(self.attn(self.n1(x), bias=bias))
        if self.ffn is not None:
            x = x + self.drop(self.ffn(self.n2(x)))
        return x


def causal_mask(t: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(t, t) additive attention mask: 0 where the source is not later than the query, -inf above the diagonal."""
    return torch.full((t, t), float("-inf"), device=device, dtype=dtype).triu(1)


class TemporalEncoder(nn.Module):
    """Self-attention over time with sinusoidal positions: (B, T, d) -> (B, T, d).

    With causal=True every token attends only to itself and earlier tokens.
    """

    def __init__(self, d, n_layers, n_heads, ffn_dim, dropout, causal=False):
        super().__init__()
        self.d, self.causal = d, bool(causal)
        self.layers = nn.ModuleList([AttentionBlock(d, n_heads, dropout, ffn_dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def _positions(self, t, device, dtype):
        pos = torch.arange(t, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / self.d))
        pe = torch.zeros(t, self.d, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe.to(dtype)

    def forward(self, x):
        x = x + self._positions(x.shape[1], x.device, x.dtype)
        mask = causal_mask(x.shape[1], x.device) if self.causal else None
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class GraphEncoder(nn.Module):
    """Attention over the node axis of (..., n_nodes, d), biased by learned edges and a fixed prior."""

    def __init__(self, n_nodes, d, n_layers, n_heads, prior=None, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(d, n_heads, dropout, n_nodes=n_nodes, prior=prior)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MuscleHead(nn.Module):
    """One token -> one node per muscle -> muscle graph -> one non-negative value per muscle."""

    def __init__(self, d_in, n_muscles=5, node_dim=32, n_layers=2, n_heads=4, prior=None, dropout=0.1):
        super().__init__()
        self.m, self.nd = n_muscles, node_dim
        self.proj = nn.Linear(d_in, n_muscles * node_dim)
        self.graph = GraphEncoder(n_muscles, node_dim, n_layers, n_heads, prior, dropout)
        self.dec_w = nn.Parameter(torch.randn(n_muscles, node_dim) * 0.05)
        self.dec_b = nn.Parameter(torch.full((n_muscles,), -2.0))      # softplus(-2) = 0.13: start near rest

    def forward(self, x):
        b, t, _ = x.shape
        n = self.graph(self.proj(x).view(b, t, self.m, self.nd))
        return F.softplus((n * self.dec_w.to(n.dtype)).sum(-1) + self.dec_b.to(n.dtype))
