"""Projection from transformer temporal states to 5 muscle-node embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class MuscleNodeProjection(nn.Module):
    """Project ``(B, T, D)`` transformer states into ``(B, T, N, F)`` nodes."""

    def __init__(self, input_dim: int, n_nodes: int = 5, node_dim: int = 32) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.proj = nn.Linear(input_dim, n_nodes * node_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got {tuple(x.shape)}")
        batch_size, steps, _ = x.shape
        out = self.proj(x)
        return out.view(batch_size, steps, self.n_nodes, self.node_dim)
