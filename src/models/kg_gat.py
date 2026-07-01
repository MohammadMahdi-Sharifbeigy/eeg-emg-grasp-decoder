"""Muscle-graph attention blocks for transformer-to-EMG decoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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
        if nodes.dim() != 4:
            raise ValueError(f"expected (B, T, N, F), got {tuple(nodes.shape)}")
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
        if nodes.dim() != 4:
            raise ValueError(f"expected nodes (B, T, N, F), got {tuple(nodes.shape)}")
        if kin.dim() != 3:
            raise ValueError(f"expected kin (B, T, K), got {tuple(kin.shape)}")

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
