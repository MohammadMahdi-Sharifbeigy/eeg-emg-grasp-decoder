"""Current runnable transformer-plus-linear-head model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import build_transformer_from_config


class TransformerRegressor(nn.Module):
    """Transformer encoder followed by a linear EMG readout head."""

    def __init__(self, cfg: dict, input_dim: int, out_channels: int = 5):
        super().__init__()
        self.encoder = build_transformer_from_config(
            cfg["model"]["transformer"],
            input_dim,
        )
        self.head = nn.Linear(self.encoder.d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)


def build_transformer_regressor(
    cfg: dict,
    input_dim: int,
    out_channels: int = 5,
) -> TransformerRegressor:
    """Build the current runnable EEG-to-EMG regressor from config."""
    return TransformerRegressor(cfg=cfg, input_dim=input_dim, out_channels=out_channels)
