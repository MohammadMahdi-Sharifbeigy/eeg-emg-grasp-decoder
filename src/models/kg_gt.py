"""Composed transformer-to-GAT EEG-to-EMG model variants."""

from __future__ import annotations

import torch
import torch.nn as nn

from .gat_projection import MuscleNodeProjection
from .kg_gat import KinematicGuidedMuscleGATEncoder, MuscleGATEncoder
from .transformer import build_transformer_from_config


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
    """Build a separate KG-GT variant from the project config."""
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
