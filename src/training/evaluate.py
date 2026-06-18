"""
Evaluation metrics for the KG-GT EMG regressor.

Collects predictions over a DataLoader on-device, then reports per-channel
RMSE, MAE, and Pearson r on the (z-scored) EMG targets. Inference runs under
autocast on CUDA for speed; metrics are computed in float64 on CPU for
numerical stability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

PrepareBatch = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class EvalMetrics:
    """Per-channel and mean regression metrics."""

    rmse: np.ndarray   # (C,)
    mae: np.ndarray    # (C,)
    pearson: np.ndarray  # (C,)
    channel_names: list[str]

    def as_table(self) -> str:
        """Format metrics as an aligned text table."""
        lines = [f'{"channel":20s} {"RMSE":>8s} {"MAE":>8s} {"Pearson":>8s}']
        for c, name in enumerate(self.channel_names):
            lines.append(
                f"{name:20s} {self.rmse[c]:8.3f} {self.mae[c]:8.3f} {self.pearson[c]:8.3f}"
            )
        lines.append(
            f'{"MEAN":20s} {self.rmse.mean():8.3f} '
            f"{self.mae.mean():8.3f} {self.pearson.mean():8.3f}"
        )
        return "\n".join(lines)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    prepare_batch: PrepareBatch,
    device: torch.device,
    n_channels: int = 5,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return stacked (pred, target).

    Returns:
        (P, Y) each (N_total, n_channels) float32 numpy arrays.
    """
    model.eval()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp = use_amp and device.type == "cuda"

    preds, targets = [], []
    for eeg, _kin, emg in loader:
        x, y = prepare_batch(eeg, emg)
        with torch.amp.autocast(device_type=amp_device, enabled=amp):
            pred = model(x)
        preds.append(pred.float().cpu().numpy().reshape(-1, n_channels))
        targets.append(y.float().cpu().numpy().reshape(-1, n_channels))

    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    channel_names: list[str] | None = None,
) -> EvalMetrics:
    """Per-channel RMSE, MAE, Pearson r between pred and target.

    Args:
        pred:   (N, C)
        target: (N, C)
        channel_names: optional labels, defaults to ch0..ch{C-1}.
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    n_channels = pred.shape[1]

    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(n_channels)]

    rmse = np.sqrt(((pred - target) ** 2).mean(0))
    mae = np.abs(pred - target).mean(0)
    pearson = np.array(
        [np.corrcoef(pred[:, c], target[:, c])[0, 1] for c in range(n_channels)]
    )
    return EvalMetrics(rmse=rmse, mae=mae, pearson=pearson, channel_names=channel_names)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    prepare_batch: PrepareBatch,
    device: torch.device,
    channel_names: list[str] | None = None,
    n_channels: int = 5,
    use_amp: bool = True,
) -> EvalMetrics:
    """End-to-end evaluation: collect predictions then compute metrics."""
    pred, target = collect_predictions(
        model, loader, prepare_batch, device, n_channels=n_channels, use_amp=use_amp
    )
    metrics = compute_metrics(pred, target, channel_names)
    logger.info("evaluation complete:\n%s", metrics.as_table())
    return metrics
