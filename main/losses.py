"""
Loss functions for the KG-GT EMG regressor.

CombinedEMGLoss was originally a mix of MSE and SoftDTW. 
Soft-DTW has been hard-removed due to O(T^2) performance issues at T=4000.
This now strictly returns pure MSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

class CombinedEMGLoss(nn.Module):
    """Pure MSE Loss wrapper (Soft-DTW removed for performance)."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.mse(pred, target)


def build_loss_from_config(cfg: dict) -> CombinedEMGLoss:
    """Build CombinedEMGLoss from the training config section."""
    return CombinedEMGLoss()

