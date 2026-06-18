"""Loss functions for KG-GT (Method 1)."""

from .soft_dtw import (
    CombinedEMGLoss,
    SoftDTWLoss,
    build_loss_from_config,
    soft_dtw,
)

__all__ = [
    "soft_dtw",
    "SoftDTWLoss",
    "CombinedEMGLoss",
    "build_loss_from_config",
]
