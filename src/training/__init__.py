"""Training and evaluation for KG-GT (Method 1)."""

from .evaluate import EvalMetrics, collect_predictions, compute_metrics, evaluate
from .train import TrainConfig, TrainResult, save_checkpoint, train_model

__all__ = [
    "TrainConfig",
    "TrainResult",
    "train_model",
    "save_checkpoint",
    "evaluate",
    "collect_predictions",
    "compute_metrics",
    "EvalMetrics",
]
