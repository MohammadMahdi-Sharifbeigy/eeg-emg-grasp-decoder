"""
Device-aware training loop for the KG-GT EMG regressor.

Runs on CUDA when available (GTX 1660 Ti / sm_75) and falls back to CPU with
the same code path. GPU-specific features:
  * Automatic Mixed Precision (AMP) via torch.amp — roughly halves activation
    memory and uses the Turing FP16 tensor cores, which matters on 6 GB VRAM.
    Disabled automatically on CPU.
  * Gradients scaled with GradScaler to avoid FP16 underflow.

The data pipeline projects EEG -> CCA and z-scores EMG outside the model, so
the caller passes a ``prepare_batch`` callable: (eeg, emg) -> (x, y) on device.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

PrepareBatch = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TrainConfig:
    """Resolved training hyperparameters (subset of configs/default.yaml)."""

    lr: float = 1e-3
    lr_patience: int = 50
    lr_factor: float = 0.5
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 30
    max_epochs: int = 500
    use_amp: bool = True

    @classmethod
    def from_config(cls, cfg: dict, max_epochs: int | None = None) -> "TrainConfig":
        """Build from the training section of default.yaml.

        Args:
            cfg: training dict.
            max_epochs: override (e.g. small value for a smoke run).
        """
        return cls(
            lr=cfg.get("lr", 1e-3),
            lr_patience=cfg.get("lr_patience", 50),
            lr_factor=cfg.get("lr_factor", 0.5),
            grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
            early_stop_patience=cfg.get("early_stop_patience", 30),
            max_epochs=max_epochs if max_epochs is not None else cfg.get("max_epochs", 500),
            use_amp=cfg.get("use_amp", True),
        )


@dataclass
class TrainResult:
    """Outcome of a training run."""

    best_val: float
    best_state: dict | None
    history: dict[str, list[float]] = field(default_factory=lambda: {"train": [], "val": []})


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    prepare_batch: PrepareBatch,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: "torch.amp.GradScaler | None",
    grad_clip: float,
    use_amp: bool,
) -> float:
    """Run one epoch. Trains when optimizer is given, else evaluates."""
    train = optimizer is not None
    model.train(train)
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    total, n = 0.0, 0
    for eeg, _kin, emg in loader:
        x, y = prepare_batch(eeg, emg)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                pred = model(x)
                loss = loss_fn(pred, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        bs = eeg.size(0)
        total += loss.item() * bs
        n += bs

    return total / max(n, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prepare_batch: PrepareBatch,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> TrainResult:
    """Train with AMP, grad-clip, LR plateau scheduling, and early stopping.

    Args:
        model:         Module already moved to ``device``.
        train_loader:  Yields (eeg, kin, emg) batches.
        val_loader:    Validation batches.
        prepare_batch: (eeg, emg) -> (x, y) on device (CCA + EMG z-score).
        loss_fn:       Loss module (e.g. CombinedEMGLoss).
        device:        Target device.
        cfg:           Resolved TrainConfig.

    Returns:
        TrainResult with best val loss, best CPU state_dict, and history.
    """
    use_amp = cfg.use_amp and device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    logger.info(
        "Training on %s | AMP=%s | epochs=%d | lr=%g",
        device, use_amp, cfg.max_epochs, cfg.lr,
    )

    result = TrainResult(best_val=float("inf"), best_state=None)
    bad = 0
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        tr = _run_epoch(
            model, train_loader, prepare_batch, loss_fn, device,
            optimizer, scaler, cfg.grad_clip_norm, use_amp,
        )
        vl = _run_epoch(
            model, val_loader, prepare_batch, loss_fn, device,
            None, None, cfg.grad_clip_norm, use_amp,
        )
        scheduler.step(vl)
        result.history["train"].append(tr)
        result.history["val"].append(vl)

        flag = ""
        if vl < result.best_val:
            result.best_val = vl
            result.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
            flag = "  <- best"
        else:
            bad += 1

        logger.info(
            "ep %3d | train %.4f | val %.4f | %5.1fs%s",
            ep, tr, vl, time.time() - t0, flag,
        )
        if bad >= cfg.early_stop_patience:
            logger.info("early stop at epoch %d", ep)
            break

    if result.best_state is not None:
        model.load_state_dict(result.best_state)
    return result


def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, best_val: float) -> None:
    """Save model weights + minimal metadata for resumption/inference."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": cfg.__dict__,
            "best_val": best_val,
        },
        path,
    )
    logger.info("checkpoint saved -> %s", path)
