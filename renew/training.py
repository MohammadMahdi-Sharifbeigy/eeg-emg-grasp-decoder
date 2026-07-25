from __future__ import annotations
"""
Device-aware training loop for the KG-GT EMG regressor.

Runs on CUDA when available (GTX 1660 Ti / sm_75) and falls back to CPU with
the same code path. GPU-specific features:
  * Automatic Mixed Precision (AMP) via torch.amp — roughly halves activation
    memory and uses the Turing FP16 tensor cores, which matters on 6 GB VRAM.
    Disabled automatically on CPU.
  * Gradients scaled with GradScaler to avoid FP16 underflow.

Crash-safe checkpointing:
  * Saves a full checkpoint (weights + optimizer + scheduler + scaler +
    history + epoch) after every epoch to checkpoint_dir/last.pt.
  * Also saves best.pt whenever validation loss improves.
  * On the next call to train_model the loop resumes from last.pt
    automatically (pass resume=True, which is the default when the file exists).

GPU-usage diagnostics:
  * print_gpu_info() logs device name, VRAM capacity, driver/CUDA version.
  * Each epoch log line includes current VRAM allocated/reserved so you can
    confirm the GPU is actually being used.
"""


import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


import csv
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from .dataset import WAYEEGDataset
from .preprocessing_eeg import preprocess_eeg_from_config
from .preprocessing_emg_kin import preprocess_emg_from_config, preprocess_kinematics_from_config
from .model import build_model_from_config


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)

ModelInputs = dict[str, Any] | torch.Tensor
PrepareBatch = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[ModelInputs, torch.Tensor]]


def _forward_model(model: nn.Module, inputs: ModelInputs):
    """Call model. Returns dict {prediction, gate, ...} or plain tensor."""
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


# ---------------------------------------------------------------------------
# GPU diagnostics
# ---------------------------------------------------------------------------

def print_gpu_info(device: torch.device) -> None:
    """Print a clear GPU status block so you can confirm training uses CUDA.

    Call this once before train_model. If the device is CPU this will tell
    you CUDA is unavailable and print the reason (useful for debugging).
    """
    sep = "=" * 60
    print(sep)
    print("  GPU / Device Status")
    print(sep)
    if device.type != "cuda":
        print(f"  WARNING  Running on CPU  (device={device})")
        if not torch.cuda.is_available():
            print("  torch.cuda.is_available() returned False")
            cuda_path = os.environ.get("CUDA_PATH", "<not set>")
            print(f"  CUDA_PATH env var : {cuda_path}")
        else:
            print("  CUDA is available but the caller chose CPU.")
        print(sep)
        return

    idx = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_mb = props.total_memory / 1024 ** 2
    alloc_mb = torch.cuda.memory_allocated(idx) / 1024 ** 2
    reserv_mb = torch.cuda.memory_reserved(idx) / 1024 ** 2

    print(f"  OK  Device index    : cuda:{idx}")
    print(f"  OK  GPU name        : {props.name}")
    print(f"  OK  CUDA capability : sm_{props.major}{props.minor}")
    print(f"  OK  VRAM (total)    : {total_mb:,.0f} MB")
    print(f"  OK  VRAM allocated  : {alloc_mb:.1f} MB")
    print(f"  OK  VRAM reserved   : {reserv_mb:.1f} MB")
    print(f"  OK  CUDA version    : {torch.version.cuda}")
    print(f"  OK  PyTorch build   : {torch.__version__}")
    has_fp16_hw = props.major >= 7
    amp_note = "YES -- Turing tensor cores active" if has_fp16_hw else "YES (no dedicated FP16 HW)"
    print(f"  OK  AMP (FP16)      : {amp_note}")
    print(sep)


def _gpu_mem_str(device: torch.device) -> str:
    """Return a compact VRAM string e.g. '1234/6144 MB', or empty on CPU."""
    if device.type != "cuda":
        return ""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(idx) / 1024 ** 2
    total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 2
    return f" | VRAM {alloc:.0f}/{total:.0f} MB"


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------

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
    gradient_accumulation_steps: int = 1
    log_memory_every: int = 50
    # --- checkpoint settings ---
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_every: int = 1   # save last.pt every N epochs (1 = every epoch)

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
            gradient_accumulation_steps=max(1, cfg.get("gradient_accumulation_steps", 1)),
            log_memory_every=max(1, cfg.get("log_memory_every", 50)),
            checkpoint_dir=cfg.get("checkpoint_dir", "outputs/checkpoints"),
            checkpoint_every=cfg.get("checkpoint_every", 1),
        )


@dataclass
class TrainResult:
    """Outcome of a training run."""

    best_val: float
    best_state: dict | None
    history: dict[str, list[float]] = field(default_factory=lambda: {"train": [], "val": []})


# ---------------------------------------------------------------------------
# Internal epoch runner
# ---------------------------------------------------------------------------

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
    gradient_accumulation_steps: int,
    epoch: int,
    phase: str,
) -> float:
    """Run one epoch. Trains when optimizer is given, else evaluates.

    Shows a per-batch tqdm progress bar (if tqdm is installed) so you can
    see real-time throughput rather than waiting for the epoch to complete.
    """
    train = optimizer is not None
    model.train(train)
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    total, n = 0.0, 0
    if train:
        optimizer.zero_grad(set_to_none=True)

    if _TQDM_AVAILABLE:
        bar = tqdm(
            loader,
            desc=f"  Ep {epoch:4d} [{phase:5s}]",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )
    else:
        bar = loader

    for batch_idx, (eeg, kin, emg) in enumerate(bar, start=1):
        model_inputs, y = prepare_batch(eeg, kin, emg)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                output = _forward_model(model, model_inputs)
                if isinstance(output, dict):
                    pred = output["prediction"]
                    gate = output.get("gate", None)
                else:
                    pred, gate = output, None
                loss = loss_fn(pred, y, gate=gate)

            if train:
                loss_for_backward = loss / gradient_accumulation_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                should_step = (
                    batch_idx % gradient_accumulation_steps == 0
                    or batch_idx == len(loader)
                )
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        bs = eeg.size(0)
        batch_loss = loss.item()
        total += batch_loss * bs
        n += bs

        if _TQDM_AVAILABLE:
            bar.set_postfix(loss=f"{batch_loss:.4f}")

    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir: str, name: str) -> Path:
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def _save_training_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val: float,
    history: dict,
    bad: int,
    cfg: TrainConfig,
) -> None:
    """Save a full resumable checkpoint (weights + all training state)."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val": best_val,
            "history": history,
            "bad_epochs": bad,
            "train_config": cfg.__dict__,
        },
        path,
    )


def _load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> tuple[int, float, dict, int]:
    """Load checkpoint and return (start_epoch, best_val, history, bad_epochs)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return (
        ckpt["epoch"],
        ckpt["best_val"],
        ckpt["history"],
        ckpt["bad_epochs"],
    )


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prepare_batch: PrepareBatch,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    resume: bool = True,
) -> TrainResult:
    """Train with AMP, grad-clip, LR plateau scheduling, and early stopping.

    Crash-safe:
        A ``last.pt`` checkpoint is written to ``cfg.checkpoint_dir`` after
        every ``cfg.checkpoint_every`` epochs. A ``best.pt`` is written
        whenever the validation loss improves. Set ``resume=True`` (default)
        and the loop will pick up from where it left off automatically.

    GPU visibility:
        ``print_gpu_info(device)`` is called at the start so you see a full
        device report. Each epoch log line shows VRAM usage so you can confirm
        the GPU is active throughout training.

    Args:
        model:         Module already moved to ``device``.
        train_loader:  Yields (eeg, kin, emg) batches.
        val_loader:    Validation batches.
        prepare_batch: (eeg, emg) -> (x, y) on device (CCA + EMG z-score).
        loss_fn:       Loss module (e.g. CombinedEMGLoss).
        device:        Target device.
        cfg:           Resolved TrainConfig.
        resume:        If True and ``last.pt`` exists in checkpoint_dir,
                       resume from that checkpoint.

    Returns:
        TrainResult with best val loss, best CPU state_dict, and history.
    """
    use_amp = cfg.use_amp and device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=getattr(cfg, 'weight_decay', 1e-2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Print GPU status so the user can visually confirm CUDA is active
    print_gpu_info(device)

    logger.info(
        "Training on %s | AMP=%s | epochs=%d | lr=%g | ckpt_dir=%s",
        device, use_amp, cfg.max_epochs, cfg.lr, cfg.checkpoint_dir,
    )

    # Resume or fresh start
    last_ckpt = _ckpt_path(cfg.checkpoint_dir, "last.pt")
    start_epoch = 0
    result = TrainResult(best_val=float("inf"), best_state=None)
    bad = 0

    if resume and last_ckpt.exists():
        start_epoch, result.best_val, result.history, bad = _load_training_checkpoint(
            last_ckpt, model, optimizer, scheduler, scaler, device
        )
        logger.info(
            "Resumed from %s  (epoch %d done, best_val=%.4f)",
            last_ckpt, start_epoch, result.best_val,
        )
        print(
            f"\nResumed from checkpoint: {last_ckpt}\n"
            f"  Completed epochs : {start_epoch}\n"
            f"  Best val loss    : {result.best_val:.4f}\n"
            f"  Early-stop bad   : {bad}/{cfg.early_stop_patience}\n"
        )
    else:
        print(f"\nStarting fresh training run  (checkpoint_dir={cfg.checkpoint_dir})\n")

    best_ckpt = _ckpt_path(cfg.checkpoint_dir, "best.pt")

    for ep in range(start_epoch + 1, cfg.max_epochs + 1):
        t0 = time.time()

        tr = _run_epoch(
            model, train_loader, prepare_batch, loss_fn, device,
            optimizer, scaler, cfg.grad_clip_norm, use_amp,
            cfg.gradient_accumulation_steps,
            epoch=ep, phase="train",
        )
        vl = _run_epoch(
            model, val_loader, prepare_batch, loss_fn, device,
            None, None, cfg.grad_clip_norm, use_amp,
            1,
            epoch=ep, phase="val",
        )
        scheduler.step(vl)
        result.history["train"].append(tr)
        result.history["val"].append(vl)

        epoch_time = time.time() - t0
        gpu_mem = _gpu_mem_str(device)
        lr_now = optimizer.param_groups[0]["lr"]

        flag = ""
        if vl < result.best_val:
            result.best_val = vl
            result.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
            flag = "  <- BEST"
            # Save best checkpoint immediately on improvement
            _save_training_checkpoint(
                best_ckpt, ep, model, optimizer, scheduler, scaler,
                result.best_val, result.history, bad, cfg,
            )
        else:
            bad += 1

        summary = (
            f"Ep {ep:4d}/{cfg.max_epochs} | "
            f"train {tr:.4f} | val {vl:.4f} | "
            f"lr {lr_now:.2e} | "
            f"{epoch_time:5.1f}s"
            f"{gpu_mem}"
            f"{flag}"
        )
        if ep % 10 == 0:
            logger.info(summary)

        # Periodic crash-safe checkpoint
        if ep % cfg.checkpoint_every == 0:
            _save_training_checkpoint(
                last_ckpt, ep, model, optimizer, scheduler, scaler,
                result.best_val, result.history, bad, cfg,
            )

        # Early stopping
        if bad >= cfg.early_stop_patience:
            msg = f"Early stop at epoch {ep} (no val improvement for {bad} epochs)."
            print(msg)
            logger.info(msg)
            # Always write last.pt on stop
            _save_training_checkpoint(
                last_ckpt, ep, model, optimizer, scheduler, scaler,
                result.best_val, result.history, bad, cfg,
            )
            break

    if result.best_state is not None:
        model.load_state_dict(result.best_state)

    print(f"\nTraining done.  Best val loss = {result.best_val:.4f}")
    print(f"Checkpoints saved to: {Path(cfg.checkpoint_dir).resolve()}")
    return result


# ---------------------------------------------------------------------------
# Standalone inference checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, best_val: float) -> None:
    """Save model weights + minimal metadata for inference.

    This is a lightweight export checkpoint (weights only). For full
    resumable checkpoints use the last.pt / best.pt files written
    automatically by train_model.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": cfg.__dict__,
            "best_val": best_val,
        },
        path,
    )
    logger.info("checkpoint saved -> %s", path)
    print(f"Saved inference checkpoint -> {path}")


def load_checkpoint(path: str, model: nn.Module, device: torch.device) -> float:
    """Load weights from a save_checkpoint file.  Returns best_val."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_val = ckpt.get("best_val", float("inf"))
    logger.info("loaded checkpoint <- %s  (best_val=%.4f)", path, best_val)
    print(f"Loaded checkpoint <- {path}  (best_val={best_val:.4f})")
    return best_val

"""
Evaluation metrics for the KG-GT EMG regressor.

Collects predictions over a DataLoader on-device, then reports per-channel
RMSE, MAE, and Pearson r on the (z-scored) EMG targets. Inference runs under
autocast on CUDA for speed; metrics are computed in float64 on CPU for
numerical stability.
"""


import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

ModelInputs = dict[str, Any] | torch.Tensor
PrepareBatch = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[ModelInputs, torch.Tensor]]


def _forward_model(model: nn.Module, inputs: ModelInputs) -> torch.Tensor:
    """Call the model with either a tensor input or a keyword-input dict."""
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


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
    for eeg, kin, emg in loader:
        model_inputs, y = prepare_batch(eeg, kin, emg)
        with torch.amp.autocast(device_type=amp_device, enabled=amp):
            pred = _forward_model(model, model_inputs)
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

def overlap_average_windows(windows, window_size, stride):
    windows = np.asarray(windows)
    n_windows, W, n_channels = windows.shape
    assert W == window_size

    total_len = stride * (n_windows - 1) + window_size
    summed = np.zeros((total_len, n_channels), dtype=np.float32)
    counts = np.zeros((total_len, 1), dtype=np.float32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        summed[start:end] += windows[i]
        counts[start:end] += 1.0

    return summed / np.clip(counts, 1.0, None)


@torch.no_grad()
def predict_full_series_from_dataset(
    model,
    ds,
    prepare_batch,
    emg_mean,
    emg_std,
    device,
    series_idx=0,
):
    eeg_series, kin_series, emg_series = unique_series_arrays(ds)

    eeg_all = eeg_series[series_idx]
    kin_all = kin_series[series_idx]
    target = emg_series[series_idx]

    W = ds.window_size
    S = ds.stride
    starts = list(range(0, len(target) - W + 1, S))

    pred_windows = []
    for start in starts:
        eeg_w = torch.from_numpy(eeg_all[start:start + W]).unsqueeze(0)
        kin_w = torch.from_numpy(kin_all[start:start + W]).unsqueeze(0)
        emg_w = torch.from_numpy(target[start:start + W]).unsqueeze(0)

        model_inputs, _ = prepare_batch(eeg_w, kin_w, emg_w)
        out = model(**model_inputs)
        pred_z = out["prediction"] if isinstance(out, dict) else out
        # pred_z = model(**model_inputs)
        pred = inverse_emg_zscore(pred_z.squeeze(0), emg_mean, emg_std)
        pred_windows.append(pred)

    pred_windows = np.stack(pred_windows, axis=0)
    pred_full = overlap_average_windows(pred_windows, window_size=W, stride=S)

    return pred_full, target[: len(pred_full)]

EMG_CHANNEL_NAMES = [
    "Ant. Deltoid",
    "Ext. Carpi Rad.",
    "Flex. Digitorum",
    "Ext. Dig. Comm.",
    "1st Dors. Inteross.",
]


def make_preprocess_fn(cfg):
    eeg_cfg = cfg["preprocessing"]["eeg"]
    emg_cfg = cfg["preprocessing"]["emg"]
    kin_cfg = cfg["preprocessing"]["kinematics"]
    def preprocess_fn(series):
        series = dict(series)
        series["eeg"] = preprocess_eeg_from_config(
            series["eeg"],
            float(series["fs_eeg"]),
            eeg_cfg,
            channel_names=series.get("eeg_names"),
        )
        series["emg"] = preprocess_emg_from_config(
            series["emg"],
            float(series["fs_emg"]),
            emg_cfg,
        )
        series["kin"] = preprocess_kinematics_from_config(                                           
              series["kin"],                                                                           
              float(series["fs_kin"]),                                                                 
              kin_cfg,                                                                                 
        ) 
        return series

    return preprocess_fn


def build_dataset_split(cfg, participants=None, split="train", root_dir=Path(".")):
    data_cfg = cfg["data"]
    if participants is None:
        participants = data_cfg["participants"]
    return WAYEEGDataset(
        data_dir=root_dir / data_cfg["raw_dir"],
        participants=participants,
        split=split,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        preprocess_fn=make_preprocess_fn(cfg),
        cache_dir=root_dir / data_cfg["cache_dir"],
    )


def unique_series_arrays(ds):
    seen, eegs, kins, emgs = set(), [], [], []
    for eeg_all, kin_all, emg_all, _ in ds._windows:
        key = id(eeg_all)
        if key in seen:
            continue
        seen.add(key)
        eegs.append(eeg_all)
        kins.append(kin_all)
        emgs.append(emg_all)
    return eegs, kins, emgs


def prepare_batch_factory(emg_mean, emg_std, kin_mean, kin_std, device, drop_kin_indices=None):
    def prepare_batch(eeg, kin, emg):                                                                
        eeg = eeg.to(device, non_blocking=True)
        kin = kin.to(device, non_blocking=True)
        emg = emg.to(device, non_blocking=True)
        
        kin_norm = (kin - kin_mean) / kin_std
        if drop_kin_indices is not None:
            keep_idx = [i for i in range(kin_norm.shape[-1]) if i not in drop_kin_indices]
            kin_norm = kin_norm[..., keep_idx]
            
        emg_norm = (emg - emg_mean) / emg_std
        return {"eeg": eeg, "kin": kin_norm}, emg_norm
    return prepare_batch 

def inverse_emg_zscore(arr, emg_mean, emg_std):
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    mean = emg_mean.detach().cpu().numpy() if torch.is_tensor(emg_mean) else np.asarray(emg_mean)
    std = emg_std.detach().cpu().numpy() if torch.is_tensor(emg_std) else np.asarray(emg_std)
    return arr * std + mean


def run_kfold_cross_validation(notebook_cfg, train_ds, batch_size, prepare_batch, loss_fn, device, train_model_func,resume=False, smoke_run=False, k_folds=5):
    """
    Runs a complete K-Fold Cross Validation sweep, logs all metrics to CSV, 
    and saves checkpoints per fold.
    """
    max_epochs = notebook_cfg["training"]["max_epochs"] if not smoke_run else 1
    config_str = f"ep{max_epochs}_st{notebook_cfg['data']['stride']}_lr{notebook_cfg['training'].get('lr', 1e-4)}_bs{batch_size}"

    # --- Per-subject folder: results/P1/<config_str>/ ---
    participants = notebook_cfg["data"].get("participants", [])
    subject_id   = participants[0] if participants else "unknown"
    subject_str  = f"P{subject_id}"

    save_dir = Path("results") / subject_str / config_str
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results and Checkpoints will be saved to: {save_dir}")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    csv_path = save_dir / "training_metrics.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Epoch", "Train_Loss", "Val_Loss"])

    all_fold_results = []
    sample_eeg, sample_kin, sample_emg = train_ds[0]

    # Wrap the fold loop in a tqdm notebook progress bar!
    fold_iterator = tqdm(kf.split(train_ds), total=k_folds, desc="K-Fold Progress")
    
    for fold, (train_idx, val_idx) in enumerate(fold_iterator):
        fold_iterator.set_postfix({"Current Fold": f"{fold + 1}/{k_folds}"})
        
        fold_dir = save_dir / f"fold_{fold+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. DataLoaders
        fold_train_subset = Subset(train_ds, train_idx)
        fold_val_subset = Subset(train_ds, val_idx)
        fold_train_loader = DataLoader(fold_train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        fold_val_loader = DataLoader(fold_val_subset, batch_size=batch_size, shuffle=False)

        # 2. Build Model (Infer dims dynamically after prepare_batch)
        dummy_eeg = torch.as_tensor(sample_eeg).unsqueeze(0)
        dummy_kin = torch.as_tensor(sample_kin).unsqueeze(0)
        dummy_emg = torch.as_tensor(sample_emg).unsqueeze(0)
        batch_inputs, _ = prepare_batch(dummy_eeg, dummy_kin, dummy_emg)
        
        fold_model = build_model_from_config(
            notebook_cfg, 
            input_dim=batch_inputs["eeg"].shape[-1], 
            kin_dim=batch_inputs["kin"].shape[-1]
        ).to(device)
        
        # 3. Setup Config
        
        train_cfg_obj = TrainConfig.from_config(
            notebook_cfg["training"], 
            max_epochs=max_epochs
        )
        train_cfg_obj.checkpoint_dir = str(fold_dir / "checkpoints")
        train_cfg_obj.use_amp = False 
 

        # 4. Train
        result = train_model_func(
            model=fold_model,
            train_loader=fold_train_loader,
            val_loader=fold_val_loader,
            prepare_batch=prepare_batch,
            loss_fn=loss_fn,
            device=device,
            cfg=train_cfg_obj,
            resume=True,
        )
        all_fold_results.append(result)
        
        # 5. Log metrics
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Extract the lists from the dictionary
            train_losses = result.history.get("train", [])
            val_losses = result.history.get("val", [])
            
            # Loop through the epochs and save them
            for ep in range(len(train_losses)):
                t_loss = train_losses[ep] if ep < len(train_losses) else 0
                v_loss = val_losses[ep] if ep < len(val_losses) else 0
                writer.writerow([fold + 1, ep + 1, t_loss, v_loss])


    print(f"\nAll {k_folds} Folds Completed! Checkpoints saved in: {save_dir}")
    return all_fold_results, save_dir
