"""
Training and evaluation loop for the KG-GT EMG regressor (nb04).

Features:
  - AMP (Automatic Mixed Precision) via torch.amp
  - Gradient accumulation and grad clipping
  - LR plateau scheduling + early stopping
  - Crash-safe checkpointing (last.pt / best.pt)
  - GPU status diagnostics
"""
from __future__ import annotations

import inspect
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from pandas._config import config
from asyncio import exceptions

import numpy as np
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


def _forward_model(model: nn.Module, inputs: ModelInputs) -> torch.Tensor:
    """Call the model with either a tensor input or a keyword-input dict."""
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


# ============================================================================
# GPU diagnostics
# ============================================================================

def print_gpu_info(device: torch.device) -> None:
    """Print a full GPU status block before training."""
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


# ============================================================================
# Config / result dataclasses
# ============================================================================

@dataclass
class TrainConfig:
    """Resolved training hyperparameters."""

    lr: float = 1e-3
    weight_decay: float = 1e-2          # used only by adamw
    lr_patience: int = 50               # used only by 'reduce' scheduler
    lr_factor: float = 0.5              # used only by 'reduce' scheduler
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 30
    max_epochs: int = 500
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    log_memory_every: int = 50
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_every: int = 1
    # ── optimizer / scheduler selection ──────────────────────────────────────
    optimizer: str = "adamw"            # 'adam' | 'adamw'
    scheduler: str = "reduce"           # 'reduce' | 'cosine'
    warmup_epochs: int = 0        # NEW — 0 disables warmup (fully backward-compatible)
    cosine_t_max: int | None = None     # cosine period (epochs); None → max_epochs
    cosine_eta_min: float = 1e-6        # cosine floor LR
    # ── differential LR for Transformer+GAT E2E training ─────────────────────
    transformer_lr_scale: float = 1.0   # encoder/node_proj LR = lr * this scale
                                        # set < 1.0 (e.g. 0.2) for E2E warmup Phase 2

    @classmethod
    def from_config(cls, cfg: dict, max_epochs: int | None = None, lr: float | None = None) -> "TrainConfig":
        """Build from the training section of default.yaml."""
        resolved_max_epochs = (
            max_epochs if max_epochs is not None else cfg.get("max_epochs", 500)
        )
        resolved_lr = (
            lr if lr is not None else cfg.get("lr", cfg.get("stage2_lr", cfg.get("stage1_lr", 1e-3)))
        )
        return cls(
            lr=resolved_lr,
            weight_decay=cfg.get("weight_decay", 1e-2),
            lr_patience=cfg.get("lr_patience", 50),
            lr_factor=cfg.get("lr_factor", 0.5),
            grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
            early_stop_patience=cfg.get("early_stop_patience", 30),
            max_epochs=resolved_max_epochs,
            use_amp=cfg.get("use_amp", True),
            gradient_accumulation_steps=max(1, cfg.get("gradient_accumulation_steps", 1)),
            log_memory_every=max(1, cfg.get("log_memory_every", 50)),
            checkpoint_dir=cfg.get("checkpoint_dir", "outputs/checkpoints"),
            checkpoint_every=cfg.get("checkpoint_every", 1),
            optimizer=cfg.get("optimizer", "adamw"),
            scheduler=cfg.get("scheduler", "reduce"),
            cosine_t_max=cfg.get("cosine_t_max", None),
            cosine_eta_min=cfg.get("cosine_eta_min", 1e-6),
            transformer_lr_scale=cfg.get("transformer_lr_scale", 1.0),
            warmup_epochs=cfg.get("warmup_epochs", 0),
        )


@dataclass
class TrainResult:
    best_val: float
    best_state: dict | None
    history: dict[str, list[float]] = field(default_factory=lambda: {"train": [], "val": [], "lr": []})
    smoothed_val_history: list[float] = field(default_factory=list)   # NEW

# ============================================================================
# Internal epoch runner
# ============================================================================

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
    """Run one epoch. Trains when optimizer is given, else evaluates."""
    train = optimizer is not None
    model.train(train)
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    total, n = 0.0, 0

    # Detect once — outside the hot batch loop — whether the loss function
    # supports a 'model' kwarg (e.g. CombinedEMGLoss with KL regularization).
    # Backward-compatible: legacy loss_fn(pred, target) calls are unaffected.
    try:
        _loss_accepts_model = "model" in inspect.signature(loss_fn.forward).parameters
    except (ValueError, TypeError):
        _loss_accepts_model = False

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

    try:
        for batch_idx, (eeg, kin, emg) in enumerate(bar, start=1):
            model_inputs, y = prepare_batch(eeg, kin, emg)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                    pred = _forward_model(model, model_inputs)
                    # Pass model to loss only during training so the KL edge-prior
                    # regularization is active. Validation loss is kept as pure
                    # reconstruction for fair cross-epoch comparison.
                    if _loss_accepts_model and train:
                        loss = loss_fn(pred, y, model=model)
                    else:
                        loss = loss_fn(pred, y)

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
                postfix = {"loss": f"{batch_loss:.4f}"}
                if hasattr(loss_fn, "last_components"):
                    comps = loss_fn.last_components
                    if "ccc" in comps and comps["ccc"] > 0:
                        postfix["ccc_loss"] = f"{comps['ccc']:.3f}"
                    if "pearson" in comps and comps["pearson"] > 0:
                        postfix["r_loss"] = f"{comps['pearson']:.3f}"
                bar.set_postfix(postfix)
    finally:
        if _TQDM_AVAILABLE:
            bar.close()

    return total / max(n, 1)

def compute_mean_baseline_loss(
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    prepare_batch: PrepareBatch | None = None,
) -> float:
    """Trivial baseline: predict the per-channel training mean for every timestep.
    If Stage-1 train loss converges to ~this value, the model has collapsed
    to a constant predictor and learned nothing beyond the channel mean.

    If prepare_batch is provided, targets are preprocessed through prepare_batch
    so the baseline MSE/loss is evaluated on the exact scale of the active loss_fn.
    """
    sums = None
    count = 0
    with torch.no_grad():
        for batch in loader:
            if prepare_batch is not None:
                _, target = prepare_batch(*batch)
            else:
                target = batch[2].to(device)
            if sums is None:
                sums = target.sum(dim=(0, 1))
            else:
                sums += target.sum(dim=(0, 1))
            count += target.shape[0] * target.shape[1]

    mean_target = (sums / max(count, 1)).view(1, 1, -1)  # (1, 1, C)

    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            if prepare_batch is not None:
                _, target = prepare_batch(*batch)
            else:
                target = batch[2].to(device)
            bs = target.shape[0]
            pred_mean = mean_target.expand_as(target)
            batch_loss = loss_fn(pred_mean, target).item()
            total_loss += batch_loss * bs
            n_samples += bs

    return total_loss / max(n_samples, 1)


# ============================================================================
# Checkpoint helpers
# ============================================================================

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
    """Save a full resumable checkpoint."""
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


# ============================================================================
# Public training API
# ============================================================================

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

    Crash-safe: last.pt is written every cfg.checkpoint_every epochs;
    best.pt is written whenever val loss improves.

    Args:
        model:         Module already moved to device.
        train_loader:  Yields (eeg, kin, emg) batches.
        val_loader:    Validation batches.
        prepare_batch: (eeg, kin, emg) -> (x, y) on device.
        loss_fn:       Loss module.
        device:        Target device.
        cfg:           Resolved TrainConfig.
        resume:        If True and last.pt exists, resume from checkpoint.

    Returns:
        TrainResult with best val loss, best CPU state_dict, and history.
    """
    use_amp = cfg.use_amp and device.type == "cuda"

    # ── Optimizer (with parameter grouping for differential LRs) ────────────
    _opt_name = cfg.optimizer.lower().strip()

    # Collect parameter sets for potential grouping
    _has_transformer = hasattr(model, "encoder") and hasattr(model, "node_projection")
    _has_kin_skip    = getattr(model, "use_kinematic_guidance", False) and getattr(model, "kin_skip_proj", None) is not None
    _use_diff_trans  = cfg.transformer_lr_scale < 1.0 and _has_transformer

    if _use_diff_trans:
        # Three-group differential LR:
        #   1. Transformer backbone (encoder + node_projection): lr * transformer_lr_scale
        #   2. KinSkip highway (if present): lr / 3.0  (already regularized)
        #   3. Everything else (GAT, decoder, …): lr
        _trans_params = set()
        for _m in [model.encoder, model.node_projection]:
            _trans_params.update(_m.parameters())

        _skip_params = set()
        if _has_kin_skip:
            _skip_params.update(model.kin_skip_proj.parameters())

        _other_params = [
            p for p in model.parameters()
            if p not in _trans_params and p not in _skip_params and p.requires_grad
        ]
        _trans_list = [
            p for p in model.parameters()
            if p in _trans_params and p.requires_grad
        ]
        _skip_list = [
            p for p in model.parameters()
            if p in _skip_params and p.requires_grad
        ]

        param_groups = [
            {"params": _other_params, "lr": cfg.lr,                                    "weight_decay": cfg.weight_decay},
            {"params": _trans_list,   "lr": cfg.lr * cfg.transformer_lr_scale,         "weight_decay": cfg.weight_decay},
        ]
        if _skip_list:
            param_groups.append(
                {"params": _skip_list, "lr": cfg.lr / 3.0, "weight_decay": max(0.05, cfg.weight_decay * 5.0)}
            )
        logger.info(
            "Differential LR: other=%.2e | transformer=%.2e (scale=%.2f) | skip=%s",
            cfg.lr, cfg.lr * cfg.transformer_lr_scale, cfg.transformer_lr_scale,
            f"{cfg.lr/3:.2e}" if _skip_list else "n/a",
        )
        print(
            f"  Differential LR groups:\n"
            f"    GAT/Decoder  : {cfg.lr:.2e}\n"
            f"    Transformer  : {cfg.lr * cfg.transformer_lr_scale:.2e}  (scale={cfg.transformer_lr_scale})\n"
            + (f"    KinSkip      : {cfg.lr/3:.2e}  (regularized)\n" if _skip_list else "")
        )
    elif _has_kin_skip:
        # Original two-group logic: neural vs kin_skip_proj
        skip_params   = set(model.kin_skip_proj.parameters())
        neural_params = [p for p in model.parameters() if p not in skip_params]
        skip_list     = list(model.kin_skip_proj.parameters())
        param_groups  = [
            {"params": neural_params, "lr": cfg.lr,       "weight_decay": cfg.weight_decay},
            {"params": skip_list,     "lr": cfg.lr / 3.0, "weight_decay": max(0.05, cfg.weight_decay * 5.0)},
        ]
        logger.info("Applying differential parameter groups: neural lr=%g, skip lr=%g, skip weight_decay=%g",
                    cfg.lr, cfg.lr / 3.0, max(0.05, cfg.weight_decay * 5.0))
    else:
        param_groups = [{"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay}]

    if _opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif _opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'. Choose 'adam' or 'adamw'.")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    _sched_name = cfg.scheduler.lower().strip()
    if _sched_name == "reduce":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
        )
    elif _sched_name == "cosine":
        t_max = cfg.cosine_t_max if cfg.cosine_t_max is not None else cfg.max_epochs
        # Reserve the warmup epochs from the cosine horizon so the anneal still
        # reaches cosine_eta_min by max_epochs rather than overshooting.
        cosine_t_max = max(1, t_max - cfg.warmup_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=cfg.cosine_eta_min
        )
    else:
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'. Choose 'reduce' or 'cosine'.")
    
    if cfg.warmup_epochs > 0 and _sched_name == "reduce":
        raise ValueError(
            "warmup_epochs > 0 is only supported with scheduler='cosine'. "
            "SequentialLR cannot forward the val-loss metric to ReduceLROnPlateau "
            "after the warmup milestone. Either set warmup_epochs=0, or use "
            "scheduler='cosine' for warmup."
        )
    if cfg.warmup_epochs > 0 and _sched_name == "cosine":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-2,          # start at 1% of base LR
            end_factor=1.0,
            total_iters=cfg.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[cfg.warmup_epochs],
        )
        logger.info("Warmup enabled: %d epochs, start_factor=0.01 -> 1.0, then %s",
                    cfg.warmup_epochs, _sched_name)
    else:
        scheduler = main_scheduler

    scaler = torch.amp.GradScaler(enabled=use_amp)

    print_gpu_info(device)
    logger.info(
        "Training on %s | AMP=%s | epochs=%d | lr=%g | optimizer=%s | scheduler=%s | ckpt_dir=%s",
        device, use_amp, cfg.max_epochs, cfg.lr, cfg.optimizer, cfg.scheduler, cfg.checkpoint_dir,
    )
    print(
        f"  Optimizer : {cfg.optimizer.upper()}  (weight_decay={cfg.weight_decay})\n"
        f"  Scheduler : {cfg.scheduler.upper()}"
        + (f"  (T_max={cfg.cosine_t_max or cfg.max_epochs}, eta_min={cfg.cosine_eta_min})"
           if cfg.scheduler == "cosine"
           else f"  (patience={cfg.lr_patience}, factor={cfg.lr_factor})")
    )

    best_ckpt = _ckpt_path(cfg.checkpoint_dir, "best.pt")
    last_ckpt = _ckpt_path(cfg.checkpoint_dir, "last.pt")
    start_epoch = 0
    result = TrainResult(best_val=float("inf"), best_state=None)
    bad = 0

    if resume and last_ckpt.exists():
        start_epoch, result.best_val, result.history, bad = _load_training_checkpoint(
            last_ckpt, model, optimizer, scheduler, scaler, device
        )
        if best_ckpt.exists():
            _b_ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            result.best_state = _b_ckpt["model_state_dict"]
            result.best_val = _b_ckpt.get("best_val", result.best_val)

        logger.info("Resumed from %s  (epoch %d done, best_val=%.4f)",
                    last_ckpt, start_epoch, result.best_val)
        print(
            f"\nResumed from checkpoint: {last_ckpt}\n"
            f"  Completed epochs : {start_epoch}\n"
            f"  Best val loss    : {result.best_val:.4f}\n"
            f"  Early-stop bad   : {bad}/{cfg.early_stop_patience}\n"
        )
        if bad >= cfg.early_stop_patience or start_epoch >= cfg.max_epochs:
            msg = f"Early stop / max epochs condition already met ({bad}/{cfg.early_stop_patience} bad epochs, {start_epoch}/{cfg.max_epochs} epochs). Skipping training."
            print(msg)
            logger.info(msg)
            if result.best_state is not None:
                model.load_state_dict(result.best_state)
            print(f"\nTraining done.  Best val loss = {result.best_val:.4f}")
            print(f"Checkpoints saved to: {Path(cfg.checkpoint_dir).resolve()}")
            return result
    else:
        print(f"\nStarting fresh training run  (checkpoint_dir={cfg.checkpoint_dir})\n")

    ep = start_epoch
    for ep in range(start_epoch + 1, cfg.max_epochs + 1):
        t0 = time.time()

        tr = _run_epoch(
            model, train_loader, prepare_batch, loss_fn, device,
            optimizer, scaler, cfg.grad_clip_norm, use_amp,
            cfg.gradient_accumulation_steps, epoch=ep, phase="train",
        )
        vl = _run_epoch(
            model, val_loader, prepare_batch, loss_fn, device,
            None, None, cfg.grad_clip_norm, use_amp,
            1, epoch=ep, phase="val",
        )
        # ReduceLROnPlateau needs the metric; CosineAnnealingLR does not
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(vl)
        else:
            scheduler.step()
        result.history["train"].append(tr)

        epoch_time = time.time() - t0
        gpu_mem = _gpu_mem_str(device)
        lr_now = optimizer.param_groups[0]["lr"]
        result.history.setdefault("lr", []).append(lr_now)

        SMOOTH_WINDOW = 5  # trailing epochs to average before treating val as "improved"

        result.history["val"].append(vl)
        window = result.history["val"][-SMOOTH_WINDOW:]
        smoothed_vl = sum(window) / len(window)
        result.smoothed_val_history.append(smoothed_vl)

        flag = ""
        n_val = len(result.history["val"])
        if n_val >= SMOOTH_WINDOW and smoothed_vl < result.best_val:
            # Normal smoothed best tracking (after warmup window is full)
            result.best_val = smoothed_vl
            result.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
            flag = "  <- BEST (smoothed)"
            _save_training_checkpoint(
                best_ckpt, ep, model, optimizer, scheduler, scaler,
                result.best_val, result.history, bad, cfg,
            )
            print(f"  [best.pt] Saved  (smoothed_val={result.best_val:.4f})")
        elif n_val < SMOOTH_WINDOW:
            # Warmup period: still save best on raw val so best.pt always exists
            if vl < result.best_val:
                result.best_val = vl
                result.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                flag = "  <- BEST (warmup)"
                _save_training_checkpoint(
                    best_ckpt, ep, model, optimizer, scheduler, scaler,
                    result.best_val, result.history, bad, cfg,
                )
            # don't increment bad during warmup
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

        # Save last.pt every single epoch so manual interruptions don't lose progress
        _save_training_checkpoint(
            last_ckpt, ep, model, optimizer, scheduler, scaler,
            result.best_val, result.history, bad, cfg,
        )

        if bad >= cfg.early_stop_patience:
            msg = f"Early stop at epoch {ep} (no val improvement for {bad} epochs)."
            print(msg)
            logger.info(msg)
            break

    # Save the final epoch state unconditionally for resumability
    _save_training_checkpoint(
        last_ckpt, ep, model, optimizer, scheduler, scaler,
        result.best_val, result.history, bad, cfg,
    )

    if result.best_state is not None:
        model.load_state_dict(result.best_state)

    print(f"\nTraining done.  Best val loss = {result.best_val:.4f}")
    print(f"Checkpoints saved to: {Path(cfg.checkpoint_dir).resolve()}")
    return result


# ============================================================================
# Standalone checkpoint helpers
# ============================================================================

def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, best_val: float) -> None:
    """Save model weights + minimal metadata for inference."""
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
    """Load weights from a save_checkpoint file. Returns best_val."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_val = ckpt.get("best_val", float("inf"))
    logger.info("loaded checkpoint <- %s  (best_val=%.4f)", path, best_val)
    print(f"Loaded checkpoint <- {path}  (best_val={best_val:.4f})")
    return best_val


# ============================================================================
# Evaluation
# ============================================================================

from dataclasses import dataclass as _dataclass


@_dataclass
class EvalMetrics:
    """Per-channel regression metrics for EMG envelope prediction.

    Biomechanics publication standard (2023–2024 literature):
      rmse    : Root Mean Square Error — magnitude of prediction error.
      nrmse   : Normalized RMSE (% of signal range) — enables cross-channel
                and cross-subject comparison regardless of amplitude scale.
      mae     : Mean Absolute Error — robust to individual outlier samples.
      pearson : Pearson r — waveform shape / temporal alignment similarity.
      r2      : Coefficient of Determination (R²) — ML convention.
      vaf     : Variance Accounted For (%) — motor control convention
                (Winter, 1990). VAF > 80 % is generally considered acceptable.
                NOTE: differs from R² in using signal variance (not mean-corrected),
                making it sensitive to both shape and mean offset errors.
    """

    rmse:    np.ndarray   # (C,)  RMSE in signal units
    mae:     np.ndarray   # (C,)  Mean Absolute Error
    pearson: np.ndarray   # (C,)  Pearson correlation coefficient (r)
    r2:      np.ndarray   # (C,)  Coefficient of Determination (R²)
    vaf:     np.ndarray   # (C,)  Variance Accounted For (%)
    nrmse:   np.ndarray   # (C,)  Normalized RMSE (% of signal range)
    channel_names: list[str]
    ccc:     np.ndarray | None = None  # (C,)  Lin's Concordance Correlation Coefficient

    def as_table(self) -> str:
        """Format all metrics as an aligned text table for publication reporting."""
        header = (
            f'{"Channel":18s} {"RMSE":>8s} {"nRMSE%":>8s} {"MAE":>8s}'
            f' {"Pearson r":>10s} {"CCC":>8s} {"R^2":>8s} {"VAF%":>8s}'
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for c, name in enumerate(self.channel_names):
            ccc_val = self.ccc[c] if self.ccc is not None else 0.0
            lines.append(
                f"{name:18s}"
                f" {self.rmse[c]:8.4f}"
                f" {self.nrmse[c]:8.2f}"
                f" {self.mae[c]:8.4f}"
                f" {self.pearson[c]:10.4f}"
                f" {ccc_val:8.4f}"
                f" {self.r2[c]:8.4f}"
                f" {self.vaf[c]:8.2f}"
            )
        lines.append(sep)
        mean_ccc = self.ccc.mean() if self.ccc is not None else 0.0
        lines.append(
            f"{' MEAN':18s}"
            f" {self.rmse.mean():8.4f}"
            f" {self.nrmse.mean():8.2f}"
            f" {self.mae.mean():8.4f}"
            f" {self.pearson.mean():10.4f}"
            f" {mean_ccc:8.4f}"
            f" {self.r2.mean():8.4f}"
            f" {self.vaf.mean():8.2f}"
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
        
        c = y.shape[-1]
        preds.append(pred.float().cpu().numpy().reshape(-1, c))
        targets.append(y.float().cpu().numpy().reshape(-1, c))

    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(
    pred:          np.ndarray,
    target:        np.ndarray,
    channel_names: list[str] | None = None,
) -> EvalMetrics:
    """Per-channel RMSE, nRMSE, MAE, Pearson r, CCC, R², and VAF — fully vectorised.

    Args:
        pred:          (N_samples, C) predicted EMG envelope.
        target:        (N_samples, C) ground-truth EMG envelope.
        channel_names: Optional list of C channel names for the metrics table.

    Returns:
        EvalMetrics dataclass with per-channel and mean values for all 7 metrics.
    """
    pred   = pred.astype(np.float64)
    target = target.astype(np.float64)
    n_ch   = pred.shape[1]

    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(n_ch)]

    # ── RMSE ─────────────────────────────────────────────────────────────────────
    rmse = np.sqrt(((pred - target) ** 2).mean(0))                          # (C,)

    # ── nRMSE (% of signal range) ─────────────────────────────────────────────────
    # Normalised RMSE enables cross-channel / cross-subject comparison.
    target_range = target.max(0) - target.min(0)                            # (C,)
    nrmse = np.where(target_range > 1e-12, rmse / target_range * 100.0, 0.0)

    # ── MAE ─────────────────────────────────────────────────────────────────────
    mae = np.abs(pred - target).mean(0)                                     # (C,)

    # ── Pearson r ─────────────────────────────────────────────────────────────
    p_mu  = pred.mean(0, keepdims=True)
    t_mu  = target.mean(0, keepdims=True)
    p_c   = pred   - p_mu
    t_c   = target - t_mu
    num   = (p_c * t_c).mean(0)
    denom = np.sqrt((p_c ** 2).mean(0)) * np.sqrt((t_c ** 2).mean(0))
    pearson = np.where(denom > 1e-12, num / denom, 0.0)                    # (C,)

    # ── CCC (Lin's Concordance Correlation Coefficient) ────────────────────────
    cov = (p_c * t_c).mean(0)
    p_var = (p_c ** 2).mean(0)
    t_var = (t_c ** 2).mean(0)
    mean_diff_sq = ((p_mu - t_mu).squeeze(0)) ** 2
    ccc_denom = p_var + t_var + mean_diff_sq
    ccc = np.where(ccc_denom > 1e-12, (2.0 * cov) / ccc_denom, 0.0)        # (C,)

    # ── R² (Coefficient of Determination) ──────────────────────────────────────
    # R² = 1 - SS_residual / SS_total  (ML / regression convention)
    # SS_total uses the mean-corrected target (same denominator as Pearson).
    ss_res = ((pred - target) ** 2).sum(0)                                  # (C,)
    ss_tot = (t_c ** 2).sum(0)                                              # (C,)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)             # (C,)

    # ── VAF (Variance Accounted For) ───────────────────────────────────────────
    # Motor control / EMG literature convention (Winter, 1990):
    #   VAF = (1 − var(pred − target) / var(target)) × 100 %
    # Uses raw signal variance (NOT mean-corrected) — stricter than R² because
    # it penalises both shape errors AND mean-offset errors simultaneously.
    # VAF > 80 % is the accepted biomechanics publication threshold.
    err_var = np.var(pred - target, axis=0, ddof=0)                         # (C,)
    tgt_var = np.var(target,        axis=0, ddof=0)                         # (C,)
    vaf = np.where(tgt_var > 1e-12, (1.0 - err_var / tgt_var) * 100.0, 0.0)

    return EvalMetrics(
        rmse=rmse,
        nrmse=nrmse,
        mae=mae,
        pearson=pearson,
        r2=r2,
        vaf=vaf,
        channel_names=channel_names,
        ccc=ccc,
    )


def prepare_batch_factory(device: torch.device, drop_kin_indices=None):
    """Create a prepare_batch function that moves batches to device.

    The returned callable expects (eeg, kin, emg) tensors from the DataLoader
    and returns (model_inputs_dict, target_emg).
    """
    def prepare_batch(eeg, kin, emg):
        eeg = eeg.to(device, dtype=torch.float32)
        kin = kin.to(device, dtype=torch.float32)
        emg = emg.to(device, dtype=torch.float32)
        if drop_kin_indices is not None:
            keep_idx = [i for i in range(kin.shape[-1]) if i not in drop_kin_indices]
            kin = kin[..., keep_idx]
        return {"eeg": eeg, "kin": kin}, emg

    return prepare_batch
