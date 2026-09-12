"""Bio-CLIP SSL Trainer — Pre-training loop for Pivot 4.

BioCLIPTrainer orchestrates the full pre-training loop:
    1. Encodes EEG windows via EEGEncoder → z_eeg (B, 128) + H_eeg (B, T, 256)
    2. Encodes EMG windows via EMGEncoder → z_emg (B, 128) + H_emg (B, T, 256)
    3. Optionally labels movement phases via PhaseLabeler(kin)
    4. Computes PhaseAwareInfoNCELoss (with false-negative masking)
    5. Optionally adds dense token-level InfoNCE on H_eeg, H_emg
    6. Logs pre-training metrics; checkpoints best val loss and last epoch (crash-safe resume)

SSLTrainConfig: dataclass-style configuration container.
SSLTrainResult: post-training result container with history and metrics.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from .eeg_encoder import EEGEncoder
from .emg_encoder import EMGEncoder
from .contrastive_loss import PhaseAwareInfoNCELoss, SymmetricInfoNCELoss
from .phase_labeler import PhaseLabeler


# ============================================================================
# Configuration & Result Containers
# ============================================================================

@dataclass
class SSLTrainConfig:
    """Configuration for Bio-CLIP pre-training.

    Fields:
        n_epochs: Total pre-training epochs (default: 100).
        learning_rate: Peak AdamW learning rate (default: 3e-4).
        weight_decay: AdamW weight decay (default: 1e-2).
        temperature: Initial InfoNCE temperature (default: 0.07).
        learnable_temp: Whether temperature is a learned parameter (default: True).
        use_phase_masking: Enable phase-aware false-negative masking (default: True).
        use_dense_loss: Add token-level dense InfoNCE on H_dense (default: True).
        dense_loss_weight: Weight of dense InfoNCE relative to global (default: 0.2).
        rest_weight: Loss upweight for REST phase windows (default: 2.0).
        warmup_epochs: Linear warmup epochs (default: 10).
        use_amp: Automatic Mixed Precision with float16 (default: True).
        clip_grad_norm: Gradient norm clipping (default: 1.0).
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_every: Save last.pt every N epochs (default: 1).
        early_stop_patience: Early stopping patience (default: 25).
        resume: Whether to resume from last.pt if available (default: True).
        log_every: Log every N steps (default: 20).
        val_every: Validate every N epochs (default: 5).
        device: Training device (default: 'auto' → cuda if available else cpu).
        save_checkpoint: Whether to save checkpoints (default: True).
    """
    n_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    temperature: float = 0.07
    learnable_temp: bool = True
    use_phase_masking: bool = True
    use_dense_loss: bool = True
    dense_loss_weight: float = 0.2
    rest_weight: float = 2.0
    warmup_epochs: int = 10
    use_amp: bool = True
    clip_grad_norm: float = 1.0
    checkpoint_dir: str = "outputs/ssl_checkpoints"
    checkpoint_every: int = 1
    early_stop_patience: int = 25
    resume: bool = True
    log_every: int = 20
    val_every: int = 5
    device: str = "auto"
    save_checkpoint: bool = True


@dataclass
class SSLTrainResult:
    """Container for Bio-CLIP pre-training results.

    Fields:
        train_losses: Per-epoch training loss history.
        val_losses: Per-epoch validation loss history (at val_every intervals).
        best_val_loss: Best validation loss achieved.
        best_epoch: Epoch of best validation loss.
        best_state: Optional dict holding best encoder state dicts.
        temperature_history: Learned temperature per epoch.
        total_time_s: Total training wall-clock time.
        checkpoint_path: Path to saved best checkpoint (if save_checkpoint=True).
        resumed_from_epoch: Starting epoch if resumed from checkpoint.
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    best_state: Optional[Dict[str, Any]] = None
    temperature_history: List[float] = field(default_factory=list)
    total_time_s: float = 0.0
    checkpoint_path: Optional[str] = None
    resumed_from_epoch: int = 0
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "temperature": [],
            "lr": [],
        }
    )

    def to_dataframe(self):
        """Convert history metrics into a pandas DataFrame."""
        import pandas as pd
        if not self.history or "epoch" not in self.history or not self.history["epoch"]:
            return pd.DataFrame()
        df = pd.DataFrame(self.history)
        if "epoch" in df.columns:
            df["epoch"] = df["epoch"].astype(int)
            df.set_index("epoch", inplace=True)
        return df

    def to_report(self) -> str:
        """Generate a formatted markdown/text report of Bio-CLIP SSL metrics."""
        lines = [
            "=" * 76,
            "                   BIO-CLIP SSL PRE-TRAINING METRICS REPORT",
            "=" * 76,
            f"  • Completed Epochs: {len(self.history.get('epoch', []))}",
            f"  • Best Epoch:       {self.best_epoch}",
            f"  • Best Val Loss:    {self.best_val_loss:.4f}",
            f"  • Total Time:       {self.total_time_s/60:.2f} min ({self.total_time_s:.1f}s)",
            f"  • Resumed From:     Epoch {self.resumed_from_epoch}",
            "-" * 76,
        ]
        epochs = self.history.get("epoch", [])
        if epochs:
            header = f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Temp τ':^10} | {'LR':^10} | {'Best':^5}"
            sep = "-" * len(header)
            lines.extend([header, sep])
            for i, ep in enumerate(epochs):
                tr = self.history.get("train_loss", [0.0]*len(epochs))[i]
                vl = self.history.get("val_loss", [float("nan")]*len(epochs))[i]
                vl_str = f"{vl:^12.4f}" if not math.isnan(vl) else f"{'-':^12s}"
                temp = self.history.get("temperature", [0.0]*len(epochs))[i]
                lr = self.history.get("lr", [0.0]*len(epochs))[i]
                star = "  *" if ep == self.best_epoch else ""
                row = f"{ep:^6d} | {tr:^12.4f} | {vl_str} | {temp:^10.4f} | {lr:^10.2e} |{star:^5s}"
                lines.append(row)
            lines.append("=" * 76)
            lines.append("  * Indicates best validation loss checkpoint")
        else:
            lines.append("  [No training history recorded]")
            lines.append("=" * 76)
        return "\n".join(lines)

    def summary(self) -> str:
        """Alias for to_report()."""
        return self.to_report()

    def save_metrics(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Save history and report to JSON, CSV, and Markdown in output_dir."""
        import json
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. JSON
        json_path = out_p / "metrics.json"
        data = {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_time_s": self.total_time_s,
            "resumed_from_epoch": self.resumed_from_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "temperature_history": self.temperature_history,
            "history": self.history,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        paths["json"] = str(json_path)

        # 2. CSV
        csv_path = out_p / "metrics.csv"
        try:
            df = self.to_dataframe()
            if not df.empty:
                df.to_csv(csv_path)
                paths["csv"] = str(csv_path)
        except Exception:
            pass

        # 3. Markdown Report
        md_path = out_p / "metrics_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_report())
        paths["report"] = str(md_path)

        return paths

    @classmethod
    def load_metrics(cls, path_or_dir: Union[str, Path]) -> "SSLTrainResult":
        """Load metrics from a saved JSON or directory to switch to past runs."""
        import json
        p = Path(path_or_dir)
        if p.is_dir():
            p = p / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"Metrics file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            train_losses=data.get("train_losses", []),
            val_losses=data.get("val_losses", []),
            best_val_loss=data.get("best_val_loss", float("inf")),
            best_epoch=data.get("best_epoch", 0),
            temperature_history=data.get("temperature_history", []),
            total_time_s=data.get("total_time_s", 0.0),
            resumed_from_epoch=data.get("resumed_from_epoch", 0),
            history=data.get("history", {}),
        )


# ============================================================================
# Dense Token-Level InfoNCE (Frame-Level Contrastive)
# ============================================================================

class DenseTokenInfoNCE(nn.Module):
    """Token-level InfoNCE between H_eeg (B, T, D) and H_emg (B, T, D).

    Aligns each EEG frame token to its temporally-paired EMG frame token.
    The loss is averaged over the temporal dimension and batch.

    Mathematically, for frame t of sample i:
        L_{i,t} = -log [ sim(h_eeg[i,t], h_emg[i,t]) / Σ_j sim(h_eeg[i,t], h_emg[j,t]) ]

    This encourages fine-grained temporal alignment of corticomuscular dynamics.

    Args:
        temperature: Scalar temperature (shared with global loss for consistency).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def update_temperature(self, new_temp: float) -> None:
        """Sync temperature from learnable global temperature parameter."""
        self.temperature.fill_(new_temp)

    def forward(self, H_eeg: Tensor, H_emg: Tensor) -> Tensor:
        """
        Args:
            H_eeg: (B, T, D) L2-normalized EEG frame tokens
            H_emg: (B, T, D) L2-normalized EMG frame tokens

        Returns:
            Scalar dense InfoNCE loss averaged over T and B
        """
        B, T, D = H_eeg.shape

        # Compute per-frame losses via a loop over T (memory efficient for large T)
        frame_losses = []
        for t in range(T):
            h_e = H_eeg[:, t, :]  # (B, D)
            h_m = H_emg[:, t, :]  # (B, D)

            logits = torch.mm(h_e, h_m.T) / self.temperature  # (B, B)
            labels = torch.arange(B, device=H_eeg.device)

            loss_t = 0.5 * (
                torch.nn.functional.cross_entropy(logits, labels) +
                torch.nn.functional.cross_entropy(logits.T, labels)
            )
            frame_losses.append(loss_t)

        return torch.stack(frame_losses).mean()


# ============================================================================
# Bio-CLIP Trainer
# ============================================================================

class BioCLIPTrainer:
    """Pre-training trainer for the Bio-CLIP framework.

    Manages the full pre-training loop including:
    - Warm-up + cosine annealing LR schedule
    - Symmetric PhaseAwareInfoNCELoss (global)
    - Optional dense token-level InfoNCE
    - AMP (Automatic Mixed Precision) with GradScaler
    - Crash-safe checkpointing on every epoch (last.pt) and best validation loss (best.pt)
    - Interactive CLI progress bars via tqdm.auto

    Args:
        eeg_encoder: EEGEncoder instance.
        emg_encoder: EMGEncoder instance.
        phase_labeler: PhaseLabeler instance (optional, for phase masking).
        config: SSLTrainConfig with all hyperparameters.
    """

    def __init__(
        self,
        eeg_encoder: EEGEncoder,
        emg_encoder: EMGEncoder,
        phase_labeler: Optional[PhaseLabeler] = None,
        config: Optional[SSLTrainConfig] = None,
    ) -> None:
        self.config = config or SSLTrainConfig()
        cfg = self.config

        # Resolve device
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        self.eeg_encoder = eeg_encoder.to(self.device)
        self.emg_encoder = emg_encoder.to(self.device)
        self.phase_labeler = phase_labeler

        # Primary contrastive loss
        self.loss_fn = PhaseAwareInfoNCELoss(
            temperature=cfg.temperature,
            learnable_temp=cfg.learnable_temp,
            rest_weight=cfg.rest_weight,
        ).to(self.device)

        # Optional dense token-level InfoNCE
        self.dense_loss_fn: Optional[DenseTokenInfoNCE] = None
        if cfg.use_dense_loss:
            self.dense_loss_fn = DenseTokenInfoNCE(temperature=cfg.temperature).to(self.device)

        # Collect all trainable parameters
        params = (
            list(self.eeg_encoder.parameters()) +
            list(self.emg_encoder.parameters()) +
            list(self.loss_fn.parameters())
        )
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Cosine annealing scheduler (with linear warmup)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.n_epochs,
            eta_min=cfg.learning_rate * 0.01,
        )

        # AMP scaler
        use_cuda_amp = cfg.use_amp and self.device.type == "cuda"
        if use_cuda_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Checkpoints setup
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.last_ckpt = self.ckpt_dir / "last.pt"
        self.best_ckpt = self.ckpt_dir / "best.pt"

        # Training state
        self._step = 0
        self._epoch = 0

    def _warmup_lr(self, epoch: int) -> None:
        """Apply linear warmup to learning rate for first warmup_epochs."""
        if epoch < self.config.warmup_epochs:
            warmup_factor = (epoch + 1) / max(self.config.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.learning_rate * warmup_factor

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        bad_epochs: int,
        result: SSLTrainResult,
    ) -> None:
        """Save a crash-safe, resumable checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "eeg_encoder_state_dict": self.eeg_encoder.state_dict(),
                "emg_encoder_state_dict": self.emg_encoder.state_dict(),
                "loss_fn_state_dict": self.loss_fn.state_dict(),
                "dense_loss_fn_state_dict": self.dense_loss_fn.state_dict() if self.dense_loss_fn is not None else None,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "bad_epochs": bad_epochs,
                "train_losses": result.train_losses,
                "val_losses": result.val_losses,
                "temperature_history": result.temperature_history,
                "step": self._step,
            },
            path,
        )

    def _load_checkpoint(
        self,
        path: Path,
        result: SSLTrainResult,
    ) -> Tuple[int, float, int]:
        """Load checkpoint and restore training states."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.eeg_encoder.load_state_dict(ckpt["eeg_encoder_state_dict"])
        self.emg_encoder.load_state_dict(ckpt["emg_encoder_state_dict"])
        self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
        if self.dense_loss_fn is not None and ckpt.get("dense_loss_fn_state_dict") is not None:
            self.dense_loss_fn.load_state_dict(ckpt["dense_loss_fn_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        result.train_losses = ckpt.get("train_losses", result.train_losses)
        result.val_losses = ckpt.get("val_losses", result.val_losses)
        result.temperature_history = ckpt.get("temperature_history", result.temperature_history)
        result.best_val_loss = ckpt.get("best_val_loss", result.best_val_loss)
        result.best_epoch = ckpt.get("best_epoch", result.best_epoch)
        self._step = ckpt.get("step", self._step)
        epoch = ckpt["epoch"]
        bad_epochs = ckpt.get("bad_epochs", 0)
        return epoch, result.best_val_loss, bad_epochs

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch with interactive CLI progress bar.

        Returns:
            Dict with 'loss', 'global_loss', 'dense_loss', 'temperature'
        """
        self.eeg_encoder.train()
        self.emg_encoder.train()
        self.loss_fn.train()

        epoch_loss = 0.0
        epoch_global_loss = 0.0
        epoch_dense_loss = 0.0
        n_batches = 0

        phase_str = "SSL-Train"
        if _TQDM_AVAILABLE:
            bar = tqdm(
                loader,
                desc=f"Epoch {epoch:02d}/{self.config.n_epochs} [{phase_str}]",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            bar = loader

        use_cuda_amp = self.config.use_amp and self.device.type == "cuda"

        for batch_idx, batch in enumerate(bar):
            eeg = batch["eeg"].to(self.device, non_blocking=True)   # (B, T, n_eeg)
            emg = batch["emg"].to(self.device, non_blocking=True)   # (B, T, n_emg)
            kin = batch["kin"].to(self.device, non_blocking=True)   # (B, T, kin_dim)
            phases = batch["phase"].to(self.device, non_blocking=True)  # (B,)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=use_cuda_amp):
                # Forward pass: encode EEG and EMG
                z_eeg, H_eeg = self.eeg_encoder(eeg, return_dense=self.config.use_dense_loss)
                z_emg, H_emg = self.emg_encoder(emg, return_dense=self.config.use_dense_loss)

                # Phase labels for masking (use kin-based or EMG-based)
                if self.config.use_phase_masking and self.phase_labeler is not None:
                    phases = self.phase_labeler(kin)

                # Global PhaseAwareInfoNCE loss
                global_loss = self.loss_fn(z_eeg, z_emg, phases=phases)

                # Dense token-level InfoNCE (optional)
                dense_loss = torch.tensor(0.0, device=self.device)
                if self.config.use_dense_loss and H_eeg is not None and H_emg is not None:
                    # Sync temperature from learnable parameter
                    if self.dense_loss_fn is not None:
                        self.dense_loss_fn.update_temperature(
                            float(self.loss_fn.temperature.detach())
                        )
                        dense_loss = self.dense_loss_fn(H_eeg, H_emg)

                # Combined loss
                total_loss = global_loss + self.config.dense_loss_weight * dense_loss

            # Backward + gradient clipping
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.eeg_encoder.parameters()) +
                list(self.emg_encoder.parameters()) +
                list(self.loss_fn.parameters()),
                max_norm=self.config.clip_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = total_loss.item()
            epoch_loss += batch_loss
            epoch_global_loss += global_loss.item()
            epoch_dense_loss += dense_loss.item()
            n_batches += 1
            self._step += 1

            if _TQDM_AVAILABLE:
                temp = float(self.loss_fn.temperature.detach())
                bar.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "glob": f"{global_loss.item():.3f}",
                    "τ": f"{temp:.3f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.1e}",
                })

        if _TQDM_AVAILABLE:
            bar.close()

        denom = max(n_batches, 1)
        return {
            "loss": epoch_loss / denom,
            "global_loss": epoch_global_loss / denom,
            "dense_loss": epoch_dense_loss / denom,
            "temperature": float(self.loss_fn.temperature.detach()),
        }

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one validation epoch with CLI progress bar.

        Returns:
            Average validation loss
        """
        self.eeg_encoder.eval()
        self.emg_encoder.eval()
        self.loss_fn.eval()

        val_loss = 0.0
        n_batches = 0

        phase_str = "SSL-Val"
        if _TQDM_AVAILABLE:
            bar = tqdm(
                loader,
                desc=f"Epoch {epoch:02d}/{self.config.n_epochs} [{phase_str}]",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            bar = loader

        use_cuda_amp = self.config.use_amp and self.device.type == "cuda"

        for batch in bar:
            eeg = batch["eeg"].to(self.device, non_blocking=True)
            emg = batch["emg"].to(self.device, non_blocking=True)
            kin = batch["kin"].to(self.device, non_blocking=True)

            phases = batch["phase"].to(self.device, non_blocking=True)
            if self.config.use_phase_masking and self.phase_labeler is not None:
                phases = self.phase_labeler(kin)

            with autocast(device_type=self.device.type, enabled=use_cuda_amp):
                z_eeg, _ = self.eeg_encoder(eeg, return_dense=False)
                z_emg, _ = self.emg_encoder(emg, return_dense=False)
                loss = self.loss_fn(z_eeg, z_emg, phases=phases)

            batch_val_loss = loss.item()
            val_loss += batch_val_loss
            n_batches += 1

            if _TQDM_AVAILABLE:
                bar.set_postfix({"val_loss": f"{batch_val_loss:.4f}"})

        if _TQDM_AVAILABLE:
            bar.close()

        return val_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> SSLTrainResult:
        """Full Bio-CLIP crash-safe, resumable pre-training loop.

        Args:
            train_loader: Training DataLoader (phase-balanced).
            val_loader: Validation DataLoader (optional).
            verbose: Print epoch-level summary.

        Returns:
            SSLTrainResult with training history and best checkpoint info.
        """
        result = SSLTrainResult()
        t_start = time.time()
        start_epoch = 0
        best_val_loss = float("inf")
        bad_epochs = 0
        cfg = self.config

        # Resume from checkpoint if available
        if cfg.resume and self.last_ckpt.exists():
            start_epoch, best_val_loss, bad_epochs = self._load_checkpoint(
                self.last_ckpt, result
            )
            result.resumed_from_epoch = start_epoch
            result.best_val_loss = best_val_loss

            if self.best_ckpt.exists():
                b_ckpt = torch.load(self.best_ckpt, map_location="cpu", weights_only=False)
                result.best_state = {
                    "eeg_encoder": b_ckpt.get("eeg_encoder_state_dict"),
                    "emg_encoder": b_ckpt.get("emg_encoder_state_dict"),
                }
                result.best_epoch = b_ckpt.get("epoch", 0)
                result.checkpoint_path = str(self.best_ckpt)

            if verbose:
                print(
                    f"\n[RESUME] Resumed from checkpoint: {self.last_ckpt}\n"
                    f"  Completed Epochs : {start_epoch}/{cfg.n_epochs}\n"
                    f"  Best Val Loss    : {best_val_loss:.4f} (at Epoch {result.best_epoch})\n"
                    f"  Bad Epochs       : {bad_epochs}/{cfg.early_stop_patience}\n"
                )

            if start_epoch >= cfg.n_epochs or bad_epochs >= cfg.early_stop_patience:
                if verbose:
                    print("Training already completed or early-stopping threshold met.")
                return result
        else:
            if verbose:
                print(f"\n[INIT] Starting fresh Bio-CLIP pre-training (checkpoints -> {self.ckpt_dir})\n")

        for epoch in range(start_epoch + 1, cfg.n_epochs + 1):
            self._epoch = epoch
            t0 = time.time()

            # Linear warmup (overrides cosine schedule during warmup)
            self._warmup_lr(epoch - 1)

            # Train epoch
            train_metrics = self._train_epoch(train_loader, epoch=epoch)
            result.train_losses.append(train_metrics["loss"])
            result.temperature_history.append(train_metrics["temperature"])

            # Cosine LR decay (after warmup)
            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            # Validation epoch
            is_val_epoch = val_loader is not None and (
                epoch % cfg.val_every == 0 or epoch == cfg.n_epochs
            )

            if is_val_epoch:
                val_loss = self._val_epoch(val_loader, epoch=epoch)
                result.val_losses.append(val_loss)

                is_best = val_loss < best_val_loss
                flag = ""
                if is_best:
                    best_val_loss = val_loss
                    result.best_val_loss = best_val_loss
                    result.best_epoch = epoch
                    result.best_state = {
                        "eeg_encoder": {k: v.detach().cpu().clone() for k, v in self.eeg_encoder.state_dict().items()},
                        "emg_encoder": {k: v.detach().cpu().clone() for k, v in self.emg_encoder.state_dict().items()},
                    }
                    bad_epochs = 0
                    flag = "  <-- BEST"
                    if cfg.save_checkpoint:
                        self._save_checkpoint(self.best_ckpt, epoch, val_loss, bad_epochs, result)
                        result.checkpoint_path = str(self.best_ckpt)
                else:
                    bad_epochs += 1
            else:
                val_loss = train_metrics["loss"]
                flag = ""

            # Save last.pt every epoch
            if cfg.save_checkpoint and epoch % cfg.checkpoint_every == 0:
                self._save_checkpoint(self.last_ckpt, epoch, val_loss, bad_epochs, result)

            epoch_time = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record full history
            result.history["epoch"].append(epoch)
            result.history["train_loss"].append(train_metrics["loss"])
            result.history["val_loss"].append(val_loss if is_val_epoch else float("nan"))
            result.history["temperature"].append(train_metrics["temperature"])
            result.history["lr"].append(current_lr)

            # Auto-save full metrics (JSON, CSV, Markdown) on each epoch
            if cfg.save_checkpoint:
                result.save_metrics(self.ckpt_dir)

            if verbose:
                if is_val_epoch:
                    print(
                        f"Epoch [{epoch:3d}/{cfg.n_epochs}] [{epoch_time:.1f}s] "
                        f"train_loss={train_metrics['loss']:.4f} "
                        f"val_loss={val_loss:.4f} "
                        f"τ={train_metrics['temperature']:.4f} "
                        f"lr={current_lr:.2e}{flag}"
                    )
                else:
                    print(
                        f"Epoch [{epoch:3d}/{cfg.n_epochs}] [{epoch_time:.1f}s] "
                        f"train_loss={train_metrics['loss']:.4f} "
                        f"τ={train_metrics['temperature']:.4f} "
                        f"lr={current_lr:.2e}"
                    )

            # Early stopping check
            if bad_epochs >= cfg.early_stop_patience:
                if verbose:
                    print(f"\n[EARLY STOP] No improvement in {cfg.early_stop_patience} validation epochs. Stopping.")
                break

        result.total_time_s = time.time() - t_start

        # Load best weights into encoders if available
        if result.best_state is not None:
            self.eeg_encoder.load_state_dict(result.best_state["eeg_encoder"])
            self.emg_encoder.load_state_dict(result.best_state["emg_encoder"])

        # Final persistent metrics save
        saved_paths = {}
        if cfg.save_checkpoint:
            saved_paths = result.save_metrics(self.ckpt_dir)

        if verbose:
            print(f"\nPre-training complete in {result.total_time_s/60:.1f} min. "
                  f"Best val_loss={result.best_val_loss:.4f} @ epoch {result.best_epoch}.")
            print(f"Checkpoints saved to: {self.ckpt_dir.resolve()}")
            if saved_paths:
                print(f"Metrics & Report saved: {saved_paths.get('json')} | {saved_paths.get('csv')} | {saved_paths.get('report')}\n")

        return result

    def load_best_checkpoint(self, checkpoint_path: str) -> None:
        """Load encoder weights from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.eeg_encoder.load_state_dict(ckpt["eeg_encoder_state_dict"])
        self.emg_encoder.load_state_dict(ckpt["emg_encoder_state_dict"])
        self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
        val_loss_str = f"{ckpt['val_loss']:.4f}" if "val_loss" in ckpt else "N/A"
        print(f"Loaded checkpoint: {checkpoint_path} (epoch={ckpt['epoch']}, "
              f"val_loss={val_loss_str})")

    @property
    def frozen_eeg_encoder(self) -> EEGEncoder:
        """Return EEGEncoder with all parameters frozen for downstream evaluation."""
        for param in self.eeg_encoder.parameters():
            param.requires_grad_(False)
        self.eeg_encoder.eval()
        return self.eeg_encoder
