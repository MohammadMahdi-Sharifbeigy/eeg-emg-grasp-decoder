"""
synergy_decoding/trainer.py
============================
Crash-safe, fully resumable trainer with CLI progress bars for Pivot 1:
Cortico-Synergy Latent Space Decoder.

Features:
- Full resumability via `last.pt` and `best.pt` checkpoints.
- CLI Progress Bar via `tqdm` tracking live CCC, Pearson, Smoothness,
  Muscle Reconstruction loss, and learned corticospinal conduction delay (ms).
- Mixed precision training (AMP) + gradient norm clipping.
- Cosine Annealing learning rate schedule + Early stopping.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


@dataclass
class SynergyTrainConfig:
    """Configuration for CorticosynergyDecoder training."""
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    checkpoint_dir: str = "outputs/synergy_checkpoints"
    checkpoint_every: int = 1
    early_stop_patience: int = 10
    resume: bool = True
    device: str = "auto"
    scheduler: str = "cosine"  # "cosine" | "reduce"
    eta_min: float = 1e-5


@dataclass
class SynergyTrainResult:
    """Artifact containing training history and best model checkpoint."""
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    best_state: Optional[Dict[str, Any]] = None
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "ccc": [],
            "pearson": [],
            "diff": [],
            "rec": [],
            "learned_lag_ms": [],
            "lr": [],
        }
    )
    total_time_s: float = 0.0
    resumed_from_epoch: int = 0

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
        """Generate a formatted markdown/text report of synergy decoding metrics."""
        lines = [
            "=" * 82,
            "                   CORTICO-SYNERGY DECODER METRICS REPORT",
            "=" * 82,
            f"  • Completed Epochs: {len(self.history.get('epoch', []))}",
            f"  • Best Epoch:       {self.best_epoch}",
            f"  • Best Val Loss:    {self.best_val_loss:.4f}",
            f"  • Total Time:       {self.total_time_s/60:.2f} min ({self.total_time_s:.1f}s)",
            f"  • Resumed From:     Epoch {self.resumed_from_epoch}",
            "-" * 82,
        ]
        epochs = self.history.get("epoch", [])
        if epochs:
            header = f"{'Epoch':^6} | {'Train':^8} | {'Val':^8} | {'CCC':^7} | {'Rec Loss':^8} | {'Lag(ms)':^7} | {'LR':^8} | {'Best':^5}"
            sep = "-" * len(header)
            lines.extend([header, sep])
            for i, ep in enumerate(epochs):
                tr = self.history.get("train_loss", [0.0]*len(epochs))[i]
                vl = self.history.get("val_loss", [0.0]*len(epochs))[i]
                ccc = self.history.get("ccc", [0.0]*len(epochs))[i]
                rec = self.history.get("rec", [0.0]*len(epochs))[i]
                lag = self.history.get("learned_lag_ms", [0.0]*len(epochs))[i]
                lr = self.history.get("lr", [0.0]*len(epochs))[i]
                star = "  *" if ep == self.best_epoch else ""
                row = f"{ep:^6d} | {tr:^8.4f} | {vl:^8.4f} | {ccc:^7.3f} | {rec:^8.4f} | {lag:^7.1f} | {lr:^8.1e} |{star:^5s}"
                lines.append(row)
            lines.append("=" * 82)
            lines.append("  * Indicates best validation loss checkpoint")
        else:
            lines.append("  [No training history recorded]")
            lines.append("=" * 82)
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
    def load_metrics(cls, path_or_dir: Union[str, Path]) -> "SynergyTrainResult":
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
            best_val_loss=data.get("best_val_loss", float("inf")),
            best_epoch=data.get("best_epoch", 0),
            history=data.get("history", {}),
            total_time_s=data.get("total_time_s", 0.0),
            resumed_from_epoch=data.get("resumed_from_epoch", 0),
        )


class SynergyTrainer:
    """Crash-safe, resumable trainer for CorticosynergyDecoder."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: Optional[SynergyTrainConfig] = None,
    ) -> None:
        self.config = config or SynergyTrainConfig()
        cfg = self.config

        # Device resolution
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler
        if cfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.eta_min,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )

        # AMP Scaler
        use_cuda_amp = cfg.use_amp and self.device.type == "cuda"
        if use_cuda_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)

        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.last_ckpt = self.ckpt_dir / "last.pt"
        self.best_ckpt = self.ckpt_dir / "best.pt"

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_val: float,
        bad_epochs: int,
        result: SynergyTrainResult,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_val_loss": best_val,
                "bad_epochs": bad_epochs,
                "history": result.history,
            },
            path,
        )

    def _load_checkpoint(
        self,
        path: Path,
        result: SynergyTrainResult,
    ) -> Tuple[int, float, int]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        result.history = ckpt.get("history", result.history)
        epoch = ckpt["epoch"]
        best_val = ckpt["best_val_loss"]
        bad_epochs = ckpt.get("bad_epochs", 0)
        return epoch, best_val, bad_epochs

    def _run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        is_train: bool = True,
    ) -> Dict[str, float]:
        self.model.train(is_train)
        total_loss = 0.0
        running_comps = {"ccc": 0.0, "pearson": 0.0, "diff": 0.0, "rec": 0.0}
        n_batches = 0

        phase_str = "Train" if is_train else "Val"
        if _TQDM_AVAILABLE:
            bar = tqdm(
                loader,
                desc=f"Epoch {epoch:02d}/{self.config.epochs} [{phase_str}]",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            bar = loader

        use_cuda_amp = self.config.use_amp and self.device.type == "cuda"

        for batch in bar:
            eeg = batch["eeg"].to(self.device, non_blocking=True)
            emg = batch["emg"].to(self.device, non_blocking=True)
            c = batch["c"].to(self.device, non_blocking=True)
            w = batch["w"].to(self.device, non_blocking=True)

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                    c_hat = self.model(eeg)
                    loss = self.criterion(c_hat, c, w, emg)

                if is_train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            n_batches += 1

            # Extract loss diagnostics if available
            comps = getattr(self.criterion, "last_components", {})
            for k in running_comps:
                running_comps[k] += comps.get(k, 0.0)

            if _TQDM_AVAILABLE:
                postfix = {"loss": f"{batch_loss:.4f}"}
                if "ccc" in comps:
                    postfix["ccc"] = f"{comps['ccc']:.3f}"
                if "rec" in comps:
                    postfix["rec"] = f"{comps['rec']:.3f}"
                if hasattr(self.model, "get_lag_ms"):
                    postfix["lag_ms"] = f"{self.model.get_lag_ms():.1f}"
                bar.set_postfix(postfix)

        if _TQDM_AVAILABLE:
            bar.close()

        denom = max(n_batches, 1)
        res = {"loss": total_loss / denom}
        for k in running_comps:
            res[k] = running_comps[k] / denom
        return res

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> SynergyTrainResult:
        """Execute full, crash-safe, resumable training loop."""
        result = SynergyTrainResult()
        t_start = time.time()
        start_epoch = 0
        best_val = float("inf")
        bad_epochs = 0
        cfg = self.config

        # Resume from checkpoint if available
        if cfg.resume and self.last_ckpt.exists():
            start_epoch, best_val, bad_epochs = self._load_checkpoint(
                self.last_ckpt, result
            )
            result.resumed_from_epoch = start_epoch
            result.best_val_loss = best_val
            if self.best_ckpt.exists():
                b_ckpt = torch.load(self.best_ckpt, map_location="cpu", weights_only=False)
                result.best_state = b_ckpt.get("model_state_dict")
                result.best_epoch = b_ckpt.get("epoch", 0)

            if verbose:
                print(
                    f"\n[RESUME] Resumed from checkpoint: {self.last_ckpt}\n"
                    f"  Completed Epochs : {start_epoch}/{cfg.epochs}\n"
                    f"  Best Val Loss    : {best_val:.4f} (at Epoch {result.best_epoch})\n"
                    f"  Bad Epochs       : {bad_epochs}/{cfg.early_stop_patience}\n"
                )

            if start_epoch >= cfg.epochs or bad_epochs >= cfg.early_stop_patience:
                if verbose:
                    print("Training already completed or early-stopping threshold met.")
                return result
        else:
            if verbose:
                print(f"\n[INIT] Starting fresh training run (checkpoints -> {self.ckpt_dir})\n")

        for epoch in range(start_epoch + 1, cfg.epochs + 1):
            t0 = time.time()

            # Train epoch
            train_metrics = self._run_epoch(train_loader, epoch=epoch, is_train=True)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validation epoch
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, epoch=epoch, is_train=False)
                val_loss = val_metrics["loss"]
            else:
                val_loss = train_metrics["loss"]
                val_metrics = train_metrics

            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Logging & history
            result.history["epoch"].append(epoch)
            result.history["train_loss"].append(train_metrics["loss"])
            result.history["val_loss"].append(val_loss)
            result.history["ccc"].append(train_metrics["ccc"])
            result.history["pearson"].append(train_metrics["pearson"])
            result.history["diff"].append(train_metrics["diff"])
            result.history["rec"].append(train_metrics["rec"])
            lag_ms = self.model.get_lag_ms() if hasattr(self.model, "get_lag_ms") else 0.0
            result.history["learned_lag_ms"].append(lag_ms)
            result.history["lr"].append(current_lr)

            # Checkpoint tracking
            is_best = val_loss < best_val
            flag = ""
            if is_best:
                best_val = val_loss
                result.best_val_loss = best_val
                result.best_epoch = epoch
                result.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                bad_epochs = 0
                flag = "  <-- BEST"
                self._save_checkpoint(self.best_ckpt, epoch, best_val, bad_epochs, result)
            else:
                bad_epochs += 1

            # Save last.pt every epoch
            if epoch % cfg.checkpoint_every == 0:
                self._save_checkpoint(self.last_ckpt, epoch, best_val, bad_epochs, result)

            # Auto-save full metrics (JSON, CSV, Markdown) on each epoch
            result.save_metrics(self.ckpt_dir)

            epoch_time = time.time() - t0
            if verbose:
                print(
                    f"Epoch {epoch:2d}/{cfg.epochs} [{epoch_time:.1f}s] | "
                    f"Train: {train_metrics['loss']:.4f} | "
                    f"Val: {val_loss:.4f} | "
                    f"CCC: {train_metrics['ccc']:.3f} | "
                    f"Rec: {train_metrics['rec']:.3f} | "
                    f"Lag: {lag_ms:.1f}ms | "
                    f"LR: {current_lr:.1e}{flag}"
                )

            # Early stopping check
            if bad_epochs >= cfg.early_stop_patience:
                if verbose:
                    print(f"\n[EARLY STOP] No improvement in {cfg.early_stop_patience} epochs. Stopping.")
                break

        result.total_time_s = time.time() - t_start

        # Load best weights back into model
        if result.best_state is not None:
            self.model.load_state_dict(result.best_state)

        # Final persistent metrics save
        saved_paths = result.save_metrics(self.ckpt_dir)

        if verbose:
            print(f"\nTraining complete in {result.total_time_s/60:.1f} min.")
            print(f"Best Val Loss: {result.best_val_loss:.4f} (Epoch {result.best_epoch})")
            print(f"Checkpoints saved to: {self.ckpt_dir.resolve()}")
            print(f"Metrics & Report saved: {saved_paths.get('json')} | {saved_paths.get('csv')} | {saved_paths.get('report')}\n")

        return result
