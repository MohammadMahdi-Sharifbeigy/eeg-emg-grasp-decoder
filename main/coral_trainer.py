"""
main/coral_trainer.py
=====================
Crash-safe, fully resumable trainer with CLI progress bars for CORAL-Net:
Corticomuscular Ordered Regression with Aligned Latents.

Features:
- Full resumability via `last.pt` and `best.pt` checkpoints.
- Interactive CLI Progress Bar via `tqdm.auto` tracking live loss, CCC, Pearson correlation,
  temporal smoothness, auxiliary synergy loss, and learned corticospinal conduction delay (ms).
- Mixed precision training (AMP) via torch.amp + gradient norm clipping.
- Cosine Annealing learning rate schedule + Early stopping.
- Memory hygiene: sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on CUDA to reduce
  fragmentation, and calls torch.cuda.empty_cache() at the start of each epoch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


@dataclass
class CORALTrainConfig:
    """Configuration for CORAL-Net training."""
    epochs: int = 25
    lr: float = 5e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    checkpoint_dir: str = "outputs/coral_checkpoints"
    checkpoint_every: int = 1
    early_stop_patience: int = 15
    resume: bool = True
    device: str = "auto"
    scheduler: str = "cosine"  # "cosine" | "reduce"
    eta_min: float = 1e-6


@dataclass
class CORALTrainResult:
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
            "syn": [],
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
        """Generate a formatted markdown/text report of training metrics."""
        lines = [
            "=" * 78,
            "                   CORAL-NET TRAINING METRICS REPORT",
            "=" * 78,
            f"  • Completed Epochs: {len(self.history.get('epoch', []))}",
            f"  • Best Epoch:       {self.best_epoch}",
            f"  • Best Val Loss:    {self.best_val_loss:.4f}",
            f"  • Total Time:       {self.total_time_s/60:.2f} min ({self.total_time_s:.1f}s)",
            f"  • Resumed From:     Epoch {self.resumed_from_epoch}",
            "-" * 78,
        ]
        epochs = self.history.get("epoch", [])
        if epochs:
            header = f"{'Epoch':^6} | {'Train':^8} | {'Val':^8} | {'CCC':^7} | {'Pearson':^7} | {'Lag(ms)':^7} | {'LR':^8} | {'Best':^5}"
            sep = "-" * len(header)
            lines.extend([header, sep])
            for i, ep in enumerate(epochs):
                tr = self.history.get("train_loss", [0.0]*len(epochs))[i]
                vl = self.history.get("val_loss", [0.0]*len(epochs))[i]
                ccc = self.history.get("ccc", [0.0]*len(epochs))[i]
                r = self.history.get("pearson", [0.0]*len(epochs))[i]
                lag = self.history.get("learned_lag_ms", [0.0]*len(epochs))[i]
                lr = self.history.get("lr", [0.0]*len(epochs))[i]
                star = "  *" if ep == self.best_epoch else ""
                row = f"{ep:^6d} | {tr:^8.4f} | {vl:^8.4f} | {ccc:^7.3f} | {r:^7.3f} | {lag:^7.1f} | {lr:^8.1e} |{star:^5s}"
                lines.append(row)
            lines.append("=" * 78)
            lines.append("  * Indicates best validation loss checkpoint")
        else:
            lines.append("  [No training history recorded]")
            lines.append("=" * 78)
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
    def load_metrics(cls, path_or_dir: Union[str, Path]) -> "CORALTrainResult":
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


class CORALTrainer:
    """Crash-safe, resumable trainer for CORAL-Net."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        prepare_batch: Optional[Callable] = None,
        config: Optional[CORALTrainConfig] = None,
    ) -> None:
        self.config = config or CORALTrainConfig()
        cfg = self.config

        # Device resolution
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # Memory hygiene: reduce CUDA allocator fragmentation
        if self.device.type == "cuda":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.prepare_batch = prepare_batch

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
        result: CORALTrainResult,
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
        result: CORALTrainResult,
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
        # Release any lingering cached allocations from the previous epoch
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.model.train(is_train)
        total_loss = 0.0
        running_comps = {"ccc": 0.0, "pearson": 0.0, "diff": 0.0, "syn": 0.0}
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
            if self.prepare_batch is not None:
                if isinstance(batch, (tuple, list)):
                    inputs, target = self.prepare_batch(*batch)
                else:
                    inputs, target = self.prepare_batch(batch)
            else:
                if isinstance(batch, (tuple, list)):
                    inputs = {"eeg": batch[0].to(self.device, non_blocking=True)}
                    if len(batch) > 2:
                        inputs["kin"] = batch[1].to(self.device, non_blocking=True)
                        target = batch[2].to(self.device, non_blocking=True)
                    else:
                        target = batch[1].to(self.device, non_blocking=True)
                else:
                    inputs = {
                        "eeg": batch["eeg"].to(self.device, non_blocking=True),
                        "kin": batch.get("kin", torch.empty(0)).to(self.device, non_blocking=True),
                    }
                    target = batch["emg"].to(self.device, non_blocking=True)

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                    kin_in = inputs.get("kin", None)
                    if kin_in is not None and kin_in.numel() > 0:
                        pred, syn = self.model(inputs["eeg"], kin_in, return_synergies=True)
                    else:
                        pred, syn = self.model(inputs["eeg"], return_synergies=True)

                    if hasattr(self.criterion, "forward"):
                        try:
                            loss, comp = self.criterion(
                                pred, target, model=self.model, synergies=syn, return_components=True
                            )
                        except TypeError:
                            loss = self.criterion(pred, target)
                            comp = getattr(self.criterion, "last_components", {})
                    else:
                        loss = self.criterion(pred, target)
                        comp = {}

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

            for k in running_comps:
                if k in comp:
                    running_comps[k] += comp[k]

            if _TQDM_AVAILABLE:
                postfix = {"loss": f"{batch_loss:.4f}"}
                if "ccc" in comp:
                    postfix["ccc"] = f"{comp['ccc']:.3f}"
                if "pearson" in comp:
                    postfix["r"] = f"{comp['pearson']:.3f}"
                if hasattr(self.model, "current_lag_ms"):
                    postfix["lag_ms"] = f"{self.model.current_lag_ms:.1f}"
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
    ) -> CORALTrainResult:
        """Execute full, crash-safe, resumable training loop."""
        result = CORALTrainResult()
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
            result.history["syn"].append(train_metrics["syn"])
            lag_ms = getattr(self.model, "current_lag_ms", 0.0)
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
                    f"Pearson: {train_metrics['pearson']:.3f} | "
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
