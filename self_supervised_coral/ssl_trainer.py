"""Bio-CLIP SSL Trainer — Pre-training loop for Pivot 4.

BioCLIPTrainer orchestrates the full pre-training loop:
    1. Encodes EEG windows via EEGEncoder → z_eeg (B, 128) + H_eeg (B, T, 256)
    2. Encodes EMG windows via EMGEncoder → z_emg (B, 128) + H_emg (B, T, 256)
    3. Optionally labels movement phases via PhaseLabeler(kin)
    4. Computes PhaseAwareInfoNCELoss (with false-negative masking)
    5. Optionally adds dense token-level InfoNCE on H_eeg, H_emg
    6. Logs pre-training metrics; checkpoints best val loss

SSLTrainConfig: dataclass-style configuration container.
SSLTrainResult: post-training result container with history and metrics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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
        checkpoint_dir: Directory to save best checkpoint.
        log_every: Log every N steps (default: 20).
        val_every: Validate every N epochs (default: 5).
        device: Training device (default: 'auto' → cuda if available else cpu).
        save_checkpoint: Whether to save best val-loss checkpoint (default: True).
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
        temperature_history: Learned temperature per epoch.
        total_time_s: Total training wall-clock time.
        checkpoint_path: Path to saved best checkpoint (if save_checkpoint=True).
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    temperature_history: List[float] = field(default_factory=list)
    total_time_s: float = 0.0
    checkpoint_path: Optional[str] = None


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

        # Average temporal loss across T frames
        total_loss = torch.tensor(0.0, device=H_eeg.device)

        # Vectorized: reshape to (B*T, D) → compute B*T × B cross-modal similarity
        # Positive pairs: (b, t) EEG with (b, t) EMG → diagonal of B-block structure
        eeg_flat = H_eeg.reshape(B * T, D)  # (B*T, D)
        emg_flat = H_emg.reshape(B * T, D)  # (B*T, D)

        # We only want intra-time-step contrastive loss (across batch, same t)
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
    - Checkpointing on best validation loss

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

        # AMP scaler — use non-deprecated torch.amp.GradScaler for PyTorch 2.x
        use_cuda_amp = cfg.use_amp and self.device.type == "cuda"
        if use_cuda_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Training state
        self._step = 0
        self._epoch = 0

    def _warmup_lr(self, epoch: int) -> None:
        """Apply linear warmup to learning rate for first warmup_epochs."""
        if epoch < self.config.warmup_epochs:
            warmup_factor = (epoch + 1) / max(self.config.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.config.learning_rate * warmup_factor

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch.

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

        for batch_idx, batch in enumerate(loader):
            eeg = batch["eeg"].to(self.device)   # (B, T, n_eeg)
            emg = batch["emg"].to(self.device)   # (B, T, n_emg)
            kin = batch["kin"].to(self.device)   # (B, T, kin_dim)
            phases = batch["phase"].to(self.device)  # (B,)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.config.use_amp and self.device.type == "cuda"):
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

            epoch_loss += total_loss.item()
            epoch_global_loss += global_loss.item()
            epoch_dense_loss += dense_loss.item()
            n_batches += 1
            self._step += 1

            if self._step % self.config.log_every == 0:
                temp = float(self.loss_fn.temperature.detach())
                print(f"  [step {self._step}] loss={total_loss.item():.4f} "
                      f"global={global_loss.item():.4f} "
                      f"dense={dense_loss.item():.4f} "
                      f"τ={temp:.4f}")

        denom = max(n_batches, 1)
        return {
            "loss": epoch_loss / denom,
            "global_loss": epoch_global_loss / denom,
            "dense_loss": epoch_dense_loss / denom,
            "temperature": float(self.loss_fn.temperature.detach()),
        }

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        """Run one validation epoch.

        Returns:
            Average validation loss
        """
        self.eeg_encoder.eval()
        self.emg_encoder.eval()
        self.loss_fn.eval()

        val_loss = 0.0
        n_batches = 0

        for batch in loader:
            eeg = batch["eeg"].to(self.device)
            emg = batch["emg"].to(self.device)
            kin = batch["kin"].to(self.device)

            phases = batch["phase"].to(self.device)
            if self.config.use_phase_masking and self.phase_labeler is not None:
                phases = self.phase_labeler(kin)

            with autocast(device_type=self.device.type, enabled=self.config.use_amp and self.device.type == "cuda"):
                z_eeg, _ = self.eeg_encoder(eeg, return_dense=False)
                z_emg, _ = self.emg_encoder(emg, return_dense=False)
                loss = self.loss_fn(z_eeg, z_emg, phases=phases)

            val_loss += loss.item()
            n_batches += 1

        return val_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> SSLTrainResult:
        """Full Bio-CLIP pre-training loop.

        Args:
            train_loader: Training DataLoader (phase-balanced).
            val_loader: Validation DataLoader (optional).
            verbose: Print epoch-level summary.

        Returns:
            SSLTrainResult with training history and best checkpoint info.
        """
        import os
        result = SSLTrainResult()
        t_start = time.time()
        best_val_loss = float("inf")
        cfg = self.config

        if cfg.save_checkpoint:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        for epoch in range(cfg.n_epochs):
            self._epoch = epoch

            # Linear warmup (overrides cosine schedule during warmup)
            self._warmup_lr(epoch)

            # Train epoch
            train_metrics = self._train_epoch(train_loader)
            result.train_losses.append(train_metrics["loss"])
            result.temperature_history.append(train_metrics["temperature"])

            # Cosine LR decay (after warmup)
            if epoch >= cfg.warmup_epochs:
                self.scheduler.step()

            # Validation
            if val_loader is not None and (epoch + 1) % cfg.val_every == 0:
                val_loss = self._val_epoch(val_loader)
                result.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    result.best_val_loss = best_val_loss
                    result.best_epoch = epoch

                    if cfg.save_checkpoint:
                        ckpt_path = os.path.join(
                            cfg.checkpoint_dir,
                            f"bioclip_best_epoch{epoch:03d}.pt"
                        )
                        torch.save({
                            "epoch": epoch,
                            "eeg_encoder_state_dict": self.eeg_encoder.state_dict(),
                            "emg_encoder_state_dict": self.emg_encoder.state_dict(),
                            "loss_fn_state_dict": self.loss_fn.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "temperature": train_metrics["temperature"],
                        }, ckpt_path)
                        result.checkpoint_path = ckpt_path

                if verbose:
                    print(
                        f"Epoch [{epoch+1:3d}/{cfg.n_epochs}] "
                        f"train_loss={train_metrics['loss']:.4f} "
                        f"val_loss={val_loss:.4f} "
                        f"τ={train_metrics['temperature']:.4f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )
            elif verbose:
                print(
                    f"Epoch [{epoch+1:3d}/{cfg.n_epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"τ={train_metrics['temperature']:.4f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

        result.total_time_s = time.time() - t_start
        if verbose:
            print(f"\nPre-training complete in {result.total_time_s/60:.1f} min. "
                  f"Best val_loss={result.best_val_loss:.4f} @ epoch {result.best_epoch}.")
        return result

    def load_best_checkpoint(self, checkpoint_path: str) -> None:
        """Load encoder weights from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.eeg_encoder.load_state_dict(ckpt["eeg_encoder_state_dict"])
        self.emg_encoder.load_state_dict(ckpt["emg_encoder_state_dict"])
        self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path} (epoch={ckpt['epoch']}, "
              f"val_loss={ckpt['val_loss']:.4f})")

    @property
    def frozen_eeg_encoder(self) -> EEGEncoder:
        """Return EEGEncoder with all parameters frozen for downstream evaluation."""
        for param in self.eeg_encoder.parameters():
            param.requires_grad_(False)
        self.eeg_encoder.eval()
        return self.eeg_encoder
