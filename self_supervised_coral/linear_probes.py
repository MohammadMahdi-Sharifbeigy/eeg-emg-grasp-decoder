"""Linear Probes for Bio-CLIP Downstream Evaluation.

After Bio-CLIP pre-training, the EEG encoder is frozen and evaluated using
lightweight probes on downstream motor tasks:

    1. LinearProbe — Standard logistic/linear regression (sklearn-style API)
       for movement phase classification (REST / ACTIVE / TRANSIT).

    2. FewShotRegressionProbe — Ridge regression probe for continuous EMG
       envelope prediction (few-shot, using only N samples per channel).

    3. evaluate_linear_probe — Full evaluation pipeline: extract embeddings,
       train probe, evaluate on validation split, return metrics dict.

    4. evaluate_few_shot_regression — Few-shot EMG regression evaluation
       with correlation (r) and CCC metrics.

All probes operate exclusively on FROZEN encoder outputs — no gradient flows
back into the encoder during evaluation.

Baseline comparison modes:
    - "bioclip": Embeddings from pre-trained EEGEncoder (Bio-CLIP).
    - "supervised": Embeddings from EEGEncoder fine-tuned end-to-end on labels.
    - "random": Randomly initialized EEGEncoder (no training).
    - "mae": Embeddings from unimodal Masked Autoencoder baseline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    """Linear classification probe on top of frozen encoder embeddings.

    Trains a single linear layer on top of the (frozen) encoder output
    using cross-entropy loss. Evaluates with balanced accuracy.

    Args:
        in_dim: Embedding dimension (proj_dim of EEGEncoder, default: 128).
        n_classes: Number of output classes (default: 3 for REST/ACTIVE/TRANSIT).
        learning_rate: Adam optimizer learning rate (default: 1e-3).
        n_epochs: Training epochs (default: 100).
        batch_size: Training batch size (default: 256).
        weight_decay: L2 regularization (default: 1e-4).
        device: Torch device to run training on.
    """

    def __init__(
        self,
        in_dim: int = 128,
        n_classes: int = 3,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device or torch.device("cpu")

        self.linear = nn.Linear(in_dim, n_classes).to(self.device)
        self._is_trained = False

    def fit(
        self,
        embeddings: Tensor,
        labels: Tensor,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Train the linear probe on frozen embeddings.

        Args:
            embeddings: (N, in_dim) embedding tensor (no grad required).
            labels: (N,) int64 class labels.
            verbose: Print training progress every 20 epochs.

        Returns:
            History dict with keys ['train_loss', 'train_acc'].
        """
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)

        dataset = TensorDataset(embeddings, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.linear.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        history: Dict[str, List[float]] = {"train_loss": [], "train_acc": []}

        self.linear.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.linear(x_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x_batch.shape[0]
                preds = logits.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += x_batch.shape[0]

            avg_loss = epoch_loss / total
            acc = correct / total
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(acc)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  LinearProbe epoch [{epoch+1}/{self.n_epochs}] "
                      f"loss={avg_loss:.4f} acc={acc:.3f}")

        self._is_trained = True
        return history

    def predict(self, embeddings: Tensor) -> Tensor:
        """Run inference and return predicted class indices.

        Args:
            embeddings: (N, in_dim) embedding tensor.

        Returns:
            (N,) int64 predicted labels.
        """
        self.linear.eval()
        with torch.no_grad():
            logits = self.linear(embeddings.to(self.device))
        return logits.argmax(dim=-1).cpu()

    def score(self, embeddings: Tensor, labels: Tensor) -> Dict[str, float]:
        """Evaluate probe on held-out embeddings.

        Args:
            embeddings: (N, in_dim)
            labels: (N,) int64 labels

        Returns:
            Dict with 'accuracy', 'balanced_accuracy'
        """
        preds = self.predict(embeddings).numpy()
        labels_np = labels.numpy() if isinstance(labels, Tensor) else labels

        acc = float((preds == labels_np).mean())

        # Balanced accuracy: mean per-class recall
        classes = np.unique(labels_np)
        per_class_acc = []
        for c in classes:
            mask = labels_np == c
            if mask.sum() > 0:
                per_class_acc.append((preds[mask] == labels_np[mask]).mean())
        balanced_acc = float(np.mean(per_class_acc)) if per_class_acc else 0.0

        return {"accuracy": acc, "balanced_accuracy": balanced_acc}

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class FewShotRegressionProbe(nn.Module):
    """Few-shot ridge regression probe for EMG envelope prediction.

    Fits a ridge regression from EEG embeddings → EMG envelope using
    only `n_shots` examples per muscle channel. Evaluates with Pearson-r
    and Lin's Concordance Correlation Coefficient (CCC).

    Args:
        in_dim: Embedding dimension.
        out_dim: Output dimension (n_muscles, default: 5).
        n_shots: Max training samples per output channel (default: 50).
        alpha: Ridge regularization strength (default: 1.0).
        device: Torch device.
    """

    def __init__(
        self,
        in_dim: int = 128,
        out_dim: int = 5,
        n_shots: int = 50,
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_shots = n_shots
        self.alpha = alpha
        self.device = device or torch.device("cpu")

        self.linear = nn.Linear(in_dim, out_dim, bias=True).to(self.device)
        self._is_trained = False

    def fit(self, embeddings: Tensor, targets: Tensor) -> None:
        """Fit ridge regression in closed form.

        Solves: W* = (X^T X + α I)^{-1} X^T Y

        Args:
            embeddings: (N, in_dim) embedding tensor — only first n_shots used.
            targets: (N, out_dim) EMG envelope values per window.
        """
        N = min(embeddings.shape[0], self.n_shots)
        X = embeddings[:N].float().to(self.device)  # (n_shots, in_dim)
        Y = targets[:N].float().to(self.device)     # (n_shots, out_dim)

        # Ridge: (X^T X + α I)^{-1} X^T Y
        XtX = X.T @ X  # (D, D)
        XtY = X.T @ Y  # (D, out_dim)
        reg = self.alpha * torch.eye(self.in_dim, device=self.device)

        W = torch.linalg.solve(XtX + reg, XtY)  # (D, out_dim)
        bias = Y.mean(dim=0) - X.mean(dim=0) @ W  # (out_dim,)

        with torch.no_grad():
            self.linear.weight.copy_(W.T)
            self.linear.bias.copy_(bias)

        self._is_trained = True

    def predict(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings: (N, in_dim)

        Returns:
            (N, out_dim) predicted EMG envelope
        """
        self.linear.eval()
        with torch.no_grad():
            return self.linear(embeddings.to(self.device)).cpu()

    def score(self, embeddings: Tensor, targets: Tensor) -> Dict[str, float]:
        """Evaluate regression probe.

        Args:
            embeddings: (N, in_dim)
            targets: (N, out_dim)

        Returns:
            Dict with 'pearson_r' (mean across channels) and 'ccc' (mean CCC)
        """
        preds = self.predict(embeddings)         # (N, out_dim)
        tgt = targets.float().cpu()              # (N, out_dim)

        r_vals = []
        ccc_vals = []

        for c in range(self.out_dim):
            p = preds[:, c]
            t = tgt[:, c]

            # Pearson r
            p_mean = p.mean()
            t_mean = t.mean()
            p_std = (p - p_mean).norm()
            t_std = (t - t_mean).norm()
            if p_std > 1e-8 and t_std > 1e-8:
                r = ((p - p_mean) * (t - t_mean)).sum() / (p_std * t_std)
                r_vals.append(float(r))
            else:
                r_vals.append(0.0)

            # Lin's CCC
            var_p = p.var()
            var_t = t.var()
            cov = ((p - p_mean) * (t - t_mean)).mean()
            denom = (var_p + var_t + (p_mean - t_mean) ** 2).clamp(min=1e-8)
            ccc = (2.0 * cov / denom).item()
            ccc_vals.append(ccc)

        return {
            "pearson_r": float(np.mean(r_vals)),
            "ccc": float(np.mean(ccc_vals)),
            "pearson_r_per_channel": r_vals,
            "ccc_per_channel": ccc_vals,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


@torch.no_grad()
def _extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_phases: bool = True,
    return_emg_targets: bool = False,
) -> Dict[str, Tensor]:
    """Extract embeddings from a frozen encoder over a DataLoader.

    Expects batches as dicts with keys: 'eeg', optionally 'kin', 'emg'.

    Returns:
        Dict with keys 'embeddings', optionally 'phases', 'emg_targets'
    """
    encoder.eval()
    all_embeddings = []
    all_phases = []
    all_emg = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        z_global = encoder.encode(eeg)         # (B, proj_dim)
        all_embeddings.append(z_global.cpu())

        if return_phases and "phase" in batch:
            all_phases.append(batch["phase"].cpu())

        if return_emg_targets and "emg" in batch:
            # Use mean over time as regression target per window
            emg = batch["emg"]  # (B, T, n_muscles)
            all_emg.append(emg.mean(dim=1).cpu())

    result = {"embeddings": torch.cat(all_embeddings, dim=0)}
    if all_phases:
        result["phases"] = torch.cat(all_phases, dim=0)
    if all_emg:
        result["emg_targets"] = torch.cat(all_emg, dim=0)

    return result


def evaluate_linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_classes: int = 3,
    n_epochs: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full linear probe evaluation pipeline.

    1. Extract frozen embeddings from train and val sets.
    2. Train a linear probe on train embeddings.
    3. Evaluate on val embeddings.

    Args:
        encoder: Frozen EEGEncoder (or other encoder with .encode() method).
        train_loader: DataLoader yielding {'eeg': ..., 'phase': ...} dicts.
        val_loader: DataLoader for validation.
        device: Torch device.
        n_classes: Number of movement phase classes.
        n_epochs: Probe training epochs.
        verbose: Print progress.

    Returns:
        Dict with 'train_accuracy', 'val_accuracy', 'val_balanced_accuracy',
        'train_time_s', and the trained 'probe' object.
    """
    if verbose:
        print("[LinearProbe] Extracting embeddings...")

    train_data = _extract_embeddings(encoder, train_loader, device, return_phases=True)
    val_data = _extract_embeddings(encoder, val_loader, device, return_phases=True)

    proj_dim = train_data["embeddings"].shape[1]
    probe = LinearProbe(
        in_dim=proj_dim,
        n_classes=n_classes,
        n_epochs=n_epochs,
        device=device,
    )

    if verbose:
        print(f"[LinearProbe] Training probe on {train_data['embeddings'].shape[0]} samples...")

    t0 = time.time()
    history = probe.fit(
        train_data["embeddings"],
        train_data["phases"],
        verbose=verbose,
    )
    train_time = time.time() - t0

    train_metrics = probe.score(train_data["embeddings"], train_data["phases"])
    val_metrics = probe.score(val_data["embeddings"], val_data["phases"])

    if verbose:
        print(f"[LinearProbe] Train acc={train_metrics['accuracy']:.3f} "
              f"Val acc={val_metrics['accuracy']:.3f} "
              f"Val balanced_acc={val_metrics['balanced_accuracy']:.3f}")

    return {
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "train_time_s": train_time,
        "history": history,
        "probe": probe,
    }


def evaluate_few_shot_regression(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_shots: int = 50,
    n_muscles: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Few-shot EMG regression probe evaluation.

    1. Extract frozen embeddings and EMG window-mean targets.
    2. Fit ridge regression on n_shots examples.
    3. Evaluate on full val set.

    Args:
        encoder: Frozen EEGEncoder.
        train_loader: DataLoader yielding {'eeg': ..., 'emg': ...} dicts.
        val_loader: DataLoader for validation.
        device: Torch device.
        n_shots: Number of training examples for few-shot regression.
        n_muscles: Number of EMG output channels.
        verbose: Print progress.

    Returns:
        Dict with 'val_pearson_r', 'val_ccc', 'val_pearson_r_per_channel',
        'val_ccc_per_channel', and the trained 'probe'.
    """
    if verbose:
        print(f"[FewShotRegression] Extracting embeddings (n_shots={n_shots})...")

    train_data = _extract_embeddings(
        encoder, train_loader, device,
        return_phases=False, return_emg_targets=True,
    )
    val_data = _extract_embeddings(
        encoder, val_loader, device,
        return_phases=False, return_emg_targets=True,
    )

    proj_dim = train_data["embeddings"].shape[1]
    probe = FewShotRegressionProbe(
        in_dim=proj_dim,
        out_dim=n_muscles,
        n_shots=n_shots,
        device=device,
    )

    probe.fit(train_data["embeddings"], train_data["emg_targets"])

    val_metrics = probe.score(val_data["embeddings"], val_data["emg_targets"])

    if verbose:
        print(f"[FewShotRegression] val pearson_r={val_metrics['pearson_r']:.4f} "
              f"val CCC={val_metrics['ccc']:.4f}")

    return {
        "val_pearson_r": val_metrics["pearson_r"],
        "val_ccc": val_metrics["ccc"],
        "val_pearson_r_per_channel": val_metrics["pearson_r_per_channel"],
        "val_ccc_per_channel": val_metrics["ccc_per_channel"],
        "probe": probe,
    }
