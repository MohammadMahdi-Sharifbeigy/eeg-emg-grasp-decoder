"""SSL Dataset for Bio-CLIP Pre-training on WAY-EEG-GAL.

SSLWindowDataset: A PyTorch Dataset that yields synchronized (EEG, EMG, kin)
windows for Bio-CLIP pre-training. Each batch item is a dict:
    {
        'eeg': Tensor (T, n_eeg_channels),
        'emg': Tensor (T, n_emg_channels),
        'kin': Tensor (T, kin_dim),
        'phase': int64 — movement phase label (0=REST, 1=ACTIVE, 2=TRANSIT),
        'subject_id': int64 — participant index,
        'trial_id': int64 — trial index within participant,
        'window_idx': int64 — global window index,
    }

Balanced sampling:
    REST windows are typically 3–5× more frequent than ACTIVE windows in
    WAY-EEG-GAL. SSLWindowDataset supports inverse-frequency weighting via
    WeightedRandomSampler to maintain REST:ACTIVE:TRANSIT ≈ 1:1:1 in each batch.

build_ssl_dataloaders() creates train/val DataLoaders with proper
phase-balanced sampling for pre-training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .phase_labeler import PhaseLabeler


class SSLWindowDataset(Dataset):
    """Sliding-window dataset for Bio-CLIP pre-training.

    Args:
        eeg_list: List of (T_i, n_eeg) numpy arrays (one per trial).
        emg_list: List of (T_i, n_emg) numpy arrays (one per trial).
        kin_list: List of (T_i, kin_dim) numpy arrays (one per trial).
        window_size: Window length in samples (default: 500 = 1.0 s at 500 Hz).
        stride: Hop size in samples (default: 50 = 100 ms).
        phase_labeler: Pre-fit PhaseLabeler instance. If None, labels are
                       all TRANSIT (phase masking disabled).
        subject_ids: Parallel list of integer subject IDs per trial.
        trial_ids: Parallel list of integer trial IDs per trial.
        fs: Sampling frequency in Hz (default: 500.0).
    """

    def __init__(
        self,
        eeg_list: List[np.ndarray],
        emg_list: List[np.ndarray],
        kin_list: List[np.ndarray],
        window_size: int = 500,
        stride: int = 50,
        phase_labeler: Optional[PhaseLabeler] = None,
        subject_ids: Optional[List[int]] = None,
        trial_ids: Optional[List[int]] = None,
        fs: float = 500.0,
    ) -> None:
        super().__init__()
        assert len(eeg_list) == len(emg_list) == len(kin_list), \
            "eeg_list, emg_list, kin_list must have equal length."

        self.window_size = window_size
        self.stride = stride
        self.phase_labeler = phase_labeler
        self.fs = fs

        # Precompute all valid (trial_idx, start_sample) windows
        self._windows: List[Tuple[int, int]] = []
        self._eeg_list = eeg_list
        self._emg_list = emg_list
        self._kin_list = kin_list
        self._subject_ids = subject_ids or list(range(len(eeg_list)))
        self._trial_ids = trial_ids or list(range(len(eeg_list)))

        for trial_idx, eeg in enumerate(eeg_list):
            T = eeg.shape[0]
            starts = range(0, T - window_size + 1, stride)
            for s in starts:
                self._windows.append((trial_idx, s))

        # Pre-compute phase labels (using kinematics) for all windows
        self._phase_labels: List[int] = self._precompute_phases()

    def _precompute_phases(self) -> List[int]:
        """Pre-compute phase labels for all windows."""
        if self.phase_labeler is None:
            return [2] * len(self._windows)  # All TRANSIT

        labels = []
        for trial_idx, start in self._windows:
            kin_win = self._kin_list[trial_idx][start : start + self.window_size]
            # Phase labeler expects (B, T, K); add batch dim
            kin_t = torch.from_numpy(kin_win).unsqueeze(0).float()  # (1, T, K)
            phase = self.phase_labeler(kin_t).item()
            labels.append(int(phase))
        return labels

    def get_phase_weights(self) -> np.ndarray:
        """Compute per-sample inverse frequency weights for balanced sampling.

        Returns:
            (N,) float array of sampling weights for WeightedRandomSampler.
        """
        labels = np.array(self._phase_labels)
        unique, counts = np.unique(labels, return_counts=True)
        freq = dict(zip(unique.tolist(), counts.tolist()))

        weights = np.zeros(len(labels), dtype=np.float32)
        for phase_id, count in freq.items():
            mask = labels == phase_id
            weights[mask] = 1.0 / count  # Inverse frequency

        # Normalize to [0, 1]
        if weights.max() > 0:
            weights = weights / weights.max()
        return weights

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial_idx, start = self._windows[idx]
        end = start + self.window_size

        eeg = torch.from_numpy(
            self._eeg_list[trial_idx][start:end].astype(np.float32)
        )  # (T, n_eeg)
        emg = torch.from_numpy(
            self._emg_list[trial_idx][start:end].astype(np.float32)
        )  # (T, n_emg)
        kin = torch.from_numpy(
            self._kin_list[trial_idx][start:end].astype(np.float32)
        )  # (T, kin_dim)

        phase = self._phase_labels[idx]
        subject_id = self._subject_ids[trial_idx]
        trial_id = self._trial_ids[trial_idx]

        return {
            "eeg": eeg,
            "emg": emg,
            "kin": kin,
            "phase": torch.tensor(phase, dtype=torch.long),
            "subject_id": torch.tensor(subject_id, dtype=torch.long),
            "trial_id": torch.tensor(trial_id, dtype=torch.long),
            "window_idx": torch.tensor(idx, dtype=torch.long),
        }


def build_ssl_dataloaders(
    eeg_train: List[np.ndarray],
    emg_train: List[np.ndarray],
    kin_train: List[np.ndarray],
    eeg_val: List[np.ndarray],
    emg_val: List[np.ndarray],
    kin_val: List[np.ndarray],
    window_size: int = 500,
    stride: int = 50,
    batch_size: int = 128,
    num_workers: int = 4,
    phase_labeler: Optional[PhaseLabeler] = None,
    subject_ids_train: Optional[List[int]] = None,
    subject_ids_val: Optional[List[int]] = None,
    trial_ids_train: Optional[List[int]] = None,
    trial_ids_val: Optional[List[int]] = None,
    use_balanced_sampling: bool = True,
    fs: float = 500.0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build balanced train/val DataLoaders for Bio-CLIP pre-training.

    Args:
        eeg_train/val: Lists of (T_i, n_eeg) numpy arrays for train/val trials.
        emg_train/val: Lists of (T_i, n_emg) numpy arrays for train/val trials.
        kin_train/val: Lists of (T_i, kin_dim) numpy arrays for train/val trials.
        window_size: Window length in samples (default: 500 = 1 s at 500 Hz).
        stride: Hop size in samples (default: 50 = 100 ms non-overlap).
        batch_size: Batch size for training (default: 128).
        num_workers: DataLoader worker processes (default: 4).
        phase_labeler: Pre-fit PhaseLabeler for phase-aware sampling.
        subject_ids_train/val: Optional subject ID lists per trial.
        trial_ids_train/val: Optional trial ID lists per trial.
        use_balanced_sampling: Use WeightedRandomSampler for balanced phase sampling.
        fs: Sampling frequency in Hz.
        pin_memory: Pin memory for faster GPU transfers.

    Returns:
        (train_loader, val_loader) DataLoader pair.
    """
    train_dataset = SSLWindowDataset(
        eeg_list=eeg_train,
        emg_list=emg_train,
        kin_list=kin_train,
        window_size=window_size,
        stride=stride,
        phase_labeler=phase_labeler,
        subject_ids=subject_ids_train,
        trial_ids=trial_ids_train,
        fs=fs,
    )

    val_dataset = SSLWindowDataset(
        eeg_list=eeg_val,
        emg_list=emg_val,
        kin_list=kin_val,
        window_size=window_size,
        stride=stride,
        phase_labeler=phase_labeler,
        subject_ids=subject_ids_val,
        trial_ids=trial_ids_val,
        fs=fs,
    )

    # Balanced sampling for training
    if use_balanced_sampling:
        weights = train_dataset.get_phase_weights()
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
