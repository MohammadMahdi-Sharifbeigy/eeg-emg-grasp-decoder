"""
synergy_decoding/dataset.py
============================
PyTorch Dataset for Muscle Synergy Decoding.

Extends the standard EEG-EMG dataset by injecting NMF-extracted ground-truth
synergy activations C(t) and the subject-specific synergy matrix W.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SynergyDataset(Dataset):
    """Dataset returning (EEG, EMG, C, W, subject_id) for Pivot 1 training.

    Args:
        eeg_list: List of (T, n_eeg) EEG trials.
        emg_list: List of (T, n_emg) EMG trials.
        C_list: List of (T, k) NMF synergy activations.
        W_dict: Dictionary mapping subject_id -> (k, n_emg) synergy matrix W.
        subject_ids: List of integer subject IDs for each trial.
        window_size: Number of temporal samples per window.
        stride: Stride between overlapping windows.
        fs: Sampling frequency in Hz.
    """

    def __init__(
        self,
        eeg_list: List[np.ndarray],
        emg_list: List[np.ndarray],
        C_list: List[np.ndarray],
        W_dict: Dict[int, np.ndarray],
        subject_ids: List[int],
        window_size: int = 500,
        stride: int = 100,
        fs: float = 500.0,
    ) -> None:
        super().__init__()
        assert len(eeg_list) == len(emg_list) == len(C_list) == len(subject_ids)
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.W_dict = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in W_dict.items()
        }

        self.windows: List[Tuple[int, int, int]] = []  # (trial_idx, start_idx, subj_id)
        self.eeg_data: List[torch.Tensor] = []
        self.emg_data: List[torch.Tensor] = []
        self.C_data: List[torch.Tensor] = []

        for i, (eeg, emg, c, sid) in enumerate(zip(eeg_list, emg_list, C_list, subject_ids)):
            T = eeg.shape[0]
            if T < window_size:
                continue

            self.eeg_data.append(torch.tensor(eeg, dtype=torch.float32))
            self.emg_data.append(torch.tensor(emg, dtype=torch.float32))
            self.C_data.append(torch.tensor(c, dtype=torch.float32))

            for start in range(0, T - window_size + 1, stride):
                self.windows.append((len(self.eeg_data) - 1, start, sid))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trial_idx, start, sid = self.windows[idx]
        end = start + self.window_size

        eeg_win = self.eeg_data[trial_idx][start:end]
        emg_win = self.emg_data[trial_idx][start:end]
        c_win = self.C_data[trial_idx][start:end]
        w_mat = self.W_dict[sid]

        return {
            "eeg": eeg_win,
            "emg": emg_win,
            "c": c_win,
            "w": w_mat,
            "subject_id": torch.tensor(sid, dtype=torch.long),
        }
