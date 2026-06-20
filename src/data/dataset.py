"""
PyTorch Dataset for WAY-EEG-GAL.

Loads continuous HS series, extracts the 13-dim kinematic state vector k_t,
slides a fixed-length window over each series, and returns
(eeg_window, kin_window, emg_window) tensors.

Kinematic column mapping in hs.kin.sig (36 cols, 0-indexed):
  0-2   Px1,Py1,Pz1  object position (mm)
  3-5   Px2,Py2,Pz2  index fingertip position (mm)
  6-8   Px3,Py3,Pz3  thumb position (mm)
  9-11  Px4,Py4,Pz4  wrist position (mm)
  12-14 FX1,FY1,FZ1  force plate 1 (index), N
  15-17 FX2,FY2,FZ2  force plate 2 (thumb), N
  18-23 TX1..TZ2     torques, N*mm
  24-35 misc env / remaining

k_t construction (13-dim):
  p_wrist (3)  = cols 9:12
  p_index (3)  = cols 3:6
  p_thumb (3)  = cols 6:9
  d_grip  (1)  = ||p_index - p_thumb||_2
  F_L     (1)  = col 12  (FX1, load force index)
  F_G     (1)  = abs(col 15)  (FZ1 negated in hardware)
  rho_GL  (1)  = F_G / (F_L + eps)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .loader import get_split_series, load_hs, load_participant

# Column indices in hs.kin.sig (0-based)
_KIN_WRIST = slice(9, 12)    # Px4, Py4, Pz4
_KIN_INDEX = slice(3, 6)     # Px2, Py2, Pz2
_KIN_THUMB = slice(6, 9)     # Px3, Py3, Pz3
_KIN_FX1   = 12              # load force index plate  (N)
_KIN_FZ1   = 15              # grip force index plate  (N, may be negated)

_EPS = 1e-8


def extract_kt(kin: np.ndarray) -> np.ndarray:
    """Build 13-dim kinematic state vector from raw 36-channel kin array.

    Args:
        kin: ndarray (T, 36)

    Returns:
        ndarray (T, 13) = [p_wrist(3), p_index(3), p_thumb(3),
                           d_grip(1), F_L(1), F_G(1), rho_GL(1)]
    """
    p_wrist = kin[:, _KIN_WRIST]               # (T, 3)
    p_index = kin[:, _KIN_INDEX]               # (T, 3)
    p_thumb = kin[:, _KIN_THUMB]               # (T, 3)

    d_grip = np.linalg.norm(p_index - p_thumb, axis=1, keepdims=True)  # (T, 1)

    F_L = kin[:, _KIN_FX1: _KIN_FX1 + 1]      # (T, 1)  load force
    F_G = np.abs(kin[:, _KIN_FZ1: _KIN_FZ1 + 1])  # (T, 1) grip force (abs)
    rho_GL = F_G / (np.abs(F_L) + _EPS)        # (T, 1)

    return np.concatenate(
        [p_wrist, p_index, p_thumb, d_grip, F_L, F_G, rho_GL], axis=1
    ).astype(np.float32)


class WAYEEGDataset(Dataset):
    """Sliding-window dataset over WAY-EEG-GAL HS series.

    Each sample is a tuple (eeg, kin, emg) of fixed-length windows:
        eeg : Tensor (window_size, 32)
        kin : Tensor (window_size, 13)   k_t kinematic state
        emg : Tensor (window_size, 5)    EMG envelope target

    EMG is at 4000 Hz in the raw files. Before windowing it must be
    downsampled to 500 Hz (8x) so that EEG, kin, and EMG share the
    same time axis. Pass a `preprocess_fn` that handles this
    (and any other preprocessing) before windowing.

    Args:
        data_dir:       Root dir with P1/, P2/, ... subdirs.
        participants:   List of participant IDs (1-12).
        split:          'train' | 'val' | 'test' | 'stability' | 'all'.
        window_size:    Number of samples per window (default 500 = 1 s @ 500 Hz).
        stride:         Step between consecutive windows (default 50 = 100 ms).
        preprocess_fn:  Optional callable (series_dict) -> series_dict applied
                        to each raw series before windowing. Must ensure
                        emg shape is (T, 5) at fs_eeg samples/s.
        cache_dir:      Optional directory to cache processed .npz files.
                        Filename: cache_dir/P{p}_{split}_S{s}.npz
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        participants: list[int],
        split: str,
        window_size: int = 500,
        stride: int = 50,
        preprocess_fn: Callable | None = None,
        cache_dir: Union[str, Path, None] = None,
    ) -> None:
        self.data_dir    = Path(data_dir)
        self.participants = participants
        self.split       = split
        self.window_size = window_size
        self.stride      = stride
        self.preprocess_fn = preprocess_fn
        self.cache_dir   = Path(cache_dir) if cache_dir else None

        # Index: list of (eeg_array, kin_array, emg_array, start_sample)
        # Built lazily in _build_index
        self._windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self._build_index()
        # Drop the preprocessing closure after indexing: it is only needed
        # during _build_index and is never called from __getitem__.
        # Keeping it would make the dataset unpicklable under Windows
        # multiprocessing (spawn), breaking DataLoader with num_workers > 0.
        self.preprocess_fn = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        target_series = get_split_series(self.split)

        for p in self.participants:
            try:
                series_list = load_participant(
                    self.data_dir,
                    participant=p,
                    file_type="hs",
                    series=target_series,
                    include_stability=("ST" in target_series),
                )
            except FileNotFoundError:
                continue

            for s in series_list:
                self._index_series(s, p)

    def _index_series(self, series: dict, participant: int) -> None:
        """Slice one series into windows and append to self._windows."""
        cache_key = f"P{participant}_{self.split}_S{series['series']}"

        # Try cache first
        if self.cache_dir is not None:
            cached = self._load_cache(cache_key)
            if cached is not None:
                eeg_all, kin_all, emg_all = cached
                self._slide_windows(eeg_all, kin_all, emg_all)
                return

        # Apply preprocessing if supplied
        if self.preprocess_fn is not None:
            series = self.preprocess_fn(series)

        eeg = series["eeg"]   # (T, 32)
        kin = series["kin"]   # (T, 36)
        emg = series["emg"]   # (T, 5)  must be at same fs as eeg after preprocess

        kt = extract_kt(kin)  # (T, 13)

        # Trim all to same length (in case preprocess left minor length diff)
        T = min(eeg.shape[0], kt.shape[0], emg.shape[0])
        eeg, kt, emg = eeg[:T], kt[:T], emg[:T]

        if self.cache_dir is not None:
            self._save_cache(cache_key, eeg, kt, emg)

        self._slide_windows(eeg, kt, emg)

    def _slide_windows(
        self,
        eeg: np.ndarray,
        kt:  np.ndarray,
        emg: np.ndarray,
    ) -> None:
        T = eeg.shape[0]
        W = self.window_size
        S = self.stride
        for start in range(0, T - W + 1, S):
            self._windows.append((eeg, kt, emg, start))

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{key}.npz"

    def _load_cache(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        data = np.load(p)
        return data["eeg"], data["kin"], data["emg"]

    def _save_cache(
        self,
        key: str,
        eeg: np.ndarray,
        kt:  np.ndarray,
        emg: np.ndarray,
    ) -> None:
        np.savez_compressed(self._cache_path(key), eeg=eeg, kin=kt, emg=emg)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        eeg_all, kt_all, emg_all, start = self._windows[idx]
        end = start + self.window_size

        eeg_w = torch.from_numpy(eeg_all[start:end])   # (W, 32)
        kin_w = torch.from_numpy(kt_all[start:end])    # (W, 13)
        emg_w = torch.from_numpy(emg_all[start:end])   # (W, 5)

        return eeg_w, kin_w, emg_w

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WAYEEGDataset("
            f"split={self.split!r}, "
            f"participants={self.participants}, "
            f"n_windows={len(self)}, "
            f"window_size={self.window_size}, "
            f"stride={self.stride})"
        )
