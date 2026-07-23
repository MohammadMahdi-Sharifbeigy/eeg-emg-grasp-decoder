"""
PyTorch Dataset for WAY-EEG-GAL (nb04 — no CCA).

Loads continuous HS series, extracts the 13-dim kinematic state vector k_t,
slides a fixed-length window over each series, and returns
(eeg_window, kin_window, emg_window) tensors.

k_t construction (13-dim):
  p_wrist (3)  = cols 9:12
  p_index (3)  = cols 3:6
  p_thumb (3)  = cols 6:9
  d_grip  (1)  = ||p_index - p_thumb||_2
  F_L     (1)  = col 12  (FX1, load force index)
  F_G     (1)  = abs(col 15)
  rho_GL  (1)  = F_G / (F_L + eps)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataloader import load_participant, get_split_series
from .preprocessing_emg_kin import extract_kt_raw


class WAYEEGDataset(Dataset):
    """Sliding-window dataset over WAY-EEG-GAL HS series.

    Each sample is a tuple (eeg, kin, emg) of fixed-length windows:
        eeg : Tensor (window_size, n_eeg)
        kin : Tensor (window_size, 13)   k_t kinematic state
        emg : Tensor (window_size, 5)    EMG envelope target

    Args:
        data_dir:       Root dir with P1/, P2/, ... subdirs.
        participants:   List of participant IDs (1-12).
        split:          'train' | 'val' | 'test' | 'stability' | 'all'.
        window_size:    Number of samples per window (default 500 = 1 s @ 500 Hz).
        stride:         Step between consecutive windows (default 50 = 100 ms).
        preprocess_fn:  Optional callable (series_dict) -> series_dict applied
                        to each raw series before windowing.
        cache_dir:      Optional directory to cache processed .npz files.
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

        self._windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self._build_index()
        # Drop closure after indexing to allow Windows multiprocessing
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
        emg = series["emg"]   # (T, 5)

        # kin may already be preprocessed k_t or raw 36-col
        kt = series["kin"]

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

        eeg_w = torch.from_numpy(eeg_all[start:end])   # (W, n_eeg)
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
