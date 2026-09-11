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
from .dataloader import load_hs, load_participant, get_split_series
from .preprocessing_emg_kin import extract_kt_raw


# ---------------------------------------------------------------------------
# Participant resolution helper
# ---------------------------------------------------------------------------

ALL_PARTICIPANTS: list[int] = list(range(1, 13))


def resolve_participants(
    participant_spec: "str | int | list[str | int]",
) -> list[int]:
    """Resolve a participant specification to a list of integer IDs.

    Accepts any of:
      ``"all"``           → [1, 2, ..., 12]
      ``"P3"`` / ``3``   → [3]
      ``[1, 2, 3]``       → [1, 2, 3]    (LOSOCV list)
      ``["P1", "P3"]``   → [1, 3]

    Passing a *list* activates LOSOCV mode: the caller is responsible for
    iterating over folds and holding out one participant per fold.

    Returns:
        Sorted list of unique participant integer IDs.
    """
    if participant_spec == "all":
        return list(ALL_PARTICIPANTS)
    if isinstance(participant_spec, (list, tuple)):
        return sorted({int(str(p).replace("P", "").replace("p", "")) for p in participant_spec})
    # Single value: "P2", 2, etc.
    return [int(str(participant_spec).replace("P", "").replace("p", ""))]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WAYEEGDataset(Dataset):
    """Sliding-window Dataset for WAY-EEG-GAL continuous series.

    Yields:
        (eeg_window, kin_window, emg_window)
        eeg_window : Tensor (window_size, n_eeg_channels)
        kin_window : Tensor (window_size, n_kin_features)  (k_t)
        emg_window : Tensor (window_size, n_emg_channels)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        participants: "int | list[int]" = 2,
        split: str = "train",
        window_size: int = 4000,
        stride: int = 250,
        latency_shift_ms: float = 0.0,   # 0.0 by architectural design: static latency shift is disabled
                                          # so that temporal Transformer self-attention discovers asymmetric
                                          # corticomuscular conduction delays (~20-100 ms) directly from data.
        fs: float = 500.0,
        preprocess_fn: "Callable[[dict], dict] | None" = None,
        cache_dir: Union[str, Path, None] = None,
    ) -> None:
        super().__init__()

        self.data_dir    = Path(data_dir) if data_dir else None
        if isinstance(participants, int):
            self.participants = [participants]
        else:
            self.participants = list(participants)

        self.split       = split
        self.window_size = window_size
        self.stride      = stride
        self.latency_shift_ms = latency_shift_ms
        self.fs          = fs
        self.latency_shift_samples = int(round((latency_shift_ms / 1000.0) * fs))
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
            p_dir = self.data_dir / f"P{p}" if self.data_dir else None

            for sid in target_series:
                cache_key = f"P{p}_{self.split}_S{sid}"

                # 1. Try cache first (ZERO memory overhead if cached)
                if self.cache_dir is not None:
                    cached = self._load_cache(cache_key)
                    if cached is not None:
                        eeg_all, kin_all, emg_all = cached
                        self._slide_windows(eeg_all, kin_all, emg_all)
                        continue

                # 2. If not cached, find and load ONLY this single series file
                if p_dir is None or not p_dir.exists():
                    continue

                file_path = p_dir / f"HS_P{p}_S{sid}.mat"
                if not file_path.exists():
                    continue

                try:
                    s = load_hs(file_path)
                except Exception:
                    continue

                self._index_series(s, p)

    def _index_series(self, series: dict, participant: int) -> None:
        """Slice one series into windows and append to self._windows."""
        cache_key = f"P{participant}_{self.split}_S{series['series']}"

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
        S = self.stride if self.stride > 0 else W   # 0 → non-overlapping (stride = window_size)
        shift = self.latency_shift_samples
        for start in range(0, T - W - shift + 1, S):
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
        shift = self.latency_shift_samples

        eeg_w = torch.from_numpy(eeg_all[start:end])                 # (W, n_eeg) at time t
        kin_w = torch.from_numpy(kt_all[start + shift:end + shift])    # (W, K) at time t + shift
        emg_w = torch.from_numpy(emg_all[start + shift:end + shift])   # (W, 5) at time t + shift

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
            f"stride={self.stride}, "
            f"latency_shift_ms={self.latency_shift_ms})"
        )