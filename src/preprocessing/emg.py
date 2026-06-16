"""
EMG preprocessing pipeline for WAY-EEG-GAL HS series.

Pipeline (per method1.tex):
  1. Bandpass  30–300 Hz   4th-order Butterworth
  2. Rectify   |s(t)|      full-wave
  3. Low-pass  10 Hz       smooth activation envelope
  4. Decimate  ×8          4000 → 500 Hz (matches EEG/kin sampling rate)
  5. Z-score               per channel (fit statistics on train set only)

Input:  raw EMG ndarray (T_emg, 5) from load_hs() at 4000 Hz
Output: envelope ndarray (T_eeg, 5) at 500 Hz, float32
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def bandpass(
    emg: np.ndarray,
    fs: float,
    low: float = 30.0,
    high: float = 300.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (axis=0)."""
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, emg, axis=0).astype(np.float32)


def rectify(emg: np.ndarray) -> np.ndarray:
    """Full-wave rectification."""
    return np.abs(emg).astype(np.float32)


def lowpass_envelope(
    emg: np.ndarray,
    fs: float,
    cutoff: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass to extract smooth activation envelope."""
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, emg, axis=0).astype(np.float32)


def downsample(
    emg: np.ndarray,
    factor: int = 8,
) -> np.ndarray:
    """Decimate EMG by integer factor (4000 → 500 Hz with factor=8).

    Uses scipy.signal.decimate which applies an anti-aliasing filter before
    down-sampling. Applied per-channel to avoid across-channel filtering.
    """
    out = np.stack(
        [decimate(emg[:, c], factor, zero_phase=True) for c in range(emg.shape[1])],
        axis=1,
    )
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Z-score normaliser (stateful — must be fit on train, applied to all splits)
# ---------------------------------------------------------------------------

class EMGNormalizer:
    """Per-channel z-score normalisation.

    Fit on the concatenated training-set EMG envelopes, then transform
    any split to zero-mean unit-variance.

    Usage:
        norm = EMGNormalizer()
        norm.fit(train_emg_list)     # list of (T, 5) arrays
        val_emg = norm.transform(val_emg)
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None   # (5,)
        self.std_:  np.ndarray | None = None   # (5,)

    def fit(self, arrays: list[np.ndarray]) -> "EMGNormalizer":
        """Compute mean and std from a list of (T, C) arrays."""
        concat = np.concatenate(arrays, axis=0)   # (T_total, C)
        self.mean_ = concat.mean(axis=0).astype(np.float32)
        self.std_  = concat.std(axis=0).astype(np.float32)
        self.std_  = np.maximum(self.std_, 1e-6)
        return self

    def transform(self, emg: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("EMGNormalizer must be fit before transform.")
        return ((emg - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        self.fit(arrays)
        return [self.transform(a) for a in arrays]

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d: dict) -> None:
        self.mean_ = np.asarray(d["mean"], dtype=np.float32)
        self.std_  = np.asarray(d["std"],  dtype=np.float32)


# ---------------------------------------------------------------------------
# Full pipeline (single series, no z-score — normaliser is separate)
# ---------------------------------------------------------------------------

def preprocess_emg(
    emg: np.ndarray,
    fs: float = 4000.0,
    bp_low: float = 30.0,
    bp_high: float = 300.0,
    filter_order: int = 4,
    lp_cutoff: float = 10.0,
    downsample_factor: int = 8,
) -> np.ndarray:
    """Apply EMG envelope extraction pipeline to one continuous HS series.

    Steps (method1.tex order):
      BP 30–300 Hz → |s(t)| → LP 10 Hz → decimate ×8

    Z-score normalisation is NOT applied here because it requires statistics
    computed across the whole training set. Use EMGNormalizer separately.

    Args:
        emg:              ndarray (T, 5) float32 at `fs` Hz from load_hs()
        fs:               EMG sampling rate (default 4000 Hz)
        bp_low/bp_high:   bandpass bounds (Hz)
        filter_order:     Butterworth filter order
        lp_cutoff:        envelope low-pass cutoff (Hz)
        downsample_factor: integer decimation factor (4000/500 = 8)

    Returns:
        ndarray (T // downsample_factor, 5) float32
        at fs // downsample_factor Hz (500 Hz), envelope only (no z-score)
    """
    emg = bandpass(emg, fs, bp_low, bp_high, filter_order)
    emg = rectify(emg)
    emg = lowpass_envelope(emg, fs, lp_cutoff, filter_order)
    emg = downsample(emg, downsample_factor)
    return emg


def preprocess_emg_from_config(emg: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Convenience wrapper accepting a config dict (preprocessing.emg section)."""
    return preprocess_emg(
        emg,
        fs=fs,
        bp_low=cfg.get("bp_low", 30.0),
        bp_high=cfg.get("bp_high", 300.0),
        filter_order=cfg.get("filter_order", 4),
        lp_cutoff=cfg.get("lp_cutoff", 10.0),
        downsample_factor=cfg.get("downsample_factor", 8),
    )
