"""
EEG preprocessing pipeline for WAY-EEG-GAL HS series.

Pipeline (per method1.tex):
  1. Bandpass  0.1–40 Hz   4th-order zero-phase Butterworth
  2. Notch     50 Hz       power-line removal
  3. ASR                   sliding-window covariance-based artifact rejection
  4. CAR                   common average reference
  5. Delta     0.1–2 Hz    4th-order zero-phase Butterworth

Input:  raw EEG ndarray (T, 32) from load_hs(), float32, µV
Output: delta-band EEG ndarray (T, 32), float32
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def bandpass(
    eeg: np.ndarray,
    fs: float,
    low: float = 0.1,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter along time axis (axis=0)."""
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, eeg, axis=0).astype(np.float32)


def notch(
    eeg: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """IIR notch filter for power-line removal."""
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, eeg, axis=0).astype(np.float32)


def asr(
    eeg: np.ndarray,
    fs: float,
    window_ms: float = 500.0,
    std_thresh: float = 5.0,
    baseline_sec: float = 30.0,
) -> np.ndarray:
    """Artifact Subspace Reconstruction (simplified sliding-window version).

    Algorithm:
      1. Estimate per-channel robust std from the first `baseline_sec` seconds
         (assumed cleaner than the rest of the recording).
      2. Slide a window of `window_ms` ms over the signal.
      3. For each window, compute the per-channel std.
      4. Channels whose std exceeds `std_thresh` × baseline_std in that window
         are replaced by the channel mean of the non-artifact channels within
         the window (linear interpolation in the channel domain).
      5. Samples within windows where *all* channels exceed the threshold are
         zeroed (no clean reference available).

    This is a lightweight approximation of the full Riemannian ASR used in
    EEGLAB. It preserves the signal shape while suppressing high-amplitude
    transients without requiring the full matrix-decomposition loop.

    Args:
        eeg:          ndarray (T, C) float32
        fs:           sampling rate (Hz)
        window_ms:    sliding window length (ms)
        std_thresh:   rejection threshold in multiples of baseline std
        baseline_sec: seconds at start of recording used to estimate baseline

    Returns:
        ndarray (T, C) float32 with artifacts attenuated
    """
    T, C = eeg.shape
    win_samples = max(1, int(round(window_ms * fs / 1000.0)))
    baseline_samples = min(T, int(round(baseline_sec * fs)))

    # Robust baseline std (median absolute deviation scaled to std)
    baseline = eeg[:baseline_samples]
    baseline_med = np.median(baseline, axis=0, keepdims=True)
    baseline_std = np.median(np.abs(baseline - baseline_med), axis=0) / 0.6745
    baseline_std = np.maximum(baseline_std, 1e-6)   # avoid division by zero

    threshold = std_thresh * baseline_std            # (C,)

    out = eeg.copy()

    for start in range(0, T - win_samples + 1, win_samples):
        end = start + win_samples
        win = out[start:end]                         # (W, C)
        win_std = win.std(axis=0)                    # (C,)

        bad_mask = win_std > threshold               # (C,) bool
        good_mask = ~bad_mask

        if bad_mask.any():
            if good_mask.any():
                # Replace bad channels with mean of good channels (per sample)
                good_mean = win[:, good_mask].mean(axis=1, keepdims=True)
                out[start:end, bad_mask] = np.broadcast_to(
                    good_mean, (win_samples, bad_mask.sum())
                )
            else:
                # All channels bad — zero out window
                out[start:end] = 0.0

    return out.astype(np.float32)


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    """Subtract mean across all channels at each time point (CAR)."""
    return (eeg - eeg.mean(axis=1, keepdims=True)).astype(np.float32)


def delta_band(
    eeg: np.ndarray,
    fs: float,
    low: float = 0.1,
    high: float = 2.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass for delta band (0.1–2 Hz)."""
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, eeg, axis=0).astype(np.float32)


def select_channels(
    eeg: np.ndarray,
    channel_names: list[str],
    target_channels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Select a specific subset of EEG channels.

    Args:
        eeg: ndarray (T, C) float32
        channel_names: list of C channel strings
        target_channels: list of channels to keep. Defaults to:
                         FC1, FC2, FC3, FC4, C3, C4, CZ, CP1, CP2, CP3, CP4, CP5, CP6, CPZ

    Returns:
        (filtered_eeg, filtered_channel_names)
    """
    if target_channels is None:
        return eeg, channel_names


    # Use case-insensitive matching in case dataset names are e.g., "Cz" instead of "CZ"
    names_lower = [str(n).lower().strip() for n in channel_names]
    indices = []
    found_names = []

    for ch in target_channels:
        ch_lower = ch.lower().strip()
        if ch_lower in names_lower:
            idx = names_lower.index(ch_lower)
            indices.append(idx)
            found_names.append(channel_names[idx])
        else:
            raise ValueError(f"Target channel '{ch}' not found in provided channel_names: {channel_names}")

    return eeg[:, indices].astype(np.float32), found_names


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_eeg(
    eeg: np.ndarray,
    fs: float = 500.0,
    bp_low: float = 0.1,
    bp_high: float = 40.0,
    filter_order: int = 4,
    notch_freq: float = 50.0,
    notch_quality: float = 30.0,
    asr_window_ms: float = 500.0,
    asr_std_thresh: float = 5.0,
    asr_baseline_sec: float = 30.0,
    delta_low: float = 0.1,
    delta_high: float = 2.0,
    channel_names: list[str] | None = None,
    target_channels: list[str] | None = None,
) -> np.ndarray:
    """Apply full EEG preprocessing pipeline to one continuous HS series.

    Steps (method1.tex order):
      Channel selection (optional) → BP 0.1–40 Hz → notch 50 Hz → ASR → CAR → delta 0.1–2 Hz

    Args:
        eeg: ndarray (T, 32) float32, raw µV from load_hs()
        fs:  EEG sampling rate (default 500 Hz)
        ... (see individual step parameters above)

    Returns:
        ndarray (T, C) float32
        Delta-band EEG ready for CCA / windowing.
    """
    if channel_names is not None:
        eeg, channel_names = select_channels(eeg, channel_names, target_channels)

    eeg = bandpass(eeg, fs, bp_low, bp_high, filter_order)
    eeg = notch(eeg, fs, notch_freq, notch_quality)
    eeg = asr(eeg, fs, asr_window_ms, asr_std_thresh, asr_baseline_sec)
    eeg = common_average_reference(eeg)
    eeg = delta_band(eeg, fs, delta_low, delta_high, filter_order)
    return eeg


def preprocess_eeg_from_config(
    eeg: np.ndarray, fs: float, cfg: dict, channel_names: list[str] | None = None
) -> np.ndarray:
    """Convenience wrapper accepting a config dict (preprocessing.eeg section).

    Example cfg:
        {bp_low: 0.1, bp_high: 40.0, filter_order: 4,
         notch_freq: 50.0, asr_window_ms: 500, asr_std_thresh: 5.0,
         delta_low: 0.1, delta_high: 2.0}
    """
    return preprocess_eeg(
        eeg,
        fs=fs,
        bp_low=cfg.get("bp_low", 0.1),
        bp_high=cfg.get("bp_high", 40.0),
        filter_order=cfg.get("filter_order", 4),
        notch_freq=cfg.get("notch_freq", 50.0),
        notch_quality=cfg.get("notch_quality", 30.0),
        asr_window_ms=cfg.get("asr_window_ms", 500.0),
        asr_std_thresh=cfg.get("asr_std_thresh", 5.0),
        asr_baseline_sec=cfg.get("asr_baseline_sec", 30.0),
        delta_low=cfg.get("delta_low", 0.1),
        delta_high=cfg.get("delta_high", 2.0),
        channel_names=channel_names,
        target_channels=cfg.get("target_channels"),
    )
