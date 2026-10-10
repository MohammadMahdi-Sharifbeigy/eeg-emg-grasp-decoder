"""main/eeg_features.py — log band-power EEG features (Package 2, Phase B).

The raw 0.5–40 Hz EEG waveform carries almost no linear information about the EMG
envelope; the slowly varying POWER in the classical bands does.  ``eeg_bandpower``
turns a whole series (T, C) into (T, n_bands · C) features:

    per band:  zero-phase Butterworth band-pass (sosfiltfilt)
               -> squared Hilbert envelope (instantaneous band power)
               -> zero-phase low-pass at ``smooth_hz`` (1st-order Butterworth run forward
                  and backward: a strictly positive kernel, so the smoothed power can
                  never ring below zero)
               -> log( max(power, floor) ),  floor = rel_floor x median power of that
                  channel and band (so a silent stretch gives no extreme log outlier)

Layout is BAND-MAJOR: column ``b * C + c`` is band ``b`` of channel ``c``
(band0 ch0..C-1, band1 ch0..C-1, …).  The features are computed on the full series,
never per window (filter edges, the median floor), and are not stored in the dataset
cache: ``WAYEEGDataset(eeg_transform=...)`` applies them after the cached EEG is loaded.

Config (``preprocessing.eeg_features``; absent / None = raw EEG):
    {"type": "bandpower", "bands": [[4, 8], [8, 13], [13, 30], [30, 40]], "smooth_hz": 5.0}
    (optional keys: "order", "smooth_order", "floor", "rel_floor")

Normalisation (per-feature z-score with TRAIN-series statistics, clip to ±5) is done in
the notebook's ``prepare_batch`` (``main.training.normalized_prepare_batch_factory``)
with ``fit_feature_stats`` of the training series.
"""

from __future__ import annotations

import functools
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import butter, hilbert, sosfiltfilt

__all__ = [
    "DEFAULT_BANDS", "eeg_bandpower", "eeg_feature_names", "band_labels",
    "eeg_electrode_groups", "eeg_band_groups", "build_eeg_transform_from_config",
    "eeg_n_bands_from_config", "fit_feature_stats", "clip_fraction",
]

DEFAULT_BANDS: tuple[tuple[float, float], ...] = ((4, 8), (8, 13), (13, 30), (30, 40))


def _check_bands(bands, fs: float) -> list[tuple[float, float]]:
    out = []
    for band in bands:
        lo, hi = float(band[0]), float(band[1])
        if not 0.0 < lo < hi:
            raise ValueError(f"band must be (low, high) with 0 < low < high, got {tuple(band)}")
        if hi >= fs / 2.0:
            raise ValueError(f"band {tuple(band)} reaches the Nyquist frequency ({fs / 2.0:g} Hz)")
        out.append((lo, hi))
    if not out:
        raise ValueError("at least one band is needed")
    return out


def eeg_bandpower(
    eeg: np.ndarray,
    fs: float,
    bands: Sequence[Sequence[float]] = DEFAULT_BANDS,
    smooth_hz: float = 5.0,
    floor: float = 1e-12,
    order: int = 4,
    rel_floor: float = 1e-3,
    smooth_order: int = 1,
) -> np.ndarray:
    """Log band power of one full EEG series.

    Args:
        eeg:       (T, C) EEG of ONE series (any float dtype).
        fs:        sampling rate in Hz.
        bands:     (low, high) band edges in Hz.
        smooth_hz: -3 dB frequency of the zero-phase low-pass applied to the
                   instantaneous power (of the forward-backward filter as a whole).
        floor:     absolute lower bound of the power before the log (keeps an all-zero
                   channel finite).
        order:     Butterworth order of the band-pass (applied forward and backward).
        rel_floor: RELATIVE lower bound: the power of every channel and band is clamped
                   at ``rel_floor`` x its median over the series.
        smooth_order: Butterworth order of the smoothing low-pass (forward and backward).
                   Default 1: its zero-phase kernel is a two-sided exponential, strictly
                   positive, so smoothed power stays positive.  A higher order rings: with
                   order 4 the smoothed power of P1's EEG (and of white noise alike) is
                   NEGATIVE in 0.3-2.4 % of the samples per band, which the floor would
                   turn into -5 SD spikes of the log feature.

    Returns:
        (T, n_bands * C) float32, band-major (column ``b * C + c``), natural log.
    """
    x = np.asarray(eeg, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"eeg must be (T, C), got shape {x.shape}")
    fs = float(fs)
    band_list = _check_bands(bands, fs)
    if not 0.0 < float(smooth_hz) < fs / 2.0:
        raise ValueError(f"smooth_hz must be in (0, {fs / 2.0:g}), got {smooth_hz}")
    T, C = x.shape
    n_fft = next_fast_len(T, real=False)
    smooth_order = int(smooth_order)
    if smooth_order < 1:
        raise ValueError(f"smooth_order must be >= 1, got {smooth_order}")
    # forward-backward filtering squares the magnitude response: design the single pass
    # so that the zero-phase filter as a whole has its -3 dB point at smooth_hz
    lp_cut = min(float(smooth_hz) / (np.sqrt(2.0) - 1.0) ** (1.0 / (2 * smooth_order)), 0.45 * fs)
    lp = butter(smooth_order, lp_cut, btype="lowpass", fs=fs, output="sos")
    out = np.empty((T, len(band_list) * C), dtype=np.float32)
    for b, (lo, hi) in enumerate(band_list):
        bp = butter(int(order), [lo, hi], btype="bandpass", fs=fs, output="sos")
        narrow = sosfiltfilt(bp, x, axis=0)
        power = np.abs(hilbert(narrow, N=n_fft, axis=0)[:T]) ** 2
        del narrow
        power = sosfiltfilt(lp, power, axis=0)                 # zero-phase smoothing
        lower = np.maximum(float(rel_floor) * np.median(power, axis=0, keepdims=True), float(floor))
        out[:, b * C:(b + 1) * C] = np.log(np.maximum(power, lower))
    return out


def band_labels(bands: Sequence[Sequence[float]] = DEFAULT_BANDS) -> list[str]:
    """["4-8Hz", "8-13Hz", …]."""
    return [f"{float(lo):g}-{float(hi):g}Hz" for lo, hi in bands]


def eeg_feature_names(eeg_names: Sequence[str], bands: Sequence[Sequence[float]] = DEFAULT_BANDS) -> list[str]:
    """Names of the ``eeg_bandpower`` columns, band-major: "<electrode> <low>-<high>Hz"."""
    return [f"{name} {label}" for label in band_labels(bands) for name in eeg_names]


def eeg_electrode_groups(electrodes, n_bands: int) -> dict:
    """Columns of every electrode in the band-major layout.

    Args:
        electrodes: electrode names, or their number (keys are then 0..C-1).
        n_bands:    number of bands (1 = raw EEG: every electrode is one column).

    Returns:
        {electrode: [c, C + c, 2C + c, …]} in electrode order — the ``electrode_groups``
        of ``main.interp.eeg_occlusion_attribution`` / ``event_locked_windows``.
    """
    keys = list(range(int(electrodes))) if isinstance(electrodes, (int, np.integer)) else list(electrodes)
    C = len(keys)
    n_bands = int(n_bands)
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    return {k: [b * C + c for b in range(n_bands)] for c, k in enumerate(keys)}


def eeg_band_groups(n_electrodes: int, bands: Sequence[Sequence[float]] = DEFAULT_BANDS) -> dict[str, list[int]]:
    """Columns of every band in the band-major layout: {"4-8Hz": [0..C-1], "8-13Hz": [C..2C-1], …}."""
    C = int(n_electrodes)
    return {label: list(range(b * C, (b + 1) * C)) for b, label in enumerate(band_labels(bands))}


def eeg_n_bands_from_config(preprocessing_cfg: Mapping | None) -> int:
    """Columns per electrode the config produces (1 = raw EEG)."""
    spec = (preprocessing_cfg or {}).get("eeg_features")
    if not spec:
        return 1
    return len(spec.get("bands", DEFAULT_BANDS))


def build_eeg_transform_from_config(preprocessing_cfg: Mapping | None, fs: float) -> Callable | None:
    """``preprocessing.eeg_features`` -> callable (T, C) -> (T, F), or None for raw EEG.

    The callable is what ``WAYEEGDataset(eeg_transform=...)`` expects; build it once and
    pass the same one to the train, val and test datasets.
    """
    spec = (preprocessing_cfg or {}).get("eeg_features")
    if not spec:
        return None
    kind = spec.get("type", "bandpower")
    if kind != "bandpower":
        raise ValueError(f"unknown preprocessing.eeg_features type {kind!r} (supported: 'bandpower')")
    kwargs = {"fs": float(fs),
              "bands": tuple(tuple(float(v) for v in b) for b in spec.get("bands", DEFAULT_BANDS)),
              "smooth_hz": float(spec.get("smooth_hz", 5.0))}
    for key in ("floor", "order", "rel_floor", "smooth_order"):
        if key in spec:
            kwargs[key] = spec[key]
    return functools.partial(eeg_bandpower, **kwargs)


def fit_feature_stats(arrays: Iterable[np.ndarray], eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and SD over the concatenation of the given (T_i, F) series.

    Pass the TRAINING series only.  Accumulated in float64 without concatenating;
    the SD is clamped at ``eps``.  Returns float32 (F,) arrays.
    """
    n, s1, s2 = 0, None, None
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"every series must be (T, F), got shape {a.shape}")
        s1 = a.sum(axis=0) if s1 is None else s1 + a.sum(axis=0)
        s2 = (a * a).sum(axis=0) if s2 is None else s2 + (a * a).sum(axis=0)
        n += a.shape[0]
    if n == 0:
        raise ValueError("fit_feature_stats: no samples given")
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    return mean.astype(np.float32), np.maximum(std, eps).astype(np.float32)


def clip_fraction(arrays: Iterable[np.ndarray], mean, std, clip: float = 5.0) -> float:
    """Fraction of all values whose z-score lies beyond ±clip (i.e. gets clipped)."""
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    n_out, n = 0, 0
    for a in arrays:
        z = (np.asarray(a, dtype=np.float64) - mean) / std
        n_out += int((np.abs(z) > float(clip)).sum())
        n += z.size
    return n_out / max(n, 1)
