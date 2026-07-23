from __future__ import annotations
"""
EEG preprocessing pipeline — all filters via MNE, artifact removal via ICA or ASR.

Pipeline:
  1. Bandpass 0.5–40 Hz (MNE FIR, zero-phase)
  2. Notch 50 Hz (MNE FIR)
  3. Artifact removal: ICA (auto/manual) OR ASR (sliding-window)
  4. CAR (numpy — fast)
  5. Delta 0.5–2 Hz (MNE FIR)
  6. Robust clip (5×MAD per channel, removes tail artifacts)

All MNE steps batched into ONE RawArray round-trip for speed.
"""

import numpy as np
import mne
from mne.preprocessing import ICA


# ─── Consolidated MNE batch: BP + Notch + Delta ───────────────────────────────

def _batch_mne_filters(
    eeg: np.ndarray, fs: float,
    bp_low: float, bp_high: float,
    notch_freq: float,
    delta_low: float, delta_high: float,
    ch_names: list | None = None,
) -> np.ndarray:
    """All MNE filter steps in ONE RawArray round-trip — faster than 5 separate."""
    T, C = eeg.shape
    if ch_names is None or len(ch_names) != C:
        ch_names = [f'EEG{i:03d}' for i in range(C)]

    info = mne.create_info(ch_names=list(ch_names), sfreq=float(fs), ch_types=['eeg'] * C)
    raw = mne.io.RawArray(eeg.T.astype(np.float64), info, verbose=False)

    # 1. Bandpass 0.5–40 Hz
    raw.filter(l_freq=bp_low, h_freq=bp_high, method='fir', fir_window='hamming', verbose=False)

    # 2. Notch 50 Hz
    raw.notch_filter(freqs=notch_freq, method='fir', verbose=False)

    # 3. Delta 0.5–2 Hz (will apply AFTER artifact removal step)
    # Kept here as placeholder — actual delta done after ASR/ICA

    return raw


# ─── ASR (custom — skip first 5s, robust baseline) ────────────────────────────

def _eeg_asr(
    eeg: np.ndarray, fs: float,
    window_ms: float = 250.0,
    std_thresh: float = 3.0,
    baseline_sec: float = 60.0,
) -> np.ndarray:
    """ASR with artifact-skipping baseline (first 5s skipped)."""
    T, C = eeg.shape
    win_samples = max(1, int(round(window_ms * fs / 1000.0)))

    # Skip first 5s (electrode settling), use next baseline_sec for threshold
    skip = int(5.0 * fs)
    start = min(skip, T // 4)
    baseline_samples = min(T - start, int(round(baseline_sec * fs)))
    baseline = eeg[start: start + baseline_samples]

    baseline_med = np.median(baseline, axis=0, keepdims=True)
    baseline_std = np.median(np.abs(baseline - baseline_med), axis=0) / 0.6745
    baseline_std = np.maximum(baseline_std, 1e-6)
    threshold = std_thresh * baseline_std

    out = eeg.copy()
    for s in range(0, T - win_samples + 1, win_samples):
        e = s + win_samples
        window = eeg[s:e]
        win_std = window.std(axis=0)
        bad = win_std > threshold
        if bad.all():
            out[s:e] = 0.0
        elif bad.any():
            clean_mean = window[:, ~bad].mean(axis=1, keepdims=True)
            out[s:e, bad] = np.broadcast_to(clean_mean, (win_samples, bad.sum()))
    return out


# ─── ICA artifact removal (MNE, auto or manual) ───────────────────────────────

def _eeg_ica(
    raw: mne.io.RawArray,
    method: str = 'auto',
    n_components: int | None = None,
    random_state: int = 42,
) -> mne.io.RawArray:
    """ICA artifact removal via MNE.

    Args:
        raw: MNE RawArray (already bandpassed 0.5–40 Hz recommended)
        method: 'auto' (ICA-AROMA-like EOG/ECG removal) or 'manual' (user selects)
        n_components: ICA components (None = min(n_channels, n_samples//5))
        random_state: reproducibility seed

    Returns:
        raw with ICA components removed (in-place modification)
    """
    ica = ICA(
        n_components=n_components,
        method='fastica',
        random_state=random_state,
        max_iter=200,
        verbose=False
    )
    ica.fit(raw)

    if method == 'auto':
        try:
            # Auto-detect EOG/ECG artifact components
            eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=2.0, verbose=False)
            ica.exclude = list(set(eog_indices))
        except RuntimeError:
            # Fallback if no EOG channel is found
            if 'Fp1' in raw.ch_names:
                eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='Fp1', threshold=2.0, verbose=False)
                ica.exclude = list(set(eog_indices))
            else:
                var_per_comp = ica.get_sources(raw).get_data().var(axis=1)
                ica.exclude = np.argsort(var_per_comp)[-2:].tolist()
        ica.apply(raw, verbose=False)
    elif method == 'manual':
        # Plot ICA components — user manually selects in notebook
        # ica.plot_components()  # interactive, not suitable for batch
        # For batch mode, auto-exclude high-variance components (artifact proxy)
        var_per_comp = ica.get_sources(raw).get_data().var(axis=1)
        # Exclude top 2 highest-variance components (often eye/muscle)
        exclude_idx = np.argsort(var_per_comp)[-2:].tolist()
        ica.exclude = exclude_idx
        ica.apply(raw, verbose=False)
    else:
        raise ValueError(f"method must be 'auto' or 'manual', got {method}")

    return raw


# ─── CAR (numpy — 10x faster than MNE for this) ───────────────────────────────

def _eeg_car(eeg: np.ndarray) -> np.ndarray:
    """Common Average Reference via numpy."""
    return (eeg - eeg.mean(axis=1, keepdims=True)).astype(np.float32)


# ─── Robust clip (5×MAD per channel) ──────────────────────────────────────────

def _robust_clip(eeg: np.ndarray) -> np.ndarray:
    """Clip to 5×MAD per channel (removes artifact tails)."""
    med = np.median(eeg, axis=0, keepdims=True)                     # (1, C)
    mad = np.median(np.abs(eeg - med), axis=0, keepdims=True)       # (1, C)
    clip_thresh = np.maximum(5.0 * mad / 0.6745, 50.0)             # min 50µV floor
    return np.clip(eeg, -clip_thresh, clip_thresh).astype(np.float32)


# ─── Channel selection (unchanged) ────────────────────────────────────────────

def select_channels(
    eeg: np.ndarray,
    channel_names: list,
    target_channels: list | None = None,
) -> tuple:
    if target_channels is None:
        return eeg, channel_names
    names_lower = [str(n).lower().strip() for n in channel_names]
    indices, found = [], []
    for ch in target_channels:
        ch_l = ch.lower().strip()
        if ch_l not in names_lower:
            raise ValueError(f"Channel '{ch}' not found in {channel_names}")
        idx = names_lower.index(ch_l)
        indices.append(idx)
        found.append(channel_names[idx])
    return eeg[:, indices].astype(np.float32), found


# ─── Full pipeline ────────────────────────────────────────────────────────────

def preprocess_eeg(
    eeg: np.ndarray,
    fs: float = 500.0,
    bp_low: float = 0.5,
    bp_high: float = 40.0,
    notch_freq: float = 50.0,
    artifact_method: str = 'asr',      # 'asr' | 'ica_auto' | 'ica_manual'
    asr_window_ms: float = 250.0,
    asr_std_thresh: float = 3.0,
    asr_baseline_sec: float = 60.0,
    ica_n_components: int | None = None,
    delta_low: float = 0.5,
    delta_high: float = 2.0,
    channel_names: list | None = None,
    target_channels: list | None = None,
    use_car: bool = True,
) -> np.ndarray:
    """Full MNE EEG preprocessing with ICA/ASR choice.

    Steps:
      1. Channel selection (optional)
      2. Bandpass 0.5–40 Hz (MNE FIR, batched)
      3. Notch 50 Hz (MNE FIR, batched)
      4. Artifact removal: ICA (auto/manual) or ASR (custom)
      5. CAR (numpy)
      6. Delta 0.5–2 Hz (MNE FIR)
      7. Robust clip (5×MAD)

    Args:
        artifact_method: 'asr' | 'ica_auto' | 'ica_manual'

    Returns:
        (T, C) float32 preprocessed EEG
    """
    # Step 1: channel selection
    if channel_names is not None and target_channels is not None:
        eeg, channel_names = select_channels(eeg, channel_names, target_channels)

    ch = list(channel_names) if channel_names is not None else None

    # Step 2+3: consolidated MNE batch (BP + Notch)
    raw = _batch_mne_filters(eeg, fs, bp_low, bp_high, notch_freq, delta_low, delta_high, ch)
    eeg = raw.get_data().T.astype(np.float32)

    # Step 4: artifact removal (ICA or ASR)
    if artifact_method.startswith('ica'):
        method = 'auto' if 'auto' in artifact_method else 'manual'
        raw = _eeg_ica(raw, method=method, n_components=ica_n_components)
        eeg = raw.get_data().T.astype(np.float32)
    elif artifact_method == 'asr':
        eeg = _eeg_asr(eeg, fs, asr_window_ms, asr_std_thresh, asr_baseline_sec)
    else:
        raise ValueError(f"artifact_method must be 'asr', 'ica_auto', or 'ica_manual'")

    # Step 5: CAR (numpy — fast)
    if use_car:
        eeg = _eeg_car(eeg)

    # Step 6: Delta band (MNE FIR)
    info = mne.create_info(ch_names=ch or [f'EEG{i:03d}' for i in range(eeg.shape[1])],
                           sfreq=float(fs), ch_types=['eeg'] * eeg.shape[1])
    raw_delta = mne.io.RawArray(eeg.T.astype(np.float64), info, verbose=False)
    raw_delta.filter(l_freq=delta_low, h_freq=delta_high, method='fir',
                     fir_window='hamming', verbose=False)
    eeg = raw_delta.get_data().T.astype(np.float32)

    # Step 7: robust clip
    eeg = _robust_clip(eeg)

    return eeg


def preprocess_eeg_from_config(
    eeg: np.ndarray, fs: float, cfg: dict,
    channel_names: list | None = None,
) -> np.ndarray:
    """Config-dict wrapper for preprocess_eeg."""
    return preprocess_eeg(
        eeg, fs=fs,
        bp_low=cfg.get('bp_low', 0.5),
        bp_high=cfg.get('bp_high', 40.0),
        notch_freq=cfg.get('notch_freq', 50.0),
        artifact_method=cfg.get('artifact_method', 'asr'),
        asr_window_ms=cfg.get('asr_window_ms', 250.0),
        asr_std_thresh=cfg.get('asr_std_thresh', 3.0),
        asr_baseline_sec=cfg.get('asr_baseline_sec', 60.0),
        ica_n_components=cfg.get('ica_n_components', None),
        delta_low=cfg.get('delta_low', 0.5),
        delta_high=cfg.get('delta_high', 2.0),
        channel_names=channel_names,
        target_channels=cfg.get('target_channels'),
        use_car=cfg.get('use_car', True),
    )
