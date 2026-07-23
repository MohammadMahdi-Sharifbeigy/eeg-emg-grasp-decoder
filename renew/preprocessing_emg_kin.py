from __future__ import annotations
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

"""
Kinematics preprocessing pipeline for WAY-EEG-GAL HS series.

Pipeline (per method1.tex):
  1. Velocity estimation        two methods (select via `velocity_method`):
       'sg'   — Savitzky-Golay first derivative (edge-preserving, default)
       'bw'   — Butterworth zero-phase differentiator (smoother, frequency-domain)
  2. Per-feature z-score        fit on train, apply to val/test
  3. k_t construction           13-dim state vector (same as dataset.py extract_kt)

Input:  raw kin ndarray (T, 36) from load_hs() at 500 Hz
Output: normalised k_t ndarray (T, 13) or (T, 26) with velocity, float32

Note: extract_kt() in dataset.py builds the raw 13-dim vector from the 36-col
signal. This module adds velocity features and per-feature normalisation.

If you only need the raw k_t vector without normalisation or velocity, use
`src/data/dataset.extract_kt` directly.
"""


import numpy as np
from scipy.signal import savgol_filter, butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Column indices in raw 36-col kin signal (0-indexed)
# ---------------------------------------------------------------------------

_KIN_WRIST = [21, 25, 29]      # Px4, Py4, Pz4 — wrist position  (mm)
_KIN_INDEX = [19, 23, 27]      # Px2, Py2, Pz2 — index fingertip (mm)
_KIN_THUMB = [20, 24, 28]      # Px3, Py3, Pz3 — thumb tip       (mm)
_KIN_FX1, _KIN_FY1, _KIN_FZ1 = 12, 14, 16   # force plate 1 (index)
_KIN_FZ2 = 17                                  # force plate 2 (thumb, Z only)
_EPS = 1.0


# ---------------------------------------------------------------------------
# Velocity methods
# ---------------------------------------------------------------------------

def sg_velocity(
    pos: np.ndarray,
    fs: float = 500.0,
    window: int = 11,
    poly: int = 3,
) -> np.ndarray:
    """Velocity via Savitzky-Golay first derivative.

    Edge-preserving — good for sharp onset/offset of grasp kinematics.

    Args:
        pos:    (T, C) or (T,) position signal
        fs:     sampling rate (Hz) — scales output to mm/s
        window: SG window length (must be odd; auto-corrected if even)
        poly:   SG polynomial order

    Returns:
        velocity ndarray same shape as pos
    """
    if window % 2 == 0:
        window += 1
    return savgol_filter(pos, window_length=window, polyorder=poly,
                         deriv=1, delta=1.0 / fs, axis=0).astype(np.float32)


def bw_velocity(
    pos: np.ndarray,
    fs: float = 500.0,
    cutoff: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Velocity via Butterworth zero-phase differentiator.

    Smoothest in the frequency domain — suppresses high-freq noise before
    numerical differentiation. Better when position signal has residual HF
    noise after filtering.

    Steps:
      1. Zero-phase Butterworth LP at `cutoff` Hz  (remove HF noise)
      2. Central finite difference  Δpos / Δt      (numerical derivative)

    Args:
        pos:    (T, C) or (T,) position signal
        fs:     sampling rate (Hz)
        cutoff: LP cutoff before differentiation (Hz)
        order:  Butterworth filter order

    Returns:
        velocity ndarray same shape as pos
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    pos_smooth = sosfiltfilt(sos, pos, axis=0)
    # Central difference; pad endpoints with one-sided difference
    vel = np.gradient(pos_smooth, 1.0 / fs, axis=0)
    return vel.astype(np.float32)


def estimate_velocity(
    pos: np.ndarray,
    fs: float = 500.0,
    method: str = "sg",
    sg_window: int = 11,
    sg_poly: int = 3,
    bw_cutoff: float = 20.0,
    bw_order: int = 4,
) -> np.ndarray:
    """Unified velocity estimator — dispatch to sg or bw method.

    Args:
        pos:       (T, C) or (T,) position signal
        fs:        sampling rate (Hz)
        method:    'sg' (Savitzky-Golay) | 'bw' (Butterworth differentiator)
        sg_window: SG window length (used when method='sg')
        sg_poly:   SG polynomial order (used when method='sg')
        bw_cutoff: LP cutoff Hz before diff (used when method='bw')
        bw_order:  Butterworth order (used when method='bw')

    Returns:
        velocity ndarray same shape as pos
    """
    if method == "sg":
        return sg_velocity(pos, fs=fs, window=sg_window, poly=sg_poly)
    if method == "bw":
        return bw_velocity(pos, fs=fs, cutoff=bw_cutoff, order=bw_order)
    raise ValueError(f"Unknown velocity method '{method}'. Choose 'sg' or 'bw'.")


# ---------------------------------------------------------------------------
# k_t extraction (raw, no normalisation)
# ---------------------------------------------------------------------------

def extract_kt(kin: np.ndarray) -> np.ndarray:
    """Build 13-dim kinematic state vector from raw 36-col kin signal.

    Mirrors dataset.extract_kt but lives here for pipeline use.

    Returns:
        (T, 13): [p_wrist(3), p_index(3), p_thumb(3), d_grip(1), F_L(1), F_G(1), rho_GL(1)]
    """
    p_wrist = kin[:, _KIN_WRIST]                                        # (T, 3)
    p_index = kin[:, _KIN_INDEX]                                        # (T, 3)
    p_thumb = kin[:, _KIN_THUMB]                                        # (T, 3)
    d_grip  = np.linalg.norm(p_index - p_thumb, axis=1, keepdims=True)  # (T, 1)
    F_vec = kin[:, [_KIN_FX1, _KIN_FY1, _KIN_FZ1]]                      # FX1, FY1, FZ1
    F_L = np.linalg.norm(F_vec, axis=1, keepdims=True)                  # sqrt(FX1²+FY1²+FZ1²)
    fz1     = np.abs(kin[:, _KIN_FZ1: _KIN_FZ1 + 1])                    # (T, 1)
    fz2     = np.abs(kin[:, _KIN_FZ2: _KIN_FZ2 + 1])                    # (T, 1)
    F_G     = (fz1 + fz2) / 2                                           # (T, 1)
    rho_GL  = F_G / (F_L + _EPS)                                        # (T, 1)
    return np.concatenate(
        [p_wrist, p_index, p_thumb, d_grip, F_L, F_G, rho_GL], axis=1
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Velocity-augmented k_t (optional, 26-dim)
# ---------------------------------------------------------------------------

def extract_kt_with_velocity(
    kin: np.ndarray,
    fs: float = 500.0,
    method: str = "sg",
    sg_window: int = 11,
    sg_poly: int = 3,
    bw_cutoff: float = 20.0,
    bw_order: int = 4,
) -> np.ndarray:
    """Build 26-dim k_t with appended velocity features.

    Args:
        kin:       (T, 36) raw kinematic signal
        fs:        sampling rate (Hz)
        method:    'sg' or 'bw' — velocity estimation method
        sg_window: SG window (used when method='sg')
        sg_poly:   SG polynomial order (used when method='sg')
        bw_cutoff: LP cutoff Hz (used when method='bw')
        bw_order:  Butterworth order (used when method='bw')

    Returns:
        (T, 26): concat of raw k_t (13) + velocity of all 13 k_t dims
    """
    kt = extract_kt(kin)                                    # (T, 13)
    vel = estimate_velocity(
        kt, fs=fs, method=method,
        sg_window=sg_window, sg_poly=sg_poly,
        bw_cutoff=bw_cutoff, bw_order=bw_order,
    )                                                           # (T, 13)
    return np.concatenate([kt, vel], axis=1).astype(np.float32)  # (T, 26)


# ---------------------------------------------------------------------------
# Normaliser (stateful, fit on train)
# ---------------------------------------------------------------------------

class KinNormalizer:
    """Per-feature z-score normalisation for k_t arrays.

    Fit on concatenated training-set k_t arrays, then apply to any split.

    Usage:
        norm = KinNormalizer()
        norm.fit(train_kt_list)
        val_kt = norm.transform(val_kt)
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, arrays: list[np.ndarray]) -> "KinNormalizer":
        concat = np.concatenate(arrays, axis=0)
        self.mean_ = concat.mean(axis=0).astype(np.float32)
        self.std_  = concat.std(axis=0).astype(np.float32)
        self.std_  = np.maximum(self.std_, 1e-6)
        return self

    def transform(self, kt: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("KinNormalizer must be fit before transform.")
        return ((kt - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        self.fit(arrays)
        return [self.transform(a) for a in arrays]

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    def load_state_dict(self, d: dict) -> None:
        self.mean_ = np.asarray(d["mean"], dtype=np.float32)
        self.std_  = np.asarray(d["std"],  dtype=np.float32)


# ---------------------------------------------------------------------------
# Full pipeline (single series, no normalisation — normaliser is separate)
# ---------------------------------------------------------------------------

def preprocess_kinematics(
    kin: np.ndarray,
    fs: float = 500.0,
    include_velocity: bool = False,
    velocity_method: str = "sg",
    sg_window: int = 11,
    sg_poly: int = 3,
    bw_cutoff: float = 20.0,
    bw_order: int = 4,
) -> np.ndarray:
    """Extract k_t from raw 36-col kin signal, optionally with velocity.

    Args:
        kin:              ndarray (T, 36) from load_hs()
        fs:               kinematic sampling rate (default 500 Hz)
        include_velocity: if True return (T, 26); if False return (T, 13)
        velocity_method:  'sg' (Savitzky-Golay) | 'bw' (Butterworth diff)
        sg_window:        SG window length (used when velocity_method='sg')
        sg_poly:          SG polynomial order (used when velocity_method='sg')
        bw_cutoff:        LP cutoff Hz before diff (used when velocity_method='bw')
        bw_order:         Butterworth order (used when velocity_method='bw')

    Returns:
        ndarray (T, 13) or (T, 26) float32 — un-normalised k_t
        Use KinNormalizer to apply z-score over the training set.
    """
    if include_velocity:
        return extract_kt_with_velocity(
            kin, fs=fs, method=velocity_method,
            sg_window=sg_window, sg_poly=sg_poly,
            bw_cutoff=bw_cutoff, bw_order=bw_order,
        )
    return extract_kt(kin)


def preprocess_kinematics_from_config(
    kin: np.ndarray,
    fs: float,
    cfg: dict,
) -> np.ndarray:
    """Convenience wrapper accepting a config dict (preprocessing.kinematics section)."""
    return preprocess_kinematics(
        kin,
        fs=fs,
        include_velocity=cfg.get("include_velocity", False),
        velocity_method=cfg.get("velocity_method", "sg"),
        sg_window=cfg.get("sg_window", 11),
        sg_poly=cfg .get("sg_poly", 3),
        bw_cutoff=cfg.get("bw_cutoff", 20.0),
        bw_order=cfg.get("bw_order", 4),
    )
