"""
EMG and kinematics preprocessing pipelines for WAY-EEG-GAL HS series.

EMG pipeline:
  1. Bandpass  30–300 Hz   4th-order Butterworth
  2. Rectify   |s(t)|      full-wave
  3. Low-pass  10 Hz       smooth activation envelope
  4. Decimate  ×8          4000 → 500 Hz

Kinematics pipeline:
  1. Extract 13-dim k_t state vector from raw 36-col signal
  2. Optional velocity estimation (Savitzky-Golay or Butterworth)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate, savgol_filter


# ============================================================================
# EMG preprocessing
# ============================================================================

def bandpass_emg(
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


def tkeo(emg: np.ndarray) -> np.ndarray:
    """Teager-Kaiser Energy Operator (TKEO)."""
    out = np.zeros_like(emg)
    out[1:-1] = emg[1:-1]**2 - emg[:-2] * emg[2:]
    out[0] = out[1]
    out[-1] = out[-2]
    return out.astype(np.float32)


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


def downsample_emg(
    emg: np.ndarray,
    factor: int = 8,
) -> np.ndarray:
    """Decimate EMG by integer factor (4000 → 500 Hz with factor=8)."""
    out = np.stack(
        [decimate(emg[:, c], factor, zero_phase=True) for c in range(emg.shape[1])],
        axis=1,
    )
    return out.astype(np.float32)


def preprocess_emg(
    emg: np.ndarray,
    fs: float = 4000.0,
    bp_low: float = 30.0,
    bp_high: float = 300.0,
    filter_order: int = 4,
    lp_cutoff: float = 10.0,
    downsample_factor: int = 8,
    use_tkeo: bool = True,
) -> np.ndarray:
    """Apply EMG envelope extraction pipeline to one continuous HS series.

    Steps: BP 30–300 Hz → TKEO (or |s(t)|) → LP 10 Hz → decimate ×8

    Z-score normalisation is NOT applied here because it requires statistics
    computed across the whole training set.

    Args:
        emg:              ndarray (T, 5) float32 at `fs` Hz from load_hs()
        fs:               EMG sampling rate (default 4000 Hz)
        use_tkeo:         whether to use TKEO instead of rectification

    Returns:
        ndarray (T // downsample_factor, 5) float32 at 500 Hz
    """
    emg = bandpass_emg(emg, fs, bp_low, bp_high, filter_order)
    if use_tkeo:
        emg = tkeo(emg)
        # Rectify AND take square root to map energy back to amplitude scale
        emg = np.sqrt(np.abs(emg))
    else:
        emg = rectify(emg)
    emg = lowpass_envelope(emg, fs, lp_cutoff, filter_order)
    emg = downsample_emg(emg, downsample_factor)
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
        use_tkeo=cfg.get("use_tkeo", True),
    )


class EMGNormalizer:
    """Per-channel z-score normalisation.

    Fit on the concatenated training-set EMG envelopes, then transform
    any split to zero-mean unit-variance.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None   # (5,)
        self.std_:  np.ndarray | None = None   # (5,)

    def fit(self, arrays: list[np.ndarray]) -> "EMGNormalizer":
        """Compute mean and std from a list of (T, C) arrays."""
        concat = np.concatenate(arrays, axis=0)
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


# ============================================================================
# Kinematics preprocessing
# ============================================================================

# Column indices in raw 36-col kin signal (0-indexed)
_KIN_WRIST = [21, 25, 29]      # Px4, Py4, Pz4 — wrist position  (mm)
_KIN_INDEX = [19, 23, 27]      # Px2, Py2, Pz2 — index fingertip (mm)
_KIN_THUMB = [20, 24, 28]      # Px3, Py3, Pz3 — thumb tip       (mm)
_KIN_FX1, _KIN_FY1, _KIN_FZ1 = 12, 14, 16   # force plate 1 (index)
_KIN_FZ2 = 17                                  # force plate 2 (thumb, Z only)
_EPS = 1.0


def extract_kt_raw(kin: np.ndarray) -> np.ndarray:
    """Build 13-dim kinematic state vector from raw 36-col kin signal.

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


# Alias used in WAYEEGDataset
extract_kt = extract_kt_raw


def sg_velocity(
    pos: np.ndarray,
    fs: float = 500.0,
    window: int = 11,
    poly: int = 3,
) -> np.ndarray:
    """Velocity via Savitzky-Golay first derivative."""
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
    """Velocity via Butterworth zero-phase differentiator."""
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    pos_smooth = sosfiltfilt(sos, pos, axis=0)
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
    """Unified velocity estimator — dispatch to sg or bw method."""
    if method == "sg":
        return sg_velocity(pos, fs=fs, window=sg_window, poly=sg_poly)
    if method == "bw":
        return bw_velocity(pos, fs=fs, cutoff=bw_cutoff, order=bw_order)
    raise ValueError(f"Unknown velocity method '{method}'. Choose 'sg' or 'bw'.")


def extract_kt_with_velocity(
    kin: np.ndarray,
    fs: float = 500.0,
    method: str = "sg",
    sg_window: int = 11,
    sg_poly: int = 3,
    bw_cutoff: float = 20.0,
    bw_order: int = 4,
) -> np.ndarray:
    """Build 26-dim k_t with appended velocity features."""
    kt = extract_kt_raw(kin)                                    # (T, 13)
    vel = estimate_velocity(
        kt, fs=fs, method=method,
        sg_window=sg_window, sg_poly=sg_poly,
        bw_cutoff=bw_cutoff, bw_order=bw_order,
    )                                                           # (T, 13)
    return np.concatenate([kt, vel], axis=1).astype(np.float32)  # (T, 26)


class KinNormalizer:
    """Per-feature z-score normalisation for k_t arrays."""

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
    """Extract k_t from raw 36-col kin signal, optionally with velocity."""
    if include_velocity:
        return extract_kt_with_velocity(
            kin, fs=fs, method=velocity_method,
            sg_window=sg_window, sg_poly=sg_poly,
            bw_cutoff=bw_cutoff, bw_order=bw_order,
        )
    return extract_kt_raw(kin)


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
        sg_poly=cfg.get("sg_poly", 3),
        bw_cutoff=cfg.get("bw_cutoff", 20.0),
        bw_order=cfg.get("bw_order", 4),
    )


# ============================================================================
# Edge-prior helper for graph attention
# ============================================================================

def compute_muscle_edge_prior(emg_arrays: list[np.ndarray]) -> np.ndarray:
    """Compute a 5×5 symmetrized Pearson correlation matrix from training EMG envelopes.

    Use this to initialise the learnable edge_bias in MuscleGATLayer /
    KinematicGuidedMuscleGATEncoder with a data-informed prior: pairs of muscles
    that co-activate strongly will start with higher edge weights.

    The matrix is:
      - Pearson correlation across all timesteps in the concatenated training set
      - Symmetrized: C = (C + C^T) / 2   (should already be symmetric, but enforced)
      - Diagonal clamped to 1.0
      - Off-diagonal clipped to [0, 1]   (negative correlations become 0 — they
        indicate inhibitory pairs; the model can learn negative edge biases itself)

    Args:
        emg_arrays: List of (T_i, 5) float32 EMG envelope arrays (training split).
                    These should be the preprocessed, *unnormalised* or normalised
                    envelopes — correlations are scale-invariant.

    Returns:
        (5, 5) float32 ndarray  in [0, 1], symmetric, diagonal ≈ 1.

    Usage (after fitting EMGNormalizer on training data):
        from main.preprocessing_emg_kin import compute_muscle_edge_prior
        import torch

        prior_np = compute_muscle_edge_prior(train_emgs)   # train_emgs: list of (T,5)
        edge_prior = torch.from_numpy(prior_np)

        model = build_kg_gt_from_config(cfg, input_dim=32, kin_dim=12, edge_prior=edge_prior)
    """
    if not emg_arrays:
        raise ValueError("emg_arrays must be a non-empty list of (T, 5) arrays.")

    # Concatenate all training EMG along the time axis
    concat = np.concatenate(emg_arrays, axis=0)  # (T_total, 5)
    if concat.ndim != 2 or concat.shape[1] != 5:
        raise ValueError(
            f"Expected each array to have shape (T, 5), got concat shape {concat.shape}"
        )

    # Pearson correlation matrix via np.corrcoef (operates on rows, so transpose)
    corr = np.corrcoef(concat.T).astype(np.float32)  # (5, 5)

    # Symmetrize (numerical safety)
    corr = (corr + corr.T) / 2.0

    # Set diagonal to 1 (self-edges = full self-attention)
    np.fill_diagonal(corr, 1.0)

    # Clip off-diagonal to [0, 1]: negative correlations → 0
    # The model can learn negative edge biases from data; the prior is a floor.
    corr = np.clip(corr, 0.0, 1.0)

    return corr
