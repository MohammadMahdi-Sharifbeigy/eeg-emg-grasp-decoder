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
from scipy.signal import butter, sosfilt, sosfiltfilt, decimate, savgol_filter, hilbert
import warnings


# ============================================================================
# EMG preprocessing
# ============================================================================

def bandpass_emg(
    emg: np.ndarray,
    fs: float,
    low: float = 30.0,
    high: float = 300.0,
    order: int = 4,
    causal: bool = True,
) -> np.ndarray:
    """Butterworth bandpass filter (axis=0).
    If causal=True, uses one-pass sosfilt to prevent non-causal lookahead.
    """
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    if causal:
        return sosfilt(sos, emg, axis=0).astype(np.float32)
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
    order: int = 2,
    causal: bool = True,
) -> np.ndarray:
    """Causal Butterworth low-pass to extract activation envelope.
    Default order=2 limits causal group delay to ~22.5 ms (at 10 Hz),
    matching physiological peripheral transmission latency while eliminating
    non-causal future leakage.
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    if causal:
        return sosfilt(sos, emg, axis=0).astype(np.float32)
    return sosfiltfilt(sos, emg, axis=0).astype(np.float32)


def downsample_emg(
    emg: np.ndarray,
    factor: int = 8,
    causal: bool = True,
) -> np.ndarray:
    """Decimate EMG by integer factor (4000 → 500 Hz with factor=8).
    causal=True sets zero_phase=False to avoid acausal IIR filtering.
    """
    out = np.stack(
        [decimate(emg[:, c], factor, zero_phase=not causal) for c in range(emg.shape[1])],
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
    lp_order: int = 2,
    downsample_factor: int = 8,
    use_tkeo: bool | None = None,
    envelope_method: str = "rectify",
    causal: bool = True,
) -> np.ndarray:
    """Apply strictly causal EMG envelope extraction pipeline.

    Steps: BP 30–300 Hz (causal) → Rectify / TKEO → LP 10 Hz (2nd-order causal) → Decimate (causal).

    Args:
        emg:              ndarray (T, 5) float32 at `fs` Hz from load_hs()
        fs:               EMG sampling rate (default 4000 Hz)
        bp_low:           Bandpass lower cutoff (default 30 Hz)
        bp_high:          Bandpass upper cutoff (default 300 Hz)
        filter_order:     Bandpass filter order (default 4)
        lp_cutoff:        Lowpass cutoff frequency (default 10 Hz)
        lp_order:         Lowpass filter order (default 2, minimizing group delay to ~22.5 ms)
        downsample_factor: Integer decimation factor (default 8)
        use_tkeo:         Legacy toggle: if explicitly True, overrides envelope_method to 'tkeo'
        envelope_method:  'rectify' (default causal) | 'tkeo' (protected) | 'hilbert' (offline only)
        causal:           Whether to enforce strict forward-only causal filtering (default True)

    Returns:
        ndarray (T // downsample_factor, 5) float32 at 500 Hz
    """
    if use_tkeo is not None:
        envelope_method = "tkeo" if use_tkeo else "rectify"

    # 1. Causal Bandpass
    emg = bandpass_emg(emg, fs, bp_low, bp_high, order=filter_order, causal=causal)

    # 2. Envelope extraction
    if envelope_method == "rectify":
        emg = rectify(emg)
    elif envelope_method == "tkeo":
        e_tkeo = tkeo(emg)
        # Protect against extreme outlier spikes before sqrt to prevent compression of genuine bursts
        p99 = np.percentile(np.abs(e_tkeo), 99.5, axis=0, keepdims=True)
        e_clipped = np.clip(np.abs(e_tkeo), 0.0, p99 * 5.0)
        emg = np.sqrt(e_clipped)
    elif envelope_method == "hilbert":
        if causal:
            warnings.warn(
                "scipy.signal.hilbert is non-causal (FFT across entire series). "
                "Use only for offline reference/benchmarking.",
                UserWarning,
                stacklevel=2,
            )
        emg = np.abs(hilbert(emg, axis=0)).astype(np.float32)
    else:
        raise ValueError(f"Unknown envelope_method: {envelope_method}. Choose 'rectify', 'tkeo', or 'hilbert'.")

    # 3. Causal Lowpass envelope (2nd-order limits group delay to ~22.5 ms)
    emg = lowpass_envelope(emg, fs, cutoff=lp_cutoff, order=lp_order, causal=causal)

    # 4. Causal Decimation
    emg = downsample_emg(emg, factor=downsample_factor, causal=causal)
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
        lp_order=cfg.get("lp_order", 2),
        downsample_factor=cfg.get("downsample_factor", 8),
        use_tkeo=cfg.get("use_tkeo", None),
        envelope_method=cfg.get("envelope_method", "rectify"),
        causal=cfg.get("causal", True),
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


def sg_acceleration(
    pos: np.ndarray,
    fs: float = 500.0,
    window: int = 11,
    poly: int = 3,
) -> np.ndarray:
    """Acceleration via Savitzky-Golay second derivative."""
    if window % 2 == 0:
        window += 1
    if poly < 2:
        poly = 2
    return savgol_filter(pos, window_length=window, polyorder=poly,
                         deriv=2, delta=1.0 / fs, axis=0).astype(np.float32)


def bw_acceleration(
    pos: np.ndarray,
    fs: float = 500.0,
    cutoff: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Acceleration via Butterworth zero-phase second differentiator."""
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    pos_smooth = sosfiltfilt(sos, pos, axis=0)
    vel = np.gradient(pos_smooth, 1.0 / fs, axis=0)
    acc = np.gradient(vel, 1.0 / fs, axis=0)
    return acc.astype(np.float32)


def estimate_acceleration(
    pos: np.ndarray,
    fs: float = 500.0,
    method: str = "sg",
    sg_window: int = 11,
    sg_poly: int = 3,
    bw_cutoff: float = 20.0,
    bw_order: int = 4,
) -> np.ndarray:
    """Unified acceleration estimator — dispatch to sg or bw method."""
    if method == "sg":
        return sg_acceleration(pos, fs=fs, window=sg_window, poly=sg_poly)
    if method == "bw":
        return bw_acceleration(pos, fs=fs, cutoff=bw_cutoff, order=bw_order)
    raise ValueError(f"Unknown acceleration method '{method}'. Choose 'sg' or 'bw'.")


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


def extract_kt_with_derivatives(
    kin: np.ndarray,
    fs: float = 500.0,
    include_velocity: bool = True,
    include_acceleration: bool = False,
    method: str = "sg",
    sg_window: int = 11,
    sg_poly: int = 3,
    bw_cutoff: float = 20.0,
    bw_order: int = 4,
) -> np.ndarray:
    """Build multi-derivative kinematic state matrix (up to 39-dim with velocity & acceleration)."""
    kt = extract_kt_raw(kin)                                    # (T, 13)
    features = [kt]
    if include_velocity or include_acceleration:
        vel = estimate_velocity(
            kt, fs=fs, method=method,
            sg_window=sg_window, sg_poly=sg_poly,
            bw_cutoff=bw_cutoff, bw_order=bw_order,
        )
        if include_velocity:
            features.append(vel)
        if include_acceleration:
            acc = estimate_acceleration(
                kt, fs=fs, method=method,
                sg_window=sg_window, sg_poly=sg_poly,
                bw_cutoff=bw_cutoff, bw_order=bw_order,
            )
            features.append(acc)
    return np.concatenate(features, axis=1).astype(np.float32)


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
    include_acceleration: bool = False,
    velocity_method: str = "sg",
    sg_window: int = 11,
    sg_poly: int = 3,
    bw_cutoff: float = 20.0,
    bw_order: int = 4,
    drop_indices: list[int] | None = None,
) -> np.ndarray:
    """Extract k_t from raw 36-col kin signal, optionally with velocity, acceleration, and index pruning."""
    if include_velocity or include_acceleration:
        kt = extract_kt_with_derivatives(
            kin, fs=fs,
            include_velocity=include_velocity,
            include_acceleration=include_acceleration,
            method=velocity_method,
            sg_window=sg_window, sg_poly=sg_poly,
            bw_cutoff=bw_cutoff, bw_order=bw_order,
        )
    else:
        kt = extract_kt_raw(kin)

    if drop_indices is not None and len(drop_indices) > 0:
        kt = np.delete(kt, drop_indices, axis=-1)
    return kt


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
        include_acceleration=cfg.get("include_acceleration", False),
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


def compute_nnmf_edge_prior(emg_arrays: list[np.ndarray], vaf_threshold: float = 0.90) -> np.ndarray:
    """Compute a 5x5 adjacency matrix from NNMF synergy loadings of training EMG.
    
    Args:
        emg_arrays: List of (T_i, 5) float32 EMG envelope arrays (training split).
        vaf_threshold: Threshold for Variance Accounted For to select rank r.
        
    Returns:
        (5, 5) float32 ndarray representing cosine similarity of synergy loadings.
    """
    import numpy as np
    from sklearn.decomposition import NMF
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not emg_arrays:
        raise ValueError("emg_arrays must be a non-empty list of (T, 5) arrays.")
        
    concat = np.concatenate(emg_arrays, axis=0)
    # Ensure non-negativity
    V = np.maximum(concat, 0.0)
    
    # Calculate total variance for VAF
    V_norm_sq = np.sum(V ** 2)
    
    best_r = 2
    best_H = None
    
    for r in [2, 3, 4, 5]:
        nmf = NMF(n_components=r, init='nndsvda', random_state=42, max_iter=500)
        W = nmf.fit_transform(V)
        H = nmf.components_  # (r, 5)
        
        V_approx = W @ H
        error_sq = np.sum((V - V_approx) ** 2)
        vaf = 1.0 - (error_sq / V_norm_sq)
        
        best_H = H
        best_r = r
        if vaf >= vaf_threshold:
            break
            
    # Compute cosine similarity between columns of H (which represent muscles)
    # H is (r, 5). We want similarity between muscles, so we treat each muscle as an r-dim vector.
    # Therefore, we compute cosine similarity of H.T
    sim = cosine_similarity(best_H.T).astype(np.float32)  # (5, 5)
    
    return sim
