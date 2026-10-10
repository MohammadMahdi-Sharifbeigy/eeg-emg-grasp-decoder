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


def preprocess_emg_from_config(
    emg: np.ndarray | dict,
    fs: float | dict | None = None,
    cfg: dict | None = None,
) -> np.ndarray:
    """Convenience wrapper accepting a config dict (preprocessing.emg section).

    Supports:
        preprocess_emg_from_config(emg_array, fs, cfg)
        preprocess_emg_from_config(series_dict, cfg)
        preprocess_emg_from_config(emg_array, cfg)
    """
    if isinstance(emg, dict):
        series = emg
        emg_arr = series["emg"]
        if cfg is None and isinstance(fs, dict):
            cfg = fs
            fs = float(series.get("fs_emg", 4000.0))
        elif fs is None:
            fs = float(series.get("fs_emg", 4000.0))
    else:
        emg_arr = emg
        if cfg is None and isinstance(fs, dict):
            cfg = fs
            fs = float(cfg.get("fs", cfg.get("fs_emg", 4000.0)))
        elif fs is None:
            fs = 4000.0

    if cfg is None:
        cfg = {}

    # Unwrap nested configs if user provided full CONFIG
    if "preprocessing" in cfg and "emg" in cfg["preprocessing"]:
        cfg = cfg["preprocessing"]["emg"]
    elif "emg" in cfg and isinstance(cfg["emg"], dict):
        cfg = cfg["emg"]

    return preprocess_emg(
        emg_arr,
        fs=float(fs),
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

# Column indices in raw 36-col kin signal (0-indexed). Positions are in cm
# (checked on P1: index-object distance ~10.5 cm while grasping, ~28 cm parked).
_KIN_WRIST = [21, 25, 29]      # Px4, Py4, Pz4 — wrist position  (cm)
_KIN_INDEX = [19, 23, 27]      # Px2, Py2, Pz2 — index fingertip (cm)
_KIN_THUMB = [20, 24, 28]      # Px3, Py3, Pz3 — thumb tip       (cm)
_KIN_OBJECT = [18, 22, 26]     # Px1, Py1, Pz1 — object (sensor 1) (cm)
_KIN_FX1, _KIN_FY1, _KIN_FZ1 = 12, 14, 16   # force plate 1 (index)
_KIN_FZ2 = 17                                  # force plate 2 (thumb, Z only)
_EPS = 1.0
_FORCE_COLS_KT = [10, 11]      # F_L, F_G in the 13-col extract_kt_raw() state

# Names of the 13 columns returned by extract_kt_raw(), in order.
# Columns 0-9 describe the hand pose; columns 10-12 are object force channels.
KT_RAW_NAMES = [
    "p_wrist_x", "p_wrist_y", "p_wrist_z",
    "p_index_x", "p_index_y", "p_index_z",
    "p_thumb_x", "p_thumb_y", "p_thumb_z",
    "d_grip", "F_L", "F_G", "rho_GL",
]

# drop_indices that keep the hand pose only, for the [position(13), velocity(13)]
# layout (include_velocity=True, include_acceleration=False → 26 columns):
#   10, 11, 12 = F_L, F_G, rho_GL          (position block)
#   23, 24, 25 = v_F_L, v_F_G, v_rho_GL    (velocity block)
# The 20 columns left are [pose(10), pose velocity(10)].
# Why: the object force channels also move when someone else loads the object
# while the hand is parked, which made the model predict EMG bursts at rest.
POSE_DROP_INDICES = [10, 11, 12, 23, 24, 25]

# drop_indices for the same 26-column layout when the forces are gated by the
# hand-near-object gate (preprocessing.kinematics.gated_force): keep F_L, F_G
# (gated), drop rho_GL and the velocities of F_L, F_G, rho_GL. The 22 columns
# left are [pose(10), F_L_gated, F_G_gated, pose velocity(10)].
POSE_FORCE_DROP_INDICES = [12, 23, 24, 25]

# Step 1 (magnitude): same 26-column cache, but keep the velocities of the GATED
# F_L, F_G (columns 23, 24 = SG derivative taken after gating, so they are 0 while
# the hand is far) as force-rate features; drop only rho_GL and its velocity.
# The 24 columns left are
#   [pose(10), F_L_gated, F_G_gated, pose velocity(10), v_F_L_gated, v_F_G_gated].
POSE_FORCE_RATE_DROP_INDICES = [12, 25]


def kin_feature_names(
    include_velocity: bool = False,
    include_acceleration: bool = False,
    drop_indices: list[int] | None = None,
    gated_force: bool | dict | None = None,
) -> list[str]:
    """Names of the kinematic features, in the column order produced by
    preprocess_kinematics() with the same arguments.

    Layout before dropping: [position(13), velocity(13), acceleration(13)], where
    the velocity / acceleration blocks exist only if requested. Velocity names are
    prefixed with "v_", acceleration names with "a_". `drop_indices` index into
    that layout, exactly like CONFIG["preprocessing"]["kinematics"]["drop_indices"].
    `gated_force` (the config value, or True) renames F_L / F_G to F_L_gated /
    F_G_gated.

    Example:
        kin_feature_names(True, False, POSE_DROP_INDICES)
        → ['p_wrist_x', ..., 'd_grip', 'v_p_wrist_x', ..., 'v_d_grip']   (20 names)
        kin_feature_names(True, False, POSE_FORCE_DROP_INDICES, gated_force=True)
        → [... 'd_grip', 'F_L_gated', 'F_G_gated', 'v_p_wrist_x', ...]   (22 names)
        kin_feature_names(True, False, POSE_FORCE_RATE_DROP_INDICES, gated_force=True)
        → [... 'F_L_gated', 'F_G_gated', 'v_p_wrist_x', ..., 'v_d_grip',
           'v_F_L_gated', 'v_F_G_gated']                                  (24 names)
    """
    base = list(KT_RAW_NAMES)
    if gated_force:
        base = [n + "_gated" if n in ("F_L", "F_G") else n for n in base]
    names = list(base)
    if include_velocity:
        names += ["v_" + n for n in base]
    if include_acceleration:
        names += ["a_" + n for n in base]
    dropped = set(drop_indices or [])
    return [n for i, n in enumerate(names) if i not in dropped]


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


def index_object_distance(kin: np.ndarray) -> np.ndarray:
    """Distance (cm) between the index fingertip (sensor 2) and the object (sensor 1).

    Returns (T,) float64; NaN where either sensor is missing.
    """
    kin = np.asarray(kin, dtype=np.float64)
    return np.linalg.norm(kin[:, _KIN_INDEX] - kin[:, _KIN_OBJECT], axis=1)


def hand_near_object_gate(
    kin_raw: np.ndarray,
    fs: float = 500.0,
    near_cm: float = 12.0,
    far_cm: float = 20.0,
    smooth_s: float | None = 0.1,
) -> np.ndarray:
    """Soft "hand is at the object" gate from the raw 36-col kin signal.

    gate = 1 while the index-object distance is <= near_cm, falls linearly to 0
    at far_cm, and is 0 beyond. A short zero-phase smoothing of `smooth_s` seconds
    then removes sharp steps: a centred running max (dilation) followed by a
    centred moving average of the same width. The dilation makes sure the
    smoothing never lowers the gate inside the near zone (gate stays exactly 1
    wherever distance <= near_cm, also right after a fast reach, when the hand
    moves ~3-4 cm per 50 ms); the price is that the gate reaches 0 up to
    `smooth_s` later/earlier outside the ramp. Values stay in [0, 1], the gate
    changes by at most 1/(smooth_s*fs) per sample. Samples with missing tracking
    (NaN distance) count as far (gate 0).

    Why: the grip plates are also loaded by someone else between trials while the
    participant's hand is parked ~28 cm away; force only describes the participant's
    grasp when the hand is at the object (~10.5 cm on P1).

    Returns:
        (T, 1) float32 in [0, 1].
    """
    if far_cm <= near_cm:
        raise ValueError(f"far_cm ({far_cm}) must be larger than near_cm ({near_cm})")
    with np.errstate(invalid="ignore"):
        d = index_object_distance(kin_raw)
        gate = np.clip((far_cm - d) / (far_cm - near_cm), 0.0, 1.0)
    gate = np.where(np.isfinite(gate), gate, 0.0)
    n = int(round((smooth_s or 0.0) * fs))
    if n > 1:
        from scipy.ndimage import maximum_filter1d, uniform_filter1d
        if n % 2 == 0:
            n += 1                      # odd width → exactly centred (zero phase)
        gate = maximum_filter1d(gate, size=n, mode="nearest")
        gate = uniform_filter1d(gate, size=n, mode="nearest")
        gate = np.clip(gate, 0.0, 1.0)
        gate[gate > 1.0 - 1e-9] = 1.0   # undo float round-off on flat 1-stretches
        gate[gate < 1e-9] = 0.0
    return gate.astype(np.float32)[:, None]


def _gate_kwargs(gated_force) -> dict | None:
    """Normalise the config value of preprocessing.kinematics.gated_force."""
    if gated_force is None or gated_force is False:
        return None
    if gated_force is True:
        return {}
    if isinstance(gated_force, dict):
        allowed = {"near_cm", "far_cm", "smooth_s"}
        unknown = set(gated_force) - allowed
        if unknown:
            raise ValueError(f"unknown gated_force keys {sorted(unknown)}; allowed {sorted(allowed)}")
        return dict(gated_force)
    raise TypeError(f"gated_force must be None, bool or dict, got {type(gated_force).__name__}")


def extract_kt_gated(kin: np.ndarray, fs: float = 500.0, gated_force=None) -> np.ndarray:
    """extract_kt_raw() with F_L, F_G (columns 10, 11) multiplied by
    hand_near_object_gate() when `gated_force` is set (dict of its kwargs or True)."""
    kt = extract_kt_raw(kin)
    kw = _gate_kwargs(gated_force)
    if kw is not None:
        gate = hand_near_object_gate(kin, fs, **kw)
        kt[:, _FORCE_COLS_KT] = kt[:, _FORCE_COLS_KT] * gate
    return kt


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
    gated_force: bool | dict | None = None,
) -> np.ndarray:
    """Build multi-derivative kinematic state matrix (up to 39-dim with velocity & acceleration).

    With `gated_force`, F_L / F_G are gated (extract_kt_gated) BEFORE the derivatives.
    """
    kt = extract_kt_gated(kin, fs=fs, gated_force=gated_force)  # (T, 13)
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
    gated_force: bool | dict | None = None,
) -> np.ndarray:
    """Extract k_t from raw 36-col kin signal, optionally with velocity, acceleration, and index pruning.

    gated_force: None/False (default, unchanged behaviour) or a dict
        {"near_cm", "far_cm", "smooth_s"} (or True for the defaults 12 / 20 / 0.1):
        F_L and F_G are multiplied by hand_near_object_gate() before the
        derivatives are computed, so object loading while the hand is away
        never reaches the features.
    """
    if include_velocity or include_acceleration:
        kt = extract_kt_with_derivatives(
            kin, fs=fs,
            include_velocity=include_velocity,
            include_acceleration=include_acceleration,
            method=velocity_method,
            sg_window=sg_window, sg_poly=sg_poly,
            bw_cutoff=bw_cutoff, bw_order=bw_order,
            gated_force=gated_force,
        )
    else:
        kt = extract_kt_gated(kin, fs=fs, gated_force=gated_force)

    if drop_indices is not None and len(drop_indices) > 0:
        kt = np.delete(kt, drop_indices, axis=-1)
    return kt


def preprocess_kinematics_from_config(
    kin: np.ndarray | dict,
    fs: float | dict | None = None,
    cfg: dict | None = None,
) -> np.ndarray:
    """Convenience wrapper accepting a config dict (preprocessing.kinematics section).

    Supports:
        preprocess_kinematics_from_config(kin_array, fs, cfg)
        preprocess_kinematics_from_config(series_dict, cfg)
        preprocess_kinematics_from_config(kin_array, cfg)
    """
    if isinstance(kin, dict):
        series = kin
        kin_arr = series["kin"]
        if cfg is None and isinstance(fs, dict):
            cfg = fs
            fs = float(series.get("fs_kin", 500.0))
        elif fs is None:
            fs = float(series.get("fs_kin", 500.0))
    else:
        kin_arr = kin
        if cfg is None and isinstance(fs, dict):
            cfg = fs
            fs = float(cfg.get("fs", cfg.get("fs_kin", 500.0)))
        elif fs is None:
            fs = 500.0

    if cfg is None:
        cfg = {}

    # Unwrap nested configs if user provided full CONFIG
    if "preprocessing" in cfg and "kinematics" in cfg["preprocessing"]:
        cfg = cfg["preprocessing"]["kinematics"]
    elif "kinematics" in cfg and isinstance(cfg["kinematics"], dict):
        cfg = cfg["kinematics"]

    return preprocess_kinematics(
        kin_arr,
        fs=float(fs),
        include_velocity=cfg.get("include_velocity", False),
        include_acceleration=cfg.get("include_acceleration", False),
        velocity_method=cfg.get("velocity_method", "sg"),
        sg_window=cfg.get("sg_window", 11),
        sg_poly=cfg.get("sg_poly", 3),
        bw_cutoff=cfg.get("bw_cutoff", 20.0),
        bw_order=cfg.get("bw_order", 4),
        gated_force=cfg.get("gated_force"),
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
