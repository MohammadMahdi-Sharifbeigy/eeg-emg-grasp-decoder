"""Causal inputs and a forecast horizon for KG-GT v2 (EEG control experiment C, off by default).

With cfg["tokens"]["causal"] = True the INPUTS are built so that the input token k depends on input
samples <= 10 k only:
  tokens       causal 4th-order Butterworth low-pass at 20 Hz, then every 10th sample (instead of the
               zero-phase FIR of v2.tokens.to_tokens)
  EEG power    causal band-pass, squared signal, two causal first-order low-passes (a strictly positive
               kernel, -3 dB at smooth_hz in total), log with the same floor as main.eeg_features
  velocities   one-sided Savitzky-Golay derivative over the last 51 samples (instead of the centred one
               of the cache)
What stays non-causal, by up to 50 ms: the hand-near-object gate of the cached forces (a centred 0.1 s
smoothing applied in the shared preprocessing). The EMG target keeps the zero-phase token reduction: it
is only scored, never an input.

With cfg["tokens"]["horizon"] = h tokens (20 ms each) the model at input token t is trained to predict the
envelope at token t + h: the inputs lose their last h tokens and the target its first h tokens.
"""
import numpy as np
from scipy.signal import butter, lfilter, savgol_coeffs, sosfilt, sosfilt_zi

from .tokens import FS_IN, Q

TOKEN_CUTOFF_HZ = 20.0


def _sosfilt_causal(sos, x):
    """Causal sosfilt along axis 0, started in the steady state of the first sample (no start-up transient)."""
    x = np.asarray(x, np.float64)
    zi = sosfilt_zi(sos)                                               # (sections, 2)
    shape = (sos.shape[0], 2) + (1,) * (x.ndim - 1)
    y, _ = sosfilt(sos, x, axis=0, zi=zi.reshape(shape) * x[:1])
    return y


def to_tokens_causal(x: np.ndarray, q: int = Q, fs: float = FS_IN) -> np.ndarray:
    """(T, ...) -> (ceil(T / q), ...): causal anti-alias low-pass, then every q-th sample (token k = sample q k)."""
    sos = butter(4, TOKEN_CUTOFF_HZ, btype="low", fs=fs, output="sos")
    y = _sosfilt_causal(sos, x)[::q]
    return np.ascontiguousarray(y, dtype=np.float32)


def bandpower_causal(eeg: np.ndarray, fs: float, bands, smooth_hz: float = 5.0, order: int = 4,
                     rel_floor: float = 1e-3, floor: float = 1e-12) -> np.ndarray:
    """Causal log band power of one series, (T, C) -> (T, n_bands * C), band-major like main.eeg_features."""
    x = np.asarray(eeg, np.float64)
    T, C = x.shape
    # two cascaded first-order passes: their product is -3 dB at smooth_hz when each is designed at this cut
    lp_cut = min(float(smooth_hz) / (np.sqrt(2.0) - 1.0) ** 0.5, 0.45 * fs)
    lp = butter(1, lp_cut, btype="low", fs=fs, output="sos")
    out = np.empty((T, len(bands) * C), dtype=np.float32)
    for b, (lo, hi) in enumerate(bands):
        bp = butter(int(order), [float(lo), float(hi)], btype="bandpass", fs=fs, output="sos")
        narrow = sosfilt(bp, x, axis=0)
        power = _sosfilt_causal(lp, _sosfilt_causal(lp, narrow ** 2))
        lower = np.maximum(float(rel_floor) * np.median(power, axis=0, keepdims=True), float(floor))
        out[:, b * C:(b + 1) * C] = np.log(np.maximum(power, lower))
    return out


def causal_velocity(pos: np.ndarray, fs: float = FS_IN, window: int = 51, poly: int = 3) -> np.ndarray:
    """One-sided Savitzky-Golay first derivative: the polynomial fitted to the last `window` samples,
    differentiated at its last sample. The first window - 1 samples use the samples available."""
    pos = np.asarray(pos, np.float64)
    c = savgol_coeffs(window, poly, deriv=1, delta=1.0 / fs, pos=window - 1, use="dot")   # c @ x[t-w+1 .. t]
    padded = np.concatenate([np.repeat(pos[:1], window - 1, axis=0), pos])               # hold the first sample
    y = lfilter(c[::-1], [1.0], padded, axis=0)[window - 1:]
    return y.astype(np.float32)


def causal_kinematics(kin26: np.ndarray, fs: float = FS_IN, window: int = 51, poly: int = 3) -> np.ndarray:
    """The cached [position(13) | velocity(13)] with the velocities recomputed one-sided from the positions."""
    kin26 = np.asarray(kin26, np.float32)
    if kin26.shape[1] != 26:
        raise ValueError(f"expected the 26-column cached kinematics, got {kin26.shape[1]} columns")
    return np.concatenate([kin26[:, :13], causal_velocity(kin26[:, :13], fs, window, poly)], axis=1)


def shift_horizon(tok: dict, h: int, q: int = Q) -> dict:
    """Pair the inputs at token t with the target at token t + h (h >= 0 tokens).

    Inputs (kin, hist, eeg) drop their last h tokens; the target (emg, hold_id, emg_full) drops its first
    h tokens (h * q samples at 500 Hz). n_samples is the length of the shifted 500 Hz target.
    """
    h = int(h)
    if h < 0:
        raise ValueError(f"the horizon must be >= 0 tokens, got {h}")
    if h == 0:
        return dict(tok, horizon=0)
    n_tok = len(tok["kin"])
    if h >= n_tok:
        raise ValueError(f"horizon {h} tokens is longer than the series ({n_tok} tokens)")
    out = dict(tok)
    for k in ("kin", "hist", "eeg"):
        out[k] = tok[k][:n_tok - h]
    out["emg"], out["hold_id"] = tok["emg"][h:], tok["hold_id"][h:]
    out["emg_full"] = np.ascontiguousarray(tok["emg_full"][h * q:])
    out["n_samples"] = int(tok["n_samples"]) - h * q
    out["horizon"] = h
    if len(out["kin"]) != -(-out["n_samples"] // q):
        raise AssertionError("token and sample lengths disagree after the horizon shift")
    return out
