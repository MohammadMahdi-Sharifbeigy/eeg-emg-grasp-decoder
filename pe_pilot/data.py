"""Lift table and epochs of the prediction-error pilot.

One row per lift from P{p}_AllLifts with the condition labels, and per lift the EEG, EMG envelope and force around the
moment both digits touch the object (tBothDigitTouch), plus window-mean features for the expectation decoder.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, hilbert, sosfiltfilt

from main.dataloader import load_ws, short_emg_label
from pe_pilot.config import CONFIG, KIN_GF, KIN_LF

DECIMATE = 5                                        # touch epochs are stored at 100 Hz (the slow EEG is below 30 Hz)
TABLE_COLS = ["Run", "Lift", "CurW", "CurS", "PrevW", "PrevS", "LEDOn", "tBothDigitTouch", "tLiftOff",
              "dGF_Max", "GF_Max", "Dur_LoadPhase", "BlockType"]


def lift_table(p: int, data_dir: str = CONFIG["data_dir"]) -> pd.DataFrame:
    """Lifts of participant p with the labels of the pilot.

    change: "expected" (same weight as the previous lift), "heavier" or "lighter"; NaN for the first lift of a series
    and when the surface changed too (a surface change would mix a friction error into the weight error).
    next_dGF: dGF_Max of the next lift of the same series (NaN at the end of a series).
    """
    m = sio.loadmat(Path(data_dir) / f"P{p}" / f"P{p}_AllLifts.mat", squeeze_me=True, struct_as_record=False)
    names = [str(c) for c in m["P"].ColNames]
    t = pd.DataFrame(np.asarray(m["P"].AllLifts), columns=names)[TABLE_COLS].copy()
    t[["Run", "Lift"]] = t[["Run", "Lift"]].astype(int)
    return label_lifts(t)


def label_lifts(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    usable = t["PrevW"].notna() & (t["CurS"] == t["PrevS"])
    change = np.where(t["CurW"] > t["PrevW"], "heavier", np.where(t["CurW"] < t["PrevW"], "lighter", "expected"))
    t["change"] = pd.Series(change, index=t.index).where(usable)
    nxt = t.groupby("Run")[["dGF_Max", "CurW"]].shift(-1)
    t["next_dGF"], t["next_W"] = nxt["dGF_Max"], nxt["CurW"]
    return t


def _sos(lo, hi, fs, order=4):
    if lo is None:
        return butter(order, hi, btype="lowpass", fs=fs, output="sos")
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def emg_envelope(emg: np.ndarray, emg_t: np.ndarray, eeg_t: np.ndarray, fs_emg: float = 4000.0) -> np.ndarray:
    """Band-pass, rectify, low-pass, then sample the envelope at the EEG times: (T_eeg, n_muscles)."""
    cfg = CONFIG["emg"]
    env = sosfiltfilt(_sos(None, cfg["lowpass"], fs_emg), np.abs(sosfiltfilt(_sos(*cfg["band"], fs_emg), emg, axis=0)),
                      axis=0)
    return np.stack([np.interp(eeg_t, emg_t, env[:, j]) for j in range(env.shape[1])], axis=1)


def band_logpower(x: np.ndarray, band, fs: float) -> np.ndarray:
    """10 log10 of the Hilbert power in `band`, per sample and channel."""
    return 10 * np.log10(np.abs(hilbert(sosfiltfilt(_sos(*band, fs), x, axis=0), axis=0)) ** 2 + 1e-12)


def _win(t: np.ndarray, t0: float, a: float, b: float) -> slice:
    i = int(np.searchsorted(t, t0 + a))
    return slice(i, i + int(round((b - a) * CONFIG["fs"])))


def _hand_speed(kin: np.ndarray, fs: float) -> np.ndarray:
    """Speed of the four position sensors (Px1..Pz4 = kin[:, 18:30]), (T, 4)."""
    pos = kin[:, 18:30].reshape(len(kin), 3, 4)
    return np.linalg.norm(np.gradient(pos, 1 / fs, axis=0), axis=1)


def trial_signals(tr: dict, row: pd.Series, eeg_names: list) -> dict | None:
    """Epochs and window features of one lift, or None when a window leaves the recorded trial."""
    cfg, fs = CONFIG, CONFIG["fs"]
    t = tr["eeg_t"].astype(float)
    eeg = tr["eeg"].astype(float)
    eeg = eeg - eeg.mean(axis=1, keepdims=True)                     # common average reference
    slow = sosfiltfilt(_sos(*cfg["eeg"]["band"], fs), eeg, axis=0)
    alpha, beta = band_logpower(eeg, cfg["eeg"]["alpha"], fs), band_logpower(eeg, cfg["eeg"]["beta"], fs)
    emg = emg_envelope(tr["emg"].astype(float), tr["emg_t"].astype(float), t)
    force = tr["kin"][:, [KIN_GF, KIN_LF]].astype(float) * [-1.0, 1.0]   # the file stores grip force as negative
    speed = _hand_speed(tr["kin"].astype(float), fs)

    touch, led = float(row["tBothDigitTouch"]), float(row["LEDOn"])
    tw, base = _win(t, touch, *cfg["touch_window"]), _win(t, led, *cfg["led_baseline"])
    pre, cue = _win(t, touch, *cfg["pre_contact"]), _win(t, led, *cfg["pre_cue"])
    if not np.isfinite(touch) or base.start < 1 or tw.stop > len(t) or pre.start <= 0:
        return None
    keep = [i for i, n in enumerate(eeg_names) if n not in cfg["peripheral"]]
    ptp = np.ptp(slow[tw][:, keep], axis=0).max()
    db = lambda x: (x[tw] - x[base].mean(axis=0))[::DECIMATE]       # dB change from rest before the cue
    feats = lambda w: np.concatenate([slow[w].mean(0), alpha[w].mean(0), beta[w].mean(0)])
    return {
        "slow": slow[tw][::DECIMATE], "alpha": db(alpha), "beta": db(beta),
        "emg": emg[tw][::DECIMATE], "force": force[tw][::DECIMATE],
        "eeg_pre": feats(pre), "eeg_cue": feats(cue),
        "emg_pre": np.log(emg[pre].mean(0) + 1e-9), "kin_pre": speed[pre].mean(0),
        "ptp": ptp,
    }


def participant_epochs(p: int, data_dir: str = CONFIG["data_dir"]) -> dict:
    """All lifts of participant p: arrays stacked over lifts plus the matching rows of the lift table."""
    table = lift_table(p, data_dir)
    rows, out, names = [], {}, None
    for run in sorted(table["Run"].unique()):
        trials = load_ws(Path(data_dir) / f"P{p}" / f"WS_P{p}_S{run}.mat")
        sub = table[table["Run"] == run].sort_values("Lift")
        if len(trials) != len(sub):
            raise ValueError(f"P{p} S{run}: {len(trials)} WS windows but {len(sub)} lifts in AllLifts")
        if names is None:
            names, emg_names = _channel_names(data_dir, p, run)
        for tr, (_, row) in zip(trials, sub.iterrows()):
            if tr["weight"] != int(row["CurW"]):
                raise ValueError(f"P{p} S{run} lift {row['Lift']}: WS weight {tr['weight']} != CurW {row['CurW']}")
            sig = trial_signals(tr, row, names)
            if sig is None:
                continue
            rows.append(row)
            for k, v in sig.items():
                out.setdefault(k, []).append(v)
    ep = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
    ep["table"] = pd.DataFrame(rows).reset_index(drop=True)
    ep["table"]["rejected"] = reject_outliers(ep.pop("ptp"))
    ep["ch_names"], ep["emg_names"] = names, emg_names
    ep["times"] = (np.arange(ep["slow"].shape[1]) * DECIMATE / CONFIG["fs"] + CONFIG["touch_window"][0]).astype(np.float32)
    return ep


def reject_outliers(ptp: np.ndarray, k: float = CONFIG["eeg"]["reject_mad"]) -> np.ndarray:
    """True where the peak-to-peak lies above median + k robust SD (1.4826 MAD) of the participant."""
    med = np.median(ptp)
    return ptp > med + k * 1.4826 * np.median(np.abs(ptp - med))


def _channel_names(data_dir, p, run) -> tuple[list, list]:
    """EEG names and short EMG labels in the channel order of the file (the EMG order is AD BR FD ED FDI, not the
    order some old notebooks assumed)."""
    m = sio.loadmat(Path(data_dir) / f"P{p}" / f"WS_P{p}_S{run}.mat", squeeze_me=True, struct_as_record=False)
    return [str(n) for n in m["ws"].names.eeg], [short_emg_label(n) for n in m["ws"].names.emg]


def cache_path(p: int, results_dir: str = CONFIG["results_dir"]) -> Path:
    return Path(results_dir) / "cache" / f"P{p}.npz"


def load_epochs(p: int, data_dir: str = CONFIG["data_dir"], results_dir: str = CONFIG["results_dir"],
                rebuild: bool = False) -> dict:
    """participant_epochs with a per-participant cache (results_dir/cache/P{p}.npz)."""
    path = cache_path(p, results_dir)
    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        ep = {k: z[k] for k in z.files if k not in ("table", "table_cols", "ch_names", "emg_names")}
        ep["table"] = pd.DataFrame(z["table"], columns=list(z["table_cols"])).infer_objects()
        ep["ch_names"], ep["emg_names"] = list(z["ch_names"]), list(z["emg_names"])
        return ep
    ep = participant_epochs(p, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in ep.items() if isinstance(v, np.ndarray)}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, table=ep["table"].to_numpy(dtype=object), table_cols=np.array(ep["table"].columns),
             ch_names=np.array(ep["ch_names"]), emg_names=np.array(ep["emg_names"]), **arrays)
    tmp.replace(path)
    return ep
