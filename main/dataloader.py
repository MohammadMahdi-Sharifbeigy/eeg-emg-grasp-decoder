"""
WAY-EEG-GAL .mat file loader.

Handles two file types:
  HS_PX_SY.mat  — continuous whole-series data (struct: hs)
  WS_PX_SY.mat  — windowed per-trial data      (struct: ws)

scipy.io.loadmat is called with squeeze_me=True, struct_as_record=False
so nested MATLAB structs are accessible as Python attributes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Split definitions (series numbers per participant)
# ---------------------------------------------------------------------------

TRAIN_SERIES: list = [1, 2, 5, 7, 8, 9]
VAL_SERIES:   list = [6]
TEST_SERIES:  list = [3, 4]
STAB_SERIES:  list = ["ST"]             # ST (stability / perturbation)

_SPLIT_MAP = {
    "train":     TRAIN_SERIES,
    "val":       VAL_SERIES,
    "test":      TEST_SERIES,
    "stability": STAB_SERIES,
    "all":       TRAIN_SERIES + VAL_SERIES + TEST_SERIES + STAB_SERIES,
}


def get_split_series(split: str) -> list:
    """Return series identifiers for the requested split.

    Args:
        split: one of 'train', 'val', 'test', 'stability', 'all'

    Returns:
        List of series identifiers (int or 'ST').
    """
    if split not in _SPLIT_MAP:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(_SPLIT_MAP)}")
    return _SPLIT_MAP[split]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mat_load(path: Union[str, Path]) -> dict:
    """Load a .mat file with squeeze_me and struct_as_record=False."""
    return sio.loadmat(
        str(path),
        squeeze_me=True,
        struct_as_record=False,
        mat_dtype=False,
    )


def _to_f32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _names_to_list(names_field) -> list[str]:
    """Convert MATLAB cell/string array of channel names to Python list."""
    if isinstance(names_field, np.ndarray):
        flat = names_field.flatten()
        return [str(n).strip() for n in flat]
    if isinstance(names_field, str):
        return [names_field.strip()]
    return list(names_field)


def _series_id_from_filename(fname: str) -> Union[int, str]:
    """Extract series ID: 'HS_P1_S3.mat' -> 3, 'HS_P1_ST.mat' -> 'ST'."""
    name = Path(fname).stem.upper()
    match = re.search(r"_S(\d+|ST)$", name)
    if match is None:
        raise ValueError(f"Cannot parse series ID from filename: {fname}")
    raw = match.group(1)
    return int(raw) if raw.isdigit() else raw


# ---------------------------------------------------------------------------
# HS loader  (continuous whole-series)
# ---------------------------------------------------------------------------

def load_hs(path: Union[str, Path]) -> dict:
    """Load one HS_PX_SY.mat file.

    Returns a dict with:
        eeg        : ndarray (T_eeg, 32)   float32, uV
        emg        : ndarray (T_emg, 5)    float32, raw units
        kin        : ndarray (T_kin, 36)   float32, mm / N / N*mm
        fs_eeg     : int    sampling rate EEG  (500 Hz)
        fs_emg     : int    sampling rate EMG  (4000 Hz)
        fs_kin     : int    sampling rate kin  (500 Hz)
        eeg_names  : list[str]  32 channel labels
        emg_names  : list[str]  5 muscle labels
        kin_names  : list[str]  36 kinematic channel labels
        participant: int
        series     : int | str  (numeric or 'ST')
        path       : str
    """
    path = Path(path)
    mat = _mat_load(path)
    hs = mat["hs"]

    eeg_sig = _to_f32(hs.eeg.sig)
    emg_sig = _to_f32(hs.emg.sig)
    kin_sig = _to_f32(hs.kin.sig)

    # Ensure 2-D even if single channel
    if eeg_sig.ndim == 1:
        eeg_sig = eeg_sig[:, None]
    if emg_sig.ndim == 1:
        emg_sig = emg_sig[:, None]
    if kin_sig.ndim == 1:
        kin_sig = kin_sig[:, None]

    fs_eeg = int(np.asarray(hs.eeg.samplingrate).flat[0])
    fs_emg = int(np.asarray(hs.emg.samplingrate).flat[0])
    fs_kin = int(np.asarray(hs.kin.samplingrate).flat[0])

    eeg_names = _names_to_list(hs.eeg.names)
    emg_names = _names_to_list(hs.emg.names)
    kin_names = _names_to_list(hs.kin.names)

    participant = int(np.asarray(hs.participant).flat[0])
    series = _series_id_from_filename(path.name)

    return {
        "eeg":         eeg_sig,
        "emg":         emg_sig,
        "kin":         kin_sig,
        "fs_eeg":      fs_eeg,
        "fs_emg":      fs_emg,
        "fs_kin":      fs_kin,
        "eeg_names":   eeg_names,
        "emg_names":   emg_names,
        "kin_names":   kin_names,
        "participant": participant,
        "series":      series,
        "path":        str(path),
    }


# ---------------------------------------------------------------------------
# WS loader  (windowed per-trial)
# ---------------------------------------------------------------------------

def _scalar(x) -> float:
    return float(np.asarray(x).flat[0])


def load_ws(path: Union[str, Path]) -> list[dict]:
    """Load one WS_PX_SY.mat file.

    Returns a list of trial dicts (one per lift), each with:
        eeg             : ndarray (T_eeg, 32)  float32
        emg             : ndarray (T_emg, 5)   float32
        kin             : ndarray (T_kin, 45)  float32  (36 raw + 9 derived)
        eeg_t           : ndarray (T_eeg,)     time vector (s)
        emg_t           : ndarray (T_emg,)     time vector (s)
        weight          : int    1=165g, 2=330g, 4=660g
        surf            : int    1=sandpaper, 2=suede, 3=silk
        weight_id       : str
        surf_id         : str
        led_on          : float  (s, relative to trial window)
        led_off         : float
        trial_start_time: float  (s, absolute within series)
        trial_idx       : int    0-based index within series
        participant     : int
        series          : int | str
        path            : str
    """
    path = Path(path)
    mat = _mat_load(path)
    ws = mat["ws"]

    wins = ws.win
    if not hasattr(wins, "__len__"):
        wins = [wins]

    participant = int(np.asarray(ws.participantnum).flat[0])
    series = _series_id_from_filename(path.name)

    trials = []
    for i, w in enumerate(wins):
        trial = {
            "eeg":              _to_f32(w.eeg),
            "emg":              _to_f32(w.emg),
            "kin":              _to_f32(w.kin),
            "eeg_t":            _to_f32(w.eeg_t),
            "emg_t":            _to_f32(w.emg_t),
            "weight":           int(_scalar(w.weight)),
            "surf":             int(_scalar(w.surf)),
            "weight_id":        str(w.weight_id).strip(),
            "surf_id":          str(w.surf_id).strip(),
            "led_on":           _scalar(w.LEDon),
            "led_off":          _scalar(w.LEDoff),
            "trial_start_time": _scalar(w.trial_start_time),
            "trial_idx":        i,
            "participant":      participant,
            "series":           series,
            "path":             str(path),
        }
        trials.append(trial)

    return trials


# ---------------------------------------------------------------------------
# Participant-level loader
# ---------------------------------------------------------------------------

def load_participant(
    data_dir: Union[str, Path],
    participant: int,
    file_type: str = "hs",
    series: list | None = None,
    include_stability: bool = True,
) -> list[dict]:
    """Load all requested series for one participant.

    Args:
        data_dir:          Root directory containing P1/, P2/, ... subdirs.
        participant:       Participant number 1-12.
        file_type:         'hs' (continuous) or 'ws' (per-trial windowed).
        series:            List of series IDs to load (ints and/or 'ST').
                           None = load all available.
        include_stability: When series=None, include the ST series.

    Returns:
        List of series dicts (load_hs) or lists of trial dicts (load_ws),
        sorted by series number with ST appended last.
    """
    data_dir = Path(data_dir)
    p_dir = data_dir / f"P{participant}"
    if not p_dir.exists():
        raise FileNotFoundError(f"Participant directory not found: {p_dir}")

    prefix = file_type.upper()
    found_files = sorted(p_dir.glob(f"{prefix}_P{participant}_S*.mat"))

    if not found_files:
        raise FileNotFoundError(
            f"No {prefix} files found for P{participant} in {p_dir}"
        )

    series_map: dict[Union[int, str], Path] = {}
    for f in found_files:
        try:
            sid = _series_id_from_filename(f.name)
        except ValueError:
            continue
        series_map[sid] = f

    if series is not None:
        target_series = series
    else:
        numeric = sorted(k for k in series_map if isinstance(k, int))
        target_series = numeric
        if include_stability and "ST" in series_map:
            target_series = target_series + ["ST"]

    loader_fn = load_hs if file_type == "hs" else load_ws

    results = []
    for sid in target_series:
        if sid not in series_map:
            continue
        results.append(loader_fn(series_map[sid]))

    return results