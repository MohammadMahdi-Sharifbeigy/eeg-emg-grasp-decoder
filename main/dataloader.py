"""
WAY-EEG-GAL .mat file loader.

Handles two file types:
  HS_PX_SY.mat  — continuous whole-series data (struct: hs)
  WS_PX_SY.mat  — windowed per-trial data      (struct: ws)

scipy.io.loadmat is called with squeeze_me=True, struct_as_record=False
so nested MATLAB structs are accessible as Python attributes.
"""

from __future__ import annotations

import functools
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


def get_split_series(
    split: str,
    participant: "int | None" = None,
    data_dir: "str | Path | None" = None,
) -> list:
    """Return series identifiers for the requested split.

    Args:
        split:       one of 'train', 'val', 'test', 'stability', 'all'
        participant: if None, return the legacy fixed lists (TRAIN_SERIES etc.).
                     Otherwise return the per-participant, condition-stratified
                     split (see ``stratified_split``).
        data_dir:    raw data root containing ``P{p}/P{p}_AllLifts.mat``
                     (default: repo ``data/way-eeg/raw``).

    Returns:
        List of series identifiers (int or 'ST').
    """
    if split not in _SPLIT_MAP:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(_SPLIT_MAP)}")
    if participant is None:
        return _SPLIT_MAP[split]
    return list(stratified_split(participant, data_dir)[split])


# ---------------------------------------------------------------------------
# Per-participant condition-stratified split
# ---------------------------------------------------------------------------
# Each participant has 9 series: 5 x W (weight varies), 2 x S (surface
# varies), 2 x B (both vary).  Series 7-9 are always W; series 1-6 hold
# 2 W + 2 S + 2 B in a participant-specific order.  Rule:
#   test  = later W of series 1-6  + later B of series 1-6       -> {W, B}
#   val   = series 9                                              -> {W}
#   train = earlier W of 1-6, series 7, 8, both S, earlier B      -> {3W, 2S, 1B}

DEFAULT_RAW_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "way-eeg" / "raw"

_BLOCKTYPE_TO_COND = {1: "W", 2: "S", 3: "B"}


def _alllifts_path(participant: int, data_dir: "str | Path | None") -> Path:
    root = Path(data_dir) if data_dir is not None else DEFAULT_RAW_DIR
    return root / f"P{int(participant)}" / f"P{int(participant)}_AllLifts.mat"


@functools.lru_cache(maxsize=None)
def _read_alllifts(participant: int, data_dir_str: str) -> tuple[dict, dict]:
    """Return ({series: cond}, {series: n_lifts}) for one participant."""
    path = _alllifts_path(participant, data_dir_str)
    if not path.exists():
        raise FileNotFoundError(
            f"AllLifts file not found for participant {participant}: {path}. "
            "It is required to derive the condition-stratified split."
        )
    P = _mat_load(path)["P"]
    cols = _names_to_list(P.ColNames)
    lifts = np.atleast_2d(np.asarray(P.AllLifts, dtype=float))
    run = lifts[:, cols.index("Run")].astype(int)
    block = lifts[:, cols.index("BlockType")].astype(int)

    cond: dict[int, str] = {}
    counts: dict[int, int] = {}
    for r in np.unique(run):
        bts = np.unique(block[run == r])
        if len(bts) != 1 or int(bts[0]) not in _BLOCKTYPE_TO_COND:
            raise ValueError(
                f"P{participant} series {r}: expected a single BlockType in "
                f"{{1,2,3}}, got {bts.tolist()}"
            )
        cond[int(r)] = _BLOCKTYPE_TO_COND[int(bts[0])]
        counts[int(r)] = int((run == r).sum())
    return cond, counts


def series_conditions(participant: int, data_dir: "str | Path | None" = None) -> dict[int, str]:
    """Map series number -> condition 'W' / 'S' / 'B' (from AllLifts BlockType 1/2/3)."""
    key = str(data_dir if data_dir is not None else DEFAULT_RAW_DIR)
    return dict(_read_alllifts(int(participant), key)[0])


def series_lift_counts(participant: int, data_dir: "str | Path | None" = None) -> dict[int, int]:
    """Map series number -> number of lifts in AllLifts."""
    key = str(data_dir if data_dir is not None else DEFAULT_RAW_DIR)
    return dict(_read_alllifts(int(participant), key)[1])


def stratified_split(participant: int, data_dir: "str | Path | None" = None) -> dict:
    """Per-participant condition-stratified split (see module comment).

    Returns dict with keys 'train', 'val', 'test' (sorted int lists),
    'stability' (['ST']) and 'all' (train + val + test + ST).

    Raises:
        FileNotFoundError: AllLifts file missing.
        ValueError: composition is not exactly 5W/2S/2B over series 1..9
                    with series 7-9 = W.
    """
    cond = series_conditions(participant, data_dir)
    comp = {c: sorted(s for s, v in cond.items() if v == c) for c in "WSB"}
    ok = (
        sorted(cond) == list(range(1, 10))
        and len(comp["W"]) == 5 and len(comp["S"]) == 2 and len(comp["B"]) == 2
        and all(cond[s] == "W" for s in (7, 8, 9))
    )
    if not ok:
        raise ValueError(
            f"P{participant}: expected series 1..9 with composition 5W/2S/2B and "
            f"series 7-9 = W; got {dict(sorted(cond.items()))}"
        )
    w16 = [s for s in comp["W"] if s <= 6]
    b16 = comp["B"]
    test = sorted([w16[1], b16[1]])
    val = [9]
    train = sorted([w16[0], 7, 8, *comp["S"], b16[0]])
    return {
        "train": train,
        "val": val,
        "test": test,
        "stability": ["ST"],
        "all": train + val + test + ["ST"],
    }


def describe_split(participant: int, data_dir: "str | Path | None" = None) -> str:
    """Short text table: split -> series ids with W/S/B labels and lift counts."""
    sp = stratified_split(participant, data_dir)
    cond = series_conditions(participant, data_dir)
    counts = series_lift_counts(participant, data_dir)
    lines = [f"P{participant} stratified split (W=weight, S=surface, B=both varies):"]
    for k in ("train", "val", "test"):
        ids = ", ".join(f"S{s}:{cond[s]}" for s in sp[k])
        n = sum(counts.get(s, 0) for s in sp[k])
        lines.append(f"  {k:<5} [{ids}]  lifts={n}")
    return "\n".join(lines)


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

    try:
        participant = int(np.asarray(hs.participant).flat[0])
    except AttributeError:
        participant = int(re.search(r"_P(\d+)_", path.name.upper()).group(1))
        
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

    try:
        participant = int(np.asarray(ws.participantnum).flat[0])
    except AttributeError:
        participant = int(re.search(r"_P(\d+)_", path.name.upper()).group(1))
        
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