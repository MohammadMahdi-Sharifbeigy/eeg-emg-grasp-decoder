"""Shared helpers of the EEG control experiments: participants, cached series, task events, segments."""
from pathlib import Path

import numpy as np

from v2.config import CONFIG_V2

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = list(range(1, 13))
EVENTS = ("led_on", "hand_start", "first_touch", "both_touch", "lift_off", "replace", "release", "hand_stop")
# Phases of one lift, in task time; a sample belongs to the first phase whose interval holds it.
SEGMENTS = (
    ("pre_move", "hand_start", -1.0, "hand_start", 0.0),     # last second before the hand moves
    ("reach", "hand_start", 0.0, "first_touch", 0.0),
    ("load", "first_touch", 0.0, "lift_off", 0.0),
    ("hold", "lift_off", 0.0, "replace", 0.0),
    ("release", "replace", 0.0, "hand_stop", 0.0),
)


def parse_subjects(spec) -> list[int]:
    """"all", "1-4", "1,3,5" or a list -> list of participant numbers."""
    if spec is None or spec == "all":
        return list(SUBJECTS)
    if isinstance(spec, (list, tuple)):
        return [int(str(s).lstrip("P")) for s in spec]
    out = []
    for part in str(spec).split(","):
        part = part.strip().lstrip("P")
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cache_dir() -> Path:
    return ROOT / CONFIG_V2["data"]["cache_dir"]


def data_dir() -> Path:
    return ROOT / CONFIG_V2["data"]["data_root"]


def lifts(participant: int, series: int, n_samples: int, fs: float = 500.0) -> list[dict]:
    """Task events of every lift of one series as sample indices (None when missing or outside the series)."""
    from main.interp import lift_events
    return lift_events(int(participant), int(series), fs=fs, data_dir=data_dir(), n_samples=int(n_samples))


def segment_labels(lift_list, n: int, fs: float, offset: int = 0) -> np.ndarray:
    """Phase name per sample (object array, "other" outside every lift phase).

    `fs` is the rate of the labelled axis and `offset` the sample of that axis where the series starts
    (a forecast horizon shifts the target axis by the horizon): sample i is series time (i + offset) / fs.
    Event values are given at 500 Hz and rescaled to `fs`.
    """
    lab = np.full(int(n), "other", dtype=object)
    scale = fs / 500.0
    for lift in lift_list:
        for name, e0, d0, e1, d1 in reversed(SEGMENTS):          # earlier phases win where two overlap
            if lift.get(e0) is None or lift.get(e1) is None:
                continue
            a = int(round(lift[e0] * scale + d0 * fs)) - offset
            b = int(round(lift[e1] * scale + d1 * fs)) - offset
            a, b = max(a, 0), min(b, int(n))
            if b > a:
                lab[a:b] = name
    return lab
