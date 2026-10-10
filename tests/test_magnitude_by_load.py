"""Per-weight / per-surface magnitude table (main.magnitude_eval).

lift_hold_segments() cuts the hold phase of every lift out of P{p}_AllLifts.mat:
    t0 = StartTime − LEDOn
    start = int((t0 + tLiftOff + 0.3) · fs),  stop = int((t0 + tReplace − 0.3) · fs)
magnitude_by_load() compares prediction and target over those samples per
weight (CurW) or surface (CurS).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.magnitude_eval import (
    LOAD_COLUMNS,
    format_magnitude_by_load,
    lift_hold_segments,
    magnitude_by_load,
)

FS = 500
NAMES = ["FDI", "APB", "ADM", "ECR", "FCR"]


# ── synthetic series ──────────────────────────────────────────────────────────

def _series(levels_by_weight: dict[int, float], n_per_weight: int = 4, seed: int = 0):
    """Rest 0.01 everywhere; hold phases of 1.5 s at a level set by the weight."""
    rng = np.random.default_rng(seed)
    weights = [w for w in levels_by_weight for _ in range(n_per_weight)]
    rng.shuffle(weights)
    hold, gap = int(1.5 * FS), int(2.0 * FS)
    T = gap + len(weights) * (hold + gap)
    target = np.full((T, 5), 0.01)
    segs = []
    for i, w in enumerate(weights):
        a = gap + i * (hold + gap)
        b = a + hold
        lvl = levels_by_weight[w] * (1.0 + 0.05 * rng.standard_normal())
        target[a:b] = lvl * np.array([1.0, 0.8, 0.6, 1.2, 1.5])
        segs.append({"lift": i + 1, "start": a, "stop": b, "CurW": w, "CurS": 1 + i % 3})
    return target, segs


def test_perfect_prediction_gives_ratio_one_per_group():
    target, segs = _series({1: 0.01, 2: 0.02, 4: 0.04})
    df = magnitude_by_load(target.copy(), target, segs, group="CurW", channel_names=NAMES)
    assert list(df.columns) == LOAD_COLUMNS
    per = df[df["value"] != "all"]
    assert sorted(per["value"].unique().tolist()) == [1, 2, 4]
    np.testing.assert_allclose(per["height_ratio"].to_numpy(dtype=float), 1.0)
    np.testing.assert_allclose(per["pred_hold"].to_numpy(dtype=float),
                               per["target_hold"].to_numpy(dtype=float))
    assert (per["n_lifts"] == 4).all()
    allrow = df[(df["value"] == "all") & (df["muscle"] == "FCR")].iloc[0]
    assert allrow["n_lifts"] == 12
    assert allrow["r_lift"] == pytest.approx(1.0)


def test_prediction_that_ignores_weight_is_too_low_for_heavy_and_too_high_for_light():
    target, segs = _series({1: 0.01, 2: 0.02, 4: 0.04})
    pred = target.copy()
    mean_hold = np.mean([target[s["start"]:s["stop"]].mean(0) for s in segs], axis=0)
    for s in segs:
        pred[s["start"]:s["stop"]] = mean_hold          # same level for every weight
    df = magnitude_by_load(pred, target, segs, group="CurW", channel_names=NAMES)
    for m in NAMES:
        h = df[df["muscle"] == m].set_index("value")["height_ratio"]
        assert h[4] < 1.0 < h[1]
        assert h[4] < h[2] < h[1]


def test_target_and_pred_hold_levels_are_lift_means():
    target = np.zeros((1000, 1))
    target[100:200] = 1.0
    target[300:500] = 3.0
    pred = target * 0.5
    segs = [{"lift": 1, "start": 100, "stop": 200, "CurW": 1, "CurS": 1},
            {"lift": 2, "start": 300, "stop": 500, "CurW": 1, "CurS": 1}]
    df = magnitude_by_load(pred, target, segs, group="CurW", channel_names=["m"])
    row = df[df["value"] == 1].iloc[0]
    assert row["target_hold"] == pytest.approx(2.0)         # mean of lift means (1, 3)
    assert row["pred_hold"] == pytest.approx(1.0)
    assert row["height_ratio"] == pytest.approx(0.5)        # Σpred / Σtarget over hold samples
    assert np.isnan(row["r_lift"])                         # < 4 lifts


def test_surface_grouping_and_segments_beyond_the_prediction_are_skipped():
    target, segs = _series({1: 0.01, 2: 0.02, 4: 0.04}, n_per_weight=3)
    cut = segs[-1]["start"] + 10                           # last lift partly beyond the end
    df = magnitude_by_load(target[:cut].copy(), target[:cut], segs, group="CurS",
                           channel_names=NAMES)
    assert sorted(v for v in df["value"].unique() if v != "all") == [1, 2, 3]
    total = df[(df["value"] == "all") & (df["muscle"] == "FDI")]["n_lifts"].iloc[0]
    assert total == len(segs) - 1


def test_bad_group_raises():
    target, segs = _series({1: 0.01})
    with pytest.raises(ValueError):
        magnitude_by_load(target, target, segs, group="weight")


def test_format_table():
    target, segs = _series({1: 0.01, 2: 0.02, 4: 0.04})
    txt = format_magnitude_by_load(magnitude_by_load(target, target, segs, channel_names=NAMES))
    assert "height" in txt and "FCR" in txt and "all" in txt


# ── fake AllLifts struct ──────────────────────────────────────────────────────

COLS = ["Part", "Run", "Lift", "CurW", "CurS", "StartTime", "LEDOn", "BlockType",
        "tLiftOff", "tReplace"]


def _write_alllifts(root: Path, participant: int, rows: list[list[float]]) -> None:
    import scipy.io as sio
    d = root / f"P{participant}"
    d.mkdir(parents=True)
    P = {"ColNames": np.array(COLS, dtype=object), "AllLifts": np.asarray(rows, dtype=float)}
    sio.savemat(str(d / f"P{participant}_AllLifts.mat"), {"P": P})


def test_segments_from_a_fake_alllifts_file(tmp_path):
    rows = [
        # Part Run Lift CurW CurS StartTime LEDOn Block tLiftOff tReplace
        [7, 2, 1, 1, 3, 10.0, 2.0, 1, 3.0, 6.0],      # t0 8: a = (11.3)·500, b = (13.7)·500
        [7, 2, 2, 4, 3, 20.0, 2.0, 1, 3.2, 6.1],      # t0 18
        [7, 2, 3, 2, 3, 30.0, 2.0, 1, 3.0, 3.7],      # hold 0.1 s → b − a < 100 → skipped
        [7, 3, 1, 1, 1, 10.0, 2.0, 2, 3.0, 6.0],      # other series
        [7, 2, 4, 1, 3, 400.0, 2.0, 1, 3.0, 6.0],     # beyond n_samples
    ]
    _write_alllifts(tmp_path, 7, rows)
    segs = lift_hold_segments(7, 2, fs=FS, data_dir=tmp_path, n_samples=100_000)
    assert [s["lift"] for s in segs] == [1, 2]
    assert segs[0] == {"lift": 1, "start": int((8 + 3.0 + 0.3) * FS), "stop": int((8 + 6.0 - 0.3) * FS),
                       "CurW": 1, "CurS": 3}
    assert segs[1]["start"] == int((18 + 3.2 + 0.3) * FS)
    assert segs[1]["stop"] == int((18 + 6.1 - 0.3) * FS)
    assert segs[1]["CurW"] == 4
    # without a length every long-enough lift is kept
    assert [s["lift"] for s in lift_hold_segments(7, 2, fs=FS, data_dir=tmp_path)] == [1, 2, 4]
    assert [s["lift"] for s in lift_hold_segments(7, 3, fs=FS, data_dir=tmp_path)] == [1]


def test_missing_alllifts_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        lift_hold_segments(5, 1, data_dir=tmp_path)


# ── real data (skipped when absent) ───────────────────────────────────────────

_RAW = ROOT / "data" / "way-eeg" / "raw"


@pytest.mark.skipif(not (_RAW / "P1" / "P1_AllLifts.mat").exists(), reason="raw data not present")
def test_real_p1_test_series_have_hold_segments():
    from main.dataloader import get_split_series
    ids = get_split_series("test", 1, _RAW)
    assert ids == [4, 5]
    counts = {}
    for sid in ids:
        segs = lift_hold_segments(1, sid, fs=FS, data_dir=_RAW)
        counts[sid] = len(segs)
        assert all(s["stop"] - s["start"] >= 100 for s in segs)
        assert all(s["start"] > 0 for s in segs)
        assert all(segs[i]["stop"] <= segs[i + 1]["start"] for i in range(len(segs) - 1))
    # 34 + 28 lifts (= 31 per series on average), none dropped by the hold-length rule
    assert counts == {4: 34, 5: 28}
    w = pd.Series([s["CurW"] for s in lift_hold_segments(1, 4, data_dir=_RAW)]).value_counts()
    assert sorted(w.index.tolist()) == [1, 2, 4]
