"""Tests for the per-participant, condition-stratified series split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import dataloader as dl  # noqa: E402
from main import dataset as ds  # noqa: E402
from main.dataloader import (  # noqa: E402
    describe_split,
    get_split_series,
    series_conditions,
    stratified_split,
)

RAW_DIR = ROOT / "data" / "way-eeg" / "raw"
PARTICIPANTS = list(range(1, 13))


def _real_participants() -> list[int]:
    return [p for p in PARTICIPANTS if (RAW_DIR / f"P{p}" / f"P{p}_AllLifts.mat").exists()]


needs_data = pytest.mark.skipif(
    not RAW_DIR.exists() or not _real_participants(),
    reason=f"WAY-EEG-GAL AllLifts files not found under {RAW_DIR}",
)


def _labels(cond: dict, series: list) -> list[str]:
    return sorted(cond[s] for s in series)


# ---------------------------------------------------------------------------
# (a) + (b) real participants
# ---------------------------------------------------------------------------

@needs_data
@pytest.mark.parametrize("p", PARTICIPANTS)
def test_split_composition(p):
    if p not in _real_participants():
        pytest.skip(f"P{p}_AllLifts.mat missing")
    cond = series_conditions(p, RAW_DIR)
    sp = stratified_split(p, RAW_DIR)

    assert _labels(cond, sp["test"]) == ["B", "W"]
    assert sp["val"] == [9] and cond[9] == "W"
    assert _labels(cond, sp["train"]) == ["B", "S", "S", "W", "W", "W"]

    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(1, 10))
    assert sp["stability"] == ["ST"]
    assert sp["all"] == sp["train"] + sp["val"] + sp["test"] + ["ST"]
    for k in ("train", "val", "test"):
        assert sp[k] == sorted(sp[k])


@needs_data
@pytest.mark.parametrize("p", PARTICIPANTS)
def test_split_later_earlier_rule(p):
    if p not in _real_participants():
        pytest.skip(f"P{p}_AllLifts.mat missing")
    cond = series_conditions(p, RAW_DIR)
    sp = stratified_split(p, RAW_DIR)
    w16 = sorted(s for s in range(1, 7) if cond[s] == "W")
    b16 = sorted(s for s in range(1, 7) if cond[s] == "B")
    assert len(w16) == 2 and len(b16) == 2
    assert w16[1] in sp["test"] and b16[1] in sp["test"]
    assert w16[0] in sp["train"] and b16[0] in sp["train"]
    for s in (7, 8):
        assert s in sp["train"]


@needs_data
def test_get_split_series_with_participant_and_describe():
    p = _real_participants()[0]
    sp = stratified_split(p, RAW_DIR)
    for k in ("train", "val", "test", "stability", "all"):
        assert get_split_series(k, p, RAW_DIR) == sp[k]
    txt = describe_split(p, RAW_DIR)
    assert f"P{p}" in txt and "test" in txt and "train" in txt


# ---------------------------------------------------------------------------
# (c) legacy behaviour
# ---------------------------------------------------------------------------

def test_legacy_fixed_lists_unchanged():
    assert get_split_series("train") == [1, 2, 5, 7, 8, 9]
    assert get_split_series("val") == [6]
    assert get_split_series("test") == [3, 4]
    assert get_split_series("stability") == ["ST"]
    assert get_split_series("all") == [1, 2, 5, 7, 8, 9, 6, 3, 4, "ST"]
    with pytest.raises(ValueError):
        get_split_series("bogus")


# ---------------------------------------------------------------------------
# (d) validation on synthetic AllLifts
# ---------------------------------------------------------------------------

_COLS = ["Part", "Run", "Lift", "BlockType"]


def _write_fake(tmp_path: Path, p: int, block_by_run: dict[int, int]) -> Path:
    rows = []
    for run, bt in block_by_run.items():
        for lift in range(1, 4):
            rows.append([p, run, lift, bt])
    P = {"AllLifts": np.asarray(rows, dtype=float), "ColNames": np.asarray(_COLS, dtype=object)}
    d = tmp_path / f"P{p}"
    d.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(d / f"P{p}_AllLifts.mat"), {"P": P})
    return tmp_path


def test_synthetic_valid_split(tmp_path):
    # S1..S6 = S B W B W S, 7-9 W
    root = _write_fake(tmp_path, 99, {1: 2, 2: 3, 3: 1, 4: 3, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1})
    sp = stratified_split(99, root)
    assert sp["test"] == [4, 5]
    assert sp["val"] == [9]
    assert sp["train"] == [1, 2, 3, 6, 7, 8]


def test_bad_composition_raises(tmp_path):
    # 4 W, 3 S, 2 B
    root = _write_fake(tmp_path, 98, {1: 2, 2: 3, 3: 2, 4: 3, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1})
    with pytest.raises(ValueError, match="5W/2S/2B"):
        stratified_split(98, root)


def test_missing_alllifts_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        series_conditions(97, tmp_path)


# ---------------------------------------------------------------------------
# (e) WAYEEGDataset indexes exactly the stratified series
# ---------------------------------------------------------------------------

def test_dataset_uses_stratified_series(tmp_path, monkeypatch):
    root = _write_fake(tmp_path, 99, {1: 2, 2: 3, 3: 1, 4: 3, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1})
    for s in list(range(1, 10)) + ["ST"]:
        (root / "P99" / f"HS_P99_S{s}.mat").touch()

    loaded: list = []

    def fake_load_hs(path):
        sid = dl._series_id_from_filename(Path(path).name)
        loaded.append(sid)
        T = 40
        return {"eeg": np.zeros((T, 2), np.float32), "emg": np.zeros((T, 1), np.float32),
                "kin": np.zeros((T, 1), np.float32), "series": sid, "participant": 99}

    monkeypatch.setattr(ds, "load_hs", fake_load_hs)
    expected = stratified_split(99, root)
    for split in ("train", "val", "test", "stability"):
        loaded.clear()
        d = ds.WAYEEGDataset(root, participants=[99], split=split, window_size=10, stride=10)
        assert loaded == expected[split], split
        assert len(d) == 4 * len(expected[split])


def test_dataset_missing_alllifts_raises(tmp_path, monkeypatch):
    (tmp_path / "P96").mkdir()
    monkeypatch.setattr(ds, "load_hs", lambda path: pytest.fail("should not load"))
    with pytest.raises(FileNotFoundError):
        ds.WAYEEGDataset(tmp_path, participants=[96], split="train", window_size=10, stride=10)
