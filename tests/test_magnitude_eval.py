"""Magnitude metrics per test series / per condition (Package 1, change 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.magnitude_eval import (
    MAGNITUDE_COLUMNS,
    format_magnitude_table,
    magnitude_report,
    magnitude_report_by_series,
    predict_series,
)

FS = 500.0
NAMES = ["m0", "m1", "m2"]


def _envelope(T=5000, C=3, seed=0):
    """Positive bursty envelopes at a resting floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / FS
    out = np.empty((T, C))
    for c in range(C):
        bursts = np.clip(np.sin(2 * np.pi * (0.3 + 0.1 * c) * t + rng.uniform(0, 6)), 0, None) ** 4
        out[:, c] = 0.01 + 0.05 * (c + 1) * bursts + 0.001 * rng.random(T)
    return out


def test_perfect_prediction():
    y = _envelope()
    df = magnitude_report(y.copy(), y, FS, channel_names=NAMES)
    assert list(df["muscle"]) == NAMES + ["MEAN"]
    m = df[df["muscle"] != "MEAN"]
    np.testing.assert_allclose(m["r"], 1.0, atol=1e-9)
    np.testing.assert_allclose(m["ccc"], 1.0, atol=1e-9)
    np.testing.assert_allclose(m["height_ratio"], 1.0, atol=1e-12)
    np.testing.assert_allclose(m["over_area"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["under_area"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["rest_pred"], m["rest_target"], atol=1e-12)
    np.testing.assert_allclose(m["vaf"], 100.0, atol=1e-9)
    np.testing.assert_allclose(m["rmse"], 0.0, atol=1e-12)
    for col in MAGNITUDE_COLUMNS:
        assert col in df.columns


def test_half_scaled_prediction_has_height_one_half():
    y = _envelope(seed=1)
    df = magnitude_report(0.5 * y, y, FS, channel_names=NAMES).set_index("muscle")
    np.testing.assert_allclose(df.loc[NAMES, "height_ratio"], 0.5, atol=1e-12)
    np.testing.assert_allclose(df.loc[NAMES, "over_area"], 0.0, atol=1e-12)
    np.testing.assert_allclose(df.loc[NAMES, "under_area"], 0.5, atol=1e-12)
    np.testing.assert_allclose(df.loc[NAMES, "r"], 1.0, atol=1e-9)


def test_constant_offset_is_over_area_only():
    y = _envelope(seed=2)
    df = magnitude_report(y + 0.02, y, FS, channel_names=NAMES).set_index("muscle")
    assert (df.loc[NAMES, "over_area"] > 0).all()
    np.testing.assert_allclose(df.loc[NAMES, "under_area"], 0.0, atol=1e-12)
    np.testing.assert_allclose(df.loc[NAMES, "over_area"], 0.02 * len(y) / y.sum(0), rtol=1e-9)
    np.testing.assert_allclose(
        df.loc[NAMES, "rest_pred"] - df.loc[NAMES, "rest_target"], 0.02, atol=1e-12)


def test_rest_and_active_masks_follow_the_quantiles():
    y = np.tile(np.arange(10, dtype=float)[:, None], (1, 1))      # 0..9
    p = y.copy()
    p[y[:, 0] >= 7] *= 2.0         # active = above 70th percentile (6.3) -> 7, 8, 9
    p[y[:, 0] <= 1] = 5.0          # rest   = below 20th percentile (1.8) -> 0, 1
    df = magnitude_report(p, y, FS, channel_names=["m"]).set_index("muscle")
    assert df.loc["m", "height_ratio"] == pytest.approx(2.0)
    assert df.loc["m", "rest_pred"] == pytest.approx(5.0)
    assert df.loc["m", "rest_target"] == pytest.approx(0.5)


def test_grouping_by_condition():
    rng = np.random.default_rng(3)
    ys = {s: _envelope(T=3000, seed=s) for s in (2, 4, 5)}
    preds = {2: ys[2] * 1.0, 4: ys[4] * 0.5, 5: ys[5] + 0.01}
    series = {s: (preds[s], ys[s]) for s in ys}
    cond = {2: "W", 4: "W", 5: "B", 9: "W"}
    df = magnitude_report_by_series(series, cond, FS, channel_names=NAMES)
    groups = list(dict.fromkeys(df["group"]))
    assert groups == ["series 2 (W)", "series 4 (W)", "series 5 (B)", "W: series 2+4", "all test"]
    # one series per group row-block equals the single-series report
    single = magnitude_report(preds[4], ys[4], FS, channel_names=NAMES)
    block = df[df["group"] == "series 4 (W)"].reset_index(drop=True)
    pd.testing.assert_frame_equal(block[single.columns], single, check_exact=False, rtol=1e-12)
    # condition / all-test blocks = report on the concatenated series
    w = magnitude_report(np.concatenate([preds[2], preds[4]]), np.concatenate([ys[2], ys[4]]),
                         FS, channel_names=NAMES)
    wb = df[df["group"] == "W: series 2+4"].reset_index(drop=True)
    pd.testing.assert_frame_equal(wb[w.columns], w, check_exact=False, rtol=1e-12)
    allb = df[df["group"] == "all test"].reset_index(drop=True)
    a = magnitude_report(np.concatenate([preds[s] for s in (2, 4, 5)]),
                         np.concatenate([ys[s] for s in (2, 4, 5)]), FS, channel_names=NAMES)
    pd.testing.assert_frame_equal(allb[a.columns], a, check_exact=False, rtol=1e-12)
    assert set(df["condition"]) == {"W", "B", "all"}

    text = format_magnitude_table(df)
    assert "series 4 (W)" in text and "all test" in text and "height" in text


class _ScaleModel(torch.nn.Module):
    """pred = 2 * first EMG-sized slice of the EEG input (in z space)."""

    def forward(self, eeg, kin):
        return 2.0 * eeg[..., :3]


def test_predict_series_stitches_non_overlapping_windows_and_denormalises():
    T, win = 1234, 100
    rng = np.random.default_rng(4)
    eeg = rng.standard_normal((T, 4)).astype(np.float32)
    kin = rng.standard_normal((T, 6)).astype(np.float32)
    emg = rng.random((T, 3)).astype(np.float32)
    mean = torch.tensor([0.1, 0.2, 0.3])
    std = torch.tensor([1.0, 2.0, 0.5])

    def pb(e, k, m):
        return {"eeg": e, "kin": k}, (m - mean) / std

    pred, target = predict_series(_ScaleModel(), eeg, kin, emg, pb, mean, std,
                                  window=win, device=torch.device("cpu"), batch_size=3)
    n = (T // win) * win
    assert pred.shape == (n, 3) and target.shape == (n, 3)
    np.testing.assert_allclose(target, emg[:n], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pred, 2.0 * eeg[:n, :3] * std.numpy() + mean.numpy(),
                               rtol=1e-5, atol=1e-6)
