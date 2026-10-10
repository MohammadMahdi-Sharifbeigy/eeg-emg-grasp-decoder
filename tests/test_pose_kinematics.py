"""Hand-pose-only kinematic input (no object forces, no acceleration, SG window 51).

Synthetic data only: no dataset files, no cache, no GPU, no training.

Run with:
    env/Scripts/python.exe -m pytest tests/test_pose_kinematics.py -q
"""
from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.model import KGGTModel
from main.preprocessing_emg_kin import (
    extract_kt_raw,
    preprocess_kinematics_from_config,
    sg_velocity,
)

FS = 500.0

# The kinematics section of CONFIG in main.ipynb (new hand-pose layout).
NEW_KIN_CFG = {
    "include_velocity": True,
    "include_acceleration": False,
    "drop_indices": [10, 11, 12, 23, 24, 25],
    "velocity_method": "sg",
    "sg_window": 51,
    "sg_poly": 3,
    "bw_cutoff": 20.0,
    "bw_order": 4,
    "normalize": True,
}
# The layout used before this change (pos + vel + acc, only rho_GL dropped).
OLD_KIN_CFG = dict(NEW_KIN_CFG, include_acceleration=True, sg_window=11,
                   drop_indices=[12, 25, 38])

POSE_NAMES = [
    "p_wrist_x", "p_wrist_y", "p_wrist_z",
    "p_index_x", "p_index_y", "p_index_z",
    "p_thumb_x", "p_thumb_y", "p_thumb_z",
    "d_grip",
]
FORCE_COLS_RAW = [12, 14, 16, 17]   # FX1, FY1, FZ1, FZ2 in the raw 36-col signal


def _synthetic_raw_kin(n: int = 3000, seed: int = 0) -> np.ndarray:
    """Smooth random-walk 'recording' with the raw 36-column layout."""
    rng = np.random.default_rng(seed)
    kin = np.cumsum(rng.normal(0.0, 0.05, size=(n, 36)), axis=0)
    kin += rng.normal(0.0, 0.01, size=(n, 36))
    return kin.astype(np.float32)


def _features(kin: np.ndarray, cfg: dict) -> np.ndarray:
    """Cached columns -> model input columns, exactly as the notebook does it."""
    cached = preprocess_kinematics_from_config(kin, FS, cfg)
    keep = [i for i in range(cached.shape[1]) if i not in cfg["drop_indices"]]
    return cached[:, keep]


# ── (a) layout: 26 cached columns → 20 pose / pose-velocity features ──────────

def test_new_config_caches_26_columns_and_keeps_20_pose_features():
    from main.preprocessing_emg_kin import POSE_DROP_INDICES

    kin = _synthetic_raw_kin()
    cached = preprocess_kinematics_from_config(kin, FS, NEW_KIN_CFG)
    assert cached.shape == (kin.shape[0], 26)
    assert POSE_DROP_INDICES == [10, 11, 12, 23, 24, 25]
    assert NEW_KIN_CFG["drop_indices"] == POSE_DROP_INDICES

    kept = np.delete(cached, POSE_DROP_INDICES, axis=-1)
    assert kept.shape == (kin.shape[0], 20)

    pose = extract_kt_raw(kin)[:, :10]
    np.testing.assert_array_equal(kept[:, :10], pose)
    np.testing.assert_allclose(
        kept[:, 10:], sg_velocity(pose, fs=FS, window=51, poly=3), rtol=1e-5, atol=1e-5
    )


def test_feature_names_follow_the_config():
    from main.preprocessing_emg_kin import KT_RAW_NAMES, POSE_DROP_INDICES, kin_feature_names

    assert KT_RAW_NAMES == POSE_NAMES + ["F_L", "F_G", "rho_GL"]

    names = kin_feature_names(True, False, POSE_DROP_INDICES)
    assert names == POSE_NAMES + ["v_" + n for n in POSE_NAMES]

    # Full layout and the previous 36-feature layout stay describable.
    full = kin_feature_names(True, True, None)
    assert len(full) == 39
    assert full[10:13] == ["F_L", "F_G", "rho_GL"]
    assert full[13] == "v_p_wrist_x" and full[26] == "a_p_wrist_x"
    old = kin_feature_names(True, True, [12, 25, 38])
    assert len(old) == 36 and "F_L" in old and "a_F_G" in old and "rho_GL" not in old
    assert kin_feature_names(False, False, None) == KT_RAW_NAMES


def test_public_names_are_exported_from_the_package():
    import main

    for name in ("KT_RAW_NAMES", "POSE_DROP_INDICES", "kin_feature_names",
                 "plot_kin_smoothing_comparison"):
        assert hasattr(main, name), name
        assert name in main.__all__, name


def test_notebook_config_uses_the_pose_layout():
    """main.ipynb CONFIG must match the layout these tests describe."""
    from main.preprocessing_emg_kin import POSE_FORCE_RATE_DROP_INDICES, kin_feature_names

    nb = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "".join(c["source"]).startswith("CONFIG = {"))
    ns = {"copy": copy}
    exec(src, ns)   # the cell only builds a dict literal
    cfg = ns["CONFIG"]
    kin_cfg = cfg["preprocessing"]["kinematics"]

    assert kin_cfg["include_velocity"] is True
    assert kin_cfg["include_acceleration"] is False
    assert kin_cfg["sg_window"] == 51
    # Package 1 + Step 1: pose + hand-near-object gated forces + their velocities
    # (see tests/test_force_gate.py, tests/test_step1_magnitude.py)
    assert kin_cfg["drop_indices"] == POSE_FORCE_RATE_DROP_INDICES
    n_features = len(kin_feature_names(
        kin_cfg["include_velocity"], kin_cfg["include_acceleration"], kin_cfg["drop_indices"],
        gated_force=kin_cfg["gated_force"],
    ))
    assert n_features == 24 == cfg["data"]["n_kin_features"]
    # A new cache folder, so caches of older layouts are never loaded.
    assert cfg["data"]["cache_dir"] == "data/cache_main_stride500_square_pose_force_sg51"


# ── (b) object forces can no longer reach the model ───────────────────────────

def test_force_columns_do_not_affect_the_pose_features():
    kin = _synthetic_raw_kin()
    loaded = kin.copy()
    rng = np.random.default_rng(1)
    # Someone else loads the object while the hand is parked.
    loaded[:, FORCE_COLS_RAW] += rng.normal(5.0, 3.0, size=(kin.shape[0], len(FORCE_COLS_RAW)))

    np.testing.assert_array_equal(_features(kin, NEW_KIN_CFG), _features(loaded, NEW_KIN_CFG))

    old_a, old_b = _features(kin, OLD_KIN_CFG), _features(loaded, OLD_KIN_CFG)
    assert old_a.shape[1] == 36
    assert not np.allclose(old_a, old_b)


# ── (c) SG 51 is quieter than SG 11 and still unbiased ────────────────────────

def test_sg51_velocity_is_quieter_at_rest_and_keeps_the_slope():
    rng = np.random.default_rng(2)
    n = 5000
    parked = (100.0 + rng.normal(0.0, 0.05, size=(n, 1))).astype(np.float32)   # mm, jitter only
    v11 = sg_velocity(parked, fs=FS, window=11, poly=3)[100:-100]
    v51 = sg_velocity(parked, fs=FS, window=51, poly=3)[100:-100]
    assert v51.std() < 0.25 * v11.std()

    slope = 40.0   # mm/s
    ramp = (slope * np.arange(n) / FS).astype(np.float64)[:, None]
    for window in (11, 51):
        v = sg_velocity(ramp, fs=FS, window=window, poly=3)[100:-100]
        np.testing.assert_allclose(v, slope, rtol=1e-3)


# ── (d) model routing ─────────────────────────────────────────────────────────

_TRANSFORMER_CFG = {
    "n_layers": 1, "n_heads": 2, "d_model": 16, "d_k": 8, "d_v": 8,
    "ffn_dim": 32, "dropout": 0.0, "chunk_size": 20,
}


def _model(kin_dim: int) -> KGGTModel:
    torch.manual_seed(0)
    return KGGTModel(
        transformer_cfg=_TRANSFORMER_CFG, input_dim=8, node_dim=16, gat_hidden_dim=16,
        gat_heads=4, out_channels=5, kin_dim=kin_dim, use_kinematic_guidance=True,
    )


def test_model_splits_20_features_into_pose_and_velocity_branches():
    model = _model(20).eval()
    assert model.gat_kin_dim == 10
    assert model.gat.kin_edge_linear.in_features == 10     # pose → GAT edge bias
    assert model.kin_skip_proj[0].in_features == 10        # velocity → skip highway

    eeg = torch.randn(2, 40, 8)
    kin = torch.randn(2, 40, 20)
    torch.testing.assert_close(model.get_gat_kin(kin), kin[..., :10])
    torch.testing.assert_close(model.get_skip_kin(kin), kin[..., 10:])
    with torch.no_grad():
        out = model(eeg, kin)
    assert out.shape == (2, 40, 5)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kin_dim, gat_in, skip_in", [
    (36, 12, 24),   # pos / vel+acc (previous notebook layout)
    (39, 13, 26),
    (26, 13, 13),
    (13, 13, 13),   # too small to split: every feature feeds both branches
    (12, 12, 12),
])
def test_existing_kin_dims_route_as_before(kin_dim, gat_in, skip_in):
    model = _model(kin_dim)
    assert model.gat_kin_dim == gat_in
    assert model.gat.kin_edge_linear.in_features == gat_in
    assert model.kin_skip_proj[0].in_features == skip_in


def test_state_dict_shapes_for_36_features_are_unchanged():
    """Checkpoints trained with the 36-feature layout must still load."""
    shapes = {k: tuple(v.shape) for k, v in _model(36).state_dict().items()}
    assert shapes["gat.kin_edge_linear.weight"] == (4 * 5 * 5, 12)
    assert shapes["gat.kin_edge_linear.bias"] == (4 * 5 * 5,)
    assert shapes["kin_skip_proj.0.weight"] == (128, 24)
    assert shapes["kin_skip_proj.4.weight"] == (5 * 16, 128)


# ── (e) smoothing-effect figure ───────────────────────────────────────────────

def _synthetic_recording(n: int = 6000) -> np.ndarray:
    """Parked hand (jitter only) with one reach in the middle, raw 36-col layout."""
    rng = np.random.default_rng(3)
    t = np.arange(n) / FS
    reach = 80.0 * np.exp(-0.5 * ((t - 6.0) / 0.4) ** 2)
    kin = rng.normal(0.0, 0.03, size=(n, 36))
    kin[:, [21, 25, 29]] += reach[:, None] * np.array([1.0, 0.5, 0.8])   # wrist
    kin[:, [19, 23, 27]] += reach[:, None] * np.array([1.0, 0.5, 0.8]) + 40.0   # index
    kin[:, [20, 24, 28]] += reach[:, None] * np.array([1.0, 0.5, 0.8])   # thumb
    kin[:, [12, 14, 16, 17]] += 4.0 * (t > 2.0)[:, None]   # object loaded at rest
    return kin.astype(np.float32)


def test_smoothing_plot_returns_a_figure_without_warnings():
    from main.plots import plot_kin_smoothing_comparison

    kin = _synthetic_recording()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_kin_smoothing_comparison(
            kin, fs=FS, sg_window_old=11, sg_window_new=51, start_sec=0.0, end_sec=10.0,
        )
        fig.canvas.draw()
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 5
        stats = fig.rest_jitter
        assert stats["std_new"] < stats["std_old"]
        assert 0.0 <= stats["rest_start_sec"] < stats["rest_end_sec"] <= 10.0
        # The parked stretch must not contain the reach at t = 6 s.
        assert not (stats["rest_start_sec"] < 6.0 < stats["rest_end_sec"])
    finally:
        plt.close(fig)


def test_smoothing_plot_accepts_a_window_that_does_not_start_at_zero():
    from main.plots import plot_kin_smoothing_comparison

    fig = plot_kin_smoothing_comparison(_synthetic_recording(), fs=FS, start_sec=4.0, end_sec=9.0)
    try:
        lo, hi = fig.axes[-1].get_xlim()
        assert lo == pytest.approx(4.0) and hi == pytest.approx(9.0, abs=0.01)
        assert 4.0 <= fig.rest_jitter["rest_start_sec"] < fig.rest_jitter["rest_end_sec"] <= 9.0
    finally:
        plt.close(fig)
