"""Hand-near-object gated force features (Package 1, change 1).

The object's grip-plate forces F_L / F_G are only kept while the index fingertip
is near the object (index-object distance <= near_cm); they ramp to zero at
far_cm. Pose features are unchanged. Library defaults stay off.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.model import KGGTModel, build_kg_gt_from_config
from main.preprocessing_emg_kin import (
    POSE_DROP_INDICES,
    POSE_FORCE_DROP_INDICES,
    extract_kt_raw,
    hand_near_object_gate,
    index_object_distance,
    kin_feature_names,
    preprocess_kinematics,
    preprocess_kinematics_from_config,
)

FS = 500.0
INDEX_COLS = [19, 23, 27]
OBJECT_COLS = [18, 22, 26]
FORCE_COLS_RAW = [12, 14, 16, 17]
GATE_CFG = {"near_cm": 12.0, "far_cm": 20.0, "smooth_s": 0.1}

KIN_CFG = {
    "include_velocity": True,
    "include_acceleration": False,
    "gated_force": dict(GATE_CFG),
    "drop_indices": [12, 23, 24, 25],
    "velocity_method": "sg",
    "sg_window": 51,
    "sg_poly": 3,
    "bw_cutoff": 20.0,
    "bw_order": 4,
    "normalize": True,
}

POSE_NAMES = [
    "p_wrist_x", "p_wrist_y", "p_wrist_z",
    "p_index_x", "p_index_y", "p_index_z",
    "p_thumb_x", "p_thumb_y", "p_thumb_z",
    "d_grip",
]


def _recording(dist_cm: np.ndarray, seed: int = 0) -> np.ndarray:
    """Raw 36-col kin whose index-object distance follows `dist_cm` (along x)."""
    n = len(dist_cm)
    rng = np.random.default_rng(seed)
    kin = np.cumsum(rng.normal(0.0, 0.05, size=(n, 36)), axis=0).astype(np.float64)
    kin[:, OBJECT_COLS] = [-50.0, 1.0, 35.0]
    kin[:, INDEX_COLS] = kin[:, OBJECT_COLS]
    kin[:, INDEX_COLS[0]] += dist_cm
    kin[:, FORCE_COLS_RAW] = 1.0 + np.abs(rng.normal(0.0, 1.0, size=(n, 4)))
    return kin.astype(np.float32)


def _step_profile() -> np.ndarray:
    """4 s far (30 cm) -> 4 s near (10 cm) -> 4 s far."""
    seg = int(4 * FS)
    return np.concatenate([np.full(seg, 30.0), np.full(seg, 10.0), np.full(seg, 30.0)])


def _features(kin: np.ndarray, cfg: dict) -> np.ndarray:
    cached = preprocess_kinematics_from_config(kin, FS, cfg)
    return np.delete(cached, cfg["drop_indices"], axis=-1)


# ── the gate ──────────────────────────────────────────────────────────────────

def test_distance_uses_index_and_object_sensors():
    d = np.linspace(5.0, 30.0, 200)
    kin = _recording(d)
    np.testing.assert_allclose(index_object_distance(kin), d, atol=1e-4)


def test_gate_is_one_near_zero_far_and_shape_t1():
    kin = _recording(_step_profile())
    g = hand_near_object_gate(kin, FS, **GATE_CFG)
    assert g.shape == (kin.shape[0], 1)
    assert g.dtype == np.float32
    assert g.min() >= 0.0 and g.max() <= 1.0
    seg = int(4 * FS)
    pad = int(0.2 * FS)
    assert np.all(g[pad:seg - pad] == 0.0)                  # far
    assert np.all(g[seg + pad:2 * seg - pad] == 1.0)        # near
    assert np.all(g[2 * seg + pad:-pad] == 0.0)             # far again


def test_gate_ramps_linearly_between_near_and_far_without_smoothing():
    d = np.linspace(8.0, 24.0, 1601)
    kin = _recording(d)
    g = hand_near_object_gate(kin, FS, near_cm=12.0, far_cm=20.0, smooth_s=0.0)[:, 0]
    expected = np.clip((20.0 - d) / 8.0, 0.0, 1.0)
    np.testing.assert_allclose(g, expected, atol=1e-5)


def test_smoothing_removes_sharp_steps():
    kin = _recording(_step_profile())
    raw = hand_near_object_gate(kin, FS, near_cm=12.0, far_cm=20.0, smooth_s=0.0)[:, 0]
    smooth = hand_near_object_gate(kin, FS, **GATE_CFG)[:, 0]
    assert np.abs(np.diff(raw)).max() == pytest.approx(1.0)       # the step profile jumps
    n_smooth = int(round(GATE_CFG["smooth_s"] * FS))
    assert np.abs(np.diff(smooth)).max() <= 1.0 / n_smooth + 1e-6  # spread over ~smooth_s
    seg = int(4 * FS)
    # never lowered inside the near zone, reaches 0 within smooth_s outside it
    assert np.all(smooth[seg:2 * seg] == 1.0)
    assert np.all(smooth[:seg - n_smooth - 1] == 0.0)
    assert np.all(smooth[2 * seg + n_smooth + 1:] == 0.0)
    # zero phase: the rising edge mirrors the falling edge
    k = np.arange(1, n_smooth + 2)
    np.testing.assert_allclose(smooth[seg - k], smooth[2 * seg - 1 + k], atol=1e-6)


def test_gate_is_zero_where_tracking_is_missing():
    d = _step_profile()
    kin = _recording(d)
    kin[int(5 * FS):int(5.5 * FS), INDEX_COLS] = np.nan
    g = hand_near_object_gate(kin, FS, **GATE_CFG)
    assert np.isfinite(g).all()
    assert np.all(g[int(5.1 * FS):int(5.4 * FS)] == 0.0)


# ── features ──────────────────────────────────────────────────────────────────

def test_default_preprocessing_is_unchanged_when_the_gate_is_absent():
    kin = _recording(_step_profile())
    cfg_off = {k: v for k, v in KIN_CFG.items() if k != "gated_force"}
    a = preprocess_kinematics_from_config(kin, FS, cfg_off)
    b = preprocess_kinematics_from_config(kin, FS, dict(cfg_off, gated_force=None))
    c = preprocess_kinematics(kin, fs=FS, include_velocity=True, sg_window=51)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_gated_forces_zero_while_far_and_unchanged_while_near():
    kin = _recording(_step_profile())
    cached = preprocess_kinematics_from_config(kin, FS, KIN_CFG)
    assert cached.shape == (kin.shape[0], 26)
    raw = extract_kt_raw(kin)
    seg, pad = int(4 * FS), int(0.2 * FS)
    far = np.r_[pad:seg - pad, 2 * seg + pad:3 * seg - pad]
    near = np.r_[seg + pad:2 * seg - pad]
    assert np.all(cached[far][:, [10, 11]] == 0.0)
    np.testing.assert_array_equal(cached[near][:, [10, 11]], raw[near][:, [10, 11]])
    # pose columns are untouched
    np.testing.assert_array_equal(cached[:, :10], raw[:, :10])


def test_layout_names_and_count_are_22():
    kin = _recording(_step_profile())
    feats = _features(kin, KIN_CFG)
    assert feats.shape[1] == 22
    assert POSE_FORCE_DROP_INDICES == [12, 23, 24, 25]
    assert POSE_DROP_INDICES == [10, 11, 12, 23, 24, 25]     # old helper unchanged
    names = kin_feature_names(True, False, POSE_FORCE_DROP_INDICES, gated_force=True)
    assert names == POSE_NAMES + ["F_L_gated", "F_G_gated"] + ["v_" + n for n in POSE_NAMES]
    # without the flag the names are the old ones
    assert kin_feature_names(True, False, POSE_FORCE_DROP_INDICES)[10:12] == ["F_L", "F_G"]
    # dict config value is accepted as "on"
    assert kin_feature_names(True, False, POSE_FORCE_DROP_INDICES, gated_force=GATE_CFG) == names
    assert kin_feature_names(True, False, POSE_FORCE_DROP_INDICES, gated_force=None)[10] == "F_L"


def test_force_changes_while_far_leave_all_22_features_unchanged():
    d = _step_profile()
    kin = _recording(d)
    kin2 = kin.copy()
    far = d > 20.0
    # someone else loads the object while the hand is parked
    kin2[np.ix_(far, FORCE_COLS_RAW)] += 7.5
    seg, pad = int(4 * FS), int(0.3 * FS)
    a, b = _features(kin, KIN_CFG), _features(kin2, KIN_CFG)
    # away from the transitions (the gate ramp itself is smoothed over 0.1 s)
    keep = np.r_[pad:seg - pad, seg + pad:2 * seg - pad, 2 * seg + pad:3 * seg - pad]
    np.testing.assert_array_equal(a[keep], b[keep])
    np.testing.assert_array_equal(a[:, :10], b[:, :10])
    np.testing.assert_array_equal(a[:, 12:], b[:, 12:])


# ── model routing ─────────────────────────────────────────────────────────────

_TRANSFORMER_CFG = {
    "n_layers": 1, "n_heads": 2, "d_model": 16, "d_k": 8, "d_v": 8,
    "ffn_dim": 32, "dropout": 0.0, "chunk_size": 50,
}


def _model(kin_dim: int, override=None) -> KGGTModel:
    torch.manual_seed(0)
    return KGGTModel(
        transformer_cfg=_TRANSFORMER_CFG, input_dim=8, node_dim=16, gat_hidden_dim=16,
        gat_heads=2, out_channels=5, kin_dim=kin_dim, use_kinematic_guidance=True,
        gat_kin_dim_override=override,
    )


def test_override_routes_12_features_to_gat_and_10_to_skip():
    m = _model(22, override=12)
    assert m.gat_kin_dim == 12
    assert m.gat.kin_edge_linear.in_features == 12
    assert m.kin_skip_proj[0].in_features == 10
    kin = torch.randn(2, 50, 22)
    assert torch.equal(m.get_gat_kin(kin), kin[..., :12])
    assert torch.equal(m.get_skip_kin(kin), kin[..., 12:])
    out = m(torch.randn(2, 50, 8), kin)
    assert out.shape == (2, 50, 5)


def test_without_override_22_would_split_11_11():
    m = _model(22)
    assert m.gat_kin_dim == 11


@pytest.mark.parametrize("kin_dim, gat_in, skip_in", [
    (36, 12, 24), (39, 13, 26), (26, 13, 13), (20, 10, 10), (13, 13, 13), (12, 12, 12),
])
def test_existing_kin_dims_unchanged_with_override_none(kin_dim, gat_in, skip_in):
    m = _model(kin_dim, override=None)
    assert m.gat_kin_dim == gat_in
    assert m.gat.kin_edge_linear.in_features == gat_in
    assert m.kin_skip_proj[0].in_features == skip_in


@pytest.mark.parametrize("bad", [0, 22, 30, -1])
def test_override_must_leave_features_for_both_branches(bad):
    with pytest.raises(ValueError):
        _model(22, override=bad)


def test_builder_reads_kin_gat_dim():
    cfg = {"model": {"type": "transformer_regressor", "transformer": _TRANSFORMER_CFG,
                     "gat": {"node_dim": 16, "hidden_dim": 16, "heads": 2,
                             "use_kinematic_guidance": True, "kin_gat_dim": 12},
                     "decoder": {"out_channels": 5}}}
    m = build_kg_gt_from_config(cfg, input_dim=8, kin_dim=22)
    assert m.gat_kin_dim == 12
    cfg_off = copy.deepcopy(cfg)
    del cfg_off["model"]["gat"]["kin_gat_dim"]
    assert build_kg_gt_from_config(cfg_off, input_dim=8, kin_dim=20).gat_kin_dim == 10
    cfg_none = copy.deepcopy(cfg)
    cfg_none["model"]["gat"]["kin_gat_dim"] = None
    assert build_kg_gt_from_config(cfg_none, input_dim=8, kin_dim=22).gat_kin_dim == 11


# ── notebook CONFIG ───────────────────────────────────────────────────────────

def _notebook_config(name: str) -> dict:
    nb = json.loads((ROOT / name).read_text(encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "".join(c["source"]).startswith("CONFIG = {"))
    ns = {"copy": copy}
    exec(src, ns)
    return ns["CONFIG"]


@pytest.mark.parametrize("name", ["main.ipynb", "main_1stage.ipynb"])
def test_notebook_config_turns_package1_on(name):
    cfg = _notebook_config(name)
    kin = cfg["preprocessing"]["kinematics"]
    assert kin["gated_force"] == GATE_CFG
    assert kin["drop_indices"] == POSE_FORCE_DROP_INDICES
    assert cfg["data"]["n_kin_features"] == 22
    assert cfg["data"]["cache_dir"] == "data/cache_main_stride500_square_pose_force_sg51"
    assert cfg["model"]["gat"]["kin_gat_dim"] == 12
    n = len(kin_feature_names(kin["include_velocity"], kin["include_acceleration"],
                              kin["drop_indices"], gated_force=kin["gated_force"]))
    assert n == 22


# ── real data (skipped when the raw folder is absent) ─────────────────────────

_REAL = ROOT / "data" / "way-eeg" / "raw" / "P1" / "HS_P1_S4.mat"


@pytest.mark.skipif(not _REAL.exists(), reason="raw WAY-EEG-GAL data not present")
def test_real_p1_s4_distance_is_in_cm_and_gate_is_one_at_grasp():
    from main.dataloader import load_hs
    hs = load_hs(str(_REAL))
    names = hs["kin_names"]
    assert names[18].startswith("Px1") and names[19].startswith("Px2")
    kin = hs["kin"]
    d = index_object_distance(kin)
    # bimodal: ~10 cm while grasping, ~28 cm when parked -> centimetres
    assert 8.0 < np.percentile(d, 25) < 12.0
    assert 24.0 < np.percentile(d, 90) < 35.0
    g = hand_near_object_gate(kin, float(hs["fs_kin"]), **GATE_CFG)[:, 0]
    assert np.all(g[d <= GATE_CFG["near_cm"]] == 1.0)
    # hand beyond far_cm for at least +-smooth_s around the sample -> gate 0
    from scipy.ndimage import minimum_filter1d
    n = int(round(GATE_CFG["smooth_s"] * float(hs["fs_kin"]))) | 1
    parked = minimum_filter1d((d > GATE_CFG["far_cm"]).astype(np.uint8), size=2 * n + 1) > 0
    assert parked.mean() > 0.2
    assert np.all(g[parked] == 0.0)
