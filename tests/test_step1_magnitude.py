"""Step 1 (magnitude): load force on the amplitude path, force-rate features,
the w_height loss term and the stage-2 convergence settings.

Library defaults stay off: kin_skip_indices=None, w_height=0 and the old
drop_indices behave exactly as before (see also tests/test_force_gate.py).
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

from main.losses import CompositeEMGLoss, build_loss_from_config
from main.model import KGGTModel, build_kg_gt_from_config
from main.preprocessing_emg_kin import (
    POSE_FORCE_DROP_INDICES,
    POSE_FORCE_RATE_DROP_INDICES,
    extract_kt_gated,
    hand_near_object_gate,
    kin_feature_names,
    preprocess_kinematics_from_config,
    sg_velocity,
)

FS = 500.0
INDEX_COLS = [19, 23, 27]
OBJECT_COLS = [18, 22, 26]
FORCE_COLS_RAW = [12, 14, 16, 17]
GATE_CFG = {"near_cm": 12.0, "far_cm": 20.0, "smooth_s": 0.1}
SKIP_IDX = list(range(10, 24))

KIN_CFG = {
    "include_velocity": True,
    "include_acceleration": False,
    "gated_force": dict(GATE_CFG),
    "drop_indices": [12, 25],
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
EXPECTED_24 = (POSE_NAMES + ["F_L_gated", "F_G_gated"]
               + ["v_" + n for n in POSE_NAMES] + ["v_F_L_gated", "v_F_G_gated"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _recording(dist_cm: np.ndarray, seed: int = 0) -> np.ndarray:
    n = len(dist_cm)
    rng = np.random.default_rng(seed)
    kin = np.cumsum(rng.normal(0.0, 0.05, size=(n, 36)), axis=0).astype(np.float64)
    kin[:, OBJECT_COLS] = [-50.0, 1.0, 35.0]
    kin[:, INDEX_COLS] = kin[:, OBJECT_COLS]
    kin[:, INDEX_COLS[0]] += dist_cm
    kin[:, FORCE_COLS_RAW] = 1.0 + np.abs(rng.normal(0.0, 1.0, size=(n, 4)))
    return kin.astype(np.float32)


def _step_profile() -> np.ndarray:
    seg = int(4 * FS)
    return np.concatenate([np.full(seg, 30.0), np.full(seg, 10.0), np.full(seg, 30.0)])


_TRANSFORMER_CFG = {
    "n_layers": 1, "n_heads": 2, "d_model": 16, "d_k": 8, "d_v": 8,
    "ffn_dim": 32, "dropout": 0.0, "chunk_size": 50,
}


def _model(kin_dim: int, override=None, skip=None) -> KGGTModel:
    torch.manual_seed(0)
    kw = {} if skip is None else {"kin_skip_indices": skip}
    return KGGTModel(
        transformer_cfg=_TRANSFORMER_CFG, input_dim=8, node_dim=16, gat_hidden_dim=16,
        gat_heads=2, out_channels=5, kin_dim=kin_dim, use_kinematic_guidance=True,
        gat_kin_dim_override=override, **kw,
    )


def _shapes(m: torch.nn.Module) -> dict:
    return {k: tuple(v.shape) for k, v in m.state_dict().items()}


# ── 1. routing load force to the amplitude path ──────────────────────────────

def test_24_features_gat_gets_12_skip_gets_forces_and_all_velocities():
    m = _model(24, override=12, skip=SKIP_IDX)
    assert m.gat_kin_dim == 12
    assert m.gat.kin_edge_linear.in_features == 12
    assert m.kin_skip_proj[0].in_features == 14
    kin = torch.randn(2, 50, 24)
    assert torch.equal(m.get_gat_kin(kin), kin[..., :12])
    assert torch.equal(m.get_skip_kin(kin), kin[..., 10:24])
    assert m(torch.randn(2, 50, 8), kin).shape == (2, 50, 5)


def test_gat_branch_is_the_same_with_and_without_skip_indices():
    a = _shapes(_model(24, override=12))
    b = _shapes(_model(24, override=12, skip=SKIP_IDX))
    assert {k: v for k, v in a.items() if not k.startswith("kin_skip_proj.0.")} == \
           {k: v for k, v in b.items() if not k.startswith("kin_skip_proj.0.")}
    assert a["kin_skip_proj.0.weight"] == (128, 12) and b["kin_skip_proj.0.weight"] == (128, 14)


def test_skip_indices_only_change_what_the_skip_branch_sees():
    """Same seed, same weights: the GAT path sees identical input; the skip
    branch output equals kin_skip_proj applied to the selected columns."""
    m = _model(24, override=12, skip=SKIP_IDX)
    kin = torch.randn(1, 50, 24)
    kin2 = kin.clone()
    kin2[..., :10] += 5.0        # pose (not in skip) changes → skip input unchanged
    assert torch.equal(m.get_skip_kin(kin), m.get_skip_kin(kin2))
    kin3 = kin.clone()
    kin3[..., 10] += 5.0         # F_L_gated → reaches the skip branch now
    assert not torch.equal(m.get_skip_kin(kin), m.get_skip_kin(kin3))


def test_non_contiguous_skip_indices():
    m = _model(24, override=12, skip=[10, 11, 22, 23])
    kin = torch.randn(2, 50, 24)
    assert m.kin_skip_proj[0].in_features == 4
    assert torch.equal(m.get_skip_kin(kin), kin[..., [10, 11, 22, 23]])
    assert m(torch.randn(2, 50, 8), kin).shape == (2, 50, 5)


@pytest.mark.parametrize("bad", [[], [24], [-1], [3, 3], [1.5]])
def test_invalid_skip_indices_raise(bad):
    with pytest.raises((ValueError, TypeError)):
        _model(24, override=12, skip=bad)


@pytest.mark.parametrize("kin_dim, override", [(36, None), (22, 12), (20, None), (13, None)])
def test_old_kin_dims_unchanged_with_default_none(kin_dim, override):
    a = _model(kin_dim, override=override)
    torch.manual_seed(0)
    b = KGGTModel(
        transformer_cfg=_TRANSFORMER_CFG, input_dim=8, node_dim=16, gat_hidden_dim=16,
        gat_heads=2, out_channels=5, kin_dim=kin_dim, use_kinematic_guidance=True,
        gat_kin_dim_override=override, kin_skip_indices=None,
    )
    assert _shapes(a) == _shapes(b)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    assert a.skip_kin_indices is None
    kin = torch.randn(1, 50, kin_dim)
    expected_skip = {36: 24, 22: 10, 20: 10, 13: 13}[kin_dim]
    assert a.kin_skip_proj[0].in_features == expected_skip
    assert a.get_skip_kin(kin).shape[-1] == expected_skip


def test_feature_names_of_each_branch():
    m = _model(24, override=12, skip=SKIP_IDX)
    assert m.gat_kin_feature_names(EXPECTED_24) == EXPECTED_24[:12]
    assert m.skip_kin_feature_names(EXPECTED_24) == EXPECTED_24[10:24]
    old = _model(20)
    names20 = kin_feature_names(True, False, [10, 11, 12, 23, 24, 25])
    assert old.skip_kin_feature_names(names20) == names20[10:]
    assert old.gat_kin_feature_names(names20) == names20[:10]


def test_builder_reads_kin_skip_indices():
    cfg = {"model": {"type": "transformer_regressor", "transformer": _TRANSFORMER_CFG,
                     "gat": {"node_dim": 16, "hidden_dim": 16, "heads": 2,
                             "use_kinematic_guidance": True, "kin_gat_dim": 12,
                             "kin_skip_indices": SKIP_IDX},
                     "decoder": {"out_channels": 5}}}
    m = build_kg_gt_from_config(cfg, input_dim=8, kin_dim=24)
    assert m.gat_kin_dim == 12 and m.kin_skip_proj[0].in_features == 14
    assert list(m.skip_kin_indices) == SKIP_IDX
    cfg_off = copy.deepcopy(cfg)
    del cfg_off["model"]["gat"]["kin_skip_indices"]
    m_off = build_kg_gt_from_config(cfg_off, input_dim=8, kin_dim=24)
    assert m_off.skip_kin_indices is None and m_off.kin_skip_proj[0].in_features == 12


# ── 2. features: 24 = pose, gated forces, their velocities ───────────────────

def test_feature_layout_is_24_with_force_rates():
    assert POSE_FORCE_RATE_DROP_INDICES == [12, 25]
    assert POSE_FORCE_DROP_INDICES == [12, 23, 24, 25]      # Package-1 constant unchanged
    names = kin_feature_names(True, False, POSE_FORCE_RATE_DROP_INDICES, gated_force=True)
    assert names == EXPECTED_24
    assert names[10:24] == ["F_L_gated", "F_G_gated"] + ["v_" + n for n in POSE_NAMES] + \
        ["v_F_L_gated", "v_F_G_gated"]


def test_cached_layout_is_pos13_vel13_and_velocity_is_taken_after_gating():
    kin = _recording(_step_profile())
    cached = preprocess_kinematics_from_config(kin, FS, KIN_CFG)
    assert cached.shape == (kin.shape[0], 26)         # = the existing cache layout
    gated = extract_kt_gated(kin, fs=FS, gated_force=GATE_CFG)
    np.testing.assert_allclose(cached[:, :13], gated, atol=1e-5)
    vel = sg_velocity(gated, fs=FS, window=51, poly=3)
    np.testing.assert_allclose(cached[:, 13:], vel, rtol=1e-4, atol=1e-4)
    feats = np.delete(cached, KIN_CFG["drop_indices"], axis=-1)
    assert feats.shape[1] == 24
    np.testing.assert_allclose(feats[:, 22:24], vel[:, 10:12], rtol=1e-4, atol=1e-4)


def test_force_rate_is_zero_while_the_hand_is_far():
    kin = _recording(_step_profile())
    feats = np.delete(preprocess_kinematics_from_config(kin, FS, KIN_CFG), [12, 25], axis=-1)
    g = hand_near_object_gate(kin, FS, **GATE_CFG)[:, 0]
    from scipy.ndimage import maximum_filter1d
    far = maximum_filter1d(g, size=51, mode="nearest") == 0.0   # gate 0 over the SG window
    assert far.sum() > 3000
    assert np.all(np.isfinite(feats))
    assert np.all(feats[far, 22:24] == 0.0)
    assert np.all(feats[far, 10:12] == 0.0)
    near = g == 1.0
    assert feats[near, 22:24].std(axis=0).min() > 0.0         # forces change while held


def test_force_rate_drop_indices_need_no_new_cache():
    """The force-rate layout is a different drop of the SAME 26 cached columns."""
    kin = _recording(_step_profile(), seed=3)
    a = preprocess_kinematics_from_config(kin, FS, dict(KIN_CFG, drop_indices=[12, 23, 24, 25]))
    b = preprocess_kinematics_from_config(kin, FS, KIN_CFG)
    np.testing.assert_array_equal(a, b)


# ── 3. w_height loss term ────────────────────────────────────────────────────

def _height_only(**kw) -> CompositeEMGLoss:
    base = dict(w_peak=0.0, w_ccc=0.0, w_pearson=0.0, w_diff=0.0, w_rest=0.0, w_reg=0.0,
                w_height=1.0, asymmetry=1.5, height_eps=1e-6)
    base.update(kw)
    return CompositeEMGLoss(**base)


def _t(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32).reshape(1, -1, 1)


# target [0, 0, 2, 4]: t_min 0, max 4 → rest where < 0.6 → active = samples 2, 3,
# mean target above t_min over active = 3.

def test_height_under_prediction_hand_computed():
    loss = _height_only()
    pred = _t([0.0, 0.0, 1.0, 2.0])            # active mean 1.5 → rel = (1.5 − 3)/3 = −0.5
    v = loss(pred, _t([0.0, 0.0, 2.0, 4.0]))
    # smooth-L1(δ=1): 0.5·0.25 = 0.125, under-prediction × asymmetry 1.5 → 0.1875
    assert v.item() == pytest.approx(0.1875, rel=1e-5)
    assert loss.last_components["height"] == pytest.approx(0.1875, rel=1e-5)


def test_height_over_prediction_hand_computed_linear_branch():
    loss = _height_only()
    pred = _t([0.0, 0.0, 6.0, 12.0])           # active mean 9 → rel = 2 → |2| − 0.5 = 1.5, ×1
    assert loss(pred, _t([0.0, 0.0, 2.0, 4.0])).item() == pytest.approx(1.5, rel=1e-5)


def test_height_is_measured_above_the_target_window_minimum():
    loss = _height_only()
    # same shape shifted by +10: t_min 10, active samples 2, 3; level above t_min = 3
    pred = _t([10.0, 10.0, 11.0, 12.0])        # (1.5 − 3)/3 = −0.5 again
    assert loss(pred, _t([10.0, 10.0, 12.0, 14.0])).item() == pytest.approx(0.1875, rel=1e-5)


def test_height_zero_when_pred_equals_target():
    torch.manual_seed(0)
    target = torch.randn(3, 40, 5)
    assert _height_only()(target.clone(), target).item() == pytest.approx(0.0, abs=1e-7)


def test_height_constant_and_zero_windows_contribute_zero_and_stay_finite():
    loss = _height_only()
    for target in (torch.zeros(2, 30, 5), torch.full((2, 30, 5), 3.0)):
        pred = torch.randn(2, 30, 5, requires_grad=True)
        v = loss(pred, target)
        assert v.item() == 0.0
        v.backward()
        assert torch.isfinite(pred.grad).all()


def test_height_mean_over_windows_and_channels_with_empty_ones_as_zero():
    loss = _height_only()
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0], [2.0, 1.0], [4.0, 1.0]]).reshape(1, 4, 2)
    pred = torch.tensor([[0.0, 9.0], [0.0, 9.0], [1.0, 9.0], [2.0, 9.0]]).reshape(1, 4, 2)
    # channel 0 → 0.1875, channel 1 is constant (no active samples) → 0; mean = 0.09375
    assert loss(pred, target).item() == pytest.approx(0.09375, rel=1e-5)
    w = _height_only(channel_weights=[2.0, 0.5])
    assert w(pred, target).item() == pytest.approx(0.1875, rel=1e-5)   # (2·0.1875 + 0)/2


def test_height_is_differentiable_and_only_active_samples_get_gradient():
    loss = _height_only()
    pred = _t([0.0, 0.0, 1.0, 2.0]).requires_grad_(True)
    loss(pred, _t([0.0, 0.0, 2.0, 4.0])).backward()
    g = pred.grad.reshape(-1)
    assert g[0] == 0.0 and g[1] == 0.0
    assert (g[2:] < 0).all()        # loss falls when the under-predicted active part rises
    # d/dpred: asym · rel / δ · (1/(mt+eps)) / n_act = 1.5 · (−0.5) · (1/3) / 2 = −0.125
    assert g[2].item() == pytest.approx(-0.125, rel=1e-4)


def test_height_one_odd_window_does_not_dominate():
    loss = _height_only(asymmetry=1.0)
    target = _t([0.0, 0.0, 2.0, 4.0])
    small = loss(_t([0.0, 0.0, 6.0, 12.0]), target).item()       # rel 2   → 1.5
    huge = loss(_t([0.0, 0.0, 60.0, 120.0]), target).item()      # rel 29  → 28.5 (linear, not 420)
    assert huge == pytest.approx(28.5, rel=1e-5) and small == pytest.approx(1.5, rel=1e-5)


def test_height_default_off_and_total_unchanged():
    torch.manual_seed(1)
    pred, target = torch.randn(2, 40, 5), torch.randn(2, 40, 5)
    a = CompositeEMGLoss(asymmetry=1.2, w_rest_over=1.0, huber_delta=1.0)
    b = CompositeEMGLoss(asymmetry=1.2, w_rest_over=1.0, huber_delta=1.0, w_height=0.0)
    assert a(pred, target).item() == b(pred, target).item()
    assert a.last_components["height"] == 0.0
    assert a.w_height == 0.0


def test_height_adds_to_total_with_its_weight():
    torch.manual_seed(2)
    pred, target = torch.randn(2, 40, 5), torch.randn(2, 40, 5).abs()
    off = CompositeEMGLoss(asymmetry=1.5, huber_delta=2.5)
    on = CompositeEMGLoss(asymmetry=1.5, huber_delta=2.5, w_height=0.5)
    v_off, v_on = off(pred, target).item(), on(pred, target).item()
    assert v_on == pytest.approx(v_off + 0.5 * on.last_components["height"], rel=1e-5)
    assert on.last_components["height"] > 0


def test_builder_reads_all_step1_loss_keys():
    cfg = {"loss": {"asymmetry": 1.5, "w_rest_over": 1.0, "huber_delta": 2.5,
                    "w_height": 0.5, "height_eps": 0.2}}
    loss = build_loss_from_config(cfg, channel_weights=[1.0] * 5)
    assert loss.peak_mse.asymmetry == 1.5
    assert loss.peak_mse.huber_delta == 2.5
    assert loss.w_rest_over == 1.0
    assert loss.w_height == 0.5
    assert loss.height_asymmetry == 1.5
    assert loss.height_eps == 0.2
    d = build_loss_from_config({"loss": {}})
    assert d.w_height == 0.0 and d.peak_mse.asymmetry == 2.0 and d.peak_mse.huber_delta is None


# ── 4. notebook CONFIGs ──────────────────────────────────────────────────────

def _notebook(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _cell_src(nb: dict, needle: str) -> str:
    hits = ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code" and needle in "".join(c["source"])]
    assert len(hits) == 1, (needle, len(hits))
    return hits[0]


def _notebook_config(name: str) -> dict:
    src = _cell_src(_notebook(name), "CONFIG = {")
    ns = {"copy": copy}
    exec(src, ns)
    return ns["CONFIG"]


@pytest.mark.parametrize("name", ["main.ipynb", "main_p2p.ipynb"])
def test_notebook_config_turns_step1_on(name):
    cfg = _notebook_config(name)
    kin = cfg["preprocessing"]["kinematics"]
    assert kin["gated_force"] == GATE_CFG
    assert kin["include_velocity"] is True and kin["include_acceleration"] is False
    assert kin["sg_window"] == 51
    assert kin["drop_indices"] == POSE_FORCE_RATE_DROP_INDICES
    assert cfg["data"]["cache_dir"] == "data/cache_main_stride500_square_pose_force_sg51"
    names = kin_feature_names(kin["include_velocity"], kin["include_acceleration"],
                              kin["drop_indices"], gated_force=kin["gated_force"])
    assert names == EXPECTED_24
    assert cfg["data"]["n_kin_features"] == 24
    gat = cfg["model"]["gat"]
    assert gat["kin_gat_dim"] == 12
    assert gat["kin_skip_indices"] == SKIP_IDX
    loss = cfg["training"]["loss"]
    assert loss["asymmetry"] == 1.5 and loss["huber_delta"] == 2.5
    assert loss["w_height"] == 0.5 and loss["w_rest_over"] == 1.0
    assert loss["channel_weights"] == "auto"
    assert cfg["training"]["stage2_lr"] == 1e-4
    assert cfg["training"]["stage2_early_stop_patience"] == 25
    m = build_kg_gt_from_config(dict(cfg, model=dict(cfg["model"], transformer=_TRANSFORMER_CFG)),
                                input_dim=8, kin_dim=24)
    assert m.gat_kin_dim == 12 and m.kin_skip_proj[0].in_features == 14
    src = _cell_src(_notebook(name), "CONFIG = {")
    assert "move aside" in src.lower() and "24 features" in src


def test_both_notebooks_share_data_preprocessing_model_and_loss():
    a, b = _notebook_config("main.ipynb"), _notebook_config("main_p2p.ipynb")
    for k in ("data", "preprocessing", "dataset", "model", "outputs"):
        assert a[k] == b[k], k
    assert a["training"]["loss"] == b["training"]["loss"]


def test_stage2_uses_its_own_early_stop_patience():
    nb = _notebook("main.ipynb")
    train = _cell_src(nb, "PER-SUBJECT TRAINING LOOP  (Stage 1 + Stage 2)")
    assert 'stage2_early_stop_patience' in train
    assert '_cfg_s2.scheduler     = "reduce"' in train
    assert "_cfg_s2.lr_patience   = 5" in train and "_cfg_s2.lr_factor     = 0.5" in train
    loso = _cell_src(nb, "[OPTIONAL] Automatic LOSOCV")
    assert "stage2_early_stop_patience" in loso


@pytest.mark.parametrize("name", ["main.ipynb", "main_p2p.ipynb"])
def test_no_stale_22_feature_texts(name):
    nb = _notebook(name)
    for c in nb["cells"]:
        s = "".join(c["source"])
        assert "22 kin" not in s and "= 22" not in s and "(22:" not in s, s[:80]
        assert "(1, T, 20)" not in s


def test_figure_folders_do_not_collide_with_the_previous_runs():
    from main.plots import get_dynamic_fig_dir
    a, b = _notebook_config("main.ipynb"), _notebook_config("main_p2p.ipynb")
    b["training"]["lr"] = b["training"]["single_stage"]["lr"]       # as the 1-stage CONFIG cell does
    a["data"]["participant"] = b["data"]["participant"] = 1
    da, db = get_dynamic_fig_dir(a).as_posix(), get_dynamic_fig_dir(b).as_posix()
    assert da != db
    # previous Package-1 runs: ep200_st500_lr1e-05_bs32 (two-stage), ep200_st500_lr0.001_bs32 (1-stage)
    for old in ("results/P1/main/ep200_st500_lr1e-05_bs32/plots",
                "results/P1/main/ep200_st500_lr0.001_bs32/plots"):
        assert da != old and db != old
    # without a run tag the folder name is unchanged (library default)
    plain = copy.deepcopy(a)
    plain["outputs"].pop("run_tag", None)
    assert get_dynamic_fig_dir(plain).as_posix() == "results/P1/main/ep200_st500_lr0.0001_bs32/plots"
