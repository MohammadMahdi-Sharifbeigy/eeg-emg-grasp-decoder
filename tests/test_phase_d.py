"""Package 2 — Phase D (interpretability rebuild): task events, exact path
decomposition, within-window attention rollout, EEG occlusion / deletion /
effect size, event-locked analysis, the two new figures and the notebook cells.

Random-weight or stub models on CPU only; no notebook cell is executed except
the CONFIG cell (a plain dict literal).
"""

from __future__ import annotations

import copy
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.dataloader import DEFAULT_RAW_DIR
from main.inference import predict_series_stitched, window_batch
from main.interp import (
    EVENT_COLUMNS,
    attention_rollout,
    concat_event_locked,
    decompose_prediction,
    decompose_series,
    deletion_curve,
    eeg_effect_size,
    eeg_occlusion_attribution,
    encoder_attention,
    event_locked_summary,
    event_locked_windows,
    event_samples,
    extract_event_windows,
    kin_column_groups,
    lift_events,
    phase_levels,
    raw_zero_kin_baseline,
    rollout_rows,
)
from main.magnitude_eval import lift_hold_segments
from main.model import KGGTModel

NOTEBOOKS = ["main.ipynb", "main_1stage.ipynb"]
NAMES = ["AD", "BR", "FD", "ED", "FDI"]

# ── tiny models ───────────────────────────────────────────────────────────────

WIN = 40
EEG_DIM, KIN_DIM, N_CH, GAT_HEADS = 6, 8, 5, 2
_TRANSFORMER_CFG = {"n_layers": 2, "n_heads": 2, "d_model": 16, "d_k": 8, "d_v": 8,
                    "ffn_dim": 32, "dropout": 0.1, "chunk_size": WIN}
FORCE_COLS, VEL_COLS = [3, 7], [4, 5, 6]            # skip highway sees columns 3..7


def _tiny_model(guided: bool = True, seed: int = 0) -> KGGTModel:
    """Random KG-GT with every zero-initialised branch given weights (GAT output,
    kin skip, kin edge bias, slope decoder, lag kernel)."""
    torch.manual_seed(seed)
    kw = dict(gat_kin_dim_override=4, kin_skip_indices=[3, 4, 5, 6, 7]) if guided else {}
    m = KGGTModel(transformer_cfg=_TRANSFORMER_CFG, input_dim=EEG_DIM, node_dim=16, gat_hidden_dim=16,
                  gat_heads=GAT_HEADS, out_channels=N_CH, kin_dim=KIN_DIM,
                  use_kinematic_guidance=guided, **kw)
    with torch.no_grad():
        if guided:
            m.kin_skip_proj[-1].weight.normal_(0.0, 0.3)
            m.gat.kin_edge_linear.weight.normal_(0.0, 0.8)
            m.gat.out.weight.normal_(0.0, 0.3)
            m.gat.out.bias.normal_(0.0, 0.1)
        else:
            for layer in m.gat.layers:
                layer.out.weight.normal_(0.0, 0.3)
        for layer in m.decoder.slope_layers:
            layer.weight.normal_(0.0, 0.3)
            layer.bias.normal_(0.0, 0.1)
        m.lag_align.conv.weight.normal_(0.0, 0.05)
    return m.eval()


def _inputs(B: int = 3, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, WIN, EEG_DIM, generator=g), torch.randn(B, WIN, KIN_DIM, generator=g)


def _series(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, EEG_DIM)).astype(np.float32),
            rng.standard_normal((T, KIN_DIM + 1)).astype(np.float32),
            rng.random((T, N_CH)).astype(np.float32))


EMG_MEAN = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
EMG_STD = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8])
KIN_MEAN = torch.linspace(-0.5, 0.5, KIN_DIM + 1)
KIN_STD = torch.linspace(0.5, 2.0, KIN_DIM + 1)


def _pb(eeg, kin, emg):
    """Shape of the notebook's prepare_batch: kin z-score + column drop, EMG z-score."""
    kn = ((kin - KIN_MEAN) / KIN_STD)[..., [i for i in range(KIN_DIM + 1) if i != 2]]
    return {"eeg": eeg, "kin": kn}, (emg - EMG_MEAN) / EMG_STD


class PlantedEEG(nn.Module):
    """Muscle 0 = mean of electrode 3 over samples 40..49 (held over the whole window);
    muscle 1 = first kinematic input.  Nothing else of the EEG is used."""

    def forward(self, eeg, kin=None):
        v = eeg[:, 40:50, 3].mean(dim=1, keepdim=True).expand(-1, eeg.shape[1])
        k = kin[..., 0] if kin is not None else torch.zeros_like(v)
        return torch.stack([v, k], dim=-1)


class IgnoresEEG(nn.Module):
    def forward(self, eeg, kin=None):
        return kin[..., :2] * 1.5 + 0.25


class UsesEEG(nn.Module):
    def forward(self, eeg, kin=None):
        return eeg[..., :2] + 0.1 * kin[..., :2]


# ════════════════════════════════════════════════════════════════════════════
# 1. Exact path decomposition
# ════════════════════════════════════════════════════════════════════════════

def test_decomposition_sums_to_model_output_kin_guided():
    model = _tiny_model()
    eeg, kin = _inputs()
    with torch.no_grad():
        ref = model(eeg, kin).numpy()
    parts = decompose_prediction(model, eeg, kin, force_cols=FORCE_COLS, vel_cols=VEL_COLS)
    assert parts["total"].dtype == np.float32 and parts["total"].shape == (3, WIN, N_CH)
    np.testing.assert_allclose(parts["total"], ref, atol=1e-6)
    np.testing.assert_allclose(parts["graph"] + parts["kin"], ref, atol=1e-5)
    assert np.abs(parts["kin"]).max() > 1e-2 and np.abs(parts["graph"]).max() > 1e-2
    # the kin path holds no constant: kin skip = 0 -> exactly the decoder bias stays in the graph path
    assert parts["decoder_bias"].shape == ref.shape
    assert parts["gat_attn"].shape == (3, WIN, N_CH, N_CH)
    np.testing.assert_allclose(parts["gat_attn"].sum(-1), 1.0, atol=1e-5)     # after softmax


def test_decomposition_non_guided_model_has_zero_kin_part():
    model = _tiny_model(guided=False)
    eeg, kin = _inputs()
    with torch.no_grad():
        ref = model(eeg, kin).numpy()
    for k_in in (kin, None):
        parts = decompose_prediction(model, eeg, k_in, force_cols=FORCE_COLS, vel_cols=VEL_COLS)
        np.testing.assert_allclose(parts["graph"] + parts["kin"], ref, atol=1e-5)
        for key in ("kin", "kin_force_effect", "kin_velocity_effect"):
            assert not parts[key].any(), key
        np.testing.assert_array_equal(parts["graph_no_kin_bias"], parts["graph"])


def test_decomposition_accepts_one_window_and_leaves_the_model_unchanged():
    model = _tiny_model()
    eeg, kin = _inputs(B=1)
    with torch.no_grad():
        before = model(eeg, kin).numpy()
    parts = decompose_prediction(model, eeg[0], kin[0])
    assert parts["total"].shape == (1, WIN, N_CH)
    assert not model.gat.kin_edge_linear._forward_hooks            # the zeroing hook is gone
    with torch.no_grad():
        np.testing.assert_array_equal(model(eeg, kin).numpy(), before)
    np.testing.assert_allclose(parts["total"], before, atol=1e-6)


def test_occlusion_effects_sanity():
    model = _tiny_model()
    eeg, kin = _inputs()
    full = decompose_prediction(model, eeg, kin, force_cols=FORCE_COLS, vel_cols=VEL_COLS)
    assert np.abs(full["kin_force_effect"]).max() > 1e-3
    assert np.abs(full["kin_velocity_effect"]).max() > 1e-3
    # no columns / columns the skip highway does not see -> no effect
    none = decompose_prediction(model, eeg, kin)
    assert not none["kin_force_effect"].any() and not none["kin_velocity_effect"].any()
    outside = decompose_prediction(model, eeg, kin, force_cols=[0, 1, 2])
    np.testing.assert_allclose(outside["kin_force_effect"], 0.0, atol=1e-6)
    # occluding with the value the input already has changes nothing
    base = torch.full((KIN_DIM,), 0.7)
    kin_c = kin.clone()
    kin_c[..., FORCE_COLS] = 0.7
    same = decompose_prediction(model, eeg, kin_c, force_cols=FORCE_COLS, kin_baseline=base)
    np.testing.assert_allclose(same["kin_force_effect"], 0.0, atol=1e-6)
    # effect of ALL skip inputs = kin path − kin path of the baseline input (sign: full − occluded)
    all_cols = [3, 4, 5, 6, 7]
    eff = decompose_prediction(model, eeg, kin, force_cols=all_cols)["kin_force_effect"]
    kin_b = kin.clone()
    kin_b[..., all_cols] = 0.0
    np.testing.assert_allclose(eff, full["kin"] - decompose_prediction(model, eeg, kin_b)["kin"], atol=1e-5)
    # the two effects are occlusions of a non-linear MLP: they need not add up to the kin path
    assert full["kin_force_effect"].shape == full["kin"].shape


def test_graph_path_without_kin_edge_bias():
    model = _tiny_model()
    eeg, kin = _inputs()
    parts = decompose_prediction(model, eeg, kin)
    assert np.abs(parts["graph_no_kin_bias"] - parts["graph"]).max() > 1e-4
    # it does not depend on the kinematics any more
    other = decompose_prediction(model, eeg, torch.randn_like(kin))
    np.testing.assert_allclose(other["graph_no_kin_bias"], parts["graph_no_kin_bias"], atol=1e-6)
    assert np.abs(other["graph"] - parts["graph"]).max() > 1e-4
    with torch.no_grad():                               # no kinematic edge bias in the model at all
        model.gat.kin_edge_linear.weight.zero_()
        model.gat.kin_edge_linear.bias.zero_()
    flat = decompose_prediction(model, eeg, kin)
    np.testing.assert_allclose(flat["graph_no_kin_bias"], flat["graph"], atol=1e-6)


def test_decompose_series_matches_the_stitched_prediction():
    model = _tiny_model()
    T = 3 * WIN + 13                                    # tail -> one end-aligned window
    eeg, kin, _ = _series(T)
    pred = predict_series_stitched(model, eeg, kin, _pb, window=WIN, batch_size=2)
    dec = decompose_series(model, eeg, kin, _pb, window=WIN, batch_size=2,
                           force_cols=FORCE_COLS, vel_cols=VEL_COLS)
    assert dec["total"].shape == (T, N_CH) and dec["gat_attn"].shape == (T, N_CH, N_CH)
    np.testing.assert_allclose(dec["total"], pred, atol=1e-6)
    np.testing.assert_allclose(dec["graph"] + dec["kin"], pred, atol=1e-5)
    assert dec["window"] == WIN and list(dec["starts"]) == [0, WIN, 2 * WIN, T - WIN]


def test_kin_column_groups_and_raw_zero_baseline():
    from main.preprocessing_emg_kin import POSE_FORCE_RATE_DROP_INDICES, kin_feature_names
    names = kin_feature_names(True, False, POSE_FORCE_RATE_DROP_INDICES, gated_force=True)
    g = kin_column_groups(names)
    assert g["force"] == [10, 11, 22, 23]               # gated F_L, F_G and their rates
    assert g["velocity"] == list(range(12, 22))         # the 10 pose velocities
    base = raw_zero_kin_baseline(_pb, EEG_DIM, KIN_DIM + 1, N_CH)
    keep = [i for i in range(KIN_DIM + 1) if i != 2]
    np.testing.assert_allclose(base.numpy(), (-KIN_MEAN / KIN_STD).numpy()[keep], atol=1e-6)


# ════════════════════════════════════════════════════════════════════════════
# 2. Attention rollout
# ════════════════════════════════════════════════════════════════════════════

def test_attention_rollout_hand_example():
    layer1 = [[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.5, 0.5]]]   # head mean = all 0.5
    layer2 = [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
    attn = np.array([layer1, layer2])                                # (L=2, H=2, T=2, T=2)
    # Â1 = 0.5·[[.5,.5],[.5,.5]] + 0.5·I = [[.75,.25],[.25,.75]]
    # Â2 = 0.5·[[1,0],[1,0]]     + 0.5·I = [[1,0],[.5,.5]]
    # R  = Â2 · Â1 = [[.75,.25],[.5,.5]]
    expected = np.array([[0.75, 0.25], [0.5, 0.5]])
    roll = attention_rollout(attn)
    assert isinstance(roll, np.ndarray)
    np.testing.assert_allclose(roll, expected, atol=1e-12)
    np.testing.assert_allclose(roll.sum(-1), 1.0)
    t = attention_rollout(torch.tensor(attn, dtype=torch.float32)[None].repeat(3, 1, 1, 1, 1))
    assert torch.is_tensor(t) and t.shape == (3, 2, 2)
    np.testing.assert_allclose(t[1].numpy(), expected, atol=1e-6)
    np.testing.assert_allclose(attention_rollout(attn.mean(1), heads_averaged=True), expected)
    # residual = 0 -> plain product of the head-averaged matrices
    np.testing.assert_allclose(attention_rollout(attn, residual=0.0), attn[1].mean(0) @ attn[0].mean(0))
    with pytest.raises(ValueError):
        attention_rollout(np.zeros((2, 3)))


def test_rollout_reads_the_weights_used_in_the_forward_pass():
    """mhsa.last_attn_weights is softmax(scores / attention_temperature), but the output
    is computed at temperature 1: the rollout must read the temperature-1 weights."""
    model = _tiny_model()
    eeg, _ = _inputs(B=2)
    ref = encoder_attention(model, eeg)
    assert ref.shape == (2, 2, 2, WIN, WIN)
    np.testing.assert_allclose(ref.sum(-1).numpy(), 1.0, atol=1e-5)
    for layer in model.encoder.layers:
        layer.mhsa.attention_temperature = 0.25                 # e.g. set by an old notebook cell
    attn = encoder_attention(model, eeg)
    np.testing.assert_allclose(attn.numpy(), ref.numpy(), atol=1e-6)
    assert all(layer.mhsa.attention_temperature == 0.25 for layer in model.encoder.layers)   # restored
    assert all(layer.mhsa.last_attn_weights is None for layer in model.encoder.layers)       # not kept
    # these weights reproduce the layer's actual output
    mhsa = model.encoder.layers[0].mhsa
    with torch.no_grad():
        z = model.encoder.pos_enc(model.encoder.embed(eeg))
        v = mhsa.W_V(z).view(2, WIN, mhsa.n_heads, mhsa.d_v).transpose(1, 2)
        manual = mhsa.W_O((attn[:, 0] @ v).transpose(1, 2).reshape(2, WIN, -1))
        np.testing.assert_allclose(manual.numpy(), mhsa(z).numpy(), atol=1e-5)
    np.testing.assert_allclose(encoder_attention(model, eeg, average_heads=True).numpy(),
                               ref.mean(2).numpy(), atol=1e-6)
    rows = rollout_rows(model, eeg, query=WIN // 2, batch_size=1)
    assert rows.shape == (2, WIN)
    np.testing.assert_allclose(rows.sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(rows, attention_rollout(ref.numpy())[:, WIN // 2], atol=1e-5)


# ════════════════════════════════════════════════════════════════════════════
# 3. EEG occlusion attribution, deletion curve, effect size
# ════════════════════════════════════════════════════════════════════════════

def _planted_inputs(B: int = 4, W: int = 100, E: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    eeg = torch.randn(B, W, E, generator=g)
    eeg[:, 40:50, 3] += 3.0                              # the planted piece is above the window mean
    return eeg, torch.randn(B, W, 2, generator=g)


def test_occlusion_attribution_finds_the_planted_dependence():
    model = PlantedEEG()
    eeg, kin = _planted_inputs()
    occ = eeg_occlusion_attribution(model, eeg, kin, time_bin=10, query=70)
    assert occ["time_abs"].shape == (4, 10, 2) and occ["electrode_abs"].shape == (4, 8, 2)
    assert occ["bins"].tolist()[4] == [40, 50] and occ["bin_centers"][4] == 44.5 and occ["query"] == 70
    t_abs = occ["time_abs"][..., 0]
    assert (t_abs.argmax(axis=1) == 4).all()
    assert np.delete(t_abs, 4, axis=1).max() < 1e-7 and t_abs[:, 4].min() > 0.5
    e_abs = occ["electrode_abs"][..., 0]
    assert (e_abs.argmax(axis=1) == 3).all() and np.delete(e_abs, 3, axis=1).max() < 1e-7
    # signed change at the query: original − occluded > 0 (the piece was above the mean)
    assert (occ["time_signed"][:, 4, 0] > 0).all() and (occ["electrode_signed"][:, 3, 0] > 0).all()
    # the kinematics-only muscle does not react to any EEG occlusion
    assert np.abs(occ["time_abs"][..., 1]).max() < 1e-7 and np.abs(occ["electrode_abs"][..., 1]).max() < 1e-7
    # electrode groups
    grp = eeg_occlusion_attribution(model, eeg, kin, time_bin=10,
                                    electrode_groups={"left": [0, 1, 2], "mid": [3, 4], "right": [5, 6, 7]})
    assert grp["electrode_labels"] == ["left", "mid", "right"]
    assert (grp["electrode_abs"][..., 0].argmax(axis=1) == 1).all()


def test_deletion_curve_beats_random_for_the_planted_model():
    model = PlantedEEG()
    eeg, kin = _planted_inputs()
    d = deletion_curve(model, eeg, kin, time_bin=10, fractions=(0.1, 0.2, 0.5, 1.0), n_random=4, seed=0)
    assert d["fractions"].tolist() == [0.0, 0.1, 0.2, 0.5, 1.0]
    assert d["top"][0] == 0.0 and d["random"][0] == 0.0
    np.testing.assert_allclose(d["top"][1], d["full"], rtol=1e-5)       # the one bin that matters goes first
    assert d["top"][1] > d["random"][1] + 1e-3
    assert (d["top"] >= d["random"] - 1e-9).all()
    assert d["area"] > 0 and d["score"] > 0.2
    # a wrong attribution (planted bin ranked last) is worse than random
    wrong = np.ones((4, 10))
    wrong[:, 4] = 0.0
    bad = deletion_curve(model, eeg, kin, attribution=wrong, time_bin=10,
                         fractions=(0.1, 0.2, 0.5, 1.0), n_random=4, seed=0)
    assert bad["score"] < 0 < d["score"]
    # a model that ignores the EEG: nothing to be faithful to
    flat = deletion_curve(IgnoresEEG(), eeg, kin, time_bin=10, n_random=2)
    assert flat["score"] == 0.0 and flat["full"] < 1e-9
    with pytest.raises(ValueError):
        deletion_curve(model, eeg, kin, attribution=np.ones((4, 3)), time_bin=10)


def test_eeg_effect_size_separates_models_that_use_the_eeg():
    eeg, kin = _planted_inputs(B=6, W=50)
    assert eeg_effect_size(IgnoresEEG(), (eeg, kin)) < 1e-9
    used = eeg_effect_size(UsesEEG(), (eeg, kin))
    assert used > 0.5                                    # shuffling moves the output about as much as its SD
    per = eeg_effect_size(UsesEEG(), (eeg, kin), per_channel=True)
    assert per.shape == (2,) and np.isclose(per.mean(), used)
    # the three accepted input forms give the same number
    as_dict = eeg_effect_size(UsesEEG(), {"eeg": eeg, "kin": kin})
    batches = eeg_effect_size(UsesEEG(), [{"eeg": eeg[:2], "kin": kin[:2]}, (eeg[2:], kin[2:])])
    assert np.isclose(as_dict, used) and np.isclose(batches, used)
    assert eeg_effect_size(_tiny_model(), _inputs(B=4)) > 0.02
    with pytest.raises(ValueError):
        eeg_effect_size(UsesEEG(), [])


# ════════════════════════════════════════════════════════════════════════════
# 4. Task events
# ════════════════════════════════════════════════════════════════════════════

_COLS = ["Part", "Run", "Lift", "CurW", "CurS", "StartTime", "LEDOn", "LEDOff", "tHandStart",
         "tFirstDigitTouch", "tBothDigitTouch", "tLiftOff", "tReplace", "tBothReleased", "tHandStop"]


def _fake_alllifts(tmp_path: Path, rows: list[list[float]], participant: int = 7) -> Path:
    d = tmp_path / f"P{participant}"
    d.mkdir(parents=True, exist_ok=True)
    P = {"ColNames": np.array(_COLS, dtype=object), "AllLifts": np.asarray(rows, dtype=float)}
    sio.savemat(str(d / f"P{participant}_AllLifts.mat"), {"P": P})
    return tmp_path


def test_lift_events_on_a_fake_alllifts(tmp_path):
    nan = float("nan")
    rows = [
        # Part Run Lift W  S  Start  LEDOn LEDOff hand  touch1 touch2 lift  repl  rel   stop
        [7, 3, 1, 4, 2, 2.002, 2.0, 6.0, 2.5, 3.1, 3.12, 3.5, 6.5, 6.9, 7.4],
        [7, 3, 2, 1, 3, 12.0, 2.0, 6.0, 2.4, nan, 3.0, 3.4, 6.4, 6.8, 7.3],      # no first touch
        [7, 4, 1, 2, 1, 5.0, 2.0, 6.0, 2.5, 3.0, 3.1, 3.5, 6.5, 6.9, 7.4],       # other series
    ]
    root = _fake_alllifts(tmp_path, rows)
    ev = lift_events(7, 3, fs=500.0, data_dir=root)
    assert [e["lift"] for e in ev] == [1, 2] and [e["CurW"] for e in ev] == [4, 1]
    assert [e["CurS"] for e in ev] == [2, 3]
    e0 = ev[0]
    assert set(EVENT_COLUMNS) <= set(e0)
    # t0 = StartTime − LEDOn = 0.002 s; sample = round((t0 + t_event) · fs)
    assert e0["t0"] == pytest.approx(0.002)
    assert e0["led_on"] == 1001 and e0["first_touch"] == 1551 and e0["both_touch"] == 1561
    assert e0["lift_off"] == 1751 and e0["replace"] == 3251 and e0["release"] == 3451
    assert e0["hand_start"] == 1251 and e0["hand_stop"] == 3701 and e0["led_off"] == 3001
    assert ev[1]["first_touch"] is None and ev[1]["lift_off"] == int(round((10.0 + 3.4) * 500))
    assert event_samples(ev, "lift_off") == [1751, 6700]
    assert event_samples(ev, "first_touch") == [1551, None]
    with pytest.raises(KeyError):
        event_samples(ev, "nonsense")
    # same time base as the validated hold-phase recipe
    seg = lift_hold_segments(7, 3, fs=500.0, data_dir=root, margin_s=0.3)
    assert seg[0]["start"] == int((0.002 + 3.5 + 0.3) * 500) and abs(seg[0]["start"] - (e0["lift_off"] + 150)) <= 1
    # events outside the series become None
    short = lift_events(7, 3, fs=500.0, data_dir=root, n_samples=3300)
    assert short[0]["replace"] == 3251 and short[0]["release"] is None and short[1]["lift_off"] is None
    assert lift_events(7, 9, data_dir=root) == []
    with pytest.raises(FileNotFoundError):
        lift_events(8, 1, data_dir=root)


def test_phase_levels_on_a_synthetic_envelope(tmp_path):
    rows = [[7, 1, 1, 2, 1, 4.0, 2.0, 6.0, 2.5, 3.0, 3.1, 3.5, 6.5, 6.9, 7.4]]
    ev = lift_events(7, 1, fs=100.0, data_dir=_fake_alllifts(tmp_path, rows))
    env = np.full((1200, 2), 0.1)
    env[ev[0]["lift_off"]:ev[0]["replace"]] = [1.0, 2.0]
    lv = phase_levels(env, ev, fs=100.0)
    assert set(lv) == {"rest_before", "reach", "load", "hold", "release", "rest_after"}
    np.testing.assert_allclose(lv["hold"], [[1.0, 2.0]])
    for k in ("rest_before", "reach", "load", "release", "rest_after"):
        np.testing.assert_allclose(lv[k], 0.1)
    assert np.isnan(phase_levels(env[:700], ev, fs=100.0)["rest_after"]).all()      # phase outside the array


_CACHE = ROOT / "data" / "cache_main_stride500_square_pose_force_sg51" / "P1_test_S4.npz"


@pytest.mark.skipif(not ((DEFAULT_RAW_DIR / "P1" / "P1_AllLifts.mat").exists() and _CACHE.exists()),
                    reason="raw data / cache not present")
def test_lift_events_real_data_fdi_envelope_follows_the_task():
    emg = np.load(_CACHE)["emg"]                          # (T, 5) preprocessed envelope at 500 Hz
    ev = lift_events(1, 4, fs=500.0, n_samples=len(emg))
    assert len(ev) >= 20
    order = ("led_on", "hand_start", "first_touch", "lift_off", "replace", "release", "hand_stop")
    ok = 0
    for e in ev:
        s = [e[k] for k in order]
        ok += all(x is not None for x in s) and all(a < b for a, b in zip(s, s[1:]))
    assert ok >= 0.9 * len(ev)
    lv = {k: np.nanmean(v[:, 4]) for k, v in phase_levels(emg, ev, fs=500.0).items()}   # channel 4 = FDI
    assert lv["load"] > 2.0 * lv["rest_before"]           # rises around touch / lift-off
    assert lv["hold"] > 2.0 * lv["rest_before"]
    assert lv["rest_after"] < 0.5 * lv["hold"]            # falls after release


# ════════════════════════════════════════════════════════════════════════════
# 5. Event-locked analysis
# ════════════════════════════════════════════════════════════════════════════

def test_extract_event_windows_alignment():
    x = np.arange(100, dtype=np.float32)[:, None] * np.ones((1, 2), dtype=np.float32)
    w, kept = extract_event_windows(x, [50, None, 5, 97, 20], pre=10, post=6)
    assert kept.tolist() == [0, 4] and w.shape == (2, 16, 2)
    assert w[0, 10, 0] == 50 and w[1, 10, 0] == 20 and w[0, 0, 0] == 40 and w[0, -1, 0] == 55
    w0, kept0 = extract_event_windows(x, [None, 2], pre=10, post=6)
    assert w0.shape == (0, 16, 2) and kept0.size == 0
    w_end, kept_end = extract_event_windows(x, [94, 95], pre=10, post=6)      # 94 + 6 = 100 just fits
    assert kept_end.tolist() == [0] and w_end[0, -1, 0] == 99


def _event_locked(model=None, T=400, samples=(60, None, 150, 395, 240, 10), **kw):
    model = model or _tiny_model()
    eeg, kin, emg = _series(T, seed=3)
    meta = [{"lift": i + 1, "CurW": (1, 2, 4)[i % 3]} for i in range(len(samples))]
    ev = event_locked_windows(model, eeg, kin, emg, list(samples), _pb, pre=WIN // 2, post=WIN // 2,
                              meta=meta, emg_mean=EMG_MEAN, emg_std=EMG_STD, force_cols=FORCE_COLS,
                              vel_cols=VEL_COLS, time_bin=10, batch_size=2,
                              deletion_kwargs={"fractions": (0.25, 0.5, 1.0), "n_random": 2}, **kw)
    return model, (eeg, kin, emg), ev


def test_event_locked_windows_shapes_and_alignment():
    model, (eeg, kin, emg), ev = _event_locked()
    pre = WIN // 2
    assert ev["kept"].tolist() == [0, 2, 4] and ev["event_samples"].tolist() == [60, 150, 240]
    assert [m["lift"] for m in ev["meta"]] == [1, 3, 5]
    assert (ev["pre"], ev["post"], ev["window"], ev["query"]) == (pre, pre, WIN, pre)
    n = 3
    for k in ("pred", "target", "pred_z", "target_z", "graph", "kin", "graph_no_kin_bias",
              "kin_force_effect", "kin_velocity_effect"):
        assert ev[k].shape == (n, WIN, N_CH), k
    assert ev["rollout"].shape == (n, WIN)
    assert ev["occ_time_abs"].shape == (n, 4, N_CH) and ev["occ_time_signed"].shape == (n, 4, N_CH)
    assert ev["occ_electrode_abs"].shape == (n, EEG_DIM, N_CH)
    np.testing.assert_allclose(ev["bin_centers"], [-15.5, -5.5, 4.5, 14.5])     # relative to the event
    # alignment: index `pre` of window i is the event sample
    for i, s in enumerate((60, 150, 240)):
        np.testing.assert_array_equal(ev["target"][i], emg[s - pre:s + pre])
        np.testing.assert_array_equal(ev["target"][i, pre], emg[s])
        inp, y = _pb(torch.as_tensor(eeg[s - pre:s + pre])[None], torch.as_tensor(kin[s - pre:s + pre])[None],
                     torch.as_tensor(emg[s - pre:s + pre])[None])
        with torch.no_grad():
            ref = model(inp["eeg"], inp["kin"])[0].numpy()
        np.testing.assert_allclose(ev["pred_z"][i], ref, atol=1e-6)
        np.testing.assert_allclose(ev["target_z"][i], y[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(ev["graph"] + ev["kin"], ev["pred_z"], atol=1e-5)
    np.testing.assert_allclose(ev["pred"], ev["pred_z"] * EMG_STD.numpy() + EMG_MEAN.numpy(), atol=1e-6)
    np.testing.assert_allclose(ev["rollout"].sum(-1), 1.0, atol=1e-5)
    d = ev["deletion"]
    assert d["fractions"].tolist() == [0.0, 0.25, 0.5, 1.0] and d["top_per_window"].shape == (n, 3)
    assert np.isfinite(d["score"])


def test_event_locked_windows_switches_and_stub_model():
    _, _, ev = _event_locked(rollout=False, occlusion=False, deletion=False)
    assert "rollout" not in ev and "occ_time_abs" not in ev and ev["deletion"] is None
    # a model that cannot be decomposed still gives prediction + EEG evidence
    eeg, kin, emg = _series(300, seed=5)
    ev = event_locked_windows(UsesEEG(), eeg, kin, emg[:, :2], [100, 200], _pb_two, pre=20, post=20, time_bin=10)
    assert ev["pred"].shape == (2, 40, 2) and "graph" not in ev and "rollout" not in ev
    assert ev["occ_time_abs"].shape == (2, 4, 2) and ev["units"].startswith("normalised")
    # no usable event at all
    empty = event_locked_windows(UsesEEG(), eeg, kin, emg[:, :2], [None, 5], _pb_two, pre=20, post=20)
    assert empty["pred"].shape == (0, 40, 2) and empty["kept"].size == 0 and empty["deletion"] is None
    assert event_locked_summary(empty)["n"] == 0


def _pb_two(eeg, kin, emg):
    return {"eeg": eeg, "kin": kin}, emg


def test_event_locked_summary_and_concat():
    _, _, a = _event_locked()
    _, _, b = _event_locked(samples=(100, 300))
    both = concat_event_locked([a, b])
    assert both["pred"].shape == (5, WIN, N_CH) and len(both["meta"]) == 5
    assert both["deletion"]["top_per_window"].shape == (5, 3)
    np.testing.assert_allclose(both["pred"][:3], a["pred"])
    s = event_locked_summary(both, group_by="CurW")
    assert s["n"] == 5 and s["t"].tolist() == list(range(-WIN // 2, WIN // 2))
    for k in ("pred", "target", "graph", "kin", "kin_force_effect", "graph_no_kin_bias"):
        assert s[k + "_mean"].shape == (WIN, N_CH) and s[k + "_sd"].shape == (WIN, N_CH), k
    np.testing.assert_allclose(s["pred_mean"], both["pred"].mean(0), atol=1e-6)
    np.testing.assert_allclose(s["pred_sd"], both["pred"].std(0), atol=1e-6)
    assert s["rollout_mean"].shape == (WIN,) and s["occ_time_abs_mean"].shape == (4, N_CH)
    assert s["occ_electrode_abs_mean"].shape == (EEG_DIM, N_CH)
    assert s["group_by"] == "CurW" and sorted(s["groups"]) == [1, 2, 4]
    assert sum(g["n"] for g in s["groups"].values()) == 5
    w1 = [i for i, m in enumerate(both["meta"]) if m["CurW"] == 1]
    np.testing.assert_allclose(s["groups"][1]["target_mean"], both["target"][w1].mean(0), atol=1e-6)
    assert event_locked_summary(both)["groups"] == {}
    with pytest.raises(ValueError):
        concat_event_locked([])


# ════════════════════════════════════════════════════════════════════════════
# 6. Figures
# ════════════════════════════════════════════════════════════════════════════

def _all_text(fig) -> str:
    out = []
    for ax in fig.axes:
        out += [ax.get_title(loc="left"), ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.texts]
        leg = ax.get_legend()
        if leg is not None:
            out += [t.get_text() for t in leg.get_texts()]
    if fig._suptitle is not None:
        out.append(fig._suptitle.get_text())
    return " ".join(" ".join(out).split())


def test_plot_interp_window_on_synthetic_data_without_warnings(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.interp_plots import plot_interp_window

    model = _tiny_model()
    T, fs = 6 * WIN, 20.0
    eeg, kin, emg = _series(T, seed=2)
    pred = predict_series_stitched(model, eeg, kin, _pb, window=WIN) * EMG_STD.numpy() + EMG_MEAN.numpy()
    dec = decompose_series(model, eeg, kin, _pb, window=WIN, force_cols=FORCE_COLS, vel_cols=VEL_COLS)
    gate = np.zeros(T)
    gate[60:150] = 1.0
    events = [{"led_on": 1040, "first_touch": 1065, "lift_off": 1075, "replace": 1130, "release": 1140,
               "hand_start": 1050, "hand_stop": 1160}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_interp_window(pred, emg, dec, NAMES, fs=fs, start=1000, gat_attn=dec["gat_attn"],
                                 events=events, gate=gate, model_window=WIN, title="synthetic")
        fig.savefig(tmp_path / "window.png", dpi=50)
        # same filtering for both curves when smoothing is asked for; no gate, no events, no attention
        fig2 = plot_interp_window(pred, emg, {"graph": dec["graph"], "kin": dec["kin"]}, NAMES, fs=fs,
                                  smooth_hz=4.0)
        fig2.savefig(tmp_path / "window2.png", dpi=50)
    text = _all_text(fig)
    assert "Panel 1" in text and "Panel 2" in text and "Panel 3" in text
    assert "it is not EEG attention" in text and "after softmax" in text and "rest mean" in text
    assert "r and RMSE on the unfiltered signals" in text and "no filtering" in text
    for label in ("LED on", "first digit touch", "lift-off", "replace", "release", "hand near object"):
        assert label in text, label
    for name in NAMES:
        assert name in text
    # Panel 1: r / RMSE of the unfiltered signals, also when the display is smoothed
    r0 = np.corrcoef(pred[:, 0], emg[:, 0])[0, 1]
    assert f"r = {r0:.3f}" in text and f"r = {r0:.3f}" in _all_text(fig2)
    assert "both low-passed at 4 Hz" in _all_text(fig2) and "not available" in _all_text(fig2)
    lines = fig2.axes[0].get_lines()                       # actual + predicted, both filtered
    assert not np.allclose(lines[0].get_ydata(), emg[:, 0]) and not np.allclose(lines[1].get_ydata(), pred[:, 0])
    # time axis = series time
    assert fig.axes[0].get_lines()[0].get_xdata()[0] == pytest.approx(1000 / fs)
    plt.close(fig)
    plt.close(fig2)


@pytest.mark.parametrize("by_group", [False, True])
def test_plot_event_locked_both_branches_without_warnings(tmp_path, by_group):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.interp_plots import plot_event_locked

    _, _, ev = _event_locked()
    summ = event_locked_summary(ev, group_by="CurW")
    eeg_names = [f"C{i}" for i in range(EEG_DIM)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shown = plot_event_locked(summ, NAMES, fs=20.0, event_label="lift-off", eeg_effect=0.4,
                                  eeg_names=eeg_names, by_group=by_group, title="synthetic")
        shown.savefig(tmp_path / "shown.png", dpi=50)
        note = plot_event_locked(summ, NAMES, fs=20.0, event_label="lift-off", eeg_effect=0.0031,
                                 by_group=by_group)
        note.savefig(tmp_path / "note.png", dpi=50)
        unmeasured = plot_event_locked(summ, NAMES, fs=20.0)
    t_shown, t_note = _all_text(shown), _all_text(note)
    assert "effect size = 0.4" in t_shown and "deletion-curve faithfulness" in t_shown
    assert "Attention rollout" in t_shown and "EEG occlusion" in t_shown and "C3" in "".join(
        lab.get_text() for ax in shown.axes for lab in ax.get_xticklabels())
    assert ("EEG input has no measurable effect on this model's output (effect size = 0.0031); "
            "attention is not shown as an explanation") in t_note
    assert "Attention rollout" not in t_note
    assert "effect size = not measured" in _all_text(unmeasured)
    for t in (t_shown, t_note):
        assert "occlusion effects, not exact shares" in t and "lift-off" in t
        assert ("weight 1" in t) == by_group
    assert len(shown.axes) > len(note.axes)
    for f in (shown, note, unmeasured):
        plt.close(f)


def test_old_triptych_is_still_available():
    import main
    from main.plots import plot_interpretability_triptych
    assert main.plot_interpretability_triptych is plot_interpretability_triptych
    for name in ("plot_interp_window", "plot_event_locked", "decompose_prediction", "attention_rollout",
                 "eeg_occlusion_attribution", "deletion_curve", "eeg_effect_size", "lift_events",
                 "event_locked_windows", "event_locked_summary"):
        assert name in main.__all__ and callable(getattr(main, name)), name


# ════════════════════════════════════════════════════════════════════════════
# 7. Notebooks
# ════════════════════════════════════════════════════════════════════════════

def _nb(name: str) -> dict:
    with open(ROOT / name, encoding="utf-8") as f:
        return json.load(f)


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _config(nb: dict) -> dict:
    ns = {"copy": copy}
    exec(next(_src(c) for c in nb["cells"] if c["cell_type"] == "code" and "CONFIG = {" in _src(c)), ns)
    return ns["CONFIG"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_code_cells_compile(name):
    for i, c in enumerate(_nb(name)["cells"]):
        if c["cell_type"] == "code":
            code = "\n".join("pass" if re.match(r"^\s*[%!]", ln) else ln for ln in _src(c).splitlines())
            compile(code, f"{name}[cell {i}]", "exec")


@pytest.mark.parametrize("name,suffix", [("main.ipynb", ""), ("main_1stage.ipynb", "_1stage")])
def test_notebook_interpretability_cells_in_order(name, suffix):
    cells = _nb(name)["cells"]
    i = next(k for k, c in enumerate(cells)
             if c["cell_type"] == "code" and "HOW MUCH DOES THE MODEL USE EEG vs KINEMATICS?" in _src(c))
    note, interp, event = cells[i + 1], cells[i + 2], cells[i + 3]
    # markdown: what every panel is and is not
    assert note["cell_type"] == "markdown"
    md = _src(note)
    for needle in ("Panel 1", "Panel 2", "Panel 3", "is not", "exact", "occlusion", "rollout",
                   "effect size", "1 s", "window"):
        assert needle in md, needle
    assert "INTERPRETABILITY — Full-Signal Triptych" not in md           # the generator looks this up
    # per-window cell
    s = _src(interp)
    assert interp["cell_type"] == "code" and "INTERPRETABILITY — Full-Signal Triptych" in s
    assert "plot_interp_window(" in s and "decompose_series(" in s and "lift_events(" in s
    assert "plot_interpretability_triptych" not in s and "encoder_key_attn" not in s
    assert "MAX_WINDOWS_PER_SERIES = None" in s
    assert f'os.path.join("outputs", "interpretability{suffix}", _subj)' in s
    assert 'f"series{_real_series:02d}_window{_w:03d}_{_t_start:.0f}s-{_t_end:.0f}s.png"' in s
    assert "smooth_hz=5.0" not in s
    for exported in ("eeg_b ", "kin_b ", "true_np ", "pred_np ", "EMG_NAMES ", "KIN_FEATURE_NAMES "):
        assert exported in s, exported                                     # the diagnostic cells need them
    # event-locked cell, right after
    e = _src(event)
    assert event["cell_type"] == "code"
    assert "Event-locked interpretability (all test lifts)" in e
    for needle in ("plot_event_locked(", "event_locked_windows(", "event_locked_summary(",
                   "eeg_effect_size(", '"lift_off"', '"first_touch"', "per_subject_results"):
        assert needle in e, needle
    assert f'os.path.join("outputs", "interpretability{suffix}_event", _subj)' in e
    other = "" if suffix else "_1stage"
    assert f'"interpretability{other}_event"' not in e
    assert "faithfulness" in e and "effect size" in e
    for cell_src in (s, e):                                               # no training, no checkpoints
        for bad in ("train_model(", "load_checkpoint(", "train_single_stage(", ".backward("):
            assert bad not in cell_src, bad
    # exactly one cell of each kind
    code = [_src(c) for c in cells if c["cell_type"] == "code"]
    assert sum("plot_interp_window(" in c for c in code) == 1
    assert sum("plot_event_locked(" in c for c in code) == 1


def test_both_notebook_configs_identical_for_data_model_loss():
    a, b = _config(_nb("main.ipynb")), _config(_nb("main_1stage.ipynb"))
    for k in ("data", "preprocessing", "dataset", "model"):
        assert a[k] == b[k], k
    assert a["training"]["loss"] == b["training"]["loss"]


def test_1stage_interpretability_cells_are_inherited_from_main():
    a, b = _nb("main.ipynb"), _nb("main_1stage.ipynb")
    for needle in ("plot_interp_window(", "plot_event_locked("):
        x = next(_src(c) for c in a["cells"] if c["cell_type"] == "code" and needle in _src(c)).splitlines()
        y = next(_src(c) for c in b["cells"] if c["cell_type"] == "code" and needle in _src(c)).splitlines()
        assert len(x) == len(y)
        diff = [(p, q) for p, q in zip(x, y) if p != q]
        assert 0 < len(diff) <= 6
        for p, q in diff:
            assert "1stage" in q, (p, q)
