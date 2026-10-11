"""Package 2 — Phase C: does EEG explain what kinematics cannot?

1. KGGTModel.forward(use_eeg=False): the encoder output is replaced by zeros (kinematics-only
   model); the default is unchanged
2. KinOnly / EEGOnly wrappers (so train_model stays unchanged)
3. residual prepare_batch: target = emg_z − frozen kinematics-only prediction
4. ResidualLoss (Huber + 0.2 · (1 − Pearson along time))
5. main.residual_eval: report, null distributions (time shuffle / window swap), per-lift
   analysis, verdict, figures
6. main_1stage_residual.ipynb = output of tools/build_main_residual_notebook.py

Synthetic data, CPU only; no notebook cell is executed except the CONFIG cell (a dict
literal plus path helpers).  Nothing is trained (train_model is only called with 0 epochs,
into tmp_path, to check the optimiser groups and the checkpoint of a wrapped model).
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.losses import CompositeEMGLoss, ResidualLoss  # noqa: E402
from main.model import EEGOnly, KGGTModel, KinOnly  # noqa: E402
from main.residual_eval import (  # noqa: E402
    NULL_KINDS,
    VERDICT_NO,
    VERDICT_YES,
    format_residual_report,
    plot_residual_delta_vaf,
    plot_residual_per_lift,
    predict_residuals,
    random_derangement,
    residual_report,
    residual_verdict,
)
from main.training import (  # noqa: E402
    TrainConfig,
    load_checkpoint,
    residual_prepare_batch_factory,
    train_model,
    zero_prediction_loss,
)

NB_NAME = "main_1stage_residual.ipynb"
RUN_TAG = "p2c"

# ════════════════════════════════════════════════════════════════════════════
# helpers (same toy model as tests/test_phase_b.py)
# ════════════════════════════════════════════════════════════════════════════

WIN, EEG_DIM, KIN_DIM, N_CH = 40, 12, 8, 5
_T_CFG = {"n_layers": 1, "n_heads": 2, "d_model": 16, "d_k": 8, "d_v": 8,
          "ffn_dim": 32, "dropout": 0.0, "chunk_size": WIN}


def _randomise(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give the zero-initialised layers (GAT out, kin skip, slope) real weights."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model


def _kggt(seed=0, **kw) -> KGGTModel:
    kw.setdefault("gat_dropout", 0.0)
    kw.setdefault("use_kinematic_guidance", True)
    torch.manual_seed(seed)
    m = KGGTModel(_T_CFG, input_dim=EEG_DIM, node_dim=8, gat_hidden_dim=8, gat_heads=2,
                  out_channels=N_CH, kin_dim=KIN_DIM, **kw)
    return _randomise(m, seed)


def _batch(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, WIN, EEG_DIM, generator=g), torch.randn(B, WIN, KIN_DIM, generator=g),
            torch.randn(B, WIN, N_CH, generator=g))


def _identity_prepare_batch(eeg, kin, emg):
    return {"eeg": eeg.float(), "kin": kin.float()}, emg.float()


# ════════════════════════════════════════════════════════════════════════════
# 1. use_eeg
# ════════════════════════════════════════════════════════════════════════════

def _old_forward(m: KGGTModel, eeg, kin):
    """The forward pass as it was before use_eeg existed (eval mode, guided model)."""
    nodes = m.node_projection(m.encoder(eeg))
    refined = m.gat(nodes, m.get_gat_kin(kin))
    B, T, _ = kin.shape
    refined = refined + m.kin_skip_proj(m.get_skip_kin(kin)).view(B, T, m.out_channels, -1)
    return m.lag_align(m.decoder(refined).squeeze(-1))


def test_use_eeg_default_is_true_and_changes_nothing():
    assert inspect.signature(KGGTModel.forward).parameters["use_eeg"].default is True
    assert inspect.signature(KGGTModel.forward).parameters["use_kin"].default is True
    eeg, kin, _ = _batch()
    m = _kggt().eval()
    keys = list(m.state_dict().keys())
    state = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        ref = _old_forward(m, eeg, kin)
        assert torch.equal(m(eeg, kin), ref)
        assert torch.equal(m(eeg, kin, use_eeg=True), ref)
        assert torch.equal(m(eeg, kin, use_kin=False), m(eeg, None, use_kin=False, use_eeg=True))
    # the flag adds no parameter / buffer and a forward pass does not touch the weights
    assert list(m.state_dict().keys()) == keys
    assert not any("use_eeg" in k for k in keys)
    assert all(torch.equal(v, state[k]) for k, v in m.state_dict().items())
    # train mode (dropout paths, EEG-only aux): same random numbers with and without the flag
    mt = _kggt(kin_dropout=0.5, eeg_channel_dropout=0.25, eeg_groups=4, eeg_only_aux=True).train()
    torch.manual_seed(3)
    a = mt(eeg, kin)
    aux_a = mt.pop_eeg_only_output()
    torch.manual_seed(3)
    b = mt(eeg, kin, use_eeg=True)
    assert torch.equal(a, b) and torch.equal(aux_a, mt.pop_eeg_only_output())


def test_use_eeg_false_does_not_depend_on_the_eeg_and_uses_the_projection_bias():
    eeg, kin, _ = _batch()
    eeg2 = torch.randn_like(eeg) * 5.0
    m = _kggt().eval()
    with torch.no_grad():
        out = m(eeg, kin, use_eeg=False)
        assert out.shape == (eeg.shape[0], WIN, N_CH) and torch.isfinite(out).all()
        assert torch.equal(out, m(eeg2, kin, use_eeg=False))
        assert not torch.allclose(out, m(eeg, kin))                 # the EEG path did contribute
        assert not torch.allclose(out, m(eeg, torch.randn_like(kin), use_eeg=False))   # kin still used
        # node embeddings = bias of the node projection, the same for every sample and time step
        bias_nodes = m.node_projection.proj.bias.view(1, 1, N_CH, -1).expand(eeg.shape[0], WIN, -1, -1)
        refined = m.gat(bias_nodes, m.get_gat_kin(kin))
        refined = refined + m.kin_skip_proj(m.get_skip_kin(kin)).view(eeg.shape[0], WIN, N_CH, -1)
        assert torch.allclose(out, m.lag_align(m.decoder(refined).squeeze(-1)), atol=1e-6)
        # both switches off: no input reaches the output at all
        both = m(eeg, kin, use_kin=False, use_eeg=False)
        assert torch.equal(both, m(eeg2, None, use_kin=False, use_eeg=False))
        assert torch.allclose(both[0], both[1], atol=1e-6)


def test_use_eeg_false_never_runs_the_encoder_and_sends_it_no_gradient():
    eeg, kin, y = _batch()
    m = _kggt(eeg_channel_dropout=0.5, eeg_groups=4).train()
    m.kin_skip_proj[3].p = 0.0                                      # the only other random layer

    def _boom(*a, **k):
        raise AssertionError("the encoder must not run with use_eeg=False")

    handle = m.encoder.register_forward_pre_hook(_boom)
    torch.manual_seed(0)
    out = m(eeg, kin, use_eeg=False)
    torch.manual_seed(1)                                            # no electrode dropout draw either
    assert torch.equal(out, m(eeg, kin, use_eeg=False))
    handle.remove()
    F.mse_loss(out, y).backward()
    assert all(p.grad is None for p in m.encoder.parameters())
    w_grad = m.node_projection.proj.weight.grad                     # its input is all zeros
    assert w_grad is None or float(w_grad.abs().sum()) == 0.0
    assert float(m.node_projection.proj.bias.grad.abs().sum()) > 0
    assert float(m.kin_skip_proj[0].weight.grad.abs().sum()) > 0
    assert float(m.gat.kin_edge_linear.weight.grad.abs().sum()) > 0
    assert float(m.decoder.layers[0].weight.grad.abs().sum()) > 0


def test_use_eeg_false_on_a_model_without_kinematic_guidance():
    eeg, _, _ = _batch()
    m = _kggt(use_kinematic_guidance=False).eval()
    with torch.no_grad():
        out = m(eeg, use_eeg=False)
        assert torch.equal(out, m(torch.randn_like(eeg), use_eeg=False))
        assert not torch.allclose(out, m(eeg))


# ════════════════════════════════════════════════════════════════════════════
# 2. KinOnly / EEGOnly wrappers
# ════════════════════════════════════════════════════════════════════════════

def test_wrappers_call_the_model_with_the_right_switch():
    eeg, kin, _ = _batch()
    m = _kggt().eval()
    k, e = KinOnly(m), EEGOnly(m)
    with torch.no_grad():
        assert torch.equal(k(eeg, kin), m(eeg, kin, use_eeg=False))
        assert torch.equal(k(**{"eeg": eeg, "kin": kin}), m(eeg, kin, use_eeg=False))
        assert torch.equal(k(torch.zeros_like(eeg), kin), k(eeg, kin))               # EEG ignored
        assert torch.equal(e(eeg, kin), m(eeg, None, use_kin=False))
        assert torch.equal(e(eeg=eeg, kin=torch.randn_like(kin)), e(eeg))            # kin ignored
        assert not torch.allclose(e(eeg), e(torch.randn_like(eeg)))
    for w in (k, e):
        assert w.model is m and w.encoder is m.encoder and w.node_projection is m.node_projection
        assert w.gat is m.gat and w.kin_skip_proj is m.kin_skip_proj
        assert w.use_kinematic_guidance is True
        assert list(w.state_dict().keys()) == [f"model.{n}" for n in m.state_dict().keys()]
        assert len(list(w.parameters())) == len(list(m.parameters()))                # nothing counted twice
        w.train()
        assert m.training
        w.eval()
        assert not m.training
        c = copy.deepcopy(w)
        assert c.model is not m and torch.equal(c.model.decoder.layers[0].weight, m.decoder.layers[0].weight)
    with pytest.raises(TypeError):
        KinOnly(nn.Linear(2, 2))                                    # needs a forward with use_eeg
    with pytest.raises(TypeError):
        EEGOnly(nn.Linear(2, 2))


def test_wrapped_models_get_the_usual_optimiser_groups_and_checkpoints(tmp_path, monkeypatch):
    """train_model is unchanged: 0 epochs -> only the optimiser groups and last.pt are made."""
    eeg, kin, y = _batch(B=4)
    loader = DataLoader(TensorDataset(eeg, kin, y), batch_size=2)
    seen = {}
    real_adamw = torch.optim.AdamW

    def _spy(groups, *a, **k):
        seen["groups"] = groups
        return real_adamw(groups, *a, **k)

    monkeypatch.setattr(torch.optim, "AdamW", _spy)
    for name, wrap in (("kin_only", KinOnly), ("eeg_residual", EEGOnly)):
        m = _kggt()
        w = wrap(m)
        cfg = TrainConfig(lr=1e-3, max_epochs=0, use_amp=False, scheduler="cosine", warmup_epochs=0,
                          transformer_lr_scale=0.1, checkpoint_dir=str(tmp_path / name / "P0"))
        res = train_model(w, loader, loader, _identity_prepare_batch, CompositeEMGLoss(),
                          torch.device("cpu"), cfg, resume=True)
        assert res.history["train"] == []                            # no epoch was run
        lrs = sorted(g["lr"] for g in seen["groups"])
        assert lrs == pytest.approx(sorted([1e-3, 1e-4, 1e-3 / 3.0]))   # other / encoder+node proj / kin skip
        n_grouped = sum(len(list(g["params"])) for g in seen["groups"])
        assert n_grouped == len(list(m.parameters()))
        last = tmp_path / name / "P0" / "last.pt"
        assert last.exists()
        fresh = wrap(_kggt(seed=5))
        load_checkpoint(str(last), fresh, torch.device("cpu"))
        assert torch.equal(fresh.model.decoder.layers[0].weight, m.decoder.layers[0].weight)


# ════════════════════════════════════════════════════════════════════════════
# 3. Residual prepare_batch
# ════════════════════════════════════════════════════════════════════════════

def test_residual_prepare_batch_gives_target_minus_the_frozen_prediction():
    eeg, kin, emg = _batch()
    kin_model = KinOnly(_kggt(kin_dropout=0.5)).train()             # left in train mode on purpose
    base = lambda e, k, m: ({"eeg": e * 2.0, "kin": k + 1.0}, m * 0.5 - 1.0)   # noqa: E731
    pb = residual_prepare_batch_factory(base, kin_model)
    assert not kin_model.training                                   # frozen: eval mode ...
    assert all(not p.requires_grad for p in kin_model.parameters())  # ... and no trainable parameter
    inputs, res = pb(eeg, kin, emg)
    base_inputs, emg_z = base(eeg, kin, emg)
    assert torch.equal(inputs["eeg"], base_inputs["eeg"]) and torch.equal(inputs["kin"], base_inputs["kin"])
    with torch.no_grad():
        expect = emg_z - kin_model(**base_inputs)
    assert torch.equal(res, expect)
    assert res.dtype == torch.float32 and not res.requires_grad and res.grad_fn is None
    assert float(res.abs().mean()) > 0 and not torch.allclose(res, emg_z)

    # the frozen model stays in eval mode whatever the caller does, and gets no gradient
    kin_model.train()
    eeg_model = EEGOnly(_kggt(seed=1)).train()
    inputs, res2 = pb(eeg, kin, emg)
    assert not kin_model.training and torch.equal(res2, res)
    loss = ResidualLoss()(eeg_model(**inputs), res2)
    loss.backward()
    assert all(p.grad is None for p in kin_model.parameters())
    assert float(eeg_model.encoder.embed.weight.grad.abs().sum()) > 0
    assert all(p.grad is None for p in eeg_model.kin_skip_proj.parameters())   # kinematic paths are off


def test_zero_prediction_loss_is_the_loss_of_predicting_zero():
    eeg, kin, emg = _batch(B=6)
    loader = DataLoader(TensorDataset(eeg, kin, emg), batch_size=4)   # batches of 4 and 2
    loss = ResidualLoss()
    got = zero_prediction_loss(loader, loss, _identity_prepare_batch)
    per_batch = [float(loss(torch.zeros_like(emg[i:j]), emg[i:j])) * (j - i) for i, j in ((0, 4), (4, 6))]
    assert got == pytest.approx(sum(per_batch) / 6.0, rel=1e-6)
    assert got == pytest.approx(float(F.huber_loss(torch.zeros_like(emg), emg, delta=1.0)) + 0.2, abs=2e-2)


# ════════════════════════════════════════════════════════════════════════════
# 4. ResidualLoss
# ════════════════════════════════════════════════════════════════════════════

def test_residual_loss_is_zero_for_a_perfect_prediction():
    _, _, t = _batch()
    loss = ResidualLoss()
    assert (loss.huber_delta, loss.w_pearson) == (1.0, 0.2)
    assert float(loss(t.clone(), t)) == pytest.approx(0.0, abs=1e-6)
    assert loss.last_components["huber"] == pytest.approx(0.0, abs=1e-7)
    assert loss.last_components["pearson"] == pytest.approx(0.0, abs=1e-6)
    assert set(loss.last_components) >= {"total", "huber", "pearson", "active_fraction"}


def test_residual_loss_hand_computed():
    # one window, two channels, T = 4
    t = torch.tensor([[[0.0, 1.0], [1.0, -1.0], [2.0, 1.0], [3.0, -1.0]]])          # (1, 4, 2)
    p = torch.tensor([[[0.0, -3.0], [0.0, 3.0], [4.0, -3.0], [4.0, 3.0]]])
    # Huber(delta 1): |e| <= 1 -> e²/2, else |e| − 1/2
    # ch0 errors 0, −1, 2, 1 -> 0, .5, 1.5, .5 ; ch1 errors −4, 4, −4, 4 -> 3.5 each
    huber = (0.0 + 0.5 + 1.5 + 0.5 + 4 * 3.5) / 8.0
    # Pearson: ch0 pred (−2, −2, 2, 2), target (−1.5, −.5, .5, 1.5) -> 8 / (4 · sqrt 5); ch1 exactly −1
    r0 = 8.0 / (4.0 * 5.0 ** 0.5)
    pearson = ((1.0 - r0) + 2.0) / 2.0
    loss = ResidualLoss()
    value, comps = loss(p, t, return_components=True)
    assert comps["huber"] == pytest.approx(huber, rel=1e-6)
    assert comps["pearson"] == pytest.approx(pearson, rel=1e-5)
    assert float(value) == pytest.approx(huber + 0.2 * pearson, rel=1e-5)
    assert comps["active_fraction"] == 1.0 and comps is loss.last_components
    # other weights
    assert float(ResidualLoss(huber_delta=10.0, w_pearson=0.0)(p, t)) == pytest.approx(
        float(0.5 * ((p - t) ** 2).mean()), rel=1e-6)


def test_residual_loss_of_predicting_zero_is_the_baseline():
    _, _, t = _batch()
    loss = ResidualLoss()
    value = float(loss(torch.zeros_like(t), t))
    # a constant prediction has r = 0 -> the Pearson part is exactly 1
    assert loss.last_components["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert value == pytest.approx(float(F.huber_loss(torch.zeros_like(t), t, delta=1.0)) + 0.2, rel=1e-6)
    loss(torch.full_like(t, 0.7), t)                                # r = 0 for any constant
    assert loss.last_components["pearson"] == pytest.approx(1.0, abs=1e-6)


def test_residual_loss_skips_windows_with_a_constant_residual_in_the_pearson_part():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(3, WIN, 2, generator=g)
    p = torch.randn(3, WIN, 2, generator=g)
    t[0] = 0.25                                                     # window 0: constant in both channels
    t[1, :, 1] = -1.0                                               # window 1: channel 1 constant
    loss = ResidualLoss()
    value = float(loss(p, t))
    assert loss.last_components["active_fraction"] == pytest.approx(3.0 / 6.0)

    def _r(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()))

    active = [(1, 0), (2, 0), (2, 1)]
    pearson = sum(1.0 - _r(p[b, :, c], t[b, :, c]) for b, c in active) / len(active)
    assert loss.last_components["pearson"] == pytest.approx(pearson, rel=1e-5)
    assert value == pytest.approx(float(F.huber_loss(p, t, delta=1.0)) + 0.2 * pearson, rel=1e-5)
    # the skipped windows still count in the Huber part, and changing the prediction there
    # changes nothing in the Pearson part
    p2 = p.clone()
    p2[0] = p2[0].flip(0)
    loss(p2, t)
    assert loss.last_components["pearson"] == pytest.approx(pearson, rel=1e-5)
    # every window constant: the Pearson part is 0 and the loss stays differentiable
    p3 = p.clone().requires_grad_(True)
    v = loss(p3, torch.zeros_like(t))
    assert loss.last_components["pearson"] == 0.0 and loss.last_components["active_fraction"] == 0.0
    v.backward()
    assert torch.isfinite(p3.grad).all()


def test_residual_loss_is_finite_under_autocast_and_has_gradients():
    eeg, kin, t = _batch()
    m = EEGOnly(_kggt()).train()
    loss = ResidualLoss()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pred = m(eeg, kin)
        value = loss(pred, t * 1e-3)                                # tiny residuals too
    assert value.dtype == torch.float32 and torch.isfinite(value)
    value.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert "model" not in inspect.signature(loss.forward).parameters   # train_model calls loss(pred, y)
    with pytest.raises(ValueError):
        ResidualLoss(huber_delta=0.0)
    with pytest.raises(ValueError):
        ResidualLoss(w_pearson=-1.0)


# ════════════════════════════════════════════════════════════════════════════
# 5. residual_eval on synthetic data
# ════════════════════════════════════════════════════════════════════════════

S_WIN, S_C, S_F = 50, 3, 6          # window, muscles, EEG features
S_LIFT = 150                        # one lift every 150 samples (3 windows)


class _ToyKin(nn.Module):
    """'Kinematics-only model': the first C kinematic columns."""

    def forward(self, eeg, kin=None):
        return kin[..., :S_C]


class _ToyEEG(nn.Module):
    """'EEG model': gain x the EEG columns [first, first + C)."""

    def __init__(self, first: int, gain: float = 1.0):
        super().__init__()
        self.first, self.gain = first, gain

    def forward(self, eeg, kin=None):
        return self.gain * eeg[..., self.first:self.first + S_C]


def _toy_series(n_lifts: int, seed: int, tail: int = 17, offset: float = 0.0):
    """One series: emg = kin part + planted residual; the residual is EEG columns 0..C-1.

    The planted residual has a level per lift (constant over the lift) plus fast noise;
    EEG columns C..2C-1 are noise that has nothing to do with the EMG.
    """
    rng = np.random.default_rng(seed)
    T = n_lifts * S_LIFT + tail
    kin = rng.standard_normal((T, S_C + 2)).astype(np.float32)
    level = np.repeat(rng.standard_normal((n_lifts, S_C)), S_LIFT, axis=0)
    level = np.concatenate([level, np.zeros((tail, S_C))]) + offset
    planted = (level + 0.5 * rng.standard_normal((T, S_C))).astype(np.float32)
    eeg = np.concatenate([planted, rng.standard_normal((T, S_C)).astype(np.float32)], axis=1)
    emg = kin[:, :S_C] + planted
    segments = [{"lift": i + 1, "start": i * S_LIFT + 10, "stop": (i + 1) * S_LIFT - 10,
                 "CurW": 1 + i % 3, "CurS": 1} for i in range(n_lifts)]
    return (eeg, kin, emg), segments


@pytest.fixture(scope="module")
def toy():
    s4, seg4 = _toy_series(8, seed=1)
    s5, seg5 = _toy_series(7, seed=2, tail=0)
    return {"series": {4: s4, 5: s5}, "segments": {4: seg4, 5: seg5},
            "mean": np.array([0.5, -1.0, 2.0]), "std": np.array([2.0, 0.5, 3.0]),
            "names": ["AD", "BR", "FD"]}


def _evaluate(toy, eeg_model, n_null=39, seed=7, **kw):
    # the "normalised" EMG is the raw toy EMG: prepare_batch is the identity
    preds = predict_residuals(_ToyKin(), eeg_model, toy["series"], _identity_prepare_batch,
                              window=S_WIN, n_null=n_null, seed=seed, batch_size=4, **kw)
    report, per_lift = residual_report(preds, toy["mean"], toy["std"], channel_names=toy["names"],
                                       segments=toy["segments"])
    return preds, report, per_lift


def _row(report, group, muscle):
    hit = report[(report["group"] == group) & (report["muscle"] == muscle)]
    assert len(hit) == 1, (group, muscle, len(hit))
    return hit.iloc[0]


def test_random_derangement_has_no_fixed_point_and_is_seeded():
    for n in (2, 3, 7, 50):
        for seed in range(20):
            d = random_derangement(n, np.random.default_rng(seed))
            assert sorted(d.tolist()) == list(range(n))
            assert not np.any(d == np.arange(n))
    a = random_derangement(30, np.random.default_rng(3))
    assert np.array_equal(a, random_derangement(30, np.random.default_rng(3)))
    assert not np.array_equal(a, random_derangement(30, np.random.default_rng(4)))
    with pytest.raises(ValueError):
        random_derangement(1, np.random.default_rng(0))


def test_predict_residuals_shapes_and_the_window_swap_null(toy):
    preds, _, _ = _evaluate(toy, _ToyEEG(0), n_null=5)
    assert preds.series_ids == [4, 5] and preds.window == S_WIN and preds.n_null == 5
    n_w = {4: (8 * S_LIFT + 17) // S_WIN, 5: (7 * S_LIFT) // S_WIN}
    for sid, (eeg, kin, emg) in toy["series"].items():
        n = n_w[sid] * S_WIN                                       # whole windows only (test loader)
        assert preds.target[sid].shape == preds.kin[sid].shape == preds.res[sid].shape == (n, S_C)
        assert np.allclose(preds.target[sid], emg[:n]) and np.allclose(preds.kin[sid], kin[:n, :S_C])
        assert np.allclose(preds.res[sid], eeg[:n, :S_C])          # the stitched 1-window inference
        for kind in NULL_KINDS:
            assert preds.null[kind][sid].shape == (5, n, S_C)
    # null (b): every window gets the EEG of a DIFFERENT test window (pooled over the series)
    total = n_w[4] + n_w[5]
    assert preds.derangements.shape == (5, total)
    pooled_eeg = np.concatenate([toy["series"][s][0][:n_w[s] * S_WIN, :S_C] for s in (4, 5)])
    pooled_eeg = pooled_eeg.reshape(total, S_WIN, S_C)
    pooled_null = np.concatenate([preds.null["swap"][s] for s in (4, 5)], axis=1).reshape(5, total, S_WIN, S_C)
    for i, d in enumerate(preds.derangements):
        assert sorted(d.tolist()) == list(range(total)) and not np.any(d == np.arange(total))
        assert np.allclose(pooled_null[i], pooled_eeg[d])          # window w was fed the EEG of window d[w]
        assert all(not np.allclose(pooled_null[i, w], pooled_eeg[w]) for w in range(total))
    assert any(np.any(preds.derangements[i, :n_w[4]] >= n_w[4]) for i in range(5))   # crosses the series
    # null (a): the same samples of each window, in another order
    sh = preds.null["shuffle"][4].reshape(5, n_w[4], S_WIN, S_C)
    own = preds.res[4].reshape(n_w[4], S_WIN, S_C)
    assert np.allclose(np.sort(sh[0], axis=1), np.sort(own, axis=1)) and not np.allclose(sh[0], own)
    assert not np.allclose(sh[0], sh[1])
    # seeded
    again, _, _ = _evaluate(toy, _ToyEEG(0), n_null=5)
    other, _, _ = _evaluate(toy, _ToyEEG(0), n_null=5, seed=8)
    assert np.array_equal(again.derangements, preds.derangements)
    assert all(np.array_equal(again.null[k][s], preds.null[k][s]) for k in NULL_KINDS for s in (4, 5))
    assert not np.array_equal(other.derangements, preds.derangements)
    with pytest.raises(ValueError):                                 # one window cannot be swapped
        predict_residuals(_ToyKin(), _ToyEEG(0), {4: tuple(a[:S_WIN] for a in toy["series"][4])},
                          _identity_prepare_batch, window=S_WIN, n_null=2)


def test_report_when_the_eeg_model_recovers_the_planted_residual(toy):
    preds, report, per_lift = _evaluate(toy, _ToyEEG(0))
    assert list(dict.fromkeys(report["group"])) == ["series 4", "series 5", "all test"]
    assert list(report[report["group"] == "all test"]["muscle"]) == toy["names"] + ["MEAN"]
    for muscle in toy["names"] + ["MEAN"]:
        r = _row(report, "all test", muscle)
        assert r["full_vaf"] == pytest.approx(100.0, abs=1e-3) and r["full_r"] == pytest.approx(1.0, abs=1e-6)
        assert 20.0 < r["kin_vaf"] < 70.0 and r["d_vaf"] == pytest.approx(r["full_vaf"] - r["kin_vaf"])
        assert r["d_vaf"] > 20.0 and r["d_r"] > 0.1 and r["d_rmse"] < 0
        assert r["res_r"] == pytest.approx(1.0, abs=1e-6) and r["res_fve"] == pytest.approx(1.0, abs=1e-5)
        assert r["full_rmse"] == pytest.approx(0.0, abs=1e-5)
        # against the window-swap null: nothing reaches the observed value
        assert r["d_vaf_p_swap"] == pytest.approx(1.0 / 40.0) and r["res_r_p_swap"] == pytest.approx(1.0 / 40.0)
        assert r["d_vaf_null_swap_mean"] < 0 and abs(r["res_r_null_swap_mean"]) < 0.2
        assert r["d_vaf_null_swap_sd"] > 0 and r["n_null"] == 39
        # per lift: the hold level of the residual is recovered
        assert r["n_lifts"] == 15 and r["lift_r"] == pytest.approx(1.0, abs=1e-6)
        assert r["lift_r_p_swap"] == pytest.approx(1.0 / 40.0) and abs(r["lift_r_null_swap_mean"]) < 0.5
        assert r["lift_r_within"] == pytest.approx(1.0, abs=1e-6)
        for sid, n in ((4, 8), (5, 7)):
            rs = _row(report, f"series {sid}", muscle)
            assert rs["n_lifts"] == n and rs["lift_r"] == pytest.approx(1.0, abs=1e-6)
            assert rs["res_r"] == pytest.approx(1.0, abs=1e-6)
    # RMSE is in original EMG units, VAF does not depend on the units
    fd = _row(report, "all test", "FD")
    true_res = np.concatenate([preds.target[s] - preds.kin[s] for s in (4, 5)])
    assert fd["kin_rmse"] == pytest.approx(3.0 * float(np.sqrt((true_res[:, 2] ** 2).mean())), rel=1e-5)
    # null (a) keeps the window mean, so it keeps part of the lift-level residual: it is the weaker null
    assert fd["d_vaf_null_shuffle_mean"] > fd["d_vaf_null_swap_mean"]
    assert fd["d_vaf_p_shuffle"] == pytest.approx(1.0 / 40.0)

    # per-lift table: one row per series, lift and muscle
    assert len(per_lift) == 15 * S_C
    assert {"series", "lift", "CurW", "CurS", "muscle", "n_samples", "true_res", "pred_res",
            "null_swap_res"} <= set(per_lift.columns)
    one = per_lift[(per_lift["series"] == 4) & (per_lift["lift"] == 2) & (per_lift["muscle"] == "BR")].iloc[0]
    sl = slice(S_LIFT + 10, 2 * S_LIFT - 10)
    assert one["n_samples"] == S_LIFT - 20
    assert one["true_res"] == pytest.approx(float(true_res[sl, 1].mean()), rel=1e-5)
    assert one["pred_res"] == pytest.approx(one["true_res"], rel=1e-5)
    assert abs(per_lift["null_swap_res"]).mean() < abs(per_lift["pred_res"]).mean()

    verdict = residual_verdict(report)
    assert list(verdict["muscle"]) == toy["names"] + ["MEAN"]
    assert (verdict["verdict"] == VERDICT_YES).all()
    assert VERDICT_YES == "EEG adds unique information" and VERDICT_NO == "no evidence"
    text = format_residual_report(report)
    assert "all test" in text and "FD" in text and "MEAN" in text


def test_report_when_the_eeg_model_predicts_noise(toy):
    _, report, per_lift = _evaluate(toy, _ToyEEG(S_C, gain=0.3))    # columns unrelated to the EMG
    for muscle in toy["names"] + ["MEAN"]:
        r = _row(report, "all test", muscle)
        assert r["d_vaf"] < 0.5 and abs(r["res_r"]) < 0.1 and r["res_fve"] < 0.01
        assert r["d_vaf_p_swap"] > 0.05 and r["res_r_p_swap"] > 0.05
        assert abs(r["lift_r"]) < 0.8
    assert _row(report, "all test", "MEAN")["lift_r_p_swap"] > 0.05
    verdict = residual_verdict(report)
    assert (verdict["verdict"] == VERDICT_NO).all()
    assert all(isinstance(s, str) and s for s in verdict["why"])


def test_verdict_rules(toy):
    _, report, _ = _evaluate(toy, _ToyEEG(0), n_null=19)
    assert (residual_verdict(report)["verdict"] == VERDICT_YES).all()      # p = 1/20 = 0.05 counts

    def _verdict(edit, muscle="FD", **kw):
        rep = report.copy()
        edit(rep, rep["muscle"] == muscle)
        v = residual_verdict(rep, **kw)
        return v[v["muscle"] == muscle].iloc[0]["verdict"]

    allt = report["group"] == "all test"
    s4, s5 = report["group"] == "series 4", report["group"] == "series 5"
    assert _verdict(lambda r, m: None) == VERDICT_YES

    def _set(col, value, where):
        def edit(r, m):
            r.loc[m & where, col] = value
        return edit

    # 1. ΔVAF must be > 0
    assert _verdict(_set("d_vaf", 0.0, allt)) == VERDICT_NO
    assert _verdict(_set("d_vaf", -1.0, allt)) == VERDICT_NO
    # 2. p against the window-swap null must be <= 0.05 (the time-shuffle null does not count)
    assert _verdict(_set("d_vaf_p_swap", 0.06, allt)) == VERDICT_NO
    assert _verdict(_set("d_vaf_p_swap", float("nan"), allt)) == VERDICT_NO
    assert _verdict(_set("d_vaf_p_shuffle", 0.9, allt)) == VERDICT_YES
    assert _verdict(_set("d_vaf_p_swap", 0.06, allt), alpha=0.1) == VERDICT_YES
    # 3. residual r > 0 in EVERY test series, per lift OR per sample
    def _both(lift4, lift5, res4, res5):
        def edit(r, m):
            r.loc[m & s4, "lift_r"], r.loc[m & s5, "lift_r"] = lift4, lift5
            r.loc[m & s4, "res_r"], r.loc[m & s5, "res_r"] = res4, res5
        return edit

    assert _verdict(_both(0.3, 0.2, -0.1, -0.1)) == VERDICT_YES      # per lift in both series
    assert _verdict(_both(-0.3, 0.2, 0.1, 0.1)) == VERDICT_YES       # per sample in both series
    assert _verdict(_both(0.3, -0.2, 0.1, -0.1)) == VERDICT_NO       # neither is positive in both
    assert _verdict(_both(0.3, -0.2, -0.1, 0.1)) == VERDICT_NO
    assert _verdict(_both(float("nan"), 0.2, -0.1, 0.1)) == VERDICT_NO
    assert _verdict(_both(0.0, 0.2, 0.0, 0.1)) == VERDICT_NO         # 0 is not > 0
    # other muscles are judged on their own rows
    rep = report.copy()
    rep.loc[(rep["muscle"] == "FD") & allt, "d_vaf"] = -1.0
    v = residual_verdict(rep).set_index("muscle")["verdict"]
    assert v["FD"] == VERDICT_NO and v["AD"] == VERDICT_YES


def test_report_without_nulls_or_segments(toy):
    preds = predict_residuals(_ToyKin(), _ToyEEG(0), toy["series"], _identity_prepare_batch,
                              window=S_WIN, n_null=0)
    report, per_lift = residual_report(preds, toy["mean"], toy["std"])
    assert list(report[report["group"] == "all test"]["muscle"]) == ["ch0", "ch1", "ch2", "MEAN"]
    assert per_lift.empty and report["lift_r"].isna().all() and report["d_vaf_p_swap"].isna().all()
    assert (report["n_null"] == 0).all() and (report["n_lifts"] == 0).all()
    assert (residual_verdict(report)["verdict"] == VERDICT_NO).all()      # no null -> no claim


def test_series_level_offset_is_not_what_the_within_series_per_lift_r_measures():
    """A between-series offset alone makes the pooled per-lift r large; lift_r_within removes it."""
    s4, seg4 = _toy_series(8, seed=1, offset=3.0)
    s5, seg5 = _toy_series(7, seed=2, tail=0, offset=-3.0)

    class _SeriesMean(nn.Module):                                   # predicts only the series offset
        def forward(self, eeg, kin=None):
            return torch.sign(eeg[..., :S_C].mean(dim=(1, 2), keepdim=True)).expand(-1, eeg.shape[1], S_C) * 3.0

    preds = predict_residuals(_ToyKin(), _SeriesMean(), {4: s4, 5: s5}, _identity_prepare_batch,
                              window=S_WIN, n_null=0)
    report, _ = residual_report(preds, np.zeros(S_C), np.ones(S_C), segments={4: seg4, 5: seg5})
    r = _row(report, "all test", "MEAN")
    assert r["lift_r"] > 0.9
    assert np.isnan(r["lift_r_within"]) or abs(r["lift_r_within"]) < 0.3
    assert np.isnan(_row(report, "series 4", "MEAN")["lift_r"])      # constant prediction within a series


def test_figures_render_without_warnings(toy, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    _, good, lift_good = _evaluate(toy, _ToyEEG(0), n_null=9)
    _, bad, lift_bad = _evaluate(toy, _ToyEEG(S_C, gain=0.3), n_null=9)
    lift_good.insert(0, "subject", "P1")
    lift_bad.insert(0, "subject", "P2")
    import pandas as pd
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig1 = plot_residual_delta_vaf({"P1": good, "P2": bad}, title="ΔVAF")
        fig2 = plot_residual_per_lift(pd.concat([lift_good, lift_bad], ignore_index=True), title="per lift")
        fig3 = plot_residual_delta_vaf(good)                         # a single report works too
        fig4 = plot_residual_per_lift(lift_good.drop(columns="subject"))
        for k, fig in enumerate((fig1, fig2, fig3, fig4)):
            fig.savefig(tmp_path / f"fig{k}.png", dpi=60)
    assert len(fig1.axes) == 4 and len(fig2.axes) == 2 * S_C
    # colours: kin-only orange, kin + EEG blue, null grey
    faces = {to_hex(p.get_facecolor()) for ax in fig1.axes for p in ax.patches}
    assert {"#eb6834", "#2a78d6"} <= faces
    texts = " ".join(t.get_text() for ax in fig1.axes for t in ax.texts)
    assert "p=" in texts.replace(" ", "")
    titles = [ax.get_title() for ax in fig2.axes]
    assert all("r" in t for t in titles) and any("FD" in t for t in titles)
    plt.close("all")


# ════════════════════════════════════════════════════════════════════════════
# 6. Notebook
# ════════════════════════════════════════════════════════════════════════════

def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _strip_comments(code: str) -> str:
    out = []
    for line in code.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(re.sub(r"\s+#.*$", "", line))
    return "\n".join(out)


@pytest.fixture(scope="module")
def nb():
    with open(ROOT / NB_NAME, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def head_1stage():
    try:
        raw = subprocess.run(["git", "show", "HEAD:main_p2p.ipynb"], cwd=ROOT,
                             check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"git HEAD:main_p2p.ipynb unavailable: {e}")
    return json.loads(raw.decode("utf-8"))


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location(
        "build_main_residual_notebook", ROOT / "tools" / "build_main_residual_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code(nb, needle: str) -> str:
    hits = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(hits) == 1, f"{needle!r}: {len(hits)} cells"
    return hits[0]


def _index(nb, needle: str, cell_type: str = "code") -> int:
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == cell_type and needle in _src(c)]
    assert len(hits) == 1, f"{needle!r}: {len(hits)} cells"
    return hits[0]


def _config_ns(nb) -> dict:
    ns = {"copy": copy}
    exec(_code(nb, "CONFIG = {"), ns)                                # dict literal + path helpers
    return ns


# `subject_list` (which participants to run) is edited by the user in the notebooks themselves:
# its value is not compared, everything else is.
_SUBJECT_LIST = re.compile(r"^subject_list = .*$", re.M)


def _no_subject_list(text: str) -> str:
    return _SUBJECT_LIST.sub("subject_list = <user>", text)


def test_notebook_is_the_generator_output(nb, builder):
    built, log = builder.build("HEAD")
    assert log
    # cell by cell (type + source); a notebook saved from the editor may order its JSON keys
    # differently, so the file is not compared byte for byte
    assert len(built["cells"]) == len(nb["cells"])
    for i, (a, b) in enumerate(zip(built["cells"], nb["cells"])):
        assert a["cell_type"] == b["cell_type"], i
        assert _no_subject_list(_src(a)) == _no_subject_list(_src(b)), f"cell {i} differs from the generator output"
    assert built["metadata"] == nb["metadata"]
    # the committed notebook (no user edits) is the generator output exactly
    try:
        raw = subprocess.run(["git", "show", f"HEAD:{NB_NAME}"], cwd=ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return
    committed = json.loads(raw.decode("utf-8"))
    assert [(c["cell_type"], _no_subject_list(_src(c))) for c in committed["cells"]] == \
        [(c["cell_type"], _no_subject_list(_src(c))) for c in built["cells"]]
    assert sum(len(_SUBJECT_LIST.findall(_src(c))) for c in nb["cells"]) == 1


def test_generator_fails_loudly_when_an_anchor_is_missing(builder, head_1stage):
    broken = copy.deepcopy(head_1stage)
    for c in broken["cells"]:
        if "PER-SUBJECT PRE-PROCESSING LOOP" in _src(c):
            c["source"] = ["# something else\n"]
    with pytest.raises(AssertionError):
        builder.build_from(broken)
    broken = copy.deepcopy(head_1stage)
    for c in broken["cells"]:
        if "CONFIG = {" in _src(c):
            c["source"] = [ln.replace('"run_tag": "p2b"', '"run_tag": "zzz"') for ln in c["source"]]
    with pytest.raises(AssertionError):
        builder.build_from(broken)


def test_notebook_has_no_outputs_and_compiles(nb):
    assert nb["nbformat"] == 4
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids) and len(set(ids)) == len(ids)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        assert c["outputs"] == [] and c["execution_count"] is None
        code = "\n".join("pass" if re.match(r"^\s*[%!]", ln) else ln for ln in _src(c).splitlines())
        compile(code, f"{NB_NAME}[cell {i}]", "exec")


def test_notebook_config_is_main_1stage_except_the_documented_keys(nb, head_1stage):
    ns, head = _config_ns(nb), _config_ns(head_1stage)
    cfg, ref = ns["CONFIG"], head["CONFIG"]
    for k in ("data", "preprocessing", "dataset", "model"):
        assert cfg[k] == ref[k], k
    assert cfg["training"]["loss"] == ref["training"]["loss"]
    assert cfg["outputs"] == {**ref["outputs"], "run_tag": RUN_TAG}
    assert cfg["training"]["checkpoint_dir"] == "outputs/checkpoints_residual_p2c"
    assert {k: v for k, v in cfg["training"].items() if k != "checkpoint_dir"} == \
        {k: v for k, v in ref["training"].items() if k != "checkpoint_dir"}
    assert cfg["residual"] == {"kin_epochs": 200, "eeg_epochs": 200, "early_stop_patience": 20,
                               "lr": 1e-3, "n_null": 20, "seed": 42}
    assert set(cfg) - set(ref) == {"residual"} and set(ref) <= set(cfg)
    assert ns["notebook_cfg"] == cfg
    # paths of this run
    assert ns["RUN_TAG"] == RUN_TAG and ns["RESULTS_DIR"] == "results/p2c"
    assert ns["tagged_output"]("x") == "outputs/x_p2c"
    assert ns["stage_checkpoint_dir"](cfg, "kin_only", "P1") == "outputs/checkpoints_residual_p2c/kin_only/P1"
    assert ns["stage_checkpoint_dir"](cfg, "eeg_residual", "P1") == \
        "outputs/checkpoints_residual_p2c/eeg_residual/P1"
    with pytest.raises(AssertionError):
        ns["stage_checkpoint_dir"](cfg, "other", "P1")
    assert 'endswith("_" + RUN_TAG)' in _code(nb, "CONFIG = {")


def test_notebook_reuses_the_helper_and_preprocessing_cells_unchanged(nb, head_1stage):
    for needle in ("def local_prepare_batch_factory(", "PER-SUBJECT PRE-PROCESSING LOOP",
                   "device = get_device()", "ROOT = Path(\".\")"):
        assert _no_subject_list(_code(nb, needle)) == _no_subject_list(_code(head_1stage, needle)), needle
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert nb["metadata"] == head_1stage["metadata"]


def test_notebook_stage_order_is_k_then_e_then_evaluation(nb):
    i_cfg = _index(nb, "CONFIG = {")
    i_help = _index(nb, "def local_prepare_batch_factory(")
    i_pre = _index(nb, "PER-SUBJECT PRE-PROCESSING LOOP")
    i_tools = _index(nb, "def train_stage(")
    i_k = _index(nb, "STAGE K")
    i_e = _index(nb, "STAGE E")
    i_ev = _index(nb, "RESIDUAL EVALUATION")
    assert i_cfg < i_help < i_pre < i_tools < i_k < i_e < i_ev
    assert i_ev == max(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    # nothing of the single-stage training / evaluation is left
    code = "\n".join(_strip_comments(_src(c)) for c in nb["cells"] if c["cell_type"] == "code")
    for gone in ("train_single_stage(_cfg", "PER-SUBJECT TRAINING LOOP", "input_ablation(",
                 "plot_interp_window", "[OPTIONAL] Automatic LOSOCV", "evaluation_report_1stage"):
        assert gone not in code, gone


def test_notebook_stage_cells(nb):
    tools = _code(nb, "def train_stage(")
    assert "train_model(" in tools and "resume=True" in tools and "load_checkpoint(" in tools
    assert 'res["early_stop_patience"]' in tools and 'res["lr"]' in tools
    assert "stage_checkpoint_dir(cfg, stage, subj)" in tools

    k = _code(nb, "STAGE K")
    assert "KinOnly(" in k and "build_kg_gt_from_config(" in k
    assert re.search(r'\["loss"\]\["w_eeg_only"\]\s*=\s*0\.0', k)
    assert re.search(r'\["model"\]\["kin_dropout"\]\s*=\s*0\.0', k)
    assert re.search(r'\["model"\]\["eeg_only_aux"\]\s*=\s*False', k)
    assert "build_loss_from_config(" in k and '"kin_only"' in k and '["kin_epochs"]' in k
    assert "EEGOnly(" not in k and "ResidualLoss(" not in k

    e = _code(nb, "STAGE E")
    assert "EEGOnly(" in e and "residual_prepare_batch_factory(" in e
    assert "ResidualLoss(huber_delta=1.0, w_pearson=0.2)" in e
    assert "zero_prediction_loss(" in e and '"eeg_residual"' in e
    assert "stage_k[_subj]" in e and "KinOnly(" not in e
    # the EEG model keeps the regularised encoder and the electrode dropout of CONFIG
    assert '["eeg_channel_dropout"]' not in _strip_comments(e)
    assert '["transformer"]' not in _strip_comments(e)


def test_notebook_evaluation_cell(nb):
    ev = _code(nb, "RESIDUAL EVALUATION")
    for needle in ("predict_residuals(", "residual_report(", "residual_verdict(", "lift_hold_segments(",
                   "emg_channel_names(", "plot_residual_delta_vaf(", "plot_residual_per_lift(",
                   'n_null=_res_cfg["n_null"]', 'seed=_res_cfg["seed"]',
                   "os.makedirs(RESULTS_DIR, exist_ok=True)", "format_residual_report("):
        assert needle in ev, needle
    assert re.search(r'RESIDUAL_CSV\s*=\s*f"\{RESULTS_DIR\}/residual_report\.csv"', ev)
    assert re.search(r'RESIDUAL_LIFT_CSV\s*=\s*f"\{RESULTS_DIR\}/residual_per_lift\.csv"', ev)
    for var in ("RESIDUAL_CSV", "RESIDUAL_LIFT_CSV", "RESIDUAL_VAF_PNG", "RESIDUAL_LIFT_PNG"):
        line = next(ln for ln in ev.splitlines() if re.match(rf"\s*{var}\s*=", ln))
        assert "RESULTS_DIR" in line, line
    assert "untrained" not in ev.lower()
    title = _src(nb["cells"][0])
    for needle in ("Phase C", "Stage K", "Stage E", "ΔVAF", "p ≤ 0.05", "EEG adds unique information",
                   "no evidence", "residual_report.csv", "residual_per_lift.csv",
                   "outputs/checkpoints_residual_p2c", "results/p2c", "derangement", "use_eeg=False",
                   "bias"):
        assert needle in title, needle
    for word in ("proves", "clearly shows", "demonstrates"):        # no overclaiming language
        assert word not in title.lower() and word not in ev.lower()


def test_notebook_has_no_untagged_output_path(nb):
    """Outside the CONFIG cell no code names results/ or outputs/ directly."""
    pattern = re.compile(r"""results/|outputs/|["'](?:results|outputs)["'](?!\s*\])""")
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = _src(c)
        code = _strip_comments(s)
        if "CONFIG = {" in s:
            for line in code.splitlines():
                if re.search(r"results/|outputs/", line):
                    assert "RUN_TAG" in line or f'_{RUN_TAG}"' in line, line
            continue
        hit = [ln for ln in code.splitlines() if pattern.search(ln)]
        assert not hit, f"cell {c.get('id')}: untagged path in {hit}"
    code = "\n".join(_strip_comments(_src(c)) for c in nb["cells"] if c["cell_type"] == "code")
    assert "p2b" not in code                                         # no path of the Phase B run
    for writer in (".to_csv(", ".savefig("):
        for line in (ln for ln in code.splitlines() if writer in ln):
            assert re.search(r"RESIDUAL_\w+", line), line


def test_other_notebooks_are_not_touched_by_the_generator(builder):
    src = Path(builder.__file__).read_text(encoding="utf-8")
    assert '"main_1stage_residual.ipynb"' in src
    assert "HEAD" in src and ":main_p2p.ipynb" in src
    assert not re.search(r'open\([^)]*main(_1stage|_p2p)?\.ipynb', src)


def test_library_exports():
    import main
    for name in ("KinOnly", "EEGOnly", "ResidualLoss", "residual_prepare_batch_factory",
                 "zero_prediction_loss", "predict_residuals", "residual_report", "residual_verdict",
                 "format_residual_report", "random_derangement", "plot_residual_delta_vaf",
                 "plot_residual_per_lift"):
        assert name in main.__all__ and callable(getattr(main, name)), name
