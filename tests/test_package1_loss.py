"""Package 1 loss changes: rest over-prediction term, Huber option, per-muscle weights.

All new terms default to off, so the previous numbers are reproduced exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.losses import (
    CompositeEMGLoss,
    PeakWeightedMSELoss,
    build_loss_from_config,
    channel_weights_from_config,
    compute_channel_weights,
)


def _previous_peak_mse(pred, target, alpha=3.0, asymmetry=2.0):
    """The PeakWeightedMSELoss formula before Package 1 (copied verbatim)."""
    if alpha == 0.0 and asymmetry == 1.0:
        return F.mse_loss(pred, target)
    t_min = target.amin(dim=1, keepdim=True)
    t_max = target.amax(dim=1, keepdim=True)
    t_range = (t_max - t_min).clamp(min=1e-4)
    y_norm = (target - t_min) / t_range
    w_t = 1.0 + alpha * (y_norm ** 2)
    under_prediction = (target > pred).float()
    asym_w = 1.0 + (asymmetry - 1.0) * under_prediction * y_norm
    mse_raw = F.mse_loss(pred, target, reduction="none")
    return (w_t * asym_w * mse_raw).mean()


def _previous_rest_l1(pred, target, rest_threshold=0.15):
    t_min = target.amin(dim=1, keepdim=True)
    target_shifted = target - t_min
    max_batch = target_shifted.amax(dim=1, keepdim=True)
    resting_mask = (target_shifted < rest_threshold * (max_batch + 1e-5)).float()
    return torch.mean(torch.abs(pred - t_min) * resting_mask)


def _data(seed=0, B=3, T=64, C=5):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(B, T, C, generator=g).cumsum(1) * 0.3
    pred = target + 0.8 * torch.randn(B, T, C, generator=g)
    return pred, target


# ── regression: defaults reproduce the previous formula ──────────────────────

@pytest.mark.parametrize("alpha, asym", [(3.0, 2.0), (3.0, 1.2), (0.0, 1.0), (1.5, 1.0)])
def test_huber_none_reproduces_previous_peak_mse(alpha, asym):
    pred, target = _data()
    new = PeakWeightedMSELoss(alpha=alpha, asymmetry=asym)(pred, target)
    old = _previous_peak_mse(pred, target, alpha, asym)
    assert torch.equal(new, old)


def test_composite_defaults_reproduce_previous_total():
    pred, target = _data(1)
    loss = CompositeEMGLoss()
    total, comps = loss(pred, target, return_components=True)
    from main.losses import CCCLoss, PearsonCorrelationLoss, TemporalSmoothnessLoss
    expected = (
        1.0 * _previous_peak_mse(pred, target)
        + 0.5 * CCCLoss()(pred, target)
        + 0.2 * PearsonCorrelationLoss()(pred, target)
        + 0.1 * TemporalSmoothnessLoss()(pred, target)
        + 0.05 * _previous_rest_l1(pred, target)
    )
    torch.testing.assert_close(total, expected, rtol=0, atol=1e-6)
    assert comps["rest_over"] == 0.0
    assert comps["rest_l1"] == pytest.approx(_previous_rest_l1(pred, target).item(), abs=1e-7)


def test_weight_zero_new_terms_give_identical_total():
    pred, target = _data(2)
    base = CompositeEMGLoss(asymmetry=1.2)(pred, target)
    same = CompositeEMGLoss(asymmetry=1.2, w_rest_over=0.0, huber_delta=None,
                            channel_weights=None)(pred, target)
    assert torch.equal(base, same)


# ── each new term is zero for a perfect prediction ───────────────────────────

def test_terms_zero_when_pred_equals_target():
    _, target = _data(3)
    w = np.array([0.5, 1.0, 1.5, 1.0, 1.0], dtype=np.float32)
    loss = CompositeEMGLoss(w_ccc=0, w_pearson=0, w_diff=0, w_rest=0, w_reg=0,
                            w_rest_over=1.0, huber_delta=1.0, channel_weights=w)
    total, comps = loss(target.clone(), target, return_components=True)
    assert comps["peak_mse"] == 0.0
    assert comps["rest_over"] == 0.0
    assert total.item() == 0.0


# ── hand-computed cases ──────────────────────────────────────────────────────

def test_rest_over_hand_computed():
    # one window, one channel, T=4: target [0, 0, 1, 0] -> rest where shifted < 0.15*1
    target = torch.tensor([0.0, 0.0, 1.0, 0.0]).view(1, 4, 1)
    pred = torch.tensor([0.5, -0.3, 2.0, 0.2]).view(1, 4, 1)
    loss = CompositeEMGLoss(w_peak=0, w_ccc=0, w_pearson=0, w_diff=0, w_rest=0, w_reg=0,
                            w_rest_over=1.0)
    total, comps = loss(pred, target, return_components=True)
    # rest samples 0,1,3: over = 0.5, 0 (under), 0.2 ; sample 2 is active (ignored)
    expected = (0.5 ** 2 + 0.0 + 0.2 ** 2) / 4
    assert comps["rest_over"] == pytest.approx(expected, abs=1e-7)
    assert total.item() == pytest.approx(expected, abs=1e-7)


def test_rest_over_uses_channel_weights():
    target = torch.zeros(1, 4, 2)
    target[0, 2, :] = 1.0
    pred = target.clone()
    pred[0, 0, :] = 1.0          # over-prediction at rest on both channels
    w = np.array([0.5, 1.5], dtype=np.float32)
    loss = CompositeEMGLoss(w_peak=0, w_ccc=0, w_pearson=0, w_diff=0, w_rest=0, w_reg=0,
                            w_rest_over=1.0, channel_weights=w)
    _, comps = loss(pred, target, return_components=True)
    assert comps["rest_over"] == pytest.approx((0.5 * 1.0 + 1.5 * 1.0) / 8, abs=1e-7)


def test_huber_hand_computed_with_flat_weights():
    # alpha=0, asymmetry=1 -> plain elementwise robust loss averaged
    target = torch.zeros(1, 3, 1)
    pred = torch.tensor([0.5, 2.0, -3.0]).view(1, 3, 1)
    l = PeakWeightedMSELoss(alpha=0.0, asymmetry=1.0, huber_delta=1.0)(pred, target)
    # rho(e) = e^2 for |e|<=d, 2d|e|-d^2 beyond (equals MSE for small errors)
    expected = (0.25 + (2 * 2.0 - 1.0) + (2 * 3.0 - 1.0)) / 3
    assert l.item() == pytest.approx(expected, abs=1e-6)


def test_huber_matches_mse_for_small_errors_and_is_smaller_for_outliers():
    pred, target = _data(4)
    small = target + 0.01 * torch.sign(pred - target)
    a = PeakWeightedMSELoss(huber_delta=1.0)(small, target)
    b = PeakWeightedMSELoss()(small, target)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)
    big = target + 5.0
    assert PeakWeightedMSELoss(huber_delta=1.0)(big, target) < PeakWeightedMSELoss()(big, target)


def test_channel_weights_scale_per_channel_error():
    target = torch.zeros(1, 2, 2)
    pred = torch.ones(1, 2, 2)
    w = np.array([0.25, 1.75], dtype=np.float32)
    l = PeakWeightedMSELoss(alpha=0.0, asymmetry=1.0, channel_weights=w)(pred, target)
    assert l.item() == pytest.approx((0.25 + 1.75) * 2 / 4, abs=1e-7)


# ── channel weights "auto" ───────────────────────────────────────────────────

def test_auto_channel_weights_are_inverse_p95_normalised_to_mean_one():
    rng = np.random.default_rng(0)
    scales = np.array([1.0, 2.0, 4.0, 0.5, 8.0])
    z = rng.standard_normal((20000, 5)) * scales
    w = compute_channel_weights(z)
    assert w.shape == (5,) and w.dtype == np.float32
    assert w.mean() == pytest.approx(1.0, abs=1e-6)
    p95 = np.percentile(np.abs(z), 95, axis=0)
    expected = (1 / p95) / (1 / p95).mean()
    np.testing.assert_allclose(w, expected, rtol=1e-5)
    assert np.argmax(w) == 3 and np.argmin(w) == 4      # spikiest muscle weighted least
    # a list of per-series arrays gives the same as their concatenation
    np.testing.assert_allclose(compute_channel_weights([z[:5000], z[5000:]]), w, rtol=1e-6)


def test_channel_weights_from_config():
    z = np.random.default_rng(1).standard_normal((1000, 5))
    assert channel_weights_from_config({"loss": {}}, z) is None
    assert channel_weights_from_config({"loss": {"channel_weights": None}}, z) is None
    np.testing.assert_allclose(
        channel_weights_from_config({"loss": {"channel_weights": "auto"}}, z),
        compute_channel_weights(z))
    np.testing.assert_allclose(
        channel_weights_from_config({"loss": {"channel_weights": [1, 2, 3, 4, 5]}}, z),
        [1, 2, 3, 4, 5])


# ── builder ──────────────────────────────────────────────────────────────────

def test_builder_reads_all_new_keys():
    w = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)
    cfg = {"loss": {"asymmetry": 1.2, "w_rest_over": 1.0, "huber_delta": 1.0,
                    "channel_weights": "auto"}}
    loss = build_loss_from_config(cfg, channel_weights=w)
    assert loss.peak_mse.asymmetry == 1.2
    assert loss.w_rest_over == 1.0
    assert loss.peak_mse.huber_delta == 1.0
    np.testing.assert_allclose(loss.channel_weights.numpy(), w)
    np.testing.assert_allclose(loss.peak_mse.channel_weights.numpy(), w)


def test_builder_auto_without_weights_raises():
    with pytest.raises(ValueError, match="auto"):
        build_loss_from_config({"loss": {"channel_weights": "auto"}})


def test_builder_explicit_list_and_defaults():
    loss = build_loss_from_config({"loss": {"channel_weights": [1, 1, 2, 1, 1]}})
    np.testing.assert_allclose(loss.channel_weights.numpy(), [1, 1, 2, 1, 1])
    d = build_loss_from_config({"loss": {}})
    assert d.w_rest_over == 0.0
    assert d.peak_mse.huber_delta is None
    assert d.channel_weights is None
    assert d.peak_mse.asymmetry == 2.0


def test_weights_follow_the_prediction_device_and_dtype():
    pred, target = _data(5)
    w = np.ones(5, dtype=np.float32)
    loss = CompositeEMGLoss(w_rest_over=1.0, channel_weights=w)
    out = loss(pred.double(), target.double())
    assert out.dtype == torch.float64
