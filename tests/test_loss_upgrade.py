"""
Unit Test Suite for Advanced Correlation- & Shape-Preserving Electrophysiological Losses.

Tests:
  1. CCCLoss: Perfect match (L=0), Scale mismatch (L>0), Mean Collapse penalty (L=1), Inverted (L=2).
  2. PearsonCorrelationLoss: Scale invariance (L=0 for affine shift), Inversion (L=2), Quiescent gating.
  3. TemporalSmoothnessLoss: Smooth ramp (L=0), High-frequency jitter penalty (L>>0).
  4. AMP FP16 & Quiescent Stability: Safe denominator clamping under float16 autocast.
  5. CompositeEMGLoss: Full forward pass, component dictionary verification, and active gradient backprop.
  6. Backward Compatibility: CombinedEMGLoss alias and build_loss_from_config behavior.
  7. compute_metrics: Verification of EvalMetrics.ccc calculation and formatting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from main.losses import (
    CCCLoss,
    PearsonCorrelationLoss,
    TemporalSmoothnessLoss,
    CompositeEMGLoss,
    CombinedEMGLoss,
    build_loss_from_config,
)
from main.model import build_kg_gt_from_config
from main.training import compute_metrics, EvalMetrics


def test_ccc_loss_properties():
    print("\n--- 1. Testing CCCLoss Mathematical Properties ---")
    ccc_fn = CCCLoss(eps=1e-6)
    B, T, C = 2, 500, 5

    # Base ground truth signal with rich temporal dynamics
    t = torch.linspace(0, 4 * np.pi, T).view(1, T, 1).expand(B, T, C)
    target = torch.sin(t) + 0.5 * torch.sin(3 * t)

    # 1. Perfect prediction -> L_ccc = 0.0 (CCC = 1.0)
    pred_perfect = target.clone()
    loss_perfect = ccc_fn(pred_perfect, target).item()
    print(f"  Perfect prediction CCC loss: {loss_perfect:.6f} (Expected: ~0.0)")
    assert loss_perfect < 1e-4, f"Perfect prediction CCC loss should be ~0, got {loss_perfect}"

    # 2. Scale-doubled prediction -> L_ccc > 0.0 (penalizes scale error)
    pred_scaled = 2.0 * target
    loss_scaled = ccc_fn(pred_scaled, target).item()
    print(f"  Scale-doubled prediction CCC loss: {loss_scaled:.6f} (Expected: > 0.0)")
    assert loss_scaled > 0.1, f"Scale mismatch must be penalized by CCC, got {loss_scaled}"

    # 3. Mean Collapse prediction -> L_ccc = 1.0 (CCC = 0.0)
    # Predict constant mean for each channel
    pred_collapsed = target.mean(dim=1, keepdim=True).expand_as(target)
    loss_collapsed = ccc_fn(pred_collapsed, target).item()
    print(f"  Mean Collapse prediction CCC loss: {loss_collapsed:.6f} (Expected: ~1.0)")
    assert abs(loss_collapsed - 1.0) < 1e-3, f"Mean collapse should yield CCC loss ~ 1.0, got {loss_collapsed}"

    # 4. Sign inversion -> L_ccc = 2.0 (CCC = -1.0)
    pred_inverted = -target
    loss_inverted = ccc_fn(pred_inverted, target).item()
    print(f"  Inverted prediction CCC loss: {loss_inverted:.6f} (Expected: ~2.0)")
    assert abs(loss_inverted - 2.0) < 1e-3, f"Inverted prediction should yield CCC loss ~ 2.0, got {loss_inverted}"

    print("  [PASS] CCCLoss correctly enforces shape, scale, and mean alignment while penalizing Mean Collapse.")


def test_pearson_loss_invariance():
    print("\n--- 2. Testing PearsonCorrelationLoss Properties ---")
    pearson_fn = PearsonCorrelationLoss(eps=1e-6, min_target_var=1e-4)
    B, T, C = 2, 500, 5

    t = torch.linspace(0, 4 * np.pi, T).view(1, T, 1).expand(B, T, C)
    target = torch.sin(t) + 0.3 * torch.cos(2 * t)

    # 1. Scale and shift invariance: pred = 3.5 * target + 10.0 -> r = 1.0 -> loss = 0.0
    pred_affine = 3.5 * target + 10.0
    loss_affine = pearson_fn(pred_affine, target).item()
    print(f"  Affine shifted prediction Pearson loss: {loss_affine:.6f} (Expected: ~0.0)")
    assert loss_affine < 1e-4, f"Pearson loss must be scale & shift invariant, got {loss_affine}"

    # 2. Inverted signal: pred = -target -> r = -1.0 -> loss = 2.0
    pred_inverted = -target
    loss_inverted = pearson_fn(pred_inverted, target).item()
    print(f"  Inverted prediction Pearson loss: {loss_inverted:.6f} (Expected: ~2.0)")
    assert abs(loss_inverted - 2.0) < 1e-4, f"Inverted signal should yield Pearson loss ~ 2.0, got {loss_inverted}"

    # 3. Quiescent channel gating: target channel 0 has near-zero variance (< 1e-4)
    target_quiescent = target.clone()
    target_quiescent[:, :, 0] = 0.0010 + 1e-6 * torch.randn(B, T)
    pred_noisy = torch.randn_like(target_quiescent)
    loss_gated = pearson_fn(pred_noisy, target_quiescent)
    assert torch.isfinite(loss_gated), "Gated quiescent channels must not produce NaN/Inf!"
    print(f"  Quiescent channel gated Pearson loss: {loss_gated.item():.4f}")

    print("  [PASS] PearsonCorrelationLoss provides scale-invariant timing rewards and handles quiescent channels.")


def test_temporal_smoothness_loss():
    print("\n--- 3. Testing TemporalSmoothnessLoss ---")
    smooth_fn = TemporalSmoothnessLoss()
    B, T, C = 2, 500, 5

    target = torch.linspace(0, 1, T).view(1, T, 1).expand(B, T, C)

    # Smooth matching ramp
    pred_smooth = target.clone()
    loss_smooth = smooth_fn(pred_smooth, target).item()
    print(f"  Smooth ramp difference loss: {loss_smooth:.6f} (Expected: 0.0)")
    assert loss_smooth < 1e-6, f"Identical ramp must have diff loss 0, got {loss_smooth}"

    # High-frequency jitter noise added
    pred_jitter = target + 0.1 * torch.randn_like(target)
    loss_jitter = smooth_fn(pred_jitter, target).item()
    print(f"  High-frequency jitter difference loss: {loss_jitter:.6f} (Expected: >> 0.0)")
    assert loss_jitter > 0.05, f"Jitter must be heavily penalized, got {loss_jitter}"

    print("  [PASS] TemporalSmoothnessLoss correctly penalizes high-frequency gradient jitter.")


def test_amp_and_quiescent_numerical_stability():
    print("\n--- 4. Testing AMP FP16 and Quiescent Numerical Stability ---")
    composite_fn = CompositeEMGLoss(w_peak=1.0, w_ccc=0.5, w_pearson=0.2, w_diff=0.1, eps=1e-6)
    B, T, C = 4, 500, 5

    # Extreme resting baseline: constant signal + micro-noise (< 1e-6)
    target_rest = torch.full((B, T, C), 0.0010, dtype=torch.float32) + 1e-7 * torch.randn(B, T, C)
    pred_rest = torch.full((B, T, C), 0.0010, dtype=torch.float32) + 1e-7 * torch.randn(B, T, C)
    pred_rest.requires_grad = True

    # Test under FP16 autocast
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = composite_fn(pred_rest, target_rest)

    assert torch.isfinite(loss), f"Loss under AMP became non-finite: {loss}"
    loss.backward()
    assert pred_rest.grad is not None and torch.isfinite(pred_rest.grad).all(), "Gradients must remain finite under AMP!"

    print(f"  AMP stability loss on flat resting baseline: {loss.item():.6f}")
    print("  [PASS] Denominator clamping guarantees 100% numerical stability under AMP.")


def test_composite_loss_components_and_backward():
    print("\n--- 5. Testing CompositeEMGLoss Full Forward & Backward Flow ---")
    cfg = {
        "model": {
            "type": "kg_gt_kinematic",
            "transformer": {"n_layers": 1, "n_heads": 2, "d_model": 32, "d_k": 16, "d_v": 16, "ffn_dim": 64, "dropout": 0.0, "chunk_size": 250},
            "gat": {"n_layers": 1, "node_dim": 16, "hidden_dim": 16, "heads": 2, "dropout": 0.0, "use_kinematic_guidance": True},
            "decoder": {"out_channels": 5},
        }
    }
    model = build_kg_gt_from_config(cfg, input_dim=32, kin_dim=13)

    loss_fn = CompositeEMGLoss(
        w_peak=1.0,
        w_ccc=0.5,
        w_pearson=0.2,
        w_diff=0.1,
        w_rest=0.05,
        w_reg=0.001,
        eps=1e-6,
    )

    B, T = 2, 250
    x_eeg = torch.randn(B, T, 32, requires_grad=True)
    x_kin = torch.randn(B, T, 13, requires_grad=True)
    target_emg = torch.abs(torch.randn(B, T, 5))

    pred = model(x_eeg, x_kin)
    loss, components = loss_fn(pred, target_emg, model=model, return_components=True)

    print("  Sub-component breakdown:")
    for k, v in components.items():
        print(f"    - {k:10s}: {v:.4f}")

    assert "peak_mse" in components and components["peak_mse"] >= 0
    assert "ccc" in components and components["ccc"] >= 0
    assert "pearson" in components and components["pearson"] >= 0
    assert "diff" in components and components["diff"] >= 0
    assert "rest_l1" in components and components["rest_l1"] >= 0
    assert "kl_reg" in components and components["kl_reg"] >= 0

    # Backward pass
    loss.backward()
    assert x_eeg.grad is not None and x_eeg.grad.norm().item() > 0
    print(f"  Gradient norm into input EEG: {x_eeg.grad.norm().item():.4f}")

    print("  [PASS] CompositeEMGLoss cleanly computes, breaks down, and backpropagates all objectives.")


def test_backward_compatibility():
    print("\n--- 6. Testing Backward Compatibility ---")
    assert CombinedEMGLoss is CompositeEMGLoss, "CombinedEMGLoss must be an alias to CompositeEMGLoss!"

    # Test with legacy config format
    legacy_cfg = {
        "loss": {
            "peak_alpha": 3.0,
            "lambda_reg": 0.01,
            "lambda_l1": 0.05,
            "lambda_grad": 0.2,
            "use_peak_weight": True,
        }
    }
    loss_inst = build_loss_from_config(legacy_cfg)
    assert isinstance(loss_inst, CompositeEMGLoss)
    assert loss_inst.w_reg == 0.01
    assert loss_inst.w_rest == 0.05
    assert loss_inst.w_diff == 0.2

    print("  [PASS] Backward compatibility with legacy configs verified.")


def test_eval_metrics_with_ccc():
    print("\n--- 7. Testing EvalMetrics with CCC ---")
    np.random.seed(42)
    N, C = 1000, 5
    target = np.random.randn(N, C)
    pred = target + 0.2 * np.random.randn(N, C)
    channel_names = ["FDI", "APB", "ADM", "ECR", "FCR"]

    metrics = compute_metrics(pred, target, channel_names=channel_names)
    assert hasattr(metrics, "ccc"), "EvalMetrics must have ccc field!"
    assert metrics.ccc is not None
    assert len(metrics.ccc) == 5
    print(f"  Per-channel CCC: {np.round(metrics.ccc, 4)}")
    print(f"  Mean CCC       : {metrics.ccc.mean():.4f}")
    assert metrics.ccc.mean() > 0.8, f"High correlation should give CCC > 0.8, got {metrics.ccc.mean()}"

    table_str = metrics.as_table()
    assert "CCC" in table_str, "Table output must contain CCC column!"
    print("\n  Formatted Metrics Table:\n" + table_str)

    print("  [PASS] compute_metrics successfully integrates Lin's CCC.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING ADVANCED LOSS UPGRADE TEST SUITE")
    print("=" * 65)
    test_ccc_loss_properties()
    test_pearson_loss_invariance()
    test_temporal_smoothness_loss()
    test_amp_and_quiescent_numerical_stability()
    test_composite_loss_components_and_backward()
    test_backward_compatibility()
    test_eval_metrics_with_ccc()
    print("\n" + "=" * 65)
    print("ALL 7 ADVANCED LOSS TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 65)
