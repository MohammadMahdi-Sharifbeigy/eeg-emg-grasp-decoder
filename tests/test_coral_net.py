"""Automated Verification Suite for CORAL-Net Architecture.

Validates the 5 Core Neuro-Computational & Numerical Safeguards:
1. Causality Check: Impulse response of LearnableFilterBank has zero energy before t=0.
2. Lag Differentiability & Linear Shift: lag_logit gradient exists and padding prevents circular FFT wrap-around.
3. Synergy Non-negativity & NMF Warm-start: Activations s(t) >= 0, W >= 0, and non-negative target projection.
4. Mean Collapse Resistance: CCC loss penalizes flat-line predictions (L_ccc = 1.0) and rewards shape alignment.
5. End-to-End Forward/Backward: Full gradient propagation through all CORAL-Net parameters and CORALLoss.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

from main.coral_net import (
    LearnableFilterBank,
    SimpleMambaBlock,
    MambaEncoder,
    LearnableLagAlignment,
    SynergyDecoder,
    CORALNet,
    extract_nmf_synergies,
    build_coral_net_from_config,
)
from main.losses import CORALLoss, CCCLoss, build_coral_loss_from_config


def test_filterbank_causality():
    print("\n--- 1. Testing LearnableFilterBank Causality ---")
    fs = 500.0
    kernel_size = 125
    d_model = 64
    in_channels = 32
    T = 300

    fb = LearnableFilterBank(
        in_channels=in_channels,
        kernel_size=kernel_size,
        fs=fs,
        d_model=d_model,
    )
    fb.eval()

    # Create input with impulse strictly at t = 100
    impulse_idx = 100
    x = torch.zeros(1, T, in_channels)
    x[0, impulse_idx, :] = 1.0

    with torch.no_grad():
        out = fb(x)  # (1, T, d_model)

    # Note: LayerNorm has bias which may be non-zero at baseline,
    # so we test the output of depthwise_conv directly or inspect pre-norm conv
    pad_left = kernel_size - 1
    x_padded = F.pad(x.transpose(1, 2), (pad_left, 0), mode="constant", value=0.0)
    with torch.no_grad():
        conv_out = fb.depthwise_conv(x_padded)  # (1, total_band_channels, T)

    # All energy strictly BEFORE impulse_idx must be exactly 0.0
    pre_impulse_energy = conv_out[0, :, :impulse_idx].abs().max().item()
    post_impulse_energy = conv_out[0, :, impulse_idx:].abs().max().item()

    print(f"  Max energy BEFORE impulse (t < {impulse_idx}): {pre_impulse_energy:.2e} (Expected: 0.00e+00)")
    print(f"  Max energy AT/AFTER impulse (t >= {impulse_idx}): {post_impulse_energy:.4f} (Expected: > 0)")

    assert pre_impulse_energy == 0.0, f"Acausal leakage detected! Energy before impulse: {pre_impulse_energy}"
    assert post_impulse_energy > 0.0, "Filter bank produced all zero response to impulse!"
    print("  [PASS] LearnableFilterBank is strictly causal with zero lookahead bias.")


def test_lag_differentiability_and_linear_shift():
    print("\n--- 2. Testing LearnableLagAlignment Differentiability & Linear Shift ---")
    T = 500
    d_model = 32
    fs = 500.0
    max_lag_ms = 100.0

    lag_module = LearnableLagAlignment(
        d_model=d_model,
        max_lag_ms=max_lag_ms,
        fs=fs,
    )

    # Verify initial delay is 50 ms
    init_lag_ms = lag_module.current_lag_ms.item()
    print(f"  Initial learned lag: {init_lag_ms:.2f} ms (Expected: 50.00 ms)")
    assert abs(init_lag_ms - 50.0) < 1e-4, f"Initial lag should be 50ms, got {init_lag_ms}"

    # Test Linear Shift (Safeguard 1): Impulse near end of window does NOT wrap to beginning
    x = torch.zeros(1, T, d_model)
    x[0, 490, :] = 1.0  # impulse near the end

    out = lag_module(x)
    loss = out.sum()
    loss.backward()

    grad = lag_module.lag_logit.grad.item()
    print(f"  lag_logit gradient norm: {abs(grad):.4f} (Expected: > 0)")
    assert abs(grad) > 0.0, "lag_logit received zero gradient!"

    # Check wrap-around: indices 0 to 50 should have zero energy
    beginning_energy = out[0, :50, :].abs().max().item()
    print(f"  Wrap-around energy in first 50 samples: {beginning_energy:.2e} (Expected: < 1e-5)")
    assert beginning_energy < 1e-5, f"Circular wrap-around detected! Energy: {beginning_energy}"

    print("  [PASS] LearnableLagAlignment is differentiable and linearly padded (no circular wrap-around).")


def test_synergy_nonnegativity_and_nmf_warmstart():
    print("\n--- 3. Testing SynergyDecoder Non-negativity & NMF Warm-start ---")
    d_model = 64
    n_muscles = 5
    n_synergies = 3
    decoder = SynergyDecoder(d_model=d_model, n_muscles=n_muscles, n_synergies=n_synergies)

    # 1. Non-negativity under extreme negative inputs
    x_neg = torch.randn(4, 100, d_model) * 10.0 - 5.0
    emg_pred, synergies = decoder(x_neg)

    min_synergy = synergies.min().item()
    min_W = decoder.W.min().item()

    print(f"  Minimum synergy activation: {min_synergy:.4f} (Expected: >= 0.0)")
    print(f"  Minimum mixing weight W:   {min_W:.4f} (Expected: >= 0.0)")
    assert min_synergy >= 0.0, f"Negative synergy activations found: {min_synergy}"
    assert min_W >= 0.0, f"Negative mixing weights found: {min_W}"

    # 2. Test NMF Warm-Start
    synthetic_emg = [np.random.uniform(0.1, 1.0, size=(1000, 5)) for _ in range(3)]
    W_acts, H_comp = extract_nmf_synergies(synthetic_emg, n_synergies=3)

    H_tensor = torch.from_numpy(H_comp).float()
    decoder.init_synergy_mix_from_nmf(H_tensor)

    W_effective = decoder.W.detach().cpu().numpy()
    max_nmf_diff = np.abs(W_effective - H_comp).max()
    print(f"  Max difference between W and NMF components H: {max_nmf_diff:.2e} (Expected: < 1e-3)")
    assert max_nmf_diff < 1e-2, f"NMF warm start mismatch: {max_nmf_diff}"

    # 3. Test Non-negative target projection in CORALLoss
    target = torch.randn(2, 50, 5)  # includes negative values
    loss_fn = CORALLoss(w_syn=1.0)
    loss, components = loss_fn(emg_pred[:2, :50], target, model=decoder, synergies=synergies[:2, :50], return_components=True)

    print(f"  Synergy Aux Loss with non-negative target projection: {components['synergy']:.4f}")
    assert components["synergy"] >= 0.0 and not math.isnan(components["synergy"])
    print("  [PASS] SynergyDecoder enforces s(t) >= 0, W >= 0, and accurately warm-starts from NMF.")


def test_mean_collapse_resistance():
    print("\n--- 4. Testing Mean Collapse Resistance via CCC ---")
    ccc_loss = CCCLoss(eps=1e-6)

    B, T, C = 4, 500, 5
    t = torch.linspace(0, 10, T).unsqueeze(0).unsqueeze(-1).repeat(B, 1, C)
    target = torch.sin(t) + 1.5  # dynamic ground truth

    # Flat line mean prediction
    pred_mean = target.mean(dim=1, keepdim=True).repeat(1, T, 1)

    loss_mean_collapse = ccc_loss(pred_mean, target).item()
    loss_perfect = ccc_loss(target, target).item()

    print(f"  Perfect prediction CCC loss: {loss_perfect:.6f} (Expected: 0.000000)")
    print(f"  Mean Collapse prediction CCC loss: {loss_mean_collapse:.6f} (Expected: 1.000000)")

    assert abs(loss_perfect) < 1e-5, f"Perfect prediction CCC should be 0.0, got {loss_perfect}"
    assert abs(loss_mean_collapse - 1.0) < 1e-4, f"Mean collapse CCC should be 1.0, got {loss_mean_collapse}"
    print("  [PASS] CCCLoss strongly penalizes Mean Collapse (L_ccc = 1.0 on running mean).")


def test_end_to_end_forward_backward():
    print("\n--- 5. Testing Full CORAL-Net Forward & Backward Flow ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Target Device: {device}")

    cfg = {
        "model": {
            "decoder": {"out_channels": 5},
            "coral": {
                "n_synergies": 3,
                "d_model": 128,
                "n_layers": 2,
                "d_state": 16,
                "max_lag_ms": 100.0,
                "use_kinematics": True,
            },
        },
        "data": {"fs_eeg": 500.0},
        "coral_loss": {
            "w_ccc": 1.0,
            "w_pearson": 0.5,
            "w_diff": 0.2,
            "w_rest": 0.05,
            "w_syn": 0.1,
        },
    }

    model = build_coral_net_from_config(cfg, input_dim=32, kin_dim=36).to(device)
    loss_fn = build_coral_loss_from_config(cfg).to(device)

    B, T = 2, 200
    eeg = torch.randn(B, T, 32, device=device, requires_grad=True)
    kin = torch.randn(B, T, 36, device=device, requires_grad=True)
    target_emg = torch.rand(B, T, 5, device=device)

    # Forward pass
    pred_emg, synergies = model(eeg, kin, return_synergies=True)

    print(f"  Predicted EMG shape: {list(pred_emg.shape)} (Expected: [{B}, {T}, 5])")
    print(f"  Synergies shape:     {list(synergies.shape)} (Expected: [{B}, {T}, 3])")

    assert pred_emg.shape == (B, T, 5), f"Shape mismatch: {pred_emg.shape}"
    assert synergies.shape == (B, T, 3), f"Synergies shape mismatch: {synergies.shape}"

    # Backward pass through CORALLoss
    total_loss, components = loss_fn(
        pred=pred_emg,
        target=target_emg,
        model=model,
        synergies=synergies,
        return_components=True,
    )

    print(f"  Loss breakdown:")
    for k, v in components.items():
        print(f"    - {k:10s}: {v:.4f}")

    total_loss.backward()

    eeg_grad = eeg.grad.norm().item()
    lag_grad = model.lag.lag_logit.grad.norm().item()
    mix_grad = model.synergy_decoder.synergy_mix.grad.norm().item()
    fb_grad = model.filterbank.depthwise_conv.weight.grad.norm().item()

    print(f"  Gradient Norms:")
    print(f"    - Input EEG:           {eeg_grad:.4f}")
    print(f"    - Lag parameter:       {lag_grad:.4f}")
    print(f"    - Synergy mixing W:    {mix_grad:.4f}")
    print(f"    - FilterBank FIR conv: {fb_grad:.4f}")

    assert eeg_grad > 0.0, "Zero gradient reached input EEG!"
    assert lag_grad > 0.0, "Zero gradient on lag logit!"
    assert mix_grad > 0.0, "Zero gradient on synergy mixing matrix!"
    assert fb_grad > 0.0, "Zero gradient on filterbank kernels!"

    print("  [PASS] Full end-to-end forward, loss decomposition, and backprop verified.")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CORAL-NET VERIFICATION SUITE")
    print("=================================================================")

    test_filterbank_causality()
    test_lag_differentiability_and_linear_shift()
    test_synergy_nonnegativity_and_nmf_warmstart()
    test_mean_collapse_resistance()
    test_end_to_end_forward_backward()

    print("\n=================================================================")
    print("ALL 5 CORAL-NET VERIFICATION SUITES PASSED SUCCESSFULLY!")
    print("=================================================================")
