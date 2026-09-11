"""
tests/test_pivot1_synergy.py
=============================
Test suite for Pivot 1: Muscle Synergy Latent Space Decoding.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from synergy_decoding.nmf_extractor import (
    compute_vaf,
    extract_nmf_synergies,
    vaf_curve,
    align_synergies,
    CrossSubjectSimilarity,
)
from synergy_decoding.synergy_model import SynergyHead, CorticosynergyDecoder
from synergy_decoding.losses import SynergyDualObjectiveLoss, SynergyLossConfig
from synergy_decoding.dataset import SynergyDataset
from synergy_decoding.evaluator import SynergyEvaluator, paired_wilcoxon_test


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. NMF Extractor Tests
# ============================================================================

class TestNMFExtractor:
    
    def test_compute_vaf_perfect(self):
        M = np.random.rand(100, 5)
        assert np.isclose(compute_vaf(M, M, np.eye(100)), 1.0)
        
    def test_compute_vaf_zero(self):
        M = np.ones((10, 5))
        M_hat = np.zeros((10, 5))
        assert compute_vaf(M, np.eye(5), M_hat) == 0.0
        
    def test_extract_nmf_synergies(self):
        np.random.seed(42)
        # Create synthetic low-rank data
        T, K, N = 500, 3, 5
        C_true = np.random.rand(T, K)
        W_true = np.random.rand(K, N)
        M = C_true @ W_true + 0.01 * np.random.rand(T, N)
        
        W, C, vaf = extract_nmf_synergies(M, k=K, n_init=2, smooth_C=False)
        assert W.shape == (K, N)
        assert C.shape == (T, K)
        assert vaf > 0.9  # Should reconstruct very well
        assert np.all(W >= 0)
        assert np.all(C >= 0)

    def test_vaf_curve(self):
        np.random.seed(42)
        T, K, N = 500, 3, 5
        C_true = np.random.rand(T, K)
        W_true = np.random.rand(K, N)
        M = C_true @ W_true
        
        curve = vaf_curve(M, k_range=(1, 5), n_init=1, max_iter=200)
        assert 1 in curve and 4 in curve
        # VAF should monotonically increase
        assert curve[2] >= curve[1]
        assert curve[4] >= curve[3]
        
    def test_align_synergies(self):
        np.random.seed(42)
        W_ref = np.random.rand(3, 5)
        # Permute W_ref
        W_tgt = W_ref[[2, 0, 1], :]
        
        W_aligned, perm = align_synergies(W_ref, W_tgt)
        assert np.allclose(W_aligned, W_ref)
        assert list(perm) == [1, 2, 0]


# ============================================================================
# 2. Synergy Model Tests
# ============================================================================

class TestSynergyModel:
    
    def test_synergy_head_non_negative(self):
        head = SynergyHead(d_model=32, k=3)
        # Extreme negative inputs
        x = -100.0 * torch.ones(4, 10, 32)
        out = head(x)
        assert out.shape == (4, 10, 3)
        assert torch.all(out >= 0), "SynergyHead output must be strictly non-negative"
        
    def test_decoder_forward(self):
        model = CorticosynergyDecoder(n_eeg=32, k=3, d_model=64, n_layers=2, max_lag_ms=100.0).to(DEVICE)
        x = torch.randn(2, 50, 32, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 50, 3)
        assert torch.all(out >= 0)
        
    def test_gradient_flow(self):
        model = CorticosynergyDecoder(n_eeg=32, k=3, d_model=64, n_layers=2).to(DEVICE)
        x = torch.randn(2, 50, 32, device=DEVICE)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients reached the filterbank
        assert model.filterbank.depthwise_conv.weight.grad is not None
        assert model.filterbank.depthwise_conv.weight.grad.abs().max() > 0


# ============================================================================
# 3. Dual-Objective Loss Tests
# ============================================================================

class TestSynergyLoss:
    
    def test_loss_forward(self):
        cfg = SynergyLossConfig()
        loss_fn = SynergyDualObjectiveLoss(cfg)
        
        B, T, K, N = 2, 50, 3, 5
        C_hat = torch.rand(B, T, K, requires_grad=True)
        C_gt = torch.rand(B, T, K)
        W = torch.rand(K, N)
        M = torch.rand(B, T, N)
        
        loss = loss_fn(C_hat, C_gt, W, M)
        assert not torch.isnan(loss)
        assert loss.item() > 0
        
        loss.backward()
        assert C_hat.grad is not None


# ============================================================================
# 4. Dataset & Evaluator Tests
# ============================================================================

class TestDatasetEvaluator:
    
    def test_synergy_dataset(self):
        eeg = [np.random.rand(1000, 32) for _ in range(2)]
        emg = [np.random.rand(1000, 5) for _ in range(2)]
        c = [np.random.rand(1000, 3) for _ in range(2)]
        w_dict = {1: np.random.rand(3, 5), 2: np.random.rand(3, 5)}
        subj_ids = [1, 2]
        
        ds = SynergyDataset(eeg, emg, c, w_dict, subj_ids, window_size=500, stride=500)
        assert len(ds) == 4  # 2 windows per trial * 2 trials
        
        item = ds[0]
        assert item["eeg"].shape == (500, 32)
        assert item["emg"].shape == (500, 5)
        assert item["c"].shape == (500, 3)
        assert item["w"].shape == (3, 5)
        assert item["subject_id"].item() == 1
        
    def test_evaluator(self):
        evaluator = SynergyEvaluator()
        
        # Perfect match
        x = np.random.rand(100, 3)
        r, vaf = evaluator.evaluate_synergy_predictions(x, x)
        assert np.isclose(r, 1.0)
        assert np.isclose(vaf, 1.0)
        
        # Mismatch
        y = np.random.rand(100, 3)
        r, vaf = evaluator.evaluate_synergy_predictions(x, y)
        assert r < 1.0
        
    def test_paired_wilcoxon(self):
        synergy = np.array([0.8, 0.85, 0.9])
        direct = np.array([0.5, 0.6, 0.7])
        
        stat, pval, sig = paired_wilcoxon_test(synergy, direct)
        assert pval < 0.5  # Should be significant/small
        assert "p=" in sig


if __name__ == "__main__":
    pytest.main(["-v", __file__])
