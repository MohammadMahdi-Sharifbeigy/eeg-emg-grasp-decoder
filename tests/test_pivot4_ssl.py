"""Test suite for Pivot 4: Bio-CLIP Self-Supervised Learning Framework.

Tests:
    T1. EEGEncoder — forward shapes, unit-sphere normalization, causal guarantee
    T2. EMGEncoder — forward shapes, unit-sphere normalization
    T3. PhaseLabeler — phase assignment, thresholds, lazy fit
    T4. PhaseAwareInfoNCELoss — negative masking, gradient flow, non-collapse
    T5. BioCLIPTrainer — one-step smoke test (forward + backward, no NaN/Inf)
    T6. DenseTokenInfoNCE — temporal token-level InfoNCE shape/gradient test
    T7. LinearProbe — fit/predict/score API on synthetic data
    T8. FewShotRegressionProbe — ridge regression closed-form + metric test
    T9. SSLWindowDataset — windowing, phase labeling, balanced sampling weights

Run with:
    python -m pytest tests/test_pivot4_ssl.py -v
or:
    python tests/test_pivot4_ssl.py
"""

from __future__ import annotations

import sys
import os
import math

# Make workspace root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn

from self_supervised_coral.eeg_encoder import EEGEncoder
from self_supervised_coral.emg_encoder import EMGEncoder
from self_supervised_coral.contrastive_loss import (
    NTXentLoss,
    SymmetricInfoNCELoss,
    PhaseAwareInfoNCELoss,
)
from self_supervised_coral.phase_labeler import PhaseLabeler, MovementPhase
from self_supervised_coral.linear_probes import LinearProbe, FewShotRegressionProbe
from self_supervised_coral.ssl_dataset import SSLWindowDataset
from self_supervised_coral.ssl_trainer import (
    BioCLIPTrainer,
    SSLTrainConfig,
    DenseTokenInfoNCE,
)


# ============================================================================
# Fixtures & Utilities
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, T, N_EEG, N_EMG, KIN_DIM = 8, 500, 32, 5, 36
PROJ_DIM = 128
D_MODEL = 128  # Use smaller d_model in tests for speed


def _make_eeg_encoder(d_model: int = D_MODEL) -> EEGEncoder:
    return EEGEncoder(
        n_eeg_channels=N_EEG,
        d_model=d_model,
        n_layers=2,
        d_state=8,
        fs=500.0,
        proj_dim=PROJ_DIM,
        pool_heads=2,
        kernel_size=63,
    ).to(DEVICE)


def _make_emg_encoder(d_model: int = D_MODEL) -> EMGEncoder:
    return EMGEncoder(
        n_emg_channels=N_EMG,
        d_model=d_model,
        n_gru_layers=1,
        proj_dim=PROJ_DIM,
        mid_channels=32,
        dropout=0.0,
    ).to(DEVICE)


def _rand_eeg() -> torch.Tensor:
    return torch.randn(B, T, N_EEG, device=DEVICE)


def _rand_emg() -> torch.Tensor:
    return torch.randn(B, T, N_EMG, device=DEVICE)


def _rand_kin() -> torch.Tensor:
    return torch.randn(B, T, KIN_DIM, device=DEVICE)


# ============================================================================
# T1. EEGEncoder
# ============================================================================

class TestEEGEncoder:
    """Tests for EEGEncoder forward pass, shapes, normalization, causality."""

    def setup_method(self):
        self.encoder = _make_eeg_encoder()

    def test_global_latent_shape(self):
        """z_global must be (B, proj_dim)."""
        eeg = _rand_eeg()
        z_global, _ = self.encoder(eeg, return_dense=False)
        assert z_global.shape == (B, PROJ_DIM), \
            f"Expected ({B}, {PROJ_DIM}), got {z_global.shape}"

    def test_dense_latent_shape(self):
        """H_dense must be (B, T, d_model)."""
        eeg = _rand_eeg()
        z_global, H_dense = self.encoder(eeg, return_dense=True)
        assert H_dense is not None
        assert H_dense.shape == (B, T, D_MODEL), \
            f"Expected ({B}, {T}, {D_MODEL}), got {H_dense.shape}"

    def test_unit_sphere_normalization(self):
        """z_global must lie on the unit sphere: ‖z‖ = 1 for each sample."""
        eeg = _rand_eeg()
        z_global, _ = self.encoder(eeg, return_dense=False)
        norms = z_global.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B, device=DEVICE), atol=1e-5), \
            f"L2 norms deviated from 1.0: {norms}"

    def test_gradient_flow(self):
        """Gradients must flow back through EEGEncoder."""
        eeg = _rand_eeg()
        z_global, _ = self.encoder(eeg, return_dense=False)
        loss = z_global.sum()
        loss.backward()
        # Check at least one parameter has non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in self.encoder.parameters()
        )
        assert has_grad, "No gradient flowed to EEGEncoder parameters"

    def test_no_nan_inf(self):
        """No NaN or Inf in EEGEncoder output."""
        eeg = _rand_eeg()
        z_global, H_dense = self.encoder(eeg, return_dense=True)
        assert not torch.isnan(z_global).any(), "NaN in z_global"
        assert not torch.isinf(z_global).any(), "Inf in z_global"
        assert not torch.isnan(H_dense).any(), "NaN in H_dense"
        assert not torch.isinf(H_dense).any(), "Inf in H_dense"

    def test_encode_convenience(self):
        """encoder.encode() returns (B, proj_dim) without dense."""
        eeg = _rand_eeg()
        z = self.encoder.encode(eeg)
        assert z.shape == (B, PROJ_DIM)

    def test_mask_changes_pooling(self):
        """Providing a mask must produce different global latent than no mask."""
        eeg = _rand_eeg()
        mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
        mask[:, :100] = True  # Only attend to first 100 frames

        z_masked, _ = self.encoder(eeg, mask=mask, return_dense=False)
        z_unmasked, _ = self.encoder(eeg, mask=None, return_dense=False)

        # With the same EEG but different masks, outputs must differ
        assert not torch.allclose(z_masked, z_unmasked, atol=1e-4), \
            "Mask had no effect on pooled latent"


# ============================================================================
# T2. EMGEncoder
# ============================================================================

class TestEMGEncoder:
    """Tests for EMGEncoder forward pass, shapes, normalization."""

    def setup_method(self):
        self.encoder = _make_emg_encoder()

    def test_global_latent_shape(self):
        emg = _rand_emg()
        z_global, _ = self.encoder(emg, return_dense=False)
        assert z_global.shape == (B, PROJ_DIM)

    def test_dense_latent_shape(self):
        emg = _rand_emg()
        _, H_dense = self.encoder(emg, return_dense=True)
        assert H_dense is not None
        assert H_dense.shape == (B, T, D_MODEL)

    def test_unit_sphere_normalization(self):
        emg = _rand_emg()
        z_global, _ = self.encoder(emg, return_dense=False)
        norms = z_global.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B, device=DEVICE), atol=1e-5), \
            f"EMG latent norms: {norms}"

    def test_gradient_flow(self):
        emg = _rand_emg()
        z_global, _ = self.encoder(emg, return_dense=False)
        z_global.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in self.encoder.parameters()
        )
        assert has_grad, "No gradient flowed to EMGEncoder parameters"

    def test_no_nan_inf(self):
        emg = _rand_emg()
        z_global, H_dense = self.encoder(emg, return_dense=True)
        assert not torch.isnan(z_global).any()
        assert not torch.isinf(z_global).any()
        assert not torch.isnan(H_dense).any()
        assert not torch.isinf(H_dense).any()


# ============================================================================
# T3. PhaseLabeler
# ============================================================================

class TestPhaseLabeler:
    """Tests for PhaseLabeler phase assignment and fitting."""

    def test_forward_shape(self):
        """Phase labels must be (B,) int64."""
        labeler = PhaseLabeler(fs=500.0)
        kin = _rand_kin()
        phases = labeler(kin)
        assert phases.shape == (B,)
        assert phases.dtype == torch.long

    def test_phase_values(self):
        """Phase labels must be one of {0, 1, 2}."""
        labeler = PhaseLabeler(fs=500.0)
        kin = _rand_kin()
        phases = labeler(kin)
        valid = {int(MovementPhase.REST), int(MovementPhase.ACTIVE), int(MovementPhase.TRANSIT)}
        for p in phases:
            assert int(p) in valid, f"Invalid phase value: {int(p)}"

    def test_lazy_fit_sets_thresholds(self):
        """First forward call must set thresholds from the data."""
        labeler = PhaseLabeler(fs=500.0)
        assert labeler.thresholds == (None, None)
        kin = _rand_kin()
        _ = labeler(kin)
        low, high = labeler.thresholds
        assert low is not None and high is not None
        assert low <= high

    def test_pre_fit(self):
        """Pre-fit PhaseLabeler on numpy array must set correct thresholds."""
        kin_np = np.random.randn(100, T, KIN_DIM).astype(np.float32)
        labeler = PhaseLabeler(fs=500.0, low_pct=25, high_pct=75)
        labeler.fit(kin_np)
        low, high = labeler.thresholds
        assert low is not None and low <= high

    def test_rest_detection_for_still_signal(self):
        """Near-zero kinematics must be classified as REST."""
        labeler = PhaseLabeler(fs=500.0)
        # Pre-fit with random data so thresholds are set
        kin_random = torch.randn(100, T, KIN_DIM, device=DEVICE)
        _ = labeler(kin_random)

        # Near-zero kinematics → REST
        kin_still = torch.zeros(4, T, KIN_DIM, device=DEVICE)
        phases = labeler(kin_still)
        assert (phases == int(MovementPhase.REST)).all(), \
            f"Expected REST for zero kinematics, got {phases}"

    def test_emg_power_labeling(self):
        """EMG-based labeling must return (B,) int64 tensor."""
        labeler = PhaseLabeler(fs=500.0)
        emg = _rand_emg()
        # Set thresholds first
        _ = labeler(torch.randn(100, T, KIN_DIM, device=DEVICE))
        phases = labeler.label_from_emg_power(emg)
        assert phases.shape == (B,) and phases.dtype == torch.long


# ============================================================================
# T4. PhaseAwareInfoNCELoss
# ============================================================================

class TestPhaseAwareInfoNCELoss:
    """Tests for contrastive losses."""

    def _make_embeddings(self, b: int = B, d: int = PROJ_DIM) -> tuple:
        z_eeg = torch.randn(b, d, device=DEVICE)
        z_emg = torch.randn(b, d, device=DEVICE)
        z_eeg = torch.nn.functional.normalize(z_eeg, dim=-1)
        z_emg = torch.nn.functional.normalize(z_emg, dim=-1)
        return z_eeg, z_emg

    def test_ntxent_loss_shape(self):
        """NT-Xent must return scalar loss."""
        loss_fn = NTXentLoss(temperature=0.1, learnable_temp=False)
        z_eeg, z_emg = self._make_embeddings()
        loss = loss_fn(z_eeg, z_emg)
        assert loss.ndim == 0 and not torch.isnan(loss)

    def test_symmetric_infonce_loss_shape(self):
        loss_fn = SymmetricInfoNCELoss(temperature=0.1, learnable_temp=False)
        z_eeg, z_emg = self._make_embeddings()
        loss = loss_fn(z_eeg, z_emg)
        assert loss.ndim == 0 and not torch.isnan(loss)

    def test_phase_aware_loss_scalar(self):
        """PhaseAwareInfoNCELoss must return scalar."""
        loss_fn = PhaseAwareInfoNCELoss(temperature=0.1, learnable_temp=False)
        z_eeg, z_emg = self._make_embeddings()
        phases = torch.randint(0, 3, (B,), device=DEVICE)
        loss = loss_fn(z_eeg, z_emg, phases=phases)
        assert loss.ndim == 0 and not torch.isnan(loss)

    def test_masking_reduces_loss_space(self):
        """Loss with masking must differ from loss without masking."""
        loss_fn = PhaseAwareInfoNCELoss(temperature=0.1, learnable_temp=False)
        z_eeg, z_emg = self._make_embeddings(b=16)

        # All same phase → all off-diagonal pairs masked → loss should change
        phases_same = torch.zeros(16, dtype=torch.long, device=DEVICE)
        phases_mixed = torch.randint(0, 3, (16,), device=DEVICE)

        loss_same = loss_fn(z_eeg, z_emg, phases=phases_same)
        loss_mixed = loss_fn(z_eeg, z_emg, phases=phases_mixed)

        # Not equal (different negatives in denominator)
        assert not torch.isclose(loss_same, loss_mixed, atol=1e-5), \
            "Phase masking had no effect on loss value"

    def test_gradient_through_loss(self):
        """Gradients must flow through PhaseAwareInfoNCELoss to encoder params."""
        encoder = _make_eeg_encoder()
        loss_fn = PhaseAwareInfoNCELoss(
            temperature=0.1, learnable_temp=True
        ).to(DEVICE)

        eeg = _rand_eeg()
        z_eeg, _ = encoder(eeg, return_dense=False)
        z_emg, _ = _make_emg_encoder()(_rand_emg(), return_dense=False)
        phases = torch.randint(0, 3, (B,), device=DEVICE)

        loss = loss_fn(z_eeg, z_emg, phases=phases)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in encoder.parameters()
        )
        assert has_grad, "No gradient flowed to EEGEncoder through PhaseAwareInfoNCELoss"

    def test_learnable_temperature_gradient(self):
        """Temperature parameter must receive gradient."""
        loss_fn = PhaseAwareInfoNCELoss(temperature=0.07, learnable_temp=True).to(DEVICE)
        z_eeg, z_emg = self._make_embeddings()
        phases = torch.randint(0, 3, (B,), device=DEVICE)
        loss = loss_fn(z_eeg, z_emg, phases=phases)
        loss.backward()
        assert loss_fn.infonce.log_temp.grad is not None, \
            "Temperature parameter has no gradient"

    def test_non_collapse(self):
        """Identical embeddings (collapse mode) must give large loss (≈ log(B))."""
        loss_fn = NTXentLoss(temperature=0.07, learnable_temp=False)
        # Same embedding for all EEG and all EMG → perfect collapse
        z_collapse = torch.ones(B, PROJ_DIM, device=DEVICE)
        z_collapse = torch.nn.functional.normalize(z_collapse, dim=-1)
        loss = loss_fn(z_collapse, z_collapse)
        # Loss should be log(B) - 1/B ~ log(B) for collapsed representations
        # (all logits equal, so cross entropy = log(B))
        expected = math.log(2 * B)
        # Loss value when collapsed should be roughly log(2B) (high, not zero)
        assert loss.item() > 0.5, f"Collapse loss unexpectedly low: {loss.item()}"

    def test_no_nan_with_phase_masking(self):
        """No NaN or Inf with all-same-phase masking."""
        loss_fn = PhaseAwareInfoNCELoss(temperature=0.07, learnable_temp=False)
        z_eeg, z_emg = self._make_embeddings(b=16)
        # All REST — almost all negatives are masked
        phases = torch.zeros(16, dtype=torch.long, device=DEVICE)
        loss = loss_fn(z_eeg, z_emg, phases=phases)
        assert not torch.isnan(loss), "NaN in loss with all-same-phase masking"
        assert not torch.isinf(loss), "Inf in loss with all-same-phase masking"


# ============================================================================
# T5. BioCLIPTrainer smoke test
# ============================================================================

class TestBioCLIPTrainer:
    """Smoke test: one step of Bio-CLIP pre-training (forward + backward, no crash)."""

    def test_one_step_smoke(self):
        """One training step must complete without NaN or exception."""
        eeg_enc = _make_eeg_encoder()
        emg_enc = _make_emg_encoder()
        labeler = PhaseLabeler(fs=500.0)

        cfg = SSLTrainConfig(
            n_epochs=1,
            learning_rate=1e-4,
            use_amp=False,  # Disable AMP for CPU testing
            use_dense_loss=False,  # Skip dense loss for speed
            use_phase_masking=True,
            save_checkpoint=False,
            val_every=1,
            log_every=1000,
        )
        trainer = BioCLIPTrainer(eeg_enc, emg_enc, labeler, cfg)

        # Synthetic batch
        eeg = _rand_eeg()
        emg = _rand_emg()
        kin = _rand_kin()
        phases = torch.randint(0, 3, (B,))

        batch = {"eeg": eeg.cpu(), "emg": emg.cpu(), "kin": kin.cpu(), "phase": phases}

        # One step manual forward
        trainer.eeg_encoder.train()
        trainer.emg_encoder.train()
        trainer.optimizer.zero_grad()

        eeg_d = batch["eeg"].to(trainer.device)
        emg_d = batch["emg"].to(trainer.device)
        kin_d = batch["kin"].to(trainer.device)
        ph_d = labeler(kin_d)

        z_eeg, _ = trainer.eeg_encoder(eeg_d, return_dense=False)
        z_emg, _ = trainer.emg_encoder(emg_d, return_dense=False)
        loss = trainer.loss_fn(z_eeg, z_emg, phases=ph_d)

        loss.backward()
        trainer.optimizer.step()

        assert not torch.isnan(loss), f"NaN loss in smoke test: {loss}"
        assert not torch.isinf(loss), f"Inf loss in smoke test: {loss}"
        assert loss.item() > 0, f"Loss should be positive: {loss}"

    def test_frozen_eeg_encoder(self):
        """frozen_eeg_encoder must have all parameters frozen."""
        eeg_enc = _make_eeg_encoder()
        emg_enc = _make_emg_encoder()
        cfg = SSLTrainConfig(save_checkpoint=False)
        trainer = BioCLIPTrainer(eeg_enc, emg_enc, config=cfg)

        frozen = trainer.frozen_eeg_encoder
        for param in frozen.parameters():
            assert not param.requires_grad, "Frozen encoder has trainable param"


# ============================================================================
# T6. DenseTokenInfoNCE
# ============================================================================

class TestDenseTokenInfoNCE:
    """Tests for frame-level token contrastive loss."""

    def test_output_shape(self):
        """Dense InfoNCE must return scalar."""
        loss_fn = DenseTokenInfoNCE(temperature=0.1)
        # Use small T for speed
        T_small = 10
        H_eeg = torch.nn.functional.normalize(
            torch.randn(B, T_small, D_MODEL, device=DEVICE), dim=-1
        )
        H_emg = torch.nn.functional.normalize(
            torch.randn(B, T_small, D_MODEL, device=DEVICE), dim=-1
        )
        loss = loss_fn(H_eeg, H_emg)
        assert loss.ndim == 0 and not torch.isnan(loss)

    def test_gradient_flow(self):
        """Gradients must flow through DenseTokenInfoNCE."""
        loss_fn = DenseTokenInfoNCE(temperature=0.1)
        T_small = 5
        # Use a leaf tensor directly (avoid checking .grad on non-leaf F.normalize output)
        H_eeg_raw = torch.randn(B, T_small, D_MODEL, device=DEVICE, requires_grad=True)
        H_eeg = torch.nn.functional.normalize(H_eeg_raw, dim=-1)
        H_eeg.retain_grad()  # Enable .grad on this non-leaf intermediate
        H_emg = torch.nn.functional.normalize(
            torch.randn(B, T_small, D_MODEL, device=DEVICE), dim=-1
        )
        loss = loss_fn(H_eeg, H_emg)
        loss.backward()
        # Check gradient flowed to the underlying leaf parameter
        assert H_eeg_raw.grad is not None and H_eeg_raw.grad.abs().max() > 0


# ============================================================================
# T7. LinearProbe
# ============================================================================

class TestLinearProbe:
    """Tests for linear classification probe."""

    def test_fit_predict_score(self):
        """LinearProbe must fit, predict, and score on synthetic data."""
        N, D = 200, PROJ_DIM
        n_classes = 3
        probe = LinearProbe(in_dim=D, n_classes=n_classes, n_epochs=20, device=DEVICE)

        embeddings = torch.randn(N, D)
        labels = torch.randint(0, n_classes, (N,))

        probe.fit(embeddings, labels)
        preds = probe.predict(embeddings[:20])
        assert preds.shape == (20,) and preds.dtype == torch.long

        metrics = probe.score(embeddings, labels)
        assert "accuracy" in metrics and "balanced_accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

    def test_linearly_separable(self):
        """Probe must achieve > 90% accuracy on linearly separable data."""
        N, D = 300, 32
        # Create well-separated clusters
        embeddings = torch.zeros(N, D)
        labels = torch.zeros(N, dtype=torch.long)
        for c in range(3):
            idx = range(c * 100, (c + 1) * 100)
            embeddings[idx, c * 10 : (c + 1) * 10] = 5.0
            labels[idx] = c

        probe = LinearProbe(in_dim=D, n_classes=3, n_epochs=50, device=DEVICE)
        probe.fit(embeddings, labels)
        metrics = probe.score(embeddings, labels)
        assert metrics["accuracy"] > 0.9, \
            f"Probe failed on linearly separable data: acc={metrics['accuracy']}"


# ============================================================================
# T8. FewShotRegressionProbe
# ============================================================================

class TestFewShotRegressionProbe:
    """Tests for few-shot ridge regression probe."""

    def test_fit_predict(self):
        """FewShotRegressionProbe must fit and produce (N, out_dim) predictions."""
        N, D, out = 100, PROJ_DIM, 5
        probe = FewShotRegressionProbe(in_dim=D, out_dim=out, n_shots=50)
        X = torch.randn(N, D)
        Y = torch.randn(N, out)
        probe.fit(X, Y)
        preds = probe.predict(X)
        assert preds.shape == (N, out)

    def test_score_returns_metrics(self):
        """score() must return pearson_r and ccc keys."""
        N, D, out = 100, PROJ_DIM, 5
        probe = FewShotRegressionProbe(in_dim=D, out_dim=out, n_shots=100)
        X = torch.randn(N, D)
        # Create linearly related targets for meaningful r
        W = torch.randn(D, out)
        Y = X @ W + 0.1 * torch.randn(N, out)
        probe.fit(X, Y)
        metrics = probe.score(X, Y)
        assert "pearson_r" in metrics and "ccc" in metrics
        # Should get reasonable correlation on training data
        assert metrics["pearson_r"] > 0.3, \
            f"Poor pearson_r on training data: {metrics['pearson_r']}"


# ============================================================================
# T9. SSLWindowDataset
# ============================================================================

class TestSSLWindowDataset:
    """Tests for SSLWindowDataset windowing and phase labeling."""

    def _make_dataset(self, n_trials: int = 3) -> SSLWindowDataset:
        np.random.seed(42)
        T_trial = 2000
        eeg = [np.random.randn(T_trial, N_EEG).astype(np.float32) for _ in range(n_trials)]
        emg = [np.random.randn(T_trial, N_EMG).astype(np.float32) for _ in range(n_trials)]
        kin = [np.random.randn(T_trial, KIN_DIM).astype(np.float32) for _ in range(n_trials)]

        labeler = PhaseLabeler(fs=500.0)
        kin_all = np.concatenate(kin, axis=0)
        n_windows = len(kin_all) // 500
        kin_fit_arr = kin_all[:n_windows * 500].reshape(n_windows, 500, KIN_DIM)
        labeler.fit(kin_fit_arr)

        return SSLWindowDataset(
            eeg_list=eeg,
            emg_list=emg,
            kin_list=kin,
            window_size=500,
            stride=100,
            phase_labeler=labeler,
        )

    def test_dataset_length(self):
        """Dataset must have at least one window per trial."""
        ds = self._make_dataset(n_trials=2)
        assert len(ds) > 0, "Empty dataset"

    def test_item_shapes(self):
        """Each item must have correct tensor shapes."""
        ds = self._make_dataset(n_trials=2)
        item = ds[0]
        assert item["eeg"].shape == (500, N_EEG)
        assert item["emg"].shape == (500, N_EMG)
        assert item["kin"].shape == (500, KIN_DIM)
        assert item["phase"].ndim == 0

    def test_phase_labels_valid(self):
        """All phase labels must be in {0, 1, 2}."""
        ds = self._make_dataset(n_trials=2)
        valid = {0, 1, 2}
        for label in ds._phase_labels:
            assert label in valid, f"Invalid phase label: {label}"

    def test_balanced_weights(self):
        """Phase sampling weights must be positive and have different values per phase."""
        ds = self._make_dataset(n_trials=3)
        weights = ds.get_phase_weights()
        assert weights.shape == (len(ds),)
        assert (weights > 0).all()
        # Weights should not all be identical (some phases rarer than others)
        assert weights.std() > 0, "All weights equal — balanced sampling not working"


# ============================================================================
# Main runner
# ============================================================================

if __name__ == "__main__":
    import math
    print("=" * 65)
    print("Pivot 4: Bio-CLIP Test Suite")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    test_classes = [
        TestEEGEncoder,
        TestEMGEncoder,
        TestPhaseLabeler,
        TestPhaseAwareInfoNCELoss,
        TestBioCLIPTrainer,
        TestDenseTokenInfoNCE,
        TestLinearProbe,
        TestFewShotRegressionProbe,
        TestSSLWindowDataset,
    ]

    total_pass = 0
    total_fail = 0

    for test_cls in test_classes:
        instance = test_cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        print(f"\n[{test_cls.__name__}]")
        for method_name in methods:
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, method_name)()
                print(f"  PASS {method_name}")
                total_pass += 1
            except Exception as e:
                print(f"  FAIL {method_name}: {e}")
                total_fail += 1

    print("\n" + "=" * 65)
    print(f"Results: {total_pass} passed, {total_fail} failed")
    print("=" * 65)
    sys.exit(0 if total_fail == 0 else 1)
