"""
Loss functions for the KG-GT EMG regressor.

COMPOSITION:
    CombinedEMGLoss = PeakWeightedMSELoss  +  EdgePriorKLDivLoss

    1. PeakWeightedMSELoss
       Standard MSE weighted by instantaneous EMG amplitude. Heavily penalizes
       missed activation bursts — the neurophysiologically informative events.

    2. EdgePriorKLDivLoss
       KL divergence between the learned GAT edge_bias distribution and the frozen
       biological prior (edge_prior_anchor). Forces the muscle graph to stay close
       to the known EMG co-activation structure and only deviate when the prediction
       gradient strongly demands it.

BACKWARD COMPATIBILITY:
    CombinedEMGLoss.forward(pred, target) still works (model defaults to None).
    Pass model explicitly to enable the KL regularization term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 1. Peak-weighted reconstruction loss
# ============================================================================

class PeakWeightedMSELoss(nn.Module):
    """MSE with higher penalty on EMG burst peaks (activated timesteps) AND Asymmetry.

    MOTIVATION:
        1. Peak Weighting: We scale each sample's squared error by a weight proportional 
           to its amplitude. This forces the model to spend capacity on bursts.
        2. Asymmetry: It is much worse to completely miss a peak (under-predict) than 
           to slightly overshoot it (over-predict).

    WEIGHT FORMULA:
        w_t = 1 + alpha * (target_shifted / (max_batch + eps))
        asym_w = asymmetry if (target > pred) else 1.0
        
        loss = w_t * asym_w * (pred - target)^2

    Args:
        alpha: Peak-emphasis factor. Typical range 1.0-5.0. 
        asymmetry: Penalty multiplier for under-predicting the signal. Default 2.0.
    """

    def __init__(self, alpha: float = 3.0, asymmetry: float = 2.0) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = alpha
        self.asymmetry = asymmetry

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (B, T, C) predicted envelope
            target: (B, T, C) ground truth envelope (raw or z-scored)
        """
        if self.alpha == 0.0 and self.asymmetry == 1.0:
            return F.mse_loss(pred, target)

        # ── 1. PEAK WEIGHTING ──────────────────────────────────────────────
        reduce_dims = tuple(range(target.dim() - 1))          # all dims except last (C)
        t_min = target.amin(dim=reduce_dims, keepdim=True)    
        target_shifted = target - t_min                       # Strictly non-negative

        max_batch = target_shifted.amax(dim=reduce_dims, keepdim=True)
        w_t = 1.0 + self.alpha * (target_shifted / (max_batch + 1e-5))

        # ── 2. ASYMMETRY ───────────────────────────────────────────────────
        # Penalize under-estimations (where the model missed the burst)
        under_prediction = (target > pred).float()
        asym_w = 1.0 + (self.asymmetry - 1.0) * under_prediction

        mse_raw = F.mse_loss(pred, target, reduction="none")  
        loss = (w_t * asym_w * mse_raw).mean()

        return loss
# ============================================================================
# 2. Edge-prior KL divergence regularization
# ============================================================================

class EdgePriorKLDivLoss(nn.Module):
    """KL divergence anchoring GAT edge_bias to the biological correlation prior.

    MECHANISM:
        For every GAT layer (MuscleGATLayer or KinematicGuidedMuscleGATEncoder)
        that has both:
          - self.edge_bias       (learnable parameter, H × N × N)
          - self.edge_prior_anchor (frozen buffer, H × N × N, registered in __init__)

        this loss computes:

            KL( P || Q )

        where:
            P = softmax(edge_bias,   dim=-1)  (learned muscle attention distribution)
            Q = softmax(edge_prior_anchor, dim=-1)  (prior muscle co-activation distribution)

        The softmax is applied over the flattened N² muscle-pair axis per head,
        treating each row as a probability distribution over the 5×5 = 25 edges.

    WHY KL OVER L2 (Q2 answer: KL is more principled):
        - L2 penalizes raw scalar differences in log-space:
              || edge_bias - anchor ||_F²
          This is scale-dependent and treats all divergences linearly.

        - KL divergence penalizes the *distributional* shift in the attention
          probabilities after softmax. This is scale-invariant and correctly
          captures how much the model's muscle routing has deviated from biology.
          Crucially, it is asymmetric: KL(P||Q) is large when the model places
          probability mass (attention) on edges that the prior considers unlikely
          — exactly the behaviour we want to penalise.

    NORMALIZATION:
        The KL is summed over heads and averaged over GAT layers, so lambda_reg
        remains meaningful regardless of model depth.

    Args:
        lambda_reg: Regularization weight multiplier. Typical range 1e-3 – 1e-1.
                    Start with 0.01. Increase if edge_bias diverges far from the
                    prior; decrease if the model is over-constrained (high RMSE).
    """

    def __init__(self, lambda_reg: float = 0.01) -> None:
        super().__init__()
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg}")
        self.lambda_reg = lambda_reg

    def forward(self, model: nn.Module) -> Tensor:
        """Scan all submodules for GAT layers and sum their KL penalties.

        Args:
            model: The full KGGTModel (or any nn.Module containing GAT layers).

        Returns:
            Scalar KL penalty = lambda_reg × mean_over_layers[ KL(P||Q) ].
            Returns a zero tensor (on the model's device) if no GAT layers found.
        """
        # Determine device from model parameters (handles CPU / CUDA transparently).
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        penalty  = torch.zeros(1, device=device)
        n_layers = 0

        for module in model.modules():
            # Only process modules that have BOTH the learnable bias and the frozen anchor.
            if not (hasattr(module, "edge_bias") and hasattr(module, "edge_prior_anchor")):
                continue
            if not isinstance(module.edge_bias, nn.Parameter):
                continue

            bias   = module.edge_bias          # (H, N, N) learnable
            anchor = module.edge_prior_anchor  # (H, N, N) frozen buffer

            H, N, _ = bias.shape

            # Flatten N×N → N² and compute softmax distributions per head.
            # P: learned, Q: biological prior (target distribution).
            p = F.softmax(bias.view(H, -1),   dim=-1)   # (H, N²) — learned
            q = F.softmax(anchor.view(H, -1), dim=-1)   # (H, N²) — prior (no grad)

            # KL(P || Q) = Σ_i P_i log(P_i / Q_i)
            # F.kl_div(log_input, target) computes: target * (log_target - log_input)
            # which equals KL(target || input). So we want:
            #   F.kl_div(log_p, q)  → q * (log_q - log_p) = KL(Q || P)   [wrong direction]
            #   F.kl_div(log_q, p)  → p * (log_p - log_q) = KL(P || Q)   [correct]
            #
            # reduction='batchmean' sums over N² and divides by H (batch dim = H).
            kl = F.kl_div(
                q.log(),              # log Q  (prior log-probabilities — targets)
                p,                    # P      (learned probabilities — input)
                reduction="batchmean",
                log_target=False,
            )  # = KL(P || Q) summed over N², averaged over H
            penalty  = penalty + kl
            n_layers += 1

        # Normalize by number of contributing GAT layers.
        # This makes lambda_reg scale-invariant to model depth:
        # a 2-layer vs 4-layer GAT with the same lambda_reg will exert the
        # same regularization pressure per layer.
        if n_layers > 0:
            penalty = penalty / n_layers

        return self.lambda_reg * penalty


# ============================================================================
# 3. Combined loss (reconstruction + regularization)
# ============================================================================

class CombinedEMGLoss(nn.Module):
    """Peak-weighted MSE reconstruction + KL edge-prior regularization + L1 Sparsity.

    TOTAL LOSS:
        L = PeakWeightedMSE(pred, target)
          + lambda_reg * EdgePriorKLDiv(model)
          + lambda_l1 * L1(pred)

    BACKWARD COMPATIBILITY:
        The signature forward(pred, target, model=None) is backward-compatible
        with any existing code calling loss_fn(pred, target). The regularization
        term is silently skipped when model=None.

    USAGE IN TRAINING LOOP:
        # Pass model explicitly in _run_epoch (see training.py):
        loss = loss_fn(pred, y, model=model)

        # Legacy / inference usage (no regularization):
        loss = loss_fn(pred, y)

    Args:
        peak_alpha:      Peak-weighting factor (PeakWeightedMSELoss). Default 3.0.
        lambda_reg:      KL regularization weight (EdgePriorKLDivLoss). Default 0.01.
        lambda_l1:       L1 Sparsity weight to suppress floating baselines. Default 1e-4.
        use_peak_weight: If False, use standard nn.MSELoss (alpha=0 equivalent).
                         Useful for ablation studies.
    """

    def __init__(
        self,
        peak_alpha:      float = 3.0,
        lambda_reg:      float = 0.01,
        lambda_l1:       float = 1e-4,
        use_peak_weight: bool  = True,
        asymmetry:       float = 2.0,
        rest_threshold:  float = 0.05,
    ) -> None:
        super().__init__()
        self.recon     = PeakWeightedMSELoss(alpha=peak_alpha, asymmetry=asymmetry) if use_peak_weight else nn.MSELoss()
        self.reg       = EdgePriorKLDivLoss(lambda_reg=lambda_reg)
        self.lambda_reg = lambda_reg
        self.lambda_l1  = lambda_l1
        self.rest_threshold = rest_threshold

    def forward(
        self,
        pred:   Tensor,
        target: Tensor,
        model:  nn.Module | None = None,  # default None -> backward-compatible
    ) -> Tensor:
        """
        Args:
            pred:   Predicted EMG envelope, shape (B, T, C).
            target: Ground-truth EMG envelope, same shape.
            model:  The KGGTModel instance (for KL edge-prior regularization).
                    Pass None to skip regularization (legacy / inference usage).

        Returns:
            Scalar loss value.
        """
        loss = self.recon(pred, target)

        # L1 Sparsity ONLY during muscle rest (Thresholded Sparsity)
        if self.lambda_l1 > 0.0:
            # Shift target to find the true baseline
            reduce_dims = tuple(range(target.dim() - 1))
            t_min = target.amin(dim=reduce_dims, keepdim=True)
            target_shifted = target - t_min

            # Find the max peak to define the threshold (e.g. 5% of max peak)
            max_batch = target_shifted.amax(dim=reduce_dims, keepdim=True)
            
            # Mask: 1.0 where muscle is resting, 0.0 during bursts
            resting_mask = (target_shifted < self.rest_threshold * (max_batch + 1e-5)).float()
            
            # Apply L1 penalty ONLY to the resting regions! 
            # We push the prediction towards the true resting baseline (t_min), not zero.
            loss = loss + self.lambda_l1 * torch.mean(torch.abs(pred - t_min) * resting_mask)

        # KL regularization is computed only when the model is explicitly provided
        # and lambda_reg > 0. This preserves strict backward compatibility.
        if model is not None and self.lambda_reg > 0.0:
            loss = loss + self.reg(model)

        return loss


# ============================================================================
# Config builder
# ============================================================================

def build_loss_from_config(cfg: dict) -> CombinedEMGLoss:
    """Build CombinedEMGLoss from the project config dictionary.

    Reads from cfg["loss"] (all keys optional with sensible defaults):

        cfg["loss"]["peak_alpha"]      float  Peak-weighting factor. Default 3.0.
        cfg["loss"]["lambda_reg"]      float  KL regularization weight. Default 0.01.
        cfg["loss"]["lambda_l1"]       float  L1 Sparsity weight. Default 1e-4.
        cfg["loss"]["use_peak_weight"] bool   Enable peak weighting. Default True.

    Example config section:
        "loss": {
            "peak_alpha": 3.0,        # 3x emphasis on burst peaks vs. baseline
            "lambda_reg": 0.01,       # mild KL anchor to biological prior
            "lambda_l1": 1e-4,        # suppress floating baseline noise
            "use_peak_weight": true,
        }

    If cfg has no "loss" key, all defaults are used (equivalent to the old
    pure-MSE CombinedEMGLoss but with peak weighting added).
    """
    loss_cfg = cfg.get("loss", {})
    return CombinedEMGLoss(
        peak_alpha=loss_cfg.get("peak_alpha", 3.0),
        lambda_reg=loss_cfg.get("lambda_reg", 0.01),
        lambda_l1=loss_cfg.get("lambda_l1", 1e-4),
        use_peak_weight=loss_cfg.get("use_peak_weight", True),
        asymmetry=loss_cfg.get("asymmetry", 2.0),
        rest_threshold=loss_cfg.get("rest_threshold", 0.05),
    )
