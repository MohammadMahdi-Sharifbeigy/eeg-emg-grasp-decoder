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
    """Scale-Invariant Superlinear (Quadratic) Peak-Weighted MSE with Activation-Gated Asymmetry.

    MOTIVATION & SOLVING THE CANOPY EFFECT:
        1. Scale Invariance: Physical EMG envelopes vary in the range [0.00, 0.02], whereas z-scored
           envelopes range [-1.0, +4.0]. Hardcoded thresholds fail on physical envelopes. We map each
           channel dynamically to normalized activation y_norm in [0, 1].
        2. Superlinear (Quadratic) Peak Weighting: A linear ramp (1 + alpha * y_norm) only mildly
           differentiates mid-level canopy predictions from sharp burst tips. Using a quadratic curve
           1 + alpha * (y_norm)^2 concentrates up to an 8x gradient boost exclusively on burst apices
           while leaving resting baseline noise (y_norm ~ 0) totally uninflated.
        3. Activation-Gated Asymmetry: Unconditional asymmetry penalizes under-predicted baseline
           noise during rest, causing predictions to float upwards. By gating asymmetry with y_norm,
           we enforce balanced zero-mean errors at rest (1.0x) while severely punishing missed contractions!

    Args:
        alpha: Peak-emphasis multiplier at burst tips. Default 3.0 (gives 4x loss at peak).
        asymmetry: Under-prediction multiplier during peak contractions. Default 2.0.
        threshold: Maintained for backwards config compatibility.
        peak_weight: Optional alias for alpha.
    """

    def __init__(self, alpha: float = 3.0, asymmetry: float = 2.0, threshold: float = 1.0, peak_weight: float | None = None) -> None:
        super().__init__()
        if peak_weight is not None:
            alpha = peak_weight
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha = float(alpha)
        self.asymmetry = float(asymmetry)
        self.threshold = float(threshold)  # kept for backwards config compatibility

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (B, T, C) predicted envelope
            target: (B, T, C) ground truth envelope (raw physical scale or z-scored)
        """
        if self.alpha == 0.0 and self.asymmetry == 1.0:
            return F.mse_loss(pred, target)

        # ── 1. SCALE-INVARIANT RELATIVE ACTIVATION (0.0 to 1.0 per channel per window) ─
        # Normalize along time dimension (dim=1) so peak weighting is evaluated relative to each window
        t_min = target.amin(dim=1, keepdim=True)   # (B, 1, C)
        t_max = target.amax(dim=1, keepdim=True)   # (B, 1, C)
        
        # Safe clamping prevents division by zero or noise explosion in resting windows
        t_range = (t_max - t_min).clamp(min=1e-4)
        y_norm = (target - t_min) / t_range

        # ── 2. SUPERLINEAR (QUADRATIC) PEAK WEIGHTING (The Canopy Eraser) ───
        # Exponential gradient acceleration on sharp burst tips without multiplying rest noise
        w_t = 1.0 + self.alpha * (y_norm ** 2)

        # ── 3. ACTIVATION-GATED ASYMMETRY ──────────────────────────────────
        # Pure symmetry at rest (y_norm -> 0) preventing upward baseline drift in FDI/ECR; 
        # aggressive under-prediction penalties during actual burst firing!
        under_prediction = (target > pred).float()
        asym_w = 1.0 + (self.asymmetry - 1.0) * under_prediction * y_norm

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
# 3. Correlation- and Shape-Preserving Electrophysiological Losses
# ============================================================================

class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient (CCC) Loss along the temporal dimension (dim=1).

    Measures agreement between prediction and ground-truth relative to the 45-degree
    line of perfect identity:
        CCC = 2 * Cov(y, y_hat) / (Var(y) + Var(y_hat) + (mean(y) - mean(y_hat))^2)
        L_ccc = 1 - CCC in [0, 2]

    Solves Mean Collapse: a flat line predictor achieves CCC = 0 (L_ccc = 1.0),
    forcing the network to reproduce both dynamic range and correlated trajectory.
    Denominator is clamped with min=eps for 100% numerical stability under AMP FP16.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mu_pred = pred.mean(dim=1, keepdim=True)        # (B, 1, C)
        mu_target = target.mean(dim=1, keepdim=True)    # (B, 1, C)

        var_pred = ((pred - mu_pred) ** 2).mean(dim=1, keepdim=True)
        var_target = ((target - mu_target) ** 2).mean(dim=1, keepdim=True)

        cov = ((pred - mu_pred) * (target - mu_target)).mean(dim=1, keepdim=True)

        # Denominator clamped for AMP FP16 safety
        denom = (var_pred + var_target + (mu_pred - mu_target) ** 2).clamp(min=self.eps)
        ccc = (2.0 * cov) / denom
        return (1.0 - ccc).mean()


class PearsonCorrelationLoss(nn.Module):
    """Temporal Pearson Correlation Loss (1 - r) with quiescent baseline gating and AMP clamp.

    Rewards burst co-activation timing and relative profile independent of amplitude scale.
    Near-zero variance channels (var_target < min_target_var) are gated out so the network
    is not penalized for failing to correlate resting noise.
    """

    def __init__(self, eps: float = 1e-6, min_target_var: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.min_target_var = min_target_var

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mu_pred = pred.mean(dim=1, keepdim=True)
        mu_target = target.mean(dim=1, keepdim=True)

        diff_pred = pred - mu_pred
        diff_target = target - mu_target

        num = (diff_pred * diff_target).sum(dim=1)     # (B, C)
        denom = (
            torch.sqrt(((diff_pred ** 2).sum(dim=1)).clamp(min=self.eps))
            * torch.sqrt(((diff_target ** 2).sum(dim=1)).clamp(min=self.eps))
        ).clamp(min=self.eps)                          # (B, C)
        r = num / denom                                # (B, C)

        # Quiescent baseline gate: only penalize when target has active variance
        var_target = (diff_target ** 2).mean(dim=1)    # (B, C)
        active_mask = (var_target > self.min_target_var).float()

        loss = 1.0 - r
        if active_mask.sum() > 0:
            return (loss * active_mask).sum() / (active_mask.sum() + self.eps)
        return loss.mean()


class TemporalSmoothnessLoss(nn.Module):
    """First-order temporal difference matching: L1(d(pred)/dt - d(target)/dt).

    Penalizes both high-frequency attention jitter and sluggish low-pass canopies.
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device)
        d_pred = pred[:, 1:, :] - pred[:, :-1, :]
        d_target = target[:, 1:, :] - target[:, :-1, :]
        return F.l1_loss(d_pred, d_target)


# ============================================================================
# 4. Composite Hybrid Loss (Peak-MSE + CCC + Pearson + Smoothness + Sparsity + KL)
# ============================================================================

class CompositeEMGLoss(nn.Module):
    """Composite objective for low-SNR neural decoding (EEG-to-EMG).

    TOTAL LOSS:
        L_total = w_peak    * L_PeakMSE
                + w_ccc     * L_CCC
                + w_pearson * L_Pearson
                + w_diff    * L_diff
                + w_rest    * L_rest (resting L1)
                + w_reg     * L_reg  (EdgePrior KL)
    """

    def __init__(
        self,
        w_peak: float = 1.0,
        w_ccc: float = 0.5,
        w_pearson: float = 0.2,
        w_diff: float = 0.1,
        w_rest: float = 0.05,
        w_reg: float = 0.001,
        peak_alpha: float = 3.0,
        asymmetry: float = 2.0,
        rest_threshold: float = 0.15,
        eps: float = 1e-6,
        # Backward compatibility arguments:
        lambda_reg: float | None = None,
        lambda_l1: float | None = None,
        lambda_grad: float | None = None,
        use_peak_weight: bool = True,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if lambda_reg is not None:
            w_reg = lambda_reg
        if lambda_l1 is not None:
            w_rest = lambda_l1
        if lambda_grad is not None:
            w_diff = lambda_grad

        self.w_peak = float(w_peak) if use_peak_weight else 0.0
        self.w_ccc = float(w_ccc)
        self.w_pearson = float(w_pearson)
        self.w_diff = float(w_diff)
        self.w_rest = float(w_rest)
        self.w_reg = float(w_reg)
        self.rest_threshold = float(rest_threshold)

        self.peak_mse = PeakWeightedMSELoss(alpha=peak_alpha, asymmetry=asymmetry, threshold=threshold)
        self.ccc = CCCLoss(eps=eps)
        self.pearson = PearsonCorrelationLoss(eps=eps)
        self.diff = TemporalSmoothnessLoss()
        self.reg = EdgePriorKLDivLoss(lambda_reg=1.0)

        self.last_components: dict[str, float] = {}

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        model: nn.Module | None = None,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        loss_components: dict[str, Tensor] = {}
        total_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)

        # 1. Peak Weighted MSE
        if self.w_peak > 0:
            l_peak = self.peak_mse(pred, target)
            loss_components["peak_mse"] = l_peak
            total_loss = total_loss + self.w_peak * l_peak
        else:
            loss_components["peak_mse"] = torch.tensor(0.0, device=pred.device)

        # 2. CCC Loss
        if self.w_ccc > 0:
            l_ccc = self.ccc(pred, target)
            loss_components["ccc"] = l_ccc
            total_loss = total_loss + self.w_ccc * l_ccc
        else:
            loss_components["ccc"] = torch.tensor(0.0, device=pred.device)

        # 3. Pearson Correlation Loss
        if self.w_pearson > 0:
            l_pearson = self.pearson(pred, target)
            loss_components["pearson"] = l_pearson
            total_loss = total_loss + self.w_pearson * l_pearson
        else:
            loss_components["pearson"] = torch.tensor(0.0, device=pred.device)

        # 4. Temporal Difference Matching
        if self.w_diff > 0 and pred.shape[1] > 1:
            l_diff = self.diff(pred, target)
            loss_components["diff"] = l_diff
            total_loss = total_loss + self.w_diff * l_diff
        else:
            loss_components["diff"] = torch.tensor(0.0, device=pred.device)

        # 5. Resting Baseline L1 Penalty (Per-window normalized threshold)
        if self.w_rest > 0:
            t_min = target.amin(dim=1, keepdim=True)
            target_shifted = target - t_min
            max_batch = target_shifted.amax(dim=1, keepdim=True)
            resting_mask = (target_shifted < self.rest_threshold * (max_batch + 1e-5)).float()
            l_rest = torch.mean(torch.abs(pred - t_min) * resting_mask)
            loss_components["rest_l1"] = l_rest
            total_loss = total_loss + self.w_rest * l_rest
        else:
            loss_components["rest_l1"] = torch.tensor(0.0, device=pred.device)

        # 6. GAT EdgePrior KL Divergence Regularization
        if model is not None and self.w_reg > 0:
            l_reg = self.reg(model)
            loss_components["kl_reg"] = l_reg
            total_loss = total_loss + self.w_reg * l_reg
        else:
            loss_components["kl_reg"] = torch.tensor(0.0, device=pred.device)

        total_loss = total_loss.squeeze()

        # Cache scalar values for epoch diagnostics
        self.last_components = {
            "total": total_loss.item(),
            "peak_mse": loss_components["peak_mse"].item(),
            "ccc": loss_components["ccc"].item(),
            "pearson": loss_components["pearson"].item(),
            "diff": loss_components["diff"].item(),
            "rest_l1": loss_components["rest_l1"].item(),
            "kl_reg": loss_components["kl_reg"].item(),
        }

        if return_components:
            return total_loss, self.last_components
        return total_loss


# Backward compatibility alias
CombinedEMGLoss = CompositeEMGLoss


# ============================================================================
# 5. Config builder
# ============================================================================

def build_loss_from_config(cfg: dict) -> CompositeEMGLoss:
    """Build CompositeEMGLoss from the project config dictionary.

    Reads from cfg["loss"] with sensible defaults:
        w_peak:          float  Peak-weighting factor. Default 1.0.
        w_ccc:           float  Concordance correlation weight. Default 0.5.
        w_pearson:       float  Pearson correlation timing weight. Default 0.2.
        w_diff:          float  First-order temporal smoothness weight. Default 0.1.
        w_rest:          float  Resting L1 penalty weight. Default 0.05.
        w_reg:           float  KL edge-prior regularization weight. Default 0.001.
        peak_alpha:      float  Multiplier for burst tips in PeakMSE. Default 3.0.
        asymmetry:       float  Under-prediction multiplier in PeakMSE. Default 2.0.
    """
    loss_cfg = cfg.get("loss", cfg if "peak_alpha" in cfg or "w_peak" in cfg else cfg.get("loss", {}))
    return CompositeEMGLoss(
        w_peak=loss_cfg.get("w_peak", 1.0 if loss_cfg.get("use_peak_weight", True) else 0.0),
        w_ccc=loss_cfg.get("w_ccc", 0.5),
        w_pearson=loss_cfg.get("w_pearson", 0.2),
        w_diff=loss_cfg.get("w_diff", loss_cfg.get("lambda_grad", 0.1)),
        w_rest=loss_cfg.get("w_rest", loss_cfg.get("lambda_l1", 0.05)),
        w_reg=loss_cfg.get("w_reg", loss_cfg.get("lambda_reg", 0.001)),
        peak_alpha=loss_cfg.get("peak_alpha", loss_cfg.get("peak_weight", 3.0)),
        asymmetry=loss_cfg.get("asymmetry", 2.0),
        rest_threshold=loss_cfg.get("rest_threshold", 0.15),
        eps=loss_cfg.get("eps", 1e-6),
        threshold=loss_cfg.get("threshold", 1.0),
    )


# ============================================================================
# 6. CORAL-Net Objective (CORALLoss)
# ============================================================================

class CORALLoss(nn.Module):
    """Compound objective for CORAL-Net corticomuscular regression.

    Formulation:
        L_total = w_ccc * L_CCC
                + w_pearson * L_Pearson
                + w_diff * L_diff
                + w_rest * L_rest
                + w_syn * L_synergy

    Safeguards:
        1. Lin's CCC operates along time (dim=1) to eliminate Mean Collapse.
        2. Gated Pearson (dim=1) ignores quiescent noise channels.
        3. Temporal smoothness matches first-order envelope derivatives.
        4. Synergy Aux Loss uses strictly non-negative ground truth projection
           s*(t) = relu(y) @ pinv(W) >= 0.
    """

    def __init__(
        self,
        w_ccc: float = 1.0,
        w_pearson: float = 0.5,
        w_diff: float = 0.2,
        w_rest: float = 0.05,
        w_syn: float = 0.1,
        rest_threshold: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.w_ccc = float(w_ccc)
        self.w_pearson = float(w_pearson)
        self.w_diff = float(w_diff)
        self.w_rest = float(w_rest)
        self.w_syn = float(w_syn)
        self.rest_threshold = float(rest_threshold)
        self.eps = float(eps)

        self.ccc = CCCLoss(eps=eps)
        self.pearson = PearsonCorrelationLoss(eps=eps, min_target_var=1e-4)
        self.diff = TemporalSmoothnessLoss()

        self.last_components: dict[str, float] = {}

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        model: nn.Module | None = None,
        synergies: Tensor | None = None,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        total_loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        loss_components: dict[str, Tensor] = {}

        # 1. CCC Loss (eliminates Mean Collapse)
        if self.w_ccc > 0:
            l_ccc = self.ccc(pred, target)
            loss_components["ccc"] = l_ccc
            total_loss = total_loss + self.w_ccc * l_ccc

        # 2. Gated Pearson Correlation Loss
        if self.w_pearson > 0:
            l_pearson = self.pearson(pred, target)
            loss_components["pearson"] = l_pearson
            total_loss = total_loss + self.w_pearson * l_pearson

        # 3. Temporal Smoothness Loss
        if self.w_diff > 0:
            l_diff = self.diff(pred, target)
            loss_components["diff"] = l_diff
            total_loss = total_loss + self.w_diff * l_diff

        # 4. Resting Baseline L1 Penalty
        if self.w_rest > 0:
            t_min = target.amin(dim=1, keepdim=True)
            t_max = target.amax(dim=1, keepdim=True)
            t_range = (t_max - t_min).clamp(min=1e-4)
            norm_target = (target - t_min) / t_range
            is_resting = (norm_target < self.rest_threshold).float()
            l_rest = (torch.abs(pred - t_min) * is_resting).sum() / (is_resting.sum() + self.eps)
            loss_components["rest"] = l_rest
            total_loss = total_loss + self.w_rest * l_rest

        # 5. Non-negative Synergy Alignment Auxiliary Loss
        if self.w_syn > 0 and synergies is not None and model is not None:
            # Extract mixing matrix W from model
            W = None
            if hasattr(model, "synergy_decoder") and hasattr(model.synergy_decoder, "W"):
                W = model.synergy_decoder.W   # (n_synergies, n_muscles)
            elif hasattr(model, "W"):
                W = model.W

            if W is not None:
                # Safeguard: Strictly non-negative ground truth projection
                y_nonneg = torch.relu(target)
                pinv_W = torch.linalg.pinv(W)  # (n_muscles, n_synergies)
                s_target = torch.clamp(torch.matmul(y_nonneg, pinv_W), min=0.0).detach()
                l_syn = F.mse_loss(synergies, s_target)
                loss_components["synergy"] = l_syn
                total_loss = total_loss + self.w_syn * l_syn

        total_loss = total_loss.squeeze()

        self.last_components = {
            "total": total_loss.item(),
            "ccc": loss_components.get("ccc", torch.tensor(0.0)).item(),
            "pearson": loss_components.get("pearson", torch.tensor(0.0)).item(),
            "diff": loss_components.get("diff", torch.tensor(0.0)).item(),
            "rest": loss_components.get("rest", torch.tensor(0.0)).item(),
            "synergy": loss_components.get("synergy", torch.tensor(0.0)).item(),
        }

        if return_components:
            return total_loss, self.last_components
        return total_loss


def build_coral_loss_from_config(cfg: dict) -> CORALLoss:
    """Builds CORALLoss from configuration dictionary."""
    coral_loss_cfg = cfg.get("coral_loss", cfg.get("loss", {}))
    return CORALLoss(
        w_ccc=coral_loss_cfg.get("w_ccc", 1.0),
        w_pearson=coral_loss_cfg.get("w_pearson", 0.5),
        w_diff=coral_loss_cfg.get("w_diff", 0.2),
        w_rest=coral_loss_cfg.get("w_rest", 0.05),
        w_syn=coral_loss_cfg.get("w_syn", 0.1),
        rest_threshold=coral_loss_cfg.get("rest_threshold", 0.15),
        eps=coral_loss_cfg.get("eps", 1e-6),
    )

