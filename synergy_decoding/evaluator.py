"""
synergy_decoding/evaluator.py
==============================
Evaluation suite for Pivot 1: Synergy Decoding.

Calculates Pearson R, VAF%, and runs statistical significance tests comparing
the synergy bottleneck approach against direct EMG regression.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.stats
import torch


def compute_vaf(M: np.ndarray, M_hat: np.ndarray) -> float:
    """Compute VAF% between ground-truth and reconstruction."""
    ss_res = np.sum((M - M_hat) ** 2)
    ss_tot = np.sum(M ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient between flattened arrays."""
    if len(x) == 0:
        return 0.0
    x_flat = x.flatten()
    y_flat = y.flatten()
    if np.var(x_flat) < 1e-12 or np.var(y_flat) < 1e-12:
        return 0.0
    r, _ = scipy.stats.pearsonr(x_flat, y_flat)
    return float(r)


class SynergyEvaluator:
    """Benchmark evaluation suite for synergy decoding vs direct regression."""

    def __init__(self) -> None:
        self.synergy_r: float = 0.0
        self.synergy_vaf: float = 0.0
        self.muscle_r: float = 0.0
        self.muscle_vaf: float = 0.0

    def evaluate_synergy_predictions(
        self, C_gt: np.ndarray, C_hat: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate latent synergy activations.

        Args:
            C_gt: (T, k) Ground-truth synergy activations.
            C_hat: (T, k) Predicted synergy activations.

        Returns:
            Pearson r, VAF
        """
        self.synergy_r = compute_pearson_r(C_gt, C_hat)
        self.synergy_vaf = compute_vaf(C_gt, C_hat)
        return self.synergy_r, self.synergy_vaf

    def evaluate_muscle_reconstruction(
        self, M_gt: np.ndarray, M_hat: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate full EMG reconstruction.

        Args:
            M_gt: (T, n_muscles) Ground-truth EMG.
            M_hat: (T, n_muscles) Reconstructed EMG (e.g. C_hat @ W).

        Returns:
            Pearson r, VAF
        """
        self.muscle_r = compute_pearson_r(M_gt, M_hat)
        self.muscle_vaf = compute_vaf(M_gt, M_hat)
        return self.muscle_r, self.muscle_vaf


def paired_wilcoxon_test(
    metric_synergy: np.ndarray, metric_direct: np.ndarray
) -> Tuple[float, float, str]:
    """Perform one-sided paired Wilcoxon signed-rank test.

    H1: Synergy decoding metric > Direct regression metric.

    Args:
        metric_synergy: Array of shape (N,) for synergy model metrics.
        metric_direct: Array of shape (N,) for direct baseline metrics.

    Returns:
        Statistic, p-value, formatted significance string.
    """
    if len(metric_synergy) != len(metric_direct):
        raise ValueError("Metric arrays must be paired and have the same length.")

    if np.allclose(metric_synergy, metric_direct):
        return 0.0, 1.0, "p=1.000 (n.s.)"

    try:
        stat, p_val = scipy.stats.wilcoxon(
            metric_synergy, metric_direct, alternative="greater"
        )
    except Exception:
        stat, p_val = 0.0, 1.0

    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."

    return stat, p_val, f"p={p_val:.4f} {sig}"
