"""
Pivot 1: Muscle Synergy Latent Space Decoding Module.

Public API:
    nmf_extractor:  extract_nmf_synergies, extract_all_subjects, compute_vaf,
                    vaf_curve, align_synergies, CrossSubjectSimilarity
    synergy_model:  CorticosynergyDecoder, SynergyHead
    losses:         SynergyActivationLoss, MuscleReconstructionLoss,
                    SynergyDualObjectiveLoss, SynergyLossConfig, CCCLoss,
                    PearsonCorrelationLoss, TemporalSmoothnessLoss
    dataset:        SynergyDataset
    trainer:        SynergyTrainer, SynergyTrainConfig, SynergyTrainResult
    evaluator:      SynergyEvaluator, paired_wilcoxon_test, compute_pearson_r
"""

from .nmf_extractor import (
    extract_nmf_synergies,
    extract_all_subjects,
    compute_vaf,
    vaf_curve,
    align_synergies,
    CrossSubjectSimilarity,
)
from .losses import (
    CCCLoss,
    PearsonCorrelationLoss,
    TemporalSmoothnessLoss,
    SynergyActivationLoss,
    MuscleReconstructionLoss,
    SynergyDualObjectiveLoss,
    SynergyLossConfig,
)
from .synergy_model import (
    CorticosynergyDecoder,
    SynergyHead,
)
from .dataset import (
    SynergyDataset,
)
from .trainer import (
    SynergyTrainer,
    SynergyTrainConfig,
    SynergyTrainResult,
)
from .evaluator import (
    SynergyEvaluator,
    paired_wilcoxon_test,
    compute_pearson_r,
)

__all__ = [
    # nmf_extractor
    "extract_nmf_synergies",
    "extract_all_subjects",
    "compute_vaf",
    "vaf_curve",
    "align_synergies",
    "CrossSubjectSimilarity",
    # losses
    "CCCLoss",
    "PearsonCorrelationLoss",
    "TemporalSmoothnessLoss",
    "SynergyActivationLoss",
    "MuscleReconstructionLoss",
    "SynergyDualObjectiveLoss",
    "SynergyLossConfig",
    # synergy_model
    "CorticosynergyDecoder",
    "SynergyHead",
    # dataset
    "SynergyDataset",
    # trainer
    "SynergyTrainer",
    "SynergyTrainConfig",
    "SynergyTrainResult",
    # evaluator
    "SynergyEvaluator",
    "paired_wilcoxon_test",
    "compute_pearson_r",
]
