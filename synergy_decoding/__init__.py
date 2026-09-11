"""
Pivot 1: Muscle Synergy Latent Space Decoding Module
"""

from .nmf_extractor import (
    extract_nmf_synergies,
    extract_all_subjects,
    compute_vaf,
    vaf_curve,
    align_synergies,
    CrossSubjectSimilarity,
)
from .losses import SynergyActivationLoss, MuscleReconstructionLoss, SynergyDualObjectiveLoss, SynergyLossConfig
from .synergy_model import CorticosynergyDecoder, SynergyHead
from .dataset import SynergyDataset
from .evaluator import SynergyEvaluator, paired_wilcoxon_test
