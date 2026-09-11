"""self_supervised_coral — Pivot 4: Bio-CLIP Framework.

Cross-modal self-supervised EEG representation learning via EMG guidance
on the WAY-EEG-GAL dataset (12 subjects, 32-ch EEG, 5-ch EMG).

Architecture:
    Twin-Tower encoders (EEGEncoder + EMGEncoder) with symmetric InfoNCE,
    Phase-Aware negative masking to prevent semantically identical windows
    from being pushed apart, and multi-granularity latents (global + dense).

Public API:
    eeg_encoder:       EEGEncoder
    emg_encoder:       EMGEncoder
    contrastive_loss:  PhaseAwareInfoNCELoss, SymmetricInfoNCELoss
    phase_labeler:     PhaseLabeler
    linear_probes:     LinearProbe, FewShotRegressionProbe, evaluate_probes
    ssl_dataset:       SSLWindowDataset, build_ssl_dataloaders
    ssl_trainer:       BioCLIPTrainer, SSLTrainConfig, SSLTrainResult
"""

from .eeg_encoder import EEGEncoder, build_eeg_encoder_from_config
from .emg_encoder import EMGEncoder, build_emg_encoder_from_config
from .contrastive_loss import (
    PhaseAwareInfoNCELoss,
    SymmetricInfoNCELoss,
    NTXentLoss,
)
from .phase_labeler import PhaseLabeler, MovementPhase
from .linear_probes import (
    LinearProbe,
    FewShotRegressionProbe,
    evaluate_linear_probe,
    evaluate_few_shot_regression,
)
from .ssl_dataset import SSLWindowDataset, build_ssl_dataloaders
from .ssl_trainer import BioCLIPTrainer, SSLTrainConfig, SSLTrainResult

__all__ = [
    # encoders
    "EEGEncoder",
    "build_eeg_encoder_from_config",
    "EMGEncoder",
    "build_emg_encoder_from_config",
    # losses
    "PhaseAwareInfoNCELoss",
    "SymmetricInfoNCELoss",
    "NTXentLoss",
    # phase labeling
    "PhaseLabeler",
    "MovementPhase",
    # linear probes
    "LinearProbe",
    "FewShotRegressionProbe",
    "evaluate_linear_probe",
    "evaluate_few_shot_regression",
    # dataset & training
    "SSLWindowDataset",
    "build_ssl_dataloaders",
    "BioCLIPTrainer",
    "SSLTrainConfig",
    "SSLTrainResult",
]
