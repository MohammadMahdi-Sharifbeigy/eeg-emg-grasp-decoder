"""
main — Method 1 KG-GT pipeline (nb04, no CCA).

Public API:
    device:                get_device, set_seed
    dataloader:            load_hs, load_ws, load_participant, get_split_series
    preprocessing_eeg:     preprocess_eeg, preprocess_eeg_from_config
    preprocessing_emg_kin: preprocess_emg, preprocess_emg_from_config,
                           preprocess_kinematics, preprocess_kinematics_from_config,
                           EMGNormalizer, KinNormalizer, extract_kt
    dataset:               WAYEEGDataset
    model:                 KGGTModel, build_kg_gt_from_config, CNN1dAligner
    losses:                CombinedEMGLoss, build_loss_from_config, SoftDTWLoss
    training:              train_model, TrainConfig, TrainResult,
                           collect_predictions, compute_metrics,
                           EvalMetrics, prepare_batch_factory,
                           save_checkpoint, load_checkpoint
"""

from .device import get_device, set_seed
from .dataloader import load_hs, load_ws, load_participant, get_split_series
from .preprocessing_eeg import (
    preprocess_eeg,
    preprocess_eeg_from_config,
    bandpass,
    notch,
    asr,
    common_average_reference,
    delta_band,
    select_channels,
)
from .preprocessing_emg_kin import (
    preprocess_emg,
    preprocess_emg_from_config,
    EMGNormalizer,
    preprocess_kinematics,
    preprocess_kinematics_from_config,
    KinNormalizer,
    extract_kt_raw,
    extract_kt,
)
from .dataset import WAYEEGDataset
from .model import (
    KGGTModel,
    build_kg_gt_from_config,
    CNN1dAligner,
    TransformerEncoder,
    build_transformer_from_config,
    MuscleGATEncoder,
    KinematicGuidedMuscleGATEncoder,
    SinusoidalPositionalEncoding,
)
from .losses import (
    CombinedEMGLoss,
    build_loss_from_config,
    SoftDTWLoss,
    soft_dtw,
)
from .training import (
    train_model,
    TrainConfig,
    TrainResult,
    EvalMetrics,
    collect_predictions,
    compute_metrics,
    prepare_batch_factory,
    save_checkpoint,
    load_checkpoint,
    print_gpu_info,
)

__all__ = [
    # device
    "get_device", "set_seed",
    # dataloader
    "load_hs", "load_ws", "load_participant", "get_split_series",
    # preprocessing
    "preprocess_eeg", "preprocess_eeg_from_config",
    "bandpass", "notch", "asr", "common_average_reference", "delta_band", "select_channels",
    "preprocess_emg", "preprocess_emg_from_config", "EMGNormalizer",
    "preprocess_kinematics", "preprocess_kinematics_from_config", "KinNormalizer",
    "extract_kt_raw", "extract_kt",
    # dataset
    "WAYEEGDataset",
    # model
    "KGGTModel", "build_kg_gt_from_config", "CNN1dAligner",
    "TransformerEncoder", "build_transformer_from_config",
    "MuscleGATEncoder", "KinematicGuidedMuscleGATEncoder",
    "SinusoidalPositionalEncoding",
    # losses
    "CombinedEMGLoss", "build_loss_from_config", "SoftDTWLoss", "soft_dtw",
    # training
    "train_model", "TrainConfig", "TrainResult", "EvalMetrics",
    "collect_predictions", "compute_metrics", "prepare_batch_factory",
    "save_checkpoint", "load_checkpoint", "print_gpu_info",
]
