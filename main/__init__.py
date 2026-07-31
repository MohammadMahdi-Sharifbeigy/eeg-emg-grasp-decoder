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
    model:                 KGGTModel, build_kg_gt_from_config, CNN1dAligner,
                           TransformerOnlyModel, build_transformer_only_from_config
    losses:                CombinedEMGLoss, PeakWeightedMSELoss, EdgePriorKLDivLoss,
                           build_loss_from_config
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
    compute_muscle_edge_prior,
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
    TransformerOnlyModel,
    build_transformer_only_from_config,
)
from .losses import (
    CombinedEMGLoss,
    PeakWeightedMSELoss,
    EdgePriorKLDivLoss,
    build_loss_from_config,
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

from .plots import(
    get_dynamic_fig_dir,
    save_fig,
    plot_multichannel_trace,
    plot_signal_heatmap,
    plot_preprocessing_comparison,
    plot_kinematic_features,
    plot_training_history,
    plot_prediction_overlay,
    plot_gate_heatmap,
    plot_attention_maps,
    plot_residual_diagnostics,
    plot_fused_pca,
    plot_gate_vs_emg_power,
    plot_emg_envelope_overlay,
    # Interpretability visualizations
    plot_interpretability_triptych,
    plot_muscle_synergy_matrix,
    plot_kin_edge_linear_weights,
)

__all__ = [
    # device
    "get_device", "set_seed",
    # dataloader
    "load_hs", "load_ws", "load_participant", "get_split_series",
    # preprocessing
    "preprocess_eeg", "preprocess_eeg_from_config", "select_channels",
    "preprocess_emg", "preprocess_emg_from_config", "EMGNormalizer",
    "preprocess_kinematics", "preprocess_kinematics_from_config", "KinNormalizer",
    "extract_kt_raw", "extract_kt", "compute_muscle_edge_prior",
    # dataset
    "WAYEEGDataset",
    # model
    "KGGTModel", "build_kg_gt_from_config", "CNN1dAligner",
    "TransformerEncoder", "build_transformer_from_config",
    "MuscleGATEncoder", "KinematicGuidedMuscleGATEncoder",
    "SinusoidalPositionalEncoding",
    # losses (SoftDTWLoss / soft_dtw removed — hard-removed from codebase)
    "CombinedEMGLoss", "PeakWeightedMSELoss", "EdgePriorKLDivLoss",
    "build_loss_from_config",
    # training
    "train_model", "TrainConfig", "TrainResult", "EvalMetrics",
    "collect_predictions", "compute_metrics", "prepare_batch_factory",
    "save_checkpoint", "load_checkpoint", "print_gpu_info",
    # plots — core
    "get_dynamic_fig_dir", "save_fig", "plot_multichannel_trace",
    "plot_signal_heatmap", "plot_preprocessing_comparison",
    "plot_kinematic_features", "plot_training_history",
    "plot_prediction_overlay", "plot_gate_heatmap",
    "plot_attention_maps", "plot_residual_diagnostics",
    "plot_fused_pca", "plot_gate_vs_emg_power", "plot_emg_envelope_overlay",
    # plots — interpretability
    "plot_interpretability_triptych",
    "plot_muscle_synergy_matrix",
    "plot_kin_edge_linear_weights",
]
