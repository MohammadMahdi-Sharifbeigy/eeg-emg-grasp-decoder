"""
renew — Method 1 Fix v2 KG-GT pipeline (nb06).

Differences from main (nb04):
  - MNE-based EEG preprocessing (ICA/ASR option, robust clip, delta 0.5–4Hz)
  - 2-layer residual CNN encoder (TemporalCNNEncoder)
  - Learnable gated fusion between cross-attention streams
  - Gate variance regularisation in CombinedEMGLoss
  - rho_GL epsilon = 1.0 N (corrected)
  - All 32 EEG channels (no ROI selection)

Public API:
    preprocessing_eeg:     preprocess_eeg, preprocess_eeg_from_config
    preprocessing_emg_kin: preprocess_emg, preprocess_emg_from_config,
                           EMGNormalizer, preprocess_kinematics,
                           preprocess_kinematics_from_config, KinNormalizer,
                           extract_kt, extract_kt_raw
    model:                 HybridKGGTModel, build_model_from_config,
                           TemporalCNNEncoder, LearnableGatedFusion,
                           OptimizedKG_GAT
    losses:                CombinedEMGLoss, build_loss_from_config,
                           SoftDTWLoss, SmoothnessLoss, PearsonLoss
    training:              train_model, TrainConfig, TrainResult,
                           collect_predictions, compute_metrics,
                           EvalMetrics, prepare_batch_factory,
                           save_checkpoint, load_checkpoint, print_gpu_info,
                           make_preprocess_fn, build_dataset_split
    device:                get_device, set_seed
    dataloader:            load_hs, load_ws, load_participant, get_split_series
    dataset:               WAYEEGDataset
"""

from .device import (
    get_device,
    set_seed,
)
from .dataloader import (
    load_hs,
    load_ws,
    load_participant,
    get_split_series,
)
from .dataset import (
    WAYEEGDataset,
)
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
    extract_kt,
    extract_kt_with_velocity,
    estimate_velocity,
    sg_velocity,
    bw_velocity,
)
from .model import (
    HybridKGGTModel,
    build_model_from_config,
    TemporalCNNEncoder,
    LearnableGatedFusion,
    OptimizedKG_GAT,
    OptimizedTransformerEncoder,
    LearnablePositionalEncoding,
)
from .losses import (
    CombinedEMGLoss,
    build_loss_from_config,
    SoftDTWLoss,
    SmoothnessLoss,
    PearsonLoss,
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
    make_preprocess_fn,
    build_dataset_split,
    run_kfold_cross_validation,
    unique_series_arrays,
    inverse_emg_zscore,
    predict_full_series_from_dataset
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
    plot_emg_envelope_overlay
)

__all__ = [
    "get_device", "set_seed",
    "load_hs", "load_ws", "load_participant", "get_split_series",
    "WAYEEGDataset",
    # preprocessing
    "preprocess_eeg", "preprocess_eeg_from_config", "select_channels",
    "preprocess_emg", "preprocess_emg_from_config", "EMGNormalizer",
    "preprocess_kinematics", "preprocess_kinematics_from_config", "KinNormalizer",
    "extract_kt", "extract_kt_with_velocity",
    "estimate_velocity", "sg_velocity", "bw_velocity",
    # model
    "HybridKGGTModel", "build_model_from_config",
    "TemporalCNNEncoder", "LearnableGatedFusion",
    "OptimizedKG_GAT", "OptimizedTransformerEncoder", "LearnablePositionalEncoding",
    # losses
    "CombinedEMGLoss", "build_loss_from_config",
    "SoftDTWLoss", "SmoothnessLoss", "PearsonLoss", "soft_dtw",
    # training
    "train_model", "TrainConfig", "TrainResult", "EvalMetrics",
    "collect_predictions", "compute_metrics", "prepare_batch_factory",
    "save_checkpoint", "load_checkpoint", "print_gpu_info",
    "make_preprocess_fn", "build_dataset_split", "run_kfold_cross_validation",
    "unique_series_arrays", "inverse_emg_zscore", "predict_full_series_from_dataset",
    # plots
    "get_dynamic_fig_dir", "save_fig","plot_multichannel_trace",
    "plot_signal_heatmap", "plot_preprocessing_comparison",
    "plot_kinematic_features", "plot_training_history",
    "plot_prediction_overlay", "plot_gate_heatmap",
    "plot_attention_maps", "plot_residual_diagnostics",
    "plot_fused_pca","plot_gate_vs_emg_power", "plot_emg_envelope_overlay"
]
