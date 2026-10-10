"""
main — Method 1 KG-GT pipeline (nb04, no CCA).

Public API:
    device:                get_device, set_seed
    dataloader:            load_hs, load_ws, load_participant, get_split_series,
                           series_conditions, stratified_split, describe_split,
                           emg_channel_names, short_emg_label
    preprocessing_eeg:     preprocess_eeg, preprocess_eeg_from_config
    preprocessing_emg_kin: preprocess_emg, preprocess_emg_from_config,
                           preprocess_kinematics, preprocess_kinematics_from_config,
                           EMGNormalizer, KinNormalizer, extract_kt,
                           KT_RAW_NAMES, POSE_DROP_INDICES, POSE_FORCE_DROP_INDICES,
                           POSE_FORCE_RATE_DROP_INDICES,
                           kin_feature_names, hand_near_object_gate, index_object_distance
    eeg_features:          (Phase B) eeg_bandpower, eeg_feature_names, eeg_electrode_groups,
                           eeg_band_groups, build_eeg_transform_from_config,
                           eeg_n_bands_from_config, fit_feature_stats, clip_fraction
    dataset:              WAYEEGDataset,
                           resolve_participants (handles "all", single, or list),
                           ALL_PARTICIPANTS
    model:                 KGGTModel, build_kg_gt_from_config, CNN1dAligner,
                           TransformerOnlyModel, build_transformer_only_from_config
    losses:                CombinedEMGLoss, PeakWeightedMSELoss, EdgePriorKLDivLoss,
                           build_loss_from_config, compute_channel_weights,
                           channel_weights_from_config
    magnitude_eval:        magnitude_report, magnitude_report_by_series,
                           format_magnitude_table, format_gain_table, predict_series,
                           plot_magnitude_report,
                           lift_hold_segments, magnitude_by_load, format_magnitude_by_load,
                           plot_magnitude_by_load
    training:              train_model, TrainConfig, TrainResult,
                           collect_predictions, compute_metrics,
                           EvalMetrics, prepare_batch_factory,
                           save_checkpoint, load_checkpoint
    ablation:              input_ablation (conditions incl. kin_off = EEG only), AblationResult, shuffle_time,
                           apply_condition, pearson_per_channel,
                           rest_threshold_from_loader, ABLATION_CONDITIONS
    inference:             predict_series_stitched (the one inference path: windows of
                           the training window size), window_batch, window_starts,
                           stitch_windows
    interp:                (Phase D) lift_events, event_samples, extract_event_windows,
                           phase_levels, decompose_prediction, decompose_series,
                           attention_rollout, encoder_attention, rollout_rows,
                           eeg_occlusion_attribution, deletion_curve, eeg_effect_size,
                           event_locked_windows, concat_event_locked, event_locked_summary,
                           kin_column_groups, raw_zero_kin_baseline, series_gate,
                           eeg_channel_names
    interp_plots:          plot_interp_window, plot_event_locked
"""

from .device import get_device, set_seed
from .dataloader import (
    load_hs, load_ws, load_participant, get_split_series,
    series_conditions, stratified_split, describe_split,
    emg_channel_names, short_emg_label, EMG_SHORT_LABELS,
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
    extract_kt_raw,
    extract_kt,
    extract_kt_with_velocity,
    estimate_velocity,
    sg_velocity,
    bw_velocity,
    compute_muscle_edge_prior,
    KT_RAW_NAMES,
    POSE_DROP_INDICES,
    POSE_FORCE_DROP_INDICES,
    POSE_FORCE_RATE_DROP_INDICES,
    kin_feature_names,
    hand_near_object_gate,
    index_object_distance,
)
from .eeg_features import (
    DEFAULT_BANDS,
    eeg_bandpower,
    eeg_feature_names,
    band_labels,
    eeg_electrode_groups,
    eeg_band_groups,
    build_eeg_transform_from_config,
    eeg_n_bands_from_config,
    fit_feature_stats,
    clip_fraction,
)
from .dataset import WAYEEGDataset, resolve_participants, ALL_PARTICIPANTS
from .model import (
    electrode_dropout,
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
    CompositeEMGLoss,
    CCCLoss,
    PearsonCorrelationLoss,
    TemporalSmoothnessLoss,
    PeakWeightedMSELoss,
    EdgePriorKLDivLoss,
    CORALLoss,
    build_loss_from_config,
    build_coral_loss_from_config,
    compute_channel_weights,
    channel_weights_from_config,
)
from .magnitude_eval import (
    MAGNITUDE_COLUMNS,
    magnitude_report,
    magnitude_report_by_series,
    format_magnitude_table,
    format_gain_table,
    predict_series,
    plot_magnitude_report,
    LOAD_COLUMNS,
    lift_hold_segments,
    magnitude_by_load,
    format_magnitude_by_load,
    plot_magnitude_by_load,
)
from .coral_net import (
    CORALNet,
    LearnableFilterBank,
    LearnableLagAlignment,
    SimpleMambaBlock,
    MambaEncoder,
    SynergyDecoder,
    extract_nmf_synergies,
    build_coral_net_from_config,
)
from .coral_trainer import (
    CORALTrainer,
    CORALTrainConfig,
    CORALTrainResult,
)
from .training import (
    train_model,
    TrainConfig,
    TrainResult,
    EvalMetrics,
    collect_predictions,
    compute_metrics,
    prepare_batch_factory,
    normalized_prepare_batch_factory,
    save_checkpoint,
    load_checkpoint,
    print_gpu_info,
)
from .ablation import (
    input_ablation,
    AblationResult,
    shuffle_time,
    apply_condition,
    pearson_per_channel,
    rest_threshold_from_loader,
    default_conditions,
    supports_use_kin,
    ABLATION_CONDITIONS,
    ALL_ABLATION_CONDITIONS,
    ABLATION_LABELS,
    ABLATION_COLORS,
)
from .inference import (
    predict_series_stitched,
    window_batch,
    window_starts,
    stitch_windows,
)
from .interp import (
    EVENT_COLUMNS, EVENT_LABELS, DECOMPOSITION_KEYS,
    lift_events, event_samples, extract_event_windows, phase_levels,
    kin_column_groups, raw_zero_kin_baseline, series_gate, eeg_channel_names,
    decompose_prediction, decompose_series,
    attention_rollout, encoder_attention, rollout_rows,
    eeg_occlusion_attribution, deletion_curve, eeg_effect_size,
    event_locked_windows, concat_event_locked, event_locked_summary,
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
    # Advanced Diagnostic Plots
    plot_neural_mechanical_latency_lag,
    plot_spectral_power_decomposition,
    plot_kinematic_velocity_acceleration_density,
    plot_kin_smoothing_comparison,
    plot_kin_force_gate,
    kin_force_gate_stats,
)
from .interp_plots import plot_interp_window, plot_event_locked

__all__ = [
    # device
    "get_device", "set_seed",
    # dataloader
    "load_hs", "load_ws", "load_participant", "get_split_series",
    "series_conditions", "stratified_split", "describe_split",
    "emg_channel_names", "short_emg_label", "EMG_SHORT_LABELS",
    # preprocessing
    "preprocess_eeg", "preprocess_eeg_from_config", "select_channels",
    "preprocess_emg", "preprocess_emg_from_config", "EMGNormalizer",
    "preprocess_kinematics", "preprocess_kinematics_from_config", "KinNormalizer",
    "extract_kt_raw", "extract_kt", "extract_kt_with_velocity",
    "estimate_velocity", "sg_velocity", "bw_velocity", "compute_muscle_edge_prior",
    "KT_RAW_NAMES", "POSE_DROP_INDICES", "POSE_FORCE_DROP_INDICES",
    "POSE_FORCE_RATE_DROP_INDICES", "kin_feature_names",
    "hand_near_object_gate", "index_object_distance",
    # EEG features (Phase B): log band power, band-major layout, train statistics
    "DEFAULT_BANDS", "eeg_bandpower", "eeg_feature_names", "band_labels",
    "eeg_electrode_groups", "eeg_band_groups", "build_eeg_transform_from_config",
    "eeg_n_bands_from_config", "fit_feature_stats", "clip_fraction",
    # dataset
    "WAYEEGDataset", "resolve_participants", "ALL_PARTICIPANTS",
    # model
    "KGGTModel", "build_kg_gt_from_config", "CNN1dAligner",
    "TransformerEncoder", "build_transformer_from_config",
    "MuscleGATEncoder", "KinematicGuidedMuscleGATEncoder",
    "SinusoidalPositionalEncoding",
    "TransformerOnlyModel", "build_transformer_only_from_config", "electrode_dropout",
    # losses
    "CombinedEMGLoss", "CompositeEMGLoss", "CCCLoss", "PearsonCorrelationLoss",
    "TemporalSmoothnessLoss", "PeakWeightedMSELoss", "EdgePriorKLDivLoss",
    "CORALLoss", "build_loss_from_config", "build_coral_loss_from_config",
    "compute_channel_weights", "channel_weights_from_config",
    # magnitude evaluation
    "MAGNITUDE_COLUMNS", "magnitude_report", "magnitude_report_by_series",
    "format_magnitude_table", "format_gain_table", "predict_series", "plot_magnitude_report",
    "LOAD_COLUMNS", "lift_hold_segments", "magnitude_by_load", "format_magnitude_by_load",
    "plot_magnitude_by_load",
    # coral-net
    "CORALNet", "LearnableFilterBank", "LearnableLagAlignment", "SimpleMambaBlock",
    "MambaEncoder", "SynergyDecoder", "extract_nmf_synergies", "build_coral_net_from_config",
    "CORALTrainer", "CORALTrainConfig", "CORALTrainResult",
    # training
    "train_model", "TrainConfig", "TrainResult", "EvalMetrics",
    "collect_predictions", "compute_metrics", "prepare_batch_factory",
    "normalized_prepare_batch_factory",
    "save_checkpoint", "load_checkpoint", "print_gpu_info",
    # ablation
    "input_ablation", "AblationResult", "shuffle_time", "apply_condition",
    "pearson_per_channel", "rest_threshold_from_loader", "default_conditions", "supports_use_kin",
    "ABLATION_CONDITIONS", "ALL_ABLATION_CONDITIONS", "ABLATION_LABELS", "ABLATION_COLORS",
    # inference (one path: windows of the training window size)
    "predict_series_stitched", "window_batch", "window_starts", "stitch_windows",
    # interpretability (Phase D): events, exact path decomposition, rollout, EEG faithfulness
    "EVENT_COLUMNS", "EVENT_LABELS", "DECOMPOSITION_KEYS",
    "lift_events", "event_samples", "extract_event_windows", "phase_levels",
    "kin_column_groups", "raw_zero_kin_baseline", "series_gate", "eeg_channel_names",
    "decompose_prediction", "decompose_series",
    "attention_rollout", "encoder_attention", "rollout_rows",
    "eeg_occlusion_attribution", "deletion_curve", "eeg_effect_size",
    "event_locked_windows", "concat_event_locked", "event_locked_summary",
    "plot_interp_window", "plot_event_locked",
    # plots — core
    "get_dynamic_fig_dir", "save_fig", "plot_multichannel_trace",
    "plot_signal_heatmap", "plot_preprocessing_comparison",
    "plot_kinematic_features", "plot_training_history",
    "plot_prediction_overlay", "plot_gate_heatmap",
    "plot_attention_maps", "plot_residual_diagnostics",
    "plot_fused_pca", "plot_gate_vs_emg_power", "plot_emg_envelope_overlay",
    # plots — interpretability & diagnostics
    "plot_interpretability_triptych",
    "plot_muscle_synergy_matrix",
    "plot_kin_edge_linear_weights",
    "plot_neural_mechanical_latency_lag",
    "plot_spectral_power_decomposition",
    "plot_kinematic_velocity_acceleration_density",
    "plot_kin_smoothing_comparison",
    "plot_kin_force_gate", "kin_force_gate_stats",
]
