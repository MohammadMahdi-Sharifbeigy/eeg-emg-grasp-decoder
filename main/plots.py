# @title PaperStylePlotHelpers
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PAPER_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
}
plt.rcParams.update(PAPER_STYLE)
sns.set_theme(style="whitegrid", context="paper")

def _make_time_axis(data, fs):
    return np.arange(len(data)) / float(fs)

def get_dynamic_fig_dir(cfg):
    """Rebuilds the dynamic model path from the config, accounting for current fold."""
    max_epochs = cfg["training"]["max_epochs"]
    stride = cfg["data"]["stride"]
    lr = cfg["training"].get("lr", 1e-4)
    bs = cfg["training"]["batch_size"]
    config_str = f"ep{max_epochs}_st{stride}_lr{lr}_bs{bs}"

    # Per-subject folder: results/P1/main/<config_str>/
    p_val = cfg["data"].get("participant", cfg["data"].get("participants", []))
    if isinstance(p_val, (list, tuple)) and len(p_val) > 0:
        p_id = p_val[0]
    else:
        p_id = p_val
    p_id = str(p_id).replace("P", "")
    subject_str = f"P{p_id}" if p_id != "" and p_id != [] else "unknown"

    base_dir = Path("results") / subject_str / "main" / config_str


    # If we injected 'current_fold', put the plot in that fold's folder!
    if "current_fold" in cfg["training"]:
        return base_dir / cfg["training"]["current_fold"] / "plots"

    return base_dir / "plots"


def save_fig(fig, name, cfg=None, dpi=300, close=False):                                                                   
    """Save a specific matplotlib figure as PNG and PDF dynamically according to model config."""                                                                          
    if cfg is None:
        return None, None
        
    fig_dir = get_dynamic_fig_dir(cfg)                                                                                         
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    import re
    stem = Path(name).stem.lower()
    # Replace anything that isn't alphanumeric, dash, or underscore with an underscore
    stem = re.sub(r'[^\w\-]', '_', stem)
    # Remove consecutive underscores
    stem = re.sub(r'_+', '_', stem).strip('_')
                                                                                                                                    
    png_path = fig_dir / f"{stem}.png"                                                                                               
    pdf_path = fig_dir / f"{stem}.pdf"                                                                                               
                                                                                                                                    
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")                                                           
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")                                                                    
                                                                                                                                    
    if close:                                                                                                                        
        plt.close(fig)                                                                                                               
                                                                                                                                    
    print(f"Saved figure:\n  {png_path}\n  {pdf_path}")                                                                              
    return png_path, pdf_path 

def plot_multichannel_trace(data, fs=500, channel_names=None, title="Signal", max_channels=8, figsize=(12, 6), cfg=None):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    n_channels = min(data.shape[1], max_channels)
    time = _make_time_axis(data, fs)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.plot(time, data[:, idx], linewidth=1.1, color=f"C{idx % 10}")
        ax.set_ylabel(channel_names[idx] if channel_names else f"ch{idx}")
        ax.grid(True, alpha=0.2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, f"multichannel_{title.replace(' ', '_').lower()}", cfg)
    return fig

def plot_signal_heatmap(data, title="Signal Heatmap", channel_names=None, figsize=(12, 4), cmap="mako", cfg=None):
    data = np.asarray(data)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data.T, cmap=cmap, cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Channels")
    if channel_names and len(channel_names) == data.shape[1]:
        ax.set_yticks(np.arange(len(channel_names)) + 0.5)
        ax.set_yticklabels(channel_names, rotation=0)
    fig.tight_layout()
    save_fig(fig, f"heatmap_{title.replace(' ', '_').lower()}", cfg)
    return fig

def plot_preprocessing_comparison(raw_data, processed_data, fs=500, channel_idx=0, title_prefix="EEG", figsize=(12, 6), cfg=None):
    raw_data = np.asarray(raw_data)
    processed_data = np.asarray(processed_data)
    time_raw = _make_time_axis(raw_data, fs)
    time_proc = _make_time_axis(processed_data, fs)
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
    axes[0].plot(time_raw, raw_data[:, channel_idx], color="tab:gray", linewidth=1.0)
    axes[0].set_title(f"{title_prefix} Raw - channel {channel_idx}")
    axes[1].plot(time_proc, processed_data[:, channel_idx], color="tab:blue", linewidth=1.0)
    axes[1].set_title(f"{title_prefix} Processed - channel {channel_idx}")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, f"preprocessing_comp_{title_prefix.lower()}_ch{channel_idx}", cfg)
    return fig

def plot_kinematic_features(kin, fs=500, feature_names=None, max_features=6, figsize=(12, 8), cfg=None):
    kin = np.asarray(kin)
    n_features = min(kin.shape[1], max_features)
    time = _make_time_axis(kin, fs)
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    if n_features == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.plot(time, kin[:, idx], linewidth=1.0, color=f"C{idx % 10}")
        ax.set_ylabel(feature_names[idx] if feature_names else f"k{idx}")
    axes[0].set_title("Kinematic State Features")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, "kinematic_features", cfg)
    return fig

def plot_training_history(history, stage="Training", figsize=(13, 5), cfg=None):
    """Plot Train/Val Loss and Learning Rate curve over epochs side-by-side."""
    if isinstance(history, list):
        train_loss = [h.get('train_loss', 0) for h in history]
        val_loss = [h.get('val_loss', 0) for h in history]
        lr_history = [h.get('lr', 0) for h in history]
    else:
        train_loss = history.get("train", [])
        val_loss = history.get("val", [])
        lr_history = history.get("lr", [])

    has_lr = len(lr_history) > 0 and any(lr > 0 for lr in lr_history)
    
    if has_lr:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] * 0.6, figsize[1]))
        ax2 = None

    epochs = range(1, len(train_loss) + 1)
    
    # Left panel: Loss
    ax1.plot(epochs, train_loss, label="Train loss", linewidth=2.0, color="#2B5B84")
    if len(val_loss) == len(train_loss):
        ax1.plot(epochs, val_loss, label="Val loss", linewidth=2.0, color="#ED553B")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title(f"{stage} â€” Loss History", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)

    # Right panel: Learning Rate
    if ax2 is not None and has_lr:
        lr_epochs = range(1, len(lr_history) + 1)
        ax2.plot(lr_epochs, lr_history, linewidth=2.0, color="#3CAEA3", label="Learning Rate")
        ax2.set_xlabel("Epoch", fontweight="bold")
        ax2.set_ylabel("Learning Rate", fontweight="bold")
        ax2.set_yscale("log")
        ax2.set_title(f"{stage} â€” LR Schedule", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    
    # Generate clean filename based on stage name
    safe_stage = "".join(c if c.isalnum() else "_" for c in str(stage)).strip("_").lower()
    fname = f"history_{safe_stage}" if safe_stage else "training_history"
    save_fig(fig, fname, cfg)
    return fig

def plot_prediction_overlay(pred, target, fs=500, channel_names=None, channels=None, figsize=(12, 8), cfg=None):
    pred = np.asarray(pred)
    target = np.asarray(target)
    if channels is None:
        channels = list(range(min(pred.shape[1], 5)))
    time = _make_time_axis(pred, fs)
    fig, axes = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        ax.plot(time, target[:, ch], label="Target", linewidth=1.2, color="black")
        ax.plot(time, pred[:, ch], label="Prediction", linewidth=1.0, color="tab:red", alpha=0.85)
        ax.set_ylabel(channel_names[ch] if channel_names else f"EMG {ch}")
        ax.legend(loc="upper right")
    axes[0].set_title("Prediction vs Target")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, "prediction_overlay", cfg)
    return fig

def plot_gate_heatmap(gate_tensor, sample_idx=0, cfg=None):
    g_t = gate_tensor[sample_idx].detach().cpu().numpy()
    g_t_mean = g_t.mean(axis=-1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 3]})
    ax1.plot(g_t_mean, color='purple', linewidth=2)
    ax1.set_title("Mean Gated Fusion Modality Weight (g_t)")
    ax1.set_ylabel("Weight (1=EEG, 0=KIN)")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    sns.heatmap(g_t.T, cmap="coolwarm", cbar=True, ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Gate Values Across Hidden Dimensions (D=64)")
    ax2.set_xlabel("Time (Samples)")
    ax2.set_ylabel("Hidden Dimension")
    plt.tight_layout()
    save_fig(fig, "gate_heatmap", cfg)
    return fig

def plot_attention_maps(attn_tensor, title_prefix, sample_idx=0, cfg=None,
                        crop: int = 200):
    """Plot self- or cross-attention maps.

    Args:
        attn_tensor : Tensor of shape (B_or_chunks, n_heads, T, T).
                      When chunking is active the batch dim is B*num_chunks;
                      sample_idx selects which chunk to visualise.
        crop        : Show only the first `crop` timesteps on each axis so
                      the diagonal structure is actually visible.  Set to
                      None to show the full matrix (may be very large).
    """
    attn = attn_tensor[sample_idx].detach().cpu().float().numpy()
    n_heads = attn.shape[0]
    T = attn.shape[-1]

    # Crop so the diagonal band is actually visible at plot resolution
    if crop is not None:
        c = min(crop, T)
        attn = attn[:, :c, :c]

    # 1. Plot Mean Attention across heads
    mean_attn = attn.mean(axis=0)
    fig_mean = plt.figure(figsize=(6, 5))
    sns.heatmap(mean_attn, cmap="viridis")
    plt.title(f"{title_prefix} - Mean Across Heads (first {attn.shape[-1]} steps)")
    plt.xlabel("Key Time")
    plt.ylabel("Query Time")
    plt.tight_layout()
    save_fig(fig_mean, f"{title_prefix.replace(' ', '_').lower()}_mean", cfg)

    # 2. Plot individual heads
    cols = 4
    rows = (n_heads + cols - 1) // cols
    fig_heads, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3))
    axes = axes.flatten()
    for h in range(n_heads):
        sns.heatmap(attn[h], cmap="viridis", ax=axes[h], cbar=False)
        axes[h].set_title(f"Head {h+1}")
        axes[h].set_xticks([])
        axes[h].set_yticks([])
    for h in range(n_heads, len(axes)):
        axes[h].axis('off')
    plt.suptitle(f"{title_prefix} - Per Head (first {attn.shape[-1]} steps)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig_heads, f"{title_prefix.replace(' ', '_').lower()}_per_head", cfg)
    return fig_mean, fig_heads

def plot_residual_diagnostics(pred, target, channel_idx=0, figsize=(10, 4), cfg=None):
    """Plots the residual error trace and distribution for a specific channel."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    residual = pred[:, channel_idx] - target[:, channel_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(residual, linewidth=0.9)
    axes[0].set_title(f"Residual trace - channel {channel_idx}")
    axes[0].set_xlabel("Time (Samples)")
    
    sns.histplot(residual, kde=True, ax=axes[1], color="tab:purple")
    axes[1].set_title(f"Residual distribution - channel {channel_idx}")
    
    fig.tight_layout()
    save_fig(fig, f"residual_diagnostics_ch{channel_idx}", cfg)
    return fig

def plot_fused_pca(fused_tensor, sample_idx=0, figsize=(6, 5), cfg=None):
    """Plots a 2D PCA representation of the latent fused state over time."""
    from sklearn.decomposition import PCA
    
    # Extract sequence [seq_len, dim]
    fused_np = fused_tensor[sample_idx].detach().cpu().numpy()
    
    pca = PCA(n_components=2)
    z = pca.fit_transform(fused_np)
    
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(z[:, 0], z[:, 1], c=range(len(z)), s=15, cmap="viridis")
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Time Step")
    
    ax.set_title("Fused Representation Latent Space (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    
    fig.tight_layout()
    save_fig(fig, "fused_pca_scatter", cfg)
    return fig

def plot_gate_vs_emg_power(gate_tensor, preds_tensor, sample_idx=0, figsize=(10, 4), cfg=None):
    """Plots the average gate value against the mean absolute EMG amplitude."""
    # Gate shape: [batch, seq_len, hidden_dim] -> Mean across hidden dims
    gate = gate_tensor[sample_idx].detach().cpu().numpy().mean(axis=1)
    
    # Preds shape: [batch, seq_len, channels] or similar -> Mean abs amplitude
    if torch.is_tensor(preds_tensor):
        emg_power = preds_tensor[sample_idx].detach().cpu().numpy()
    else:
        emg_power = np.asarray(preds_tensor)[sample_idx]
        
    emg_power = np.abs(emg_power).mean(axis=1)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot Gate on primary Y-axis
    l1 = ax1.plot(gate, label="Gate (1=EEG, 0=KIN)", linewidth=1.5, color="tab:blue")
    ax1.set_xlabel("Time (Samples)")
    ax1.set_ylabel("Gate Value", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    
    # Plot EMG Power on secondary Y-axis (since amplitudes might have different scales)
    ax2 = ax1.twinx()  
    l2 = ax2.plot(emg_power, label="EMG Amplitude (Mean Abs)", linewidth=1.5, color="tab:orange")
    ax2.set_ylabel("EMG Amplitude", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")
    
    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    
    plt.title("Information Gating vs EMG Power Over Time")
    fig.tight_layout()
    save_fig(fig, "gate_vs_emg_power", cfg)
    return fig

def plot_emg_envelope_overlay(raw_emg, env_emg, fs_raw=4000, fs_env=500, channel_idx=0, channel_name="EMG 1", title="EMG Envelope Overlay", figsize=(10, 4), cfg=None):
    """Plots raw EMG signal overlaid with its processed envelope or RMS for a specific channel."""
    raw_emg = np.asarray(raw_emg)
    env_emg = np.asarray(env_emg)
    
    time_raw = _make_time_axis(raw_emg, fs_raw)
    time_env = _make_time_axis(env_emg, fs_env)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw signal in blue
    ax.plot(time_raw, raw_emg[:, channel_idx], color="blue", linewidth=0.5, label=f"{channel_name} (Raw)")
    
    # Plot envelope/RMS in red
    ax.plot(time_env, env_emg[:, channel_idx], color="red", linewidth=2.0, label=f"{channel_name} (Envelope/RMS)")
    
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_fig(fig, f"emg_overlay_ch{channel_idx}", cfg)
    return fig


# ============================================================================
# Interpretability Visualizations
# ============================================================================

def plot_interpretability_triptych(
    pred:              np.ndarray,
    target:            np.ndarray,
    eeg_attn:          np.ndarray,
    kin_edge_bias:     np.ndarray,
    muscle_names:      list[str],
    kin_feature_names: list[str] | None = None,
    fs:                float = 500.0,
    cfg:               dict | None = None,
    figsize:           tuple = (15, 13),
) -> plt.Figure:
    """Publication-quality 3-panel interpretability 'money plot' for a single inference window.

    Panels (top â†’ bottom):
      1. Actual vs Predicted EMG Envelope â€” all muscle channels overlaid with per-channel
         Pearson r and RMSE annotations and error-fill shading.
      2. EEG Temporal Attention Heatmap â€” (H Ã— T) matrix showing which EEG timeframes
         each attention head focuses on, averaged over query positions.
      3. Dynamic Kinematic Edge Bias â€” 10 time-series lines for the upper-triangle muscle
         pairs, showing how kinematic state modulates the muscle graph over the window.
      All panels share the same time axis for direct temporal alignment.

    Args:
        pred:              (T, C) predicted EMG envelope.
        target:            (T, C) ground-truth EMG envelope.
        eeg_attn:          (H, T, T) EEG self-attention weights from the last encoder layer.
                           Obtain via: model.encoder.layers[-1].mhsa.last_attn_weights[0]
        kin_edge_bias:     (T, H, N, N) per-timestep kinematic edge bias (dynamic component).
                           Obtain via: model.gat.last_kin_edge_bias[0]
        muscle_names:      List of C muscle channel names, e.g. ['FDI', 'APB', 'ADM', 'ECR', 'FCR'].
        kin_feature_names: Optional list of kin_dim feature names for tooltips/legends.
        fs:                Sampling rate in Hz (for time axis). Default 500.
        cfg:               Optional config dict passed to save_fig().
        figsize:           (width, height) in inches.

    Returns:
        matplotlib Figure. Call plt.show() or fig.savefig(...) afterwards.
    """
    import matplotlib.gridspec as gridspec
    from scipy.stats import pearsonr as _pearsonr

    pred   = np.asarray(pred,   dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    T, n_ch = pred.shape
    time = np.arange(T) / fs   # seconds

    # â”€â”€ EEG attention processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # eeg_attn: (H, T_q, T_k). Mean over query positions â†’ (H, T) "key received".
    eeg_attn  = np.asarray(eeg_attn, dtype=np.float64)   # (H, T, T)
    H_attn    = eeg_attn.shape[0]
    key_attn  = eeg_attn.mean(axis=1)                     # (H, T) key attention density

    # Min-max normalise per head to [0, 1] for a clean, comparable heatmap.
    attn_min = key_attn.min(axis=1, keepdims=True)
    attn_max = key_attn.max(axis=1, keepdims=True)
    key_norm  = (key_attn - attn_min) / (attn_max - attn_min + 1e-8)   # (H, T)

    # â”€â”€ Kinematic edge bias processing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # kin_edge_bias: (T, H, N, N). Mean over H â†’ (T, N, N).
    kin_edge_bias = np.asarray(kin_edge_bias, dtype=np.float64)
    n_muscles  = kin_edge_bias.shape[-1]
    edge_mean  = kin_edge_bias.mean(axis=1)   # (T, N, N) â€” mean over H

    # Upper-triangle muscle pairs (10 unique for N=5).
    pairs      = [(i, j) for i in range(n_muscles) for j in range(i + 1, n_muscles)]
    pair_labels = [f"{muscle_names[i]}\u2013{muscle_names[j]}" for i, j in pairs]
    pair_data   = np.stack([edge_mean[:, i, j] for i, j in pairs], axis=1)   # (T, n_pairs)
    n_pairs     = len(pairs)

    # Peak EMG activation time (mean over channels) â€” used for a reference marker.
    peak_t = float(target.argmax(axis=0).mean()) / fs

    # â”€â”€ Color palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    EMG_ACTUAL    = "#1a1a2e"      # near-black
    EMG_PRED      = "#e94040"      # vivid red
    EMG_FILL      = "#e94040"
    PEAK_MARKER   = "#00b4d8"      # cyan â€” peak reference line
    EDGE_CMAP     = plt.cm.tab10
    edge_colors   = [EDGE_CMAP(k / max(n_pairs - 1, 1)) for k in range(n_pairs)]

    # â”€â”€ Figure / GridSpec â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig = plt.figure(figsize=figsize, facecolor="white")
    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[n_ch * 1.1, 1.8, 2.8],
        hspace=0.42,
    )

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # PANEL 1 â€” EMG Envelopes
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    inner_emg = gridspec.GridSpecFromSubplotSpec(
        n_ch, 1, subplot_spec=outer[0], hspace=0.06,
    )
    emg_axes = [fig.add_subplot(inner_emg[c]) for c in range(n_ch)]

    for c, ax in enumerate(emg_axes):
        # Actual and predicted traces.
        ax.plot(time, target[:, c], color=EMG_ACTUAL, lw=1.6, zorder=3, label="Actual")
        ax.plot(time, pred[:, c],   color=EMG_PRED,   lw=1.0, ls="--", zorder=2,
                alpha=0.90, label="Predicted")

        # Error fill â€” highlights discrepancy regions.
        ax.fill_between(time, target[:, c], pred[:, c],
                        color=EMG_FILL, alpha=0.12, zorder=1)

        # Peak reference marker.
        ax.axvline(peak_t, color=PEAK_MARKER, lw=0.9, ls=":", alpha=0.55, zorder=4)

        # Per-channel performance annotation.
        try:
            r_val, _ = _pearsonr(pred[:, c], target[:, c])
        except Exception:
            r_val = float("nan")
        rmse_val = float(np.sqrt(np.mean((pred[:, c] - target[:, c]) ** 2)))
        ax.text(
            0.993, 0.86,
            f"r = {r_val:.3f}  |  RMSE = {rmse_val:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="#2d2d2d",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.82, ec="none"),
        )

        ch_label = muscle_names[c] if c < len(muscle_names) else f"ch{c}"
        ax.set_ylabel(ch_label, fontsize=9, labelpad=4, rotation=0, ha="right", va="center")
        ax.set_xlim(time[0], time[-1])
        ax.tick_params(axis="x", labelbottom=(c == n_ch - 1), labelsize=8)
        ax.tick_params(axis="y", labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    emg_axes[0].legend(
        loc="upper left", fontsize=8.5, framealpha=0.85, ncol=2,
        borderpad=0.35, handlelength=1.6,
    )
    emg_axes[0].set_title(
        "Panel 1 \u2014 EMG Envelope: Actual vs Predicted",
        fontsize=11, fontweight="bold", pad=5,
    )
    emg_axes[-1].set_xlabel("Time (s)", fontsize=9)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # PANEL 2 â€” EEG Temporal Attention Heatmap
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax_attn = fig.add_subplot(outer[1])

    im = ax_attn.imshow(
        key_norm,
        aspect="auto",
        cmap="plasma",
        interpolation="nearest",
        extent=[time[0], time[-1], H_attn + 0.5, 0.5],
        vmin=0.0, vmax=1.0,
    )
    # Peak marker on attention panel.
    ax_attn.axvline(peak_t, color=PEAK_MARKER, lw=1.0, ls=":", alpha=0.80,
                    label=f"Peak t={peak_t:.2f}s")

    cbar = fig.colorbar(im, ax=ax_attn, pad=0.01, shrink=0.90, aspect=12)
    cbar.set_label("Attention\n(norm.)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax_attn.set_yticks(range(1, H_attn + 1))
    ax_attn.set_yticklabels([f"H{h}" for h in range(1, H_attn + 1)], fontsize=8.5)
    ax_attn.set_xlabel("Time (s)", fontsize=9)
    ax_attn.set_ylabel("Attn Head", fontsize=9)
    ax_attn.set_xlim(time[0], time[-1])
    ax_attn.tick_params(axis="x", labelsize=8)
    ax_attn.set_title(
        "Panel 2 \u2014 EEG Temporal Attention (per head \u00d7 key timestep, mean over query positions)",
        fontsize=11, fontweight="bold", pad=5,
    )
    ax_attn.legend(loc="upper left", fontsize=8, framealpha=0.7, borderpad=0.3)

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # PANEL 3 â€” Dynamic Kinematic Edge Bias
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax_edge = fig.add_subplot(outer[2])

    for k, (label, color) in enumerate(zip(pair_labels, edge_colors)):
        ax_edge.plot(time, pair_data[:, k], lw=1.2, color=color,
                     label=label, alpha=0.88)

    ax_edge.axhline(0.0, color="#888888", lw=0.7, ls=":", alpha=0.7)
    ax_edge.axvline(peak_t, color=PEAK_MARKER, lw=0.9, ls=":", alpha=0.55)

    ax_edge.set_xlabel("Time (s)", fontsize=9)
    ax_edge.set_ylabel("Edge Bias\n(kinematic component)", fontsize=9)
    ax_edge.set_title(
        "Panel 3 \u2014 Dynamic Kinematic Edge Bias: muscle-graph modulation over window"
        "\n(mean over attention heads; upper-triangle pairs only)",
        fontsize=11, fontweight="bold", pad=5,
    )
    ax_edge.legend(
        loc="upper right", fontsize=7.5, ncol=2,
        framealpha=0.85, borderpad=0.35, handlelength=1.4,
    )
    ax_edge.set_xlim(time[0], time[-1])
    for sp in ("top", "right"):
        ax_edge.spines[sp].set_visible(False)
    ax_edge.tick_params(labelsize=8)

    # â”€â”€ Global title â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig.suptitle(
        "KG-GT Model \u2014 Interpretability Triptych (Single Inference Window)",
        fontsize=13, fontweight="bold", y=1.010,
    )
    fig.tight_layout()
    save_fig(fig, "interpretability_triptych", cfg)
    return fig


def plot_muscle_synergy_matrix(
    attn_weights:  np.ndarray,
    muscle_names:  list[str],
    title:         str = "Learned Muscle Synergy\n(Mean GAT Attention over All Windows)",
    figsize:       tuple = (7, 6),
    cfg:           dict | None = None,
) -> plt.Figure:
    """Annotated heatmap of the mean GAT attention matrix â€” reveals muscle synergies.

    The learned attention weights encode which muscle pairs co-activate.
    Averaged over all timesteps and attention heads, the (N Ã— N) matrix is
    analogous to an NMF synergy matrix but derived end-to-end from the data.

    Args:
        attn_weights:  (B*T, H, N, N) GAT attention from model.gat.last_attn_weights.
                       Concatenate across all test batches for a dataset-level view.
        muscle_names:  List of N muscle names for axis labels.
        title:         Figure title.
        figsize:       (width, height) in inches.
        cfg:           Optional config dict passed to save_fig().

    Returns:
        matplotlib Figure.
    """
    attn = np.asarray(attn_weights, dtype=np.float64)   # (B*T, H, N, N)
    # Mean over all timesteps AND all heads â†’ (N, N)
    synergy = attn.mean(axis=(0, 1))
    N = synergy.shape[0]
    names = muscle_names[:N]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    mask_diag = np.zeros_like(synergy, dtype=bool)
    np.fill_diagonal(mask_diag, False)   # keep diagonal

    hm = sns.heatmap(
        synergy,
        ax=ax,
        cmap="mako",
        vmin=0.0,
        vmax=synergy.max(),
        annot=True,
        fmt=".3f",
        annot_kws={"size": 10.5, "weight": "semibold", "color": "white"},
        linewidths=1.0,
        linecolor="white",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Mean Attention Weight", "shrink": 0.88},
    )

    # Highlight diagonal (self-connections) with a different annotation colour.
    for i in range(N):
        ax.add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False,
                           edgecolor="#FFD700", lw=2.0, zorder=5)
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Key Muscle (source)", fontsize=10)
    ax.set_ylabel("Query Muscle (target)", fontsize=10)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Footnote with interpretation guidance.
    fig.text(
        0.5, -0.01,
        "Off-diagonal high values = stable co-activation (synergy)."
        "  Gold border = self-loop.",
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    fig.tight_layout()
    save_fig(fig, "muscle_synergy_matrix", cfg)
    return fig


def plot_kin_edge_linear_weights(
    weight_matrix:     np.ndarray,
    muscle_names:      list[str],
    kin_feature_names: list[str] | None = None,
    num_heads:         int = 4,
    n_nodes:           int = 5,
    figsize:           tuple = (15, 8),
    cfg:               dict | None = None,
) -> plt.Figure:
    """Heatmap of the transparent kin_edge_linear weight matrix.

    Directly reveals which kinematic features drive which muscle-pair edge
    connections in each attention head â€” the primary neurophysiological
    interpretation output of the transparent linear mapping (Q1 design choice).

    W \u2208 R^{H\u00b7N\u00b2 \u00d7 kin_dim}. Entry W[h*N\u00b2 + i*N + j, k] is the direct linear
    contribution of kinematic feature k to the i\u2192j edge in head h.

    Displayed as: one (N\u00b2 \u00d7 kin_dim) diverging heatmap per head + a mean-head
    panel. Horizontal dashed lines separate source muscle groups (every N rows).

    Args:
        weight_matrix:     (H*N*N, kin_dim) numpy array â€” e.g.
                           model.gat.kin_edge_linear.weight.detach().cpu().numpy()
        muscle_names:      List of N muscle names.
        kin_feature_names: List of kin_dim kinematic feature names.
                           Default: ['k0', 'k1', ...].
        num_heads:         Number of GAT attention heads H.
        n_nodes:           Number of muscle nodes N.
        figsize:           (width, height) in inches.
        cfg:               Optional config dict passed to save_fig().

    Returns:
        matplotlib Figure (mean + H per-head subplots).
    """
    W    = np.asarray(weight_matrix, dtype=np.float64)   # (H*N\u00b2, kin_dim)
    kin_dim = W.shape[1]
    W_4d = W.reshape(num_heads, n_nodes, n_nodes, kin_dim)   # (H, N, N, kin_dim)

    # Row labels: "Muscle_i \u2192 Muscle_j" for all N\u00b2 source-target combos.
    row_labels = [
        f"{muscle_names[i]}\u2192{muscle_names[j]}"
        for i in range(n_nodes)
        for j in range(n_nodes)
    ]
    col_labels = kin_feature_names or [f"k{k}" for k in range(kin_dim)]
    # Truncate to 6 chars to avoid overlap on x-axis.
    col_short  = [f[:7] for f in col_labels]

    n_plots = num_heads + 1   # one per head + one mean
    n_cols  = min(n_plots, 3)
    n_rows  = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor="white")
    axes_flat = np.array(axes).flatten()

    def _draw_panel(ax: plt.Axes, data_n2_k: np.ndarray, panel_title: str) -> None:
        """data_n2_k: (N\u00b2, kin_dim)."""
        vabs = max(np.abs(data_n2_k).max(), 1e-6)
        sns.heatmap(
            data_n2_k,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vabs,
            vmax=vabs,
            annot=False,
            xticklabels=col_short,
            yticklabels=row_labels,
            cbar_kws={"shrink": 0.85, "label": "Weight"},
            linewidths=0.0,
        )
        ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=4)
        ax.set_xlabel("Kinematic Feature", fontsize=8, labelpad=3)
        ax.set_ylabel("Muscle Edge (i\u2192j)", fontsize=8, labelpad=3)
        ax.tick_params(axis="x", labelsize=7, rotation=55)
        ax.tick_params(axis="y", labelsize=7, rotation=0)

        # Horizontal dashed lines separating source-muscle groups (every N rows).
        for boundary in range(n_nodes, n_nodes * n_nodes, n_nodes):
            ax.axhline(boundary, color="#888888", lw=0.9, ls="--", alpha=0.55)

    # Mean over heads.
    W_mean = W_4d.mean(axis=0).reshape(n_nodes * n_nodes, kin_dim)
    _draw_panel(axes_flat[0], W_mean, "Mean over All Heads")

    # Per-head panels.
    for h in range(num_heads):
        W_h = W_4d[h].reshape(n_nodes * n_nodes, kin_dim)
        _draw_panel(axes_flat[h + 1], W_h, f"Head {h + 1}")

    # Hide surplus subplot axes.
    for k in range(n_plots, len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        r"Kinematic $\rightarrow$ Muscle Edge Weights  "
        r"($W_{h,\,i\to j,\,k}$: contribution of feature $k$ to edge $i\to j$ in head $h$)"
        "\n\u25ba Red = positive bias (feature increases this edge)"
        "   \u25ba Blue = negative bias (feature suppresses this edge)"
        "\n\u25ba High |weight| on d_grip / F_L / F_G rows confirms task-relevant synergy routing",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_fig(fig, "kin_edge_linear_weights", cfg)
    return fig


# DIAGNOSTIC PLOT 1: Residual Decomposition â€” GAT nodes vs. Kinematic Skip
def plot_residual_decomposition(model, eeg_window, kin_window, emg_target,
                                 emg_names=None, fs=500,
                                 emg_mean=None, emg_std=None,
                                 smooth_hz=10.0,
                                 cfg=None):
    """Decomposes a single window's model output into two additive contributions:
       - GAT-only component: what the graph attention network produces from EEG alone
       - Kinematic skip component: what kin_skip_proj adds on top
    This reveals whether the kinematic highway is contributing meaningfully to burst
    peaks or whether the GAT is doing all the work (or vice versa).

    Args:
        model:       Trained KGGTModel with use_kinematic_guidance=True.
        eeg_window:  Tensor (1, T, C_eeg) â€” a single EEG window.
        kin_window:  Tensor (1, T, K)     â€” matching kinematic window.
        emg_target:  Array  (T, 5)        â€” ground-truth EMG envelope.
        emg_names:   List[str] of 5 muscle names.
        fs:          Sampling frequency (Hz).
        cfg:         Project config dict for save_fig.

    Usage (notebook):
        from main.plots import plot_residual_decomposition
        eeg_t = eeg_window.unsqueeze(0).to(device)
        kin_t = kin_window.unsqueeze(0).to(device)
        fig = plot_residual_decomposition(model, eeg_t, kin_t, emg_target_np, emg_names=EMG_NAMES, cfg=notebook_cfg)
    """
    import torch
    model.eval()
    emg_names = emg_names or [f"Ch{i}" for i in range(5)]

    with torch.no_grad():
        eeg_window = eeg_window.to(next(model.parameters()).device)
        kin_window  = kin_window.to(next(model.parameters()).device)

        temporal = model.encoder(eeg_window)
        nodes    = model.node_projection(temporal)

        # GAT refined node embedding (graph attention output ONLY)
        gat_out  = model.gat(nodes, kin_window)          # (1, T, n_nodes, node_dim)

        # Kinematic skip contribution
        B, T, K = kin_window.shape
        kin_skip = model.kin_skip_proj(kin_window).view(B, T, model.out_channels, -1)

        # Decode each component independently — all in normalized model space
        full_out  = model.decoder(gat_out + kin_skip).squeeze(-1).squeeze(0).cpu().numpy()  # (T, 5)
        gat_only  = model.decoder(gat_out).squeeze(-1).squeeze(0).cpu().numpy()             # (T, 5)
        skip_only = model.decoder(kin_skip).squeeze(-1).squeeze(0).cpu().numpy()            # (T, 5)

    # Un-normalize to physical EMG scale so they match emg_target
    if emg_mean is not None and emg_std is not None:
        if hasattr(emg_mean, "cpu"):
            emg_mean = emg_mean.cpu().numpy()
        if hasattr(emg_std, "cpu"):
            emg_std = emg_std.cpu().numpy()
        _m = np.asarray(emg_mean).reshape(1, -1)
        _s = np.asarray(emg_std).reshape(1, -1)
        full_out  = full_out  * _s + _m
        gat_only  = gat_only  * _s + _m
        skip_only = skip_only * _s + _m

    # Optional light smoothing to suppress high-frequency noise in the decoded traces
    from scipy.signal import butter, sosfiltfilt
    if smooth_hz is not None and smooth_hz > 0:
        sos = butter(4, smooth_hz / (fs / 2.0), btype="low", output="sos")
        full_out = sosfiltfilt(sos, full_out, axis=0)
        gat_only = sosfiltfilt(sos, gat_only, axis=0)

    time = np.arange(full_out.shape[0]) / float(fs)
    n_ch = full_out.shape[1]

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.8 * n_ch), sharex=True)
    colors = {"actual": "#1a1a2e", "full": "#e94560", "gat": "#0f3460", "skip": "#f5a623"}

    for i, ax in enumerate(axes):
        tgt = emg_target[:, i] if emg_target is not None else None
        if tgt is not None:
            ax.plot(time, tgt,         color=colors["actual"], lw=1.5, label="Ground Truth",   zorder=4)
        ax.plot(time, full_out[:, i],  color=colors["full"],   lw=1.2, ls="--", label="Full Pred (smoothed)",  zorder=3)
        ax.plot(time, gat_only[:, i],  color=colors["gat"],    lw=1.0, ls=":",  label="GAT component (smoothed)", zorder=2, alpha=0.85)
        ax.fill_between(time, 0, skip_only[:, i],
                        color=colors["skip"], alpha=0.35, label="Kin-Skip component")
        ax.set_ylabel(emg_names[i], fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=4, framealpha=0.7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Residual Decomposition: GAT Contribution vs. Kinematic Skip Highway\n"
                 "(Check if 'Kin-Skip' orange aligns with burst peaks in 'Ground Truth')",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "residual_decomposition", cfg)
    return fig


# DIAGNOSTIC PLOT 2: EMG Prediction Error Analysis (histogram + scatter + temporal)
def plot_emg_error_analysis(y_pred, y_true, emg_names=None, fs=500, cfg=None):
    """Three-panel error analysis per EMG channel:
       Left:   Error distribution histogram (residuals) â€” reveals systematic bias.
       Center: Predicted vs. Actual scatter with identity line â€” reveals canopy/compression.
       Right:  Temporal error trace â€” reveals where (in time) the model fails most.

    Args:
        y_pred:    np.ndarray (N_windows*T, 5) or (T, 5) â€” flattened predictions.
        y_true:    np.ndarray same shape as y_pred â€” ground truth.
        emg_names: List[str] of 5 muscle names.
        fs:        Sampling rate (Hz), used for x-axis label.
        cfg:       Project config dict for save_fig.

    Usage (notebook):
        from main.plots import plot_emg_error_analysis
        fig = plot_emg_error_analysis(all_preds, all_targets, emg_names=EMG_NAMES, cfg=notebook_cfg)
    """
    emg_names = emg_names or [f"Ch{i}" for i in range(y_pred.shape[-1])]
    n_ch = y_pred.shape[-1]
    residuals = y_pred - y_true
    time = np.arange(y_pred.shape[0]) / float(fs)

    fig, axes = plt.subplots(n_ch, 3, figsize=(16, 2.6 * n_ch))
    if n_ch == 1:
        axes = axes[np.newaxis, :]

    import matplotlib
    palette = matplotlib.colormaps["tab10"]

    for i in range(n_ch):
        c = palette(i)
        res = residuals[:, i]
        bias = res.mean()

        # --- Left: Error histogram ---
        ax_h = axes[i, 0]
        ax_h.hist(res, bins=60, color=c, alpha=0.75, edgecolor="none")
        ax_h.axvline(0,    color="black", lw=1.2, ls="--", label="Zero")
        ax_h.axvline(bias, color="red",   lw=1.2, ls="-",  label=f"Bias={bias:.4f}")
        ax_h.set_title(f"{emg_names[i]} â€” Error Distribution", fontsize=9)
        ax_h.set_xlabel("Residual (pred âˆ’ true)", fontsize=8)
        ax_h.legend(fontsize=7)

        # --- Center: Scatter pred vs true ---
        ax_s = axes[i, 1]
        # subsample for speed if large
        n_pts = min(5000, len(y_pred))
        idx = np.random.choice(len(y_pred), n_pts, replace=False)
        ax_s.scatter(y_true[idx, i], y_pred[idx, i], s=3, alpha=0.25, color=c)
        lims = [min(y_true[:, i].min(), y_pred[:, i].min()),
                max(y_true[:, i].max(), y_pred[:, i].max())]
        ax_s.plot(lims, lims, "k--", lw=1.2, label="Ideal")
        ax_s.set_xlabel("Actual", fontsize=8)
        ax_s.set_ylabel("Predicted", fontsize=8)
        ax_s.set_title(f"{emg_names[i]} â€” Pred vs Actual (n={n_pts})", fontsize=9)
        ax_s.legend(fontsize=7)

        # --- Right: Temporal error trace ---
        ax_t = axes[i, 2]
        ax_t.plot(time, np.abs(res), color=c, lw=0.7, alpha=0.8)
        ax_t.set_xlabel("Time (s)", fontsize=8)
        ax_t.set_ylabel("|Error|", fontsize=8)
        ax_t.set_title(f"{emg_names[i]} â€” Temporal |Error| Trace", fontsize=9)

    fig.suptitle("EMG Prediction Error Analysis\n"
                 "Left: Bias check | Center: Compression/Canopy check | Right: Temporal error hotspots",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "emg_error_analysis", cfg)
    return fig


# DIAGNOSTIC PLOT 3: Predicted vs. Actual Inter-Channel Correlation Matrix
def plot_channel_correlation_matrix(y_pred, y_true, emg_names=None, cfg=None):
    """Compares the inter-channel Pearson correlation structure between
    ground-truth EMG and predicted EMG.
    If the model has learned true muscle synergies, the two matrices should match closely.
    Large deviations reveal which co-activations the model fails to decode.

    Args:
        y_pred:    np.ndarray (T, 5) â€” predictions for a single window.
        y_true:    np.ndarray (T, 5) â€” ground truth for the same window.
        emg_names: List[str] of 5 muscle names.
        cfg:       Project config dict for save_fig.

    Usage (notebook):
        from main.plots import plot_channel_correlation_matrix
        fig = plot_channel_correlation_matrix(pred_window, true_window, emg_names=EMG_NAMES, cfg=notebook_cfg)
    """
    emg_names = emg_names or [f"Ch{i}" for i in range(y_pred.shape[-1])]

    corr_pred = np.corrcoef(y_pred.T)
    corr_true = np.corrcoef(y_true.T)
    corr_diff = corr_pred - corr_true

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    kw = dict(annot=True, fmt=".2f", cmap="RdBu_r", center=0,
              vmin=-1, vmax=1, square=True, linewidths=0.5,
              xticklabels=emg_names, yticklabels=emg_names,
              cbar_kws={"shrink": 0.8})

    sns.heatmap(corr_true, ax=axes[0], **kw)
    axes[0].set_title("Ground Truth\nInter-Muscle Correlation", fontweight="bold")

    sns.heatmap(corr_pred, ax=axes[1], **kw)
    axes[1].set_title("Predicted\nInter-Muscle Correlation", fontweight="bold")

    kw_diff = dict(annot=True, fmt=".2f", cmap="PiYG", center=0,
                   vmin=-0.5, vmax=0.5, square=True, linewidths=0.5,
                   xticklabels=emg_names, yticklabels=emg_names,
                   cbar_kws={"shrink": 0.8, "label": "Pred âˆ’ True"})
    sns.heatmap(corr_diff, ax=axes[2], **kw_diff)
    axes[2].set_title("Deviation (Pred âˆ’ True)\n|Green| = over-corr | |Purple| = under-corr",
                       fontweight="bold")

    fig.suptitle("Predicted vs. Actual Inter-Channel EMG Correlation Structure\n"
                 "Deviation matrix reveals which muscle synergies the model fails to decode",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "channel_correlation_matrix", cfg)
    return fig


# DIAGNOSTIC PLOT 4: Kinematic Skip Contribution Magnitude Over Time
def plot_kin_skip_over_time(model, eeg_window, kin_window, emg_target,
                             emg_names=None, fs=500,
                             emg_mean=None, emg_std=None,
                             smooth_hz=10.0,
                             cfg=None):
    """Plots the L2 norm of the kinematic skip vector at each timestep alongside
    the ground-truth EMG. This reveals whether kin_skip_proj actually fires at burst
    onset times, or whether it is contributing a constant smooth background offset.

    Args:
        model:       Trained KGGTModel with kin_skip_proj initialized.
        eeg_window:  Tensor (1, T, C_eeg).
        kin_window:  Tensor (1, T, K).
        emg_target:  np.ndarray (T, 5) â€” ground truth EMG in PHYSICAL scale (un-normalized).
        emg_names:   List[str] of 5 muscle names.
        fs:          Sampling frequency (Hz).
        emg_mean:    np.ndarray (5,) or Tensor â€” per-channel EMG mean used during training
                     normalization. Required to un-normalize kin_skip_decoded to physical scale.
        emg_std:     np.ndarray (5,) or Tensor â€” per-channel EMG std used during training.
        cfg:         Project config dict for save_fig.

    Usage (notebook):
        from main.plots import plot_kin_skip_over_time
        fig = plot_kin_skip_over_time(
            model=best_model, eeg_window=eeg_b, kin_window=kin_b,
            emg_target=true_np,           # already un-normalized (from triptych cell)
            emg_mean=emg_mean, emg_std=emg_std,
            smooth_hz=10.0,           # low-pass cutoff Hz for decoded output smoothing
            emg_names=EMG_NAMES, fs=fs, cfg=CONFIG,
        )
    """
    import torch
    model.eval()
    emg_names = emg_names or [f"Ch{i}" for i in range(5)]

    with torch.no_grad():
        eeg_window = eeg_window.to(next(model.parameters()).device)
        kin_window  = kin_window.to(next(model.parameters()).device)

        B, T, K = kin_window.shape
        kin_skip = model.kin_skip_proj(kin_window).view(B, T, model.out_channels, -1)  # (1, T, 5, node_dim)

        # L2 norm over node_dim â†’ (T, 5): how strongly does kin highway fire per muscle per timestep?
        kin_skip_norm = kin_skip.squeeze(0).norm(dim=-1).cpu().numpy()                  # (T, 5)

        # Final decoded skip contribution â€” in model's normalized output space
        kin_skip_decoded = model.decoder(kin_skip).squeeze(-1).squeeze(0).cpu().numpy() # (T, 5)

    # Un-normalize kin_skip_decoded to physical EMG scale so it is comparable to emg_target
    if emg_mean is not None and emg_std is not None:
        if hasattr(emg_mean, "cpu"):
            emg_mean = emg_mean.cpu().numpy()
        if hasattr(emg_std, "cpu"):
            emg_std = emg_std.cpu().numpy()
        emg_mean = np.asarray(emg_mean).reshape(1, -1)
        emg_std  = np.asarray(emg_std).reshape(1, -1)
        kin_skip_decoded = kin_skip_decoded * emg_std + emg_mean

    # Optional light smoothing on the decoded skip output
    from scipy.signal import butter, sosfiltfilt
    if smooth_hz is not None and smooth_hz > 0:
        sos = butter(4, smooth_hz / (fs / 2.0), btype="low", output="sos")
        kin_skip_decoded = sosfiltfilt(sos, kin_skip_decoded, axis=0)

    time = np.arange(T) / float(fs)
    n_ch = model.out_channels

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.5 * n_ch), sharex=True)

    for i, ax in enumerate(axes):
        ax2 = ax.twinx()

        tgt = emg_target[:, i]
        ax.plot(time, tgt, color="#1a1a2e", lw=1.5, label="Ground Truth EMG", zorder=4)
        ax.plot(time, kin_skip_decoded[:, i], color="#e94560", lw=1.0, ls="--",
                label="Decoded Skip Output (physical scale)", alpha=0.9, zorder=3)
        ax2.fill_between(time, 0, kin_skip_norm[:, i],
                         color="#f5a623", alpha=0.4, label="Skip L2 Norm")

        ax.set_ylabel(emg_names[i], fontsize=9, color="#1a1a2e")
        ax2.set_ylabel("Skip Norm", fontsize=8, color="#f5a623")
        ax2.tick_params(axis="y", labelcolor="#f5a623")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, ncol=3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Kinematic Skip Highway: Activation Magnitude vs. EMG Ground Truth\n"
        "If orange spikes align with black burst peaks â†’ highway fires at correct times\n"
        "If orange is flat/constant â†’ highway contributes baseline offset, not burst sharpness",
        fontweight="bold", fontsize=11
    )
    fig.tight_layout()
    save_fig(fig, "kin_skip_over_time", cfg)
    return fig