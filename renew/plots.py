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
    
    base_dir = Path("results") / config_str
    
    # MAGIC HAPPENS HERE: If we injected 'current_fold', put the plot in that fold's folder!
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

def plot_training_history(history, figsize=(10, 4), cfg=None):
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(history, list):
        train_loss = [h.get('train_loss', 0) for h in history]
        val_loss = [h.get('val_loss', 0) for h in history]
    else:
        train_loss = history.get("train", [])
        val_loss = history.get("val", [])
        
    ax.plot(train_loss, label="Train loss", linewidth=1.8, color="blue")
    ax.plot(val_loss, label="Val loss", linewidth=1.8, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "training_history", cfg)
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

def plot_attention_maps(attn_tensor, title_prefix, sample_idx=0, cfg=None):
    attn = attn_tensor[sample_idx].detach().cpu().numpy()
    n_heads = attn.shape[0]
    
    # 1. Plot Mean Attention across heads
    mean_attn = attn.mean(axis=0)
    fig_mean = plt.figure(figsize=(6, 5))
    sns.heatmap(mean_attn, cmap="viridis")
    plt.title(f"{title_prefix} - Mean Across Heads")
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
    plt.suptitle(f"{title_prefix} - Per Head", fontsize=14, y=1.02)
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