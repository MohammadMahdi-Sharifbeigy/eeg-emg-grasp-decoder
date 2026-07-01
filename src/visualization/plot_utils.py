import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(data, time=None, channels=None, title="Time Series Data", xlabel="Time", ylabel="Amplitude", figsize=(12, 6)):
    """
    Plots multi-channel time series data.
    
    Args:
        data (np.ndarray): 2D array of shape (n_samples, n_channels) or 1D array.
        time (np.ndarray, optional): 1D array of time points.
        channels (list, optional): List of channel names.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
        
    n_samples, n_channels = data.shape
    
    if time is None:
        time = np.arange(n_samples)
        
    for i in range(n_channels):
        label = channels[i] if channels else f"Channel {i+1}"
        plt.plot(time, data[:, i], label=label, alpha=0.8)
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if n_channels <= 10:  # Only show legend if there aren't too many channels
        plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_heatmaps(X, Y, X_labels=None, Y_labels=None, figsize=(14, 6)):
    """
    Plots heatmaps for two feature sets to visualize their structure over time.
    
    Args:
        X (np.ndarray): Feature set 1 (e.g., EEG), shape (n_samples, n_features1).
        Y (np.ndarray): Feature set 2 (e.g., EMG), shape (n_samples, n_features2).
        X_labels (list, optional): Labels for X features.
        Y_labels (list, optional): Labels for Y features.
        figsize (tuple): Figure size.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Transpose so time is on x-axis, features on y-axis
    sns.heatmap(X.T, ax=ax1, cmap="viridis", cbar=True)
    ax1.set_title("X Features (e.g., EEG) Heatmap")
    ax1.set_xlabel("Samples")
    if X_labels and len(X_labels) == X.shape[1]:
        ax1.set_yticks(np.arange(len(X_labels)) + 0.5)
        ax1.set_yticklabels(X_labels, rotation=0)
    else:
        ax1.set_ylabel("Features / Channels")
        
    sns.heatmap(Y.T, ax=ax2, cmap="magma", cbar=True)
    ax2.set_title("Y Features (e.g., EMG) Heatmap")
    ax2.set_xlabel("Samples")
    if Y_labels and len(Y_labels) == Y.shape[1]:
        ax2.set_yticks(np.arange(len(Y_labels)) + 0.5)
        ax2.set_yticklabels(Y_labels, rotation=0)
    else:
        ax2.set_ylabel("Features / Channels")
        
    plt.tight_layout()
    plt.show()

def plot_cca_correlations(correlations, title="CCA Canonical Correlations", figsize=(8, 5)):
    """
    Plots the canonical correlation coefficients from CCA.
    
    Args:
        correlations (np.ndarray): 1D array of correlation coefficients.
        title (str): Plot title.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    components = np.arange(1, len(correlations) + 1)
    
    plt.bar(components, correlations, color='skyblue', edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add values on top of bars
    for i, v in enumerate(correlations):
        y_pos = v + 0.02 if v >= 0 else v - 0.05
        plt.text(i + 1, y_pos, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
        
    plt.title(title)
    plt.xlabel("Canonical Component")
    plt.ylabel("Correlation Coefficient (r)")
    plt.ylim(min(0, np.min(correlations)) - 0.1, 1.1)
    plt.xticks(components)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cca_scatter(X_c, Y_c, component_idx=0, title=None, figsize=(7, 7)):
    """
    Creates a scatter plot of the X and Y canonical variates for a specific component.
    
    Args:
        X_c (np.ndarray): Transformed X data (canonical variates).
        Y_c (np.ndarray): Transformed Y data (canonical variates).
        component_idx (int): Index of the canonical component to plot (0-indexed).
        title (str, optional): Plot title.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    
    x_var = X_c[:, component_idx]
    y_var = Y_c[:, component_idx]
    
    correlation = np.corrcoef(x_var, y_var)[0, 1]
    
    sns.scatterplot(x=x_var, y=y_var, alpha=0.6, edgecolor='w')
    
    # Add a trend line
    m, b = np.polyfit(x_var, y_var, 1)
    plt.plot(x_var, m*x_var + b, color='red', linestyle='--', alpha=0.8, 
             label=f'Trend line (y = {m:.2f}x + {b:.2f})')
    
    if title is None:
        title = f"CCA Component {component_idx + 1} Scatter (r = {correlation:.3f})"
        
    plt.title(title)
    plt.xlabel(f"X Canonical Variate {component_idx + 1}")
    plt.ylabel(f"Y Canonical Variate {component_idx + 1}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_cca_components_timeseries(X_c, Y_c, num_components=3, time=None, figsize=(12, 8)):
    """
    Plots the time series of the first few CCA components for both X and Y.
    
    Args:
        X_c (np.ndarray): Transformed X data.
        Y_c (np.ndarray): Transformed Y data.
        num_components (int): Number of components to plot.
        time (np.ndarray, optional): Time array.
        figsize (tuple): Figure size.
    """
    num_components = min(num_components, X_c.shape[1], Y_c.shape[1])
    
    if time is None:
        time = np.arange(X_c.shape[0])
        
    fig, axes = plt.subplots(num_components, 1, figsize=figsize, sharex=True)
    if num_components == 1:
        axes = [axes]
        
    for i in range(num_components):
        ax = axes[i]
        
        # Standardize for better visualization comparison
        x_norm = (X_c[:, i] - np.mean(X_c[:, i])) / (np.std(X_c[:, i]) + 1e-8)
        y_norm = (Y_c[:, i] - np.mean(Y_c[:, i])) / (np.std(Y_c[:, i]) + 1e-8)
        
        ax.plot(time, x_norm, label=f'X Variate {i+1}', alpha=0.8)
        ax.plot(time, y_norm, label=f'Y Variate {i+1}', alpha=0.8, linestyle='--')
        
        corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        ax.set_title(f'Component {i+1} (r={corr:.3f})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    plt.xlabel("Time / Samples")
    plt.tight_layout()
    plt.show()
