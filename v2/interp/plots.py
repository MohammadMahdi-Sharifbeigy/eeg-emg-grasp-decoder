"""Figures for v2.interp. Every function returns the Figure; none saves or shows it.

Colour by job. Views are categorical, in a fixed order (rollout blue, attention flow orange,
gradient-weighted rollout aqua, gradient x input yellow). Magnitudes use one hue, light to dark.
Signed values use blue <-> red around a neutral grey. Text is always ink, never a series colour.

Every figure is built inside `style()`: matplotlib's own defaults plus _RC, so a theme set by the caller
(seaborn, plt.rcParams.update) does not change it. Save a figure inside `style()` as well.
"""
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

VIEW_COLORS = {"rollout": "#2a78d6", "flow": "#eb6834", "relevance": "#1baf7a", "saliency": "#eda100"}
VIEW_LABELS = {"rollout": "rollout", "flow": "attention flow", "relevance": "gradient-weighted rollout",
               "saliency": "gradient x input"}
VIEW_MARKERS = {"rollout": "o", "flow": "s", "relevance": "^", "saliency": "D"}
ACCENT, INK, MUTED, GRID, CONTEXT = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "v2_sequential", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("v2_diverging", ["#2a78d6", "#f0efec", "#e34948"])
_RC = {
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": CONTEXT, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.axisbelow": True, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.titlesize": 10, "font.family": "DejaVu Sans",
    "lines.linewidth": 1.5, "axes.linewidth": 0.8, "grid.alpha": 1.0, "figure.dpi": 100, "savefig.dpi": "figure",
    "savefig.bbox": None,
}
_MIN_WIDTH = 7.5            # inches: room for the figure title


@contextmanager
def style(**extra):
    """matplotlib's defaults + _RC (+ extra) for the length of the block, whatever the caller has set."""
    with plt.rc_context():
        matplotlib.rcdefaults()                     # keeps the backend; rc_context puts the caller's settings back
        plt.rcParams.update({**_RC, **{k.replace("_", "."): v for k, v in extra.items()}})
        yield


def _pin(axes):
    """Give the tick label size to the axes themselves: matplotlib otherwise reads it from rcParams again
    when the figure is drawn (it sets how many ticks fit), which may be outside `style()`."""
    for ax in axes:
        ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])


def _title(fig, title):
    with style():                                   # the font family is taken when the text is made
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
    return fig


def _legend(fig, ax):
    """One legend for the figure, under the panels (the title owns the top edge)."""
    handles, labels = ax.get_legend_handles_labels()
    ncol = min(len(labels), max(2, int(fig.get_figwidth() // 1.8)))
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)


def plot_time_views(curves, panel_names, title, event_label="event", flow=None, own_share=None):
    """One panel per muscle, one line per view with a +-SEM band. curves: {view: (t, mean, sem)}, mean and
    sem are (n_panels, n_t); views may use different time axes. NaN in a curve leaves a gap.

    own_share: {view: (n_panels,)} share held by the explained token itself, written into each panel when the
               curves show the share among the other tokens only.
    flow:      {"t", "mean", "sem", "uniform"} raw attention flow per time bin and the flow of a model that
               attends to everything equally; drawn in a second row on its own axis (flow is not a share).
    """
    n = len(panel_names)
    with style():
        fig = plt.figure(figsize=(max(2.7 * n, _MIN_WIDTH), 3.2 if flow is None else 5.6), layout="constrained")
        grid = fig.add_gridspec(1 if flow is None else 2, n, height_ratios=None if flow is None else [3, 2])
        axes = [fig.add_subplot(grid[0, 0])]
        axes += [fig.add_subplot(grid[0, p], sharex=axes[0], sharey=axes[0]) for p in range(1, n)]
        _pin(axes)
        for p, ax in enumerate(axes):
            for view, (t, mean, sem) in curves.items():
                ax.fill_between(t, mean[p] - sem[p], mean[p] + sem[p], color=VIEW_COLORS[view], alpha=0.18, linewidth=0)
                ax.plot(t, mean[p], color=VIEW_COLORS[view], linewidth=1.5, label=VIEW_LABELS[view])
            ax.axvline(0.0, color=MUTED, linewidth=0.8)
            if own_share:
                held = ", ".join(f"{VIEW_LABELS[v].split()[0]} {100 * s[p]:.0f}%" for v, s in own_share.items())
                ax.set_title(f"{panel_names[p]}  (own token: {held})", loc="left", fontsize=7.5, fontweight="bold")
            else:
                ax.set_title(panel_names[p], loc="left", fontweight="bold")
            ax.set_xlabel(f"time from {event_label} (s)")
            if p:
                ax.tick_params(labelleft=False)
        axes[0].set_ylabel("share among the other tokens" if own_share else "share per token")
        _legend(fig, axes[0])
        if flow is not None:
            ax = fig.add_subplot(grid[1, :])
            _pin([ax])
            t, mean, sem = flow["t"], np.asarray(flow["mean"], float), np.asarray(flow["sem"], float)
            ax.plot(t, flow["uniform"], color=CONTEXT, linewidth=1.5, label="uniform attention (carries no information)")
            ax.fill_between(t, mean - sem, mean + sem, color=VIEW_COLORS["flow"], alpha=0.18, linewidth=0)
            ax.plot(t, mean, color=VIEW_COLORS["flow"], linewidth=1.5, label=VIEW_LABELS["flow"])
            ax.axvline(0.0, color=MUTED, linewidth=0.8)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("maximum flow (0-1)")
            ax.set_xlabel(f"time from {event_label} (s)")
            ax.set_title("Attention flow per time bin (same for all muscles). Where it follows the grey line, flow is "
                         "saturated and does not localise.", loc="left", fontsize=7.5, fontweight="bold")
            ax.legend(loc="lower right", frameon=False, ncol=2)
    return _title(fig, title)


def plot_layer_heads(rows, t, title, event_label="event"):
    """One panel per layer: every head as a thin grey line, the mean of the heads in colour. rows: (L, H, n_t)."""
    rows = np.asarray(rows, dtype=float)
    with style():
        fig, axes = plt.subplots(1, rows.shape[0], figsize=(max(3.0 * rows.shape[0], _MIN_WIDTH), 3.1), sharex=True,
                                 sharey=True, squeeze=False, layout="constrained")
        _pin(axes[0])
        for i, ax in enumerate(axes[0]):
            for h in range(rows.shape[1]):
                ax.plot(t, rows[i, h], color=CONTEXT, linewidth=0.8, label="one head" if h == 0 else None)
            ax.plot(t, rows[i].mean(0), color=ACCENT, linewidth=1.5, label="mean of the heads")
            ax.axvline(0.0, color=MUTED, linewidth=0.8)
            ax.set_title(f"layer {i + 1}", loc="left", fontweight="bold")
            ax.set_xlabel(f"time from {event_label} (s)")
        axes[0, 0].set_ylabel("attention weight")
        _legend(fig, axes[0, 0])
    return _title(fig, title)


def plot_matrices(mats, title, row_names=None, col_names=None, signed=False, xlabel="source", ylabel="receiver",
                  value_label="share"):
    """A row of heat maps on one colour scale with one colour bar. mats: {label: (n_rows, n_cols)}."""
    arrs = {k: np.asarray(v, dtype=float) for k, v in mats.items()}
    top = max(float(np.abs(a).max()) for a in arrs.values()) or 1.0
    with style(axes_grid=False):
        fig, axes = plt.subplots(1, len(arrs), figsize=(max(2.9 * len(arrs) + 0.6, _MIN_WIDTH), 3.3), squeeze=False,
                                 layout="constrained")
        _pin(axes[0])
        for ax, (label, a) in zip(axes[0], arrs.items()):
            im = ax.imshow(a, cmap=DIVERGING if signed else SEQUENTIAL, vmin=-top if signed else 0.0, vmax=top, aspect="auto")
            ax.set_title(label, loc="left", fontweight="bold")
            ax.set_xlabel(xlabel)
            if col_names is not None:
                ax.set_xticks(range(len(col_names)), list(col_names), fontsize=5 if len(col_names) > 12 else 7,
                              rotation=90 if len(col_names) > 12 else 0)
            if row_names is not None:
                ax.set_yticks(range(len(row_names)), list(row_names), fontsize=5 if len(row_names) > 12 else 7)
        axes[0, 0].set_ylabel(ylabel)
        _pin([fig.colorbar(im, ax=list(axes[0]), fraction=0.04, pad=0.02, label=value_label).ax])
    return _title(fig, title)


def plot_scalp(maps, ch_names, title, signed=False, masks=None):
    """A row of scalp maps, each with its own colour bar. maps: {label: (n_electrodes,)};
    masks: {label: bool (n_electrodes,)} marks electrodes (for example corrected p < 0.05)."""
    import mne

    info = mne.create_info(list(ch_names), 500.0, "eeg")
    info.set_montage("standard_1005")
    with style(axes_grid=False):
        fig, axes = plt.subplots(1, len(maps), figsize=(max(2.0 * len(maps), _MIN_WIDTH), 2.7), squeeze=False,
                                 layout="constrained")
        _pin(axes[0])
        for ax, (label, values) in zip(axes[0], maps.items()):
            v = np.asarray(values, dtype=float)
            top = float(np.abs(v).max()) or 1.0
            mask = None if masks is None or label not in masks else np.asarray(masks[label], dtype=bool)
            im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=DIVERGING if signed else SEQUENTIAL,
                                         vlim=(-top if signed else 0.0, top), contours=0, mask=mask)
            ax.set_title(label, fontsize=8)
            _pin([fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04,
                               ticks=MaxNLocator(4)).ax])              # few ticks: the bars sit side by side
    return _title(fig, title)


def plot_deletion(curves, title):
    """Deletion curves, one panel per output. curves: {panel: {view: (fractions, top, random)}}: the change of
    the output when the top-scored inputs are removed; grey is the same number of random inputs."""
    with style():
        fig, axes = plt.subplots(1, len(curves), figsize=(max(3.6 * len(curves), _MIN_WIDTH), 3.5), squeeze=False,
                                 layout="constrained")
        _pin(axes[0])
        for ax, (panel, views) in zip(axes[0], curves.items()):
            frac = 100 * np.asarray(next(iter(views.values()))[0], dtype=float)
            rand = np.mean([np.asarray(r, dtype=float) for _, _, r in views.values()], axis=0)
            ax.plot(frac, rand, color=CONTEXT, linewidth=1.5, marker="x", markersize=6, label="random inputs")
            for view in VIEW_COLORS:
                if view in views:
                    ax.plot(100 * np.asarray(views[view][0], dtype=float), views[view][1], color=VIEW_COLORS[view],
                            linewidth=1.5, marker=VIEW_MARKERS[view], markersize=6, label=VIEW_LABELS[view])
            ax.set_title(panel, loc="left", fontweight="bold")
            ax.set_xlabel("inputs removed (%)")
        axes[0, 0].set_ylabel("change of the output (model units)")
        _legend(fig, axes[0, 0])
    return _title(fig, title)


def plot_shares(groups, title, xlabel="share of |gradient x input|"):
    """Horizontal bars, one panel per group. groups: {panel: (labels, values)}. One series, so one colour.
    Every panel has room for the longest group, so the bars are equally thin in all of them."""
    n_rows = max(len(labels) for labels, _ in groups.values())
    with style():
        fig, axes = plt.subplots(1, len(groups), figsize=(max(3.6 * len(groups), _MIN_WIDTH), 1.0 + 0.17 * n_rows),
                                 squeeze=False, layout="constrained")
        _pin(axes[0])
        for ax, (panel, (labels, values)) in zip(axes[0], groups.items()):
            y = np.arange(len(labels))
            ax.barh(y, np.asarray(values, dtype=float), height=0.6, color=ACCENT)
            ax.set_yticks(y, list(labels), fontsize=6.5)
            ax.set_ylim(n_rows - 0.5, -0.5)                              # first label on top
            ax.grid(axis="y", visible=False)
            ax.set_title(panel, loc="left", fontweight="bold")
            ax.set_xlabel(xlabel)
    return _title(fig, title)
