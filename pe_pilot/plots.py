"""Figures of the pilot."""
import matplotlib.pyplot as plt
import numpy as np

from pe_pilot.config import CONFIG

SHOW = ["C3", "Cz", "C4", "Pz", "Fz"]


def _shade(ax, times, clusters, alpha_level, ch=None):
    for _, c in clusters[clusters["p"] < alpha_level].iterrows():
        if ch is None or ch in c["channels"]:
            ax.axvspan(times[c["t_first"]], times[c["t_last"]], color="0.85", zorder=0)


def error_figure(err_detail: dict, kind: str):
    """Group-mean matched contrast (unexpected minus expected) for a few channels, EMG and force, touch at 0 s.
    Grey bands: significant clusters (for EEG, clusters that include the plotted channel)."""
    times, names, det = err_detail["times"], err_detail["ch_names"], err_detail["contrasts"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    for ax, signal, unit in zip(axes[:3], ("slow", "alpha", "beta"), ("uV", "dB", "dB")):
        d = det[(kind, signal)]
        for ch in SHOW:
            ax.plot(times, d["mean"][:, names.index(ch)], label=ch)
        _shade(ax, times, d["clusters"], CONFIG["cluster_p"], names.index("C3"))
        ax.set_ylabel(f"{signal} ({unit})")
    axes[0].legend(ncol=5, fontsize=8)
    d = det[(kind, "emg")]
    for j, m in enumerate(err_detail["emg_names"]):
        axes[3].plot(times, d["mean"][:, j], label=m)
    _shade(axes[3], times, d["clusters"], CONFIG["cluster_p"])
    axes[3].set_ylabel("EMG envelope")
    axes[3].legend(ncol=5, fontsize=8)
    d = det[(kind, "force")]
    for j, m in enumerate(["grip", "load"]):
        axes[4].plot(times, d["mean"][:, j], label=m)
    _shade(axes[4], times, d["clusters"], CONFIG["cluster_p"])
    axes[4].set_ylabel("force (N)")
    axes[4].legend(fontsize=8)
    axes[4].set_xlabel("time from both digits touching (s)")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle(f"unexpectedly {kind} minus expected, same object weight (grey: cluster p < {CONFIG['cluster_p']})")
    fig.tight_layout()
    return fig


def expectation_figure(per):
    fig, ax = plt.subplots(figsize=(8, 4))
    decoders = list(dict.fromkeys(per["decoder"]))
    for i, d in enumerate(decoders):
        v = per.loc[per["decoder"] == d, "auc"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=18)
        ax.plot([i - 0.25, i + 0.25], [np.nanmean(v)] * 2, color="k")
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(len(decoders)), decoders, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("AUC, previous weight light vs heavy")
    ax.set_title("expectation before contact: one dot per participant")
    fig.tight_layout()
    return fig


def behaviour_figure(per):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, m in zip(axes, ("dGF_Max", "GF_Max", "Dur_LoadPhase")):
        for i, k in enumerate(("heavier", "lighter")):
            v = per.loc[(per["measure"] == m) & (per["kind"] == k), "diff"].to_numpy()
            ax.scatter(np.full(len(v), i), v, s=18)
            ax.plot([i - 0.2, i + 0.2], [np.nanmean(v)] * 2, color="k")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks([0, 1], ["heavier", "lighter"])
        ax.set_title(f"{m}: unexpected - expected", fontsize=9)
    fig.tight_layout()
    return fig
