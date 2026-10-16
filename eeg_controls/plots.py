"""Figures of the EEG control experiments. Every function returns the Figure; none saves or shows it.

Same look as v2.interp.plots (its style() and colours): E is blue, K orange, the task-phase models grey
from light (least information) to dark, signed values blue <-> red.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v2.electrodes import EEG_NAMES
from v2.interp import plots as P

E_COLOR, K_COLOR, F_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
PHASE_COLORS = {"cue": "#c3c2b7", "coarse": "#9a9890", "phase": "#5f5d57", "phase_weight": "#2f2e2b"}
MODEL_ORDER = ["cue", "coarse", "phase", "phase_weight", "E", "K"]
MODEL_LABELS = {"cue": "LED cue\nonly", "coarse": "coarse\ntiming", "phase": "task\ntiming",
                "phase_weight": "timing\n+ weight", "E": "E\n(EEG only)", "K": "K\n(kinematics)"}
EXAMPLE_COLORS = {"cue": "#5f5d57", "coarse": "#7b4fb5", "phase": "#0b0b0b", "phase_weight": "#7b4fb5"}
SEG_ORDER = ["pre_move", "reach", "load", "hold", "release", "other", "all"]


def _color(model):
    return {"E": E_COLOR, "K": K_COLOR, **PHASE_COLORS}.get(model, P.MUTED)


def _sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# =============================================================================== experiment 2: phase baseline
def phase_summary(cmp: pd.DataFrame, metric: str = "r"):
    """Mean over muscles per participant (grey lines) and over participants (dots), for every model."""
    per = cmp.groupby("subject").mean(numeric_only=True)
    models = [m for m in MODEL_ORDER if f"{m}_{metric}" in per]
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
        x = np.arange(len(models))
        vals = np.column_stack([per[f"{m}_{metric}"] for m in models])
        for row in vals:
            ax.plot(x, row, color=P.GRID, linewidth=0.9, zorder=1)
        for i, m in enumerate(models):
            v = vals[:, i]
            ax.scatter(np.full(len(v), x[i]), v, s=12, color=_color(m), alpha=0.55, zorder=2)
            ax.errorbar(x[i], np.nanmean(v), yerr=_sem(v), fmt="o", ms=8, color=_color(m), capsize=3, zorder=3)
            ax.annotate(f"{np.nanmean(v):.2f}", (x[i], np.nanmean(v)), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=P.INK)
        ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
        ax.set_ylabel(f"mean {metric} over muscles" + (" (%)" if metric == "vaf" else ""))
        ax.set_title(f"Does E hold more than the timing of the task? ({metric}, {len(per)} participants)",
                     loc="left", fontweight="bold", pad=14)
        ax.text(0, 1.01, "big dot: mean +- SEM over participants; small dots and grey lines: participants",
                transform=ax.transAxes, va="bottom", fontsize=6.5, color=P.MUTED)
    return fig


def phase_muscles(cmp: pd.DataFrame, metric: str = "r"):
    """Mean over participants per muscle, one bar per model."""
    models = [m for m in MODEL_ORDER if f"{m}_{metric}" in cmp]
    muscles = list(dict.fromkeys(cmp["muscle"]))
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            g = cmp.groupby("muscle")[f"{m}_{metric}"]
            mean, sem = g.mean().reindex(muscles), g.apply(_sem).reindex(muscles)
            ax.bar(np.arange(len(muscles)) + (i - (len(models) - 1) / 2) * w, mean, w, yerr=sem, capsize=2,
                   color=_color(m), label=MODEL_LABELS[m].replace("\n", " "), error_kw={"linewidth": 0.8})
        ax.set_xticks(np.arange(len(muscles)), muscles)
        ax.set_ylabel(f"mean {metric} over participants")
        ax.set_title("Per muscle: task-phase models against E and K", loc="left", fontweight="bold")
        ax.legend(ncol=len(models), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7)
    return fig


def phase_example(preds: dict, target: np.ndarray, names, fs: float = 500.0, start_s: float = 30.0,
                  length_s: float = 20.0, title: str = ""):
    """Target and the predictions of each phase model on one stretch of one test series (one panel per muscle).

    preds: {variant: (T, 5) prediction of the same series at 500 Hz}.
    """
    a, b = int(start_s * fs), int((start_s + length_s) * fs)
    t = np.arange(a, min(b, len(target))) / fs
    with P.style():
        fig, axes = plt.subplots(len(names), 1, figsize=(7.5, 1.25 * len(names) + 0.6), sharex=True,
                                 layout="constrained")
        for c, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
            ax.fill_between(t, target[a:a + len(t), c], color=P.GRID, linewidth=0)
            ax.plot(t, target[a:a + len(t), c], color=P.CONTEXT, linewidth=0.8, label="EMG envelope")
            for v, y in preds.items():
                ax.plot(t, y[a:a + len(t), c], color=EXAMPLE_COLORS.get(v, P.INK), linewidth=1.2,
                        linestyle="--" if v == "cue" else "-", label=MODEL_LABELS.get(v, v).replace("\n", " "))
            ax.set_ylabel(name, rotation=0, ha="right", va="center")
            ax.tick_params(labelleft=False)
        np.atleast_1d(axes)[-1].set_xlabel("time in series (s)")
        P._legend(fig, np.atleast_1d(axes)[0])
    return P._title(fig, title or "Phase models on a test series")


# =============================================================================== experiment 4: reference maps
def _group_map(ref, kind, band=None, window=None):
    d = ref[ref.kind == kind]
    if band is not None:
        d = d[d.band == band]
    if window is not None:
        d = d[d.window == window]
    return d.groupby("electrode")["value"].mean().reindex(EEG_NAMES).to_numpy()


def erd_maps(ref: pd.DataFrame, windows=("first_touch", "lift_off", "hold")):
    maps = {f"{band} ERD %\n{win}": _group_map(ref, "erd", band, win)
            for band in ("alpha", "beta") for win in windows}
    return P.plot_scalp(maps, EEG_NAMES, "Group ERD (blue: power falls = desynchronisation; red: power rises)",
                        signed=True)


def cmc_maps(ref: pd.DataFrame):
    c = ref[ref.kind == "cmc"]
    maps = {f"CMC {m}": d.groupby("electrode")["value"].mean().reindex(EEG_NAMES).to_numpy()
            for m, d in c.groupby("muscle", sort=False)}
    sig = c["cmc_sig_level"].mean() if "cmc_sig_level" in c else np.nan
    return P.plot_scalp(maps, EEG_NAMES, f"Group CMC, 15-30 Hz, hold (one-participant 95% level about {sig:.3f})",
                        tight=tuple(maps))


def rollout_vs_reference(cmp: pd.DataFrame, group: pd.DataFrame | None = None, event: str = "lift_off"):
    """Spearman rho (rollout map of E vs reference map) per participant (dots) and mean +- 95% CI."""
    d = cmp[(cmp.event == event) & (cmp.muscle == "mean")]
    refs = sorted(d.reference.unique(), key=lambda r: (not r.startswith("-ERD"), r))
    views = ["rollout", "relevance"]
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 3.4), layout="constrained")
        y = np.arange(len(refs))
        for j, v in enumerate(views):
            off = (j - 0.5) * 0.3
            for i, r in enumerate(refs):
                x = d[(d.view == v) & (d.reference == r)]["rho"].to_numpy()
                if not len(x):
                    continue
                ax.scatter(x, np.full(len(x), y[i] + off), s=10, color=P.VIEW_COLORS[v], alpha=0.45)
                m, ci = x.mean(), 1.96 * _sem(x)
                ax.errorbar(m, y[i] + off, xerr=ci, fmt="o", ms=6, color=P.VIEW_COLORS[v], capsize=3,
                            label=P.VIEW_LABELS[v] if i == 0 else None)
                if group is not None:
                    g = group[(group.level == "participants") & (group.view == v) & (group.reference == r)
                              & (group.event == event) & (group.muscle == "mean")]
                    if len(g):
                        ax.annotate(f"p={g.p.iloc[0]:.3f}", (1.0, y[i] + off), xycoords=("axes fraction", "data"),
                                    ha="right", va="center", fontsize=6.5, color=P.MUTED)
        ax.axvline(0, color=P.INK, linewidth=0.8)
        ax.set_yticks(y, refs)
        ax.invert_yaxis()
        ax.set_xlabel("Spearman rho over the 32 electrodes (rollout share of E vs reference map)")
        ax.set_title(f"Do the rollout maps of E agree with classical maps? ({event}; dots: participants)",
                     loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower left", fontsize=7)
    return fig


def rollout_reference_scatter(ref: pd.DataFrame, rollout_group: np.ndarray, band: str = "beta",
                              window: str = "lift_off"):
    """Group rollout share of E per electrode against group -ERD and group CMC (mean over muscles)."""
    from scipy.stats import spearmanr
    panels = {f"-ERD % {band}, {window}": -_group_map(ref, "erd", band, window),
              "CMC 15-30 Hz, hold (mean over muscles)": _group_map(ref, "cmc")}
    with P.style():
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), layout="constrained")
        for ax, (lbl, v) in zip(axes, panels.items()):
            ax.scatter(v, rollout_group, s=14, color=E_COLOR)
            for name, a, b in zip(EEG_NAMES, v, rollout_group):
                ax.annotate(name, (a, b), xytext=(2, 2), textcoords="offset points", fontsize=5.5, color=P.MUTED)
            rho = spearmanr(v, rollout_group)[0]
            ax.set_xlabel(lbl)
            ax.set_title(f"rho = {rho:+.2f}", loc="left")
        axes[0].set_ylabel("group rollout share of E")
    return P._title(fig, "Electrode by electrode: rollout of E against the reference maps (group means)")


# =============================================================================== experiment 1: forecast horizon
def horizon_gain(summary: pd.DataFrame):
    """EEG gain against the horizon: F - K, residual r, and the pre-movement skill of the ridge."""
    panels = [("F_minus_K_vaf", "F - K (VAF points)"), ("res_r_tokens", "residual r (EEG tokens)"),
              ("pre_move_skill_ridge", "skill before the hand moves\n1 - MSE(K + EEG) / MSE(K)")]
    hs = sorted(summary.horizon_ms.unique())
    with P.style():
        fig, axes = plt.subplots(1, len(panels), figsize=(7.5, 3.0), layout="constrained")
        for ax, (col, lbl) in zip(axes, panels):
            for _, d in summary.groupby("subject"):
                d = d.sort_values("horizon_ms")
                ax.plot(d.horizon_ms, d[col], color=P.GRID, linewidth=0.9)
            g = summary.groupby("horizon_ms")[col]
            ax.errorbar(hs, g.mean().reindex(hs), yerr=g.apply(_sem).reindex(hs), color=E_COLOR, marker="o",
                        capsize=3)
            ax.axhline(0, color=P.INK, linewidth=0.8)
            ax.set_xlabel("horizon (ms)")
            ax.set_title(lbl, loc="left", fontsize=8)
    return P._title(fig, "EEG gain against the forecast horizon (mean +- SEM; grey: participants)")


def horizon_accuracy(summary: pd.DataFrame):
    """How forecasting costs accuracy: K VAF and E r against the horizon."""
    hs = sorted(summary.horizon_ms.unique())
    with P.style():
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), layout="constrained")
        for ax, (col, lbl, colr) in zip(axes, [("K_vaf", "K: VAF (%)", K_COLOR), ("E_r", "E: r", E_COLOR)]):
            for _, d in summary.groupby("subject"):
                d = d.sort_values("horizon_ms")
                ax.plot(d.horizon_ms, d[col], color=P.GRID, linewidth=0.9)
            g = summary.groupby("horizon_ms")[col]
            ax.errorbar(hs, g.mean().reindex(hs), yerr=g.apply(_sem).reindex(hs), color=colr, marker="o", capsize=3)
            ax.set_xlabel("horizon (ms)")
            ax.set_title(lbl, loc="left")
    return P._title(fig, "Accuracy of the causal model against the forecast horizon")


def horizon_segments(results_dir, value: str = "skill_ridge"):
    """Heat map, task phase x horizon: mean EEG skill over participants and how many are significant."""
    frames = [pd.read_csv(p) for p in Path(results_dir).glob("h*ms/P*/segments.csv")]
    if not frames:
        raise FileNotFoundError(f"no segments.csv under {results_dir}")
    seg = pd.concat(frames, ignore_index=True)
    mean = seg.pivot_table(index="segment", columns="horizon_ms", values=value, aggfunc="mean")
    mean = mean.reindex([s for s in SEG_ORDER if s in mean.index])
    nsig = None
    if value == "skill_ridge":
        seg["sig"] = (seg.p_skill_ridge <= 0.05) & (seg.skill_ridge > 0)
        nsig = seg.pivot_table(index="segment", columns="horizon_ms", values="sig", aggfunc="sum").reindex(mean.index)
    n = seg.groupby("horizon_ms")["subject"].nunique()
    top = float(np.nanmax(np.abs(mean.to_numpy()))) or 1.0
    with P.style(axes_grid=False):
        fig, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
        im = ax.imshow(mean.to_numpy(), cmap=P.DIVERGING, vmin=-top, vmax=top, aspect="auto")
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                txt = f"{mean.iat[i, j]:+.3f}"
                if nsig is not None:
                    txt += f"\n{int(nsig.iat[i, j])}/{int(n.iloc[j])} sig."
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=P.INK)
        ax.set_xticks(range(mean.shape[1]), [f"{int(h)} ms" for h in mean.columns])
        ax.set_yticks(range(mean.shape[0]), mean.index)
        ax.set_xlabel("forecast horizon")
        fig.colorbar(im, ax=ax, label="1 - MSE(K + EEG) / MSE(K)", shrink=0.8)
        ax.set_title(f"EEG gain per task phase ({value}; red = the EEG helps)", loc="left", fontweight="bold")
    return fig
