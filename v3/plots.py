"""Figures of the v3 notebook, one or more per run step. Every function takes the step's saved table, returns the
Figure, and neither saves nor shows it. Same look as v2 and eeg_controls (v2.interp.plots.style).

Colours: the six conditions keep one colour everywhere (registry names of v3.conditions); the EEG sources of F use
the real EEG's colour for `eeg` and greys / muted colours for the controls.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v2.interp import plots as P

CONDITION_COLORS = {"causal_cue": "#c3c2b7", "K": "#eb6834", "B": "#5f5d57", "F": "#2a78d6", "E0": "#7b4fb5",
                    "Eguided": "#1baf7a"}
SOURCE_COLORS = {"eeg": "#2a78d6", "phase_swap": "#e34948", "swap": "#eda100", "bandpower": "#5f86b3",
                 "peripheral": "#9a9890", "student": "#2a78d6"}
SOURCE_LABELS = {"eeg": "real EEG (F)", "phase_swap": "phase-matched swap", "swap": "cue-aligned swap",
                 "bandpower": "band power, all channels", "peripheral": "band power, peripheral channels",
                 "student": "student tokens"}
TIMING_COLORS = {"cue": "#c3c2b7", "phase": "#5f5d57", "cue_causal": "#86b6ef", "events_causal": "#1c5cab"}
PHASES = ["pre_move", "reach", "load", "hold", "release", "other", "all"]


def _sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def _empty(text: str):
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 1.2))
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color=P.MUTED)
    return fig


# ---- step 2 ----------------------------------------------------------------------------------------------------------

def timing_baselines(table: pd.DataFrame, rate: int = 500):
    """Mean r over participants of every timing model, per horizon (left) and per muscle at horizon 0 (right)."""
    t = table[table.rate == rate]
    if t.empty:
        return _empty("no timing baseline yet")
    per = t.groupby(["variant", "horizon_ms", "subject"]).r.mean().reset_index()
    with P.style():
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for v, d in per.groupby("variant"):
            g = d.groupby("horizon_ms").r
            ax[0].errorbar(g.mean().index, g.mean().values, yerr=g.apply(_sem).values, marker="o", capsize=3,
                           color=TIMING_COLORS.get(v, P.INK), label=v)
        ax[0].set_xticks(sorted(per.horizon_ms.unique()))
        ax[0].set_xlabel("horizon (ms)"); ax[0].set_ylabel("mean r over participants and muscles")
        ax[0].set_title("timing models without EEG: how much of the envelope is the task clock")
        ax[0].legend()
        m = t[t.horizon_ms == 0].groupby(["variant", "muscle"]).r.mean().unstack("variant")
        m.plot.bar(ax=ax[1], color=[TIMING_COLORS.get(c, P.INK) for c in m.columns], width=0.8)
        ax[1].set_title("per muscle, horizon 0"); ax[1].set_ylabel("mean r"); ax[1].set_xlabel("")
        ax[1].tick_params(axis="x", rotation=0)
        fig.tight_layout()
    return fig


# ---- step 3 ----------------------------------------------------------------------------------------------------------

def b_vs_f_horizon(summary: pd.DataFrame):
    """eeg_skill = 1 - MSE(F) / MSE(B) against the horizon, one line per EEG source (all tokens, mean +- SEM)."""
    s = summary[summary.segment == "all"]
    if s.empty:
        return _empty("no B vs F summary yet")
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for src, d in s.groupby("source"):
            d = d.sort_values("horizon_ms")
            ax.errorbar(d.horizon_ms, d.eeg_skill_mean, yerr=d.eeg_skill_sem, marker="o", capsize=3,
                        color=SOURCE_COLORS.get(src, P.INK), label=SOURCE_LABELS.get(src, src),
                        linewidth=2.2 if src == "eeg" else 1.2)
            for _, r in d.iterrows():
                if src == "eeg" and r.p_eeg_skill_holm_horizons < 0.05:
                    ax.annotate("*", (r.horizon_ms, r.eeg_skill_mean), textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=12, color=SOURCE_COLORS["eeg"])
        ax.axhline(0, color=P.MUTED, linewidth=1)
        ax.set_xticks(sorted(s.horizon_ms.unique()))
        ax.set_xlabel("horizon (ms)"); ax.set_ylabel("eeg_skill = 1 - MSE(F) / MSE(B)")
        ax.set_title("what EEG history adds to past movement and past events (* Holm p < 0.05 over horizons)")
        ax.legend()
        fig.tight_layout()
    return fig


def b_vs_f_phases(summary: pd.DataFrame, source: str = "eeg"):
    """Heatmap of eeg_skill per task phase and horizon for one EEG source; * marks Holm p < 0.05 over phases."""
    s = summary[(summary.source == source) & (summary.segment.isin(PHASES))]
    if s.empty:
        return _empty(f"no rows for source {source}")
    m = s.pivot_table(index="segment", columns="horizon_ms", values="eeg_skill_mean").reindex(
        [p for p in PHASES if p in set(s.segment)])
    p = s.pivot_table(index="segment", columns="horizon_ms", values="p_eeg_skill_holm_phases").reindex(m.index)
    lim = float(np.nanmax(np.abs(m.values))) if np.isfinite(m.values).any() else 1.0
    with P.style():
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        im = ax.imshow(m.values, cmap=P.DIVERGING.reversed(), vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(m.columns)), [f"{c} ms" for c in m.columns]); ax.set_yticks(range(len(m.index)), m.index)
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v, q = m.values[i, j], p.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.3f}" + ("*" if np.isfinite(q) and q < 0.05 else ""), ha="center",
                            va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="eeg_skill")
        ax.set_title(f"eeg_skill per phase and horizon: {SOURCE_LABELS.get(source, source)}")
        fig.tight_layout()
    return fig


def b_vs_f_mse_ladder(summary: pd.DataFrame):
    """Absolute MSE of causal_cue, K, B and F per horizon (all tokens): the ladder from the task clock to EEG."""
    s = summary[(summary.segment == "all") & (summary.source == "eeg")].sort_values("horizon_ms")
    if s.empty:
        return _empty("no B vs F summary yet")
    conds = [("causal_cue", "mse_causal_cue_mean"), ("K", "mse_K_mean"), ("B", "mse_B_mean"), ("F", "mse_F_mean")]
    x = np.arange(len(s))
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for i, (c, col) in enumerate(conds):
            ax.bar(x + (i - 1.5) * 0.2, s[col], width=0.2, color=CONDITION_COLORS[c], label=c)
        ax.set_xticks(x, [f"{h} ms" for h in s.horizon_ms]); ax.set_ylabel("MSE (model domain, mean over muscles)")
        ax.set_title("absolute error of every condition: a pooled correlation alone is not enough")
        ax.legend(ncol=4)
        fig.tight_layout()
    return fig


def b_vs_f_participants(table: pd.DataFrame, source: str = "eeg"):
    """eeg_skill of every participant against the horizon (all tokens): how consistent is the group result?"""
    t = table[(table.source == source) & (table.segment == "all")]
    if t.empty:
        return _empty("no B vs F table yet")
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for subj, d in t.groupby("subject"):
            d = d.sort_values("horizon_ms")
            ax.plot(d.horizon_ms, d.eeg_skill, color=P.CONTEXT, marker="o", ms=3)
            ax.annotate(subj, (d.horizon_ms.iloc[-1], d.eeg_skill.iloc[-1]), fontsize=6, color=P.MUTED,
                        xytext=(3, 0), textcoords="offset points")
        g = t.groupby("horizon_ms").eeg_skill.mean()
        ax.plot(g.index, g.values, color=SOURCE_COLORS["eeg"], linewidth=2.4, marker="o", label="mean")
        ax.axhline(0, color=P.MUTED, linewidth=1)
        ax.set_xticks(sorted(t.horizon_ms.unique()))
        ax.set_xlabel("horizon (ms)"); ax.set_ylabel("eeg_skill"); ax.set_title("every participant, real EEG")
        ax.legend()
        fig.tight_layout()
    return fig


def onset_errors(onset: pd.DataFrame, by: str = "horizon_ms"):
    """Median absolute onset error (ms) per condition, against `by` (horizon for B vs F, scale for the students)."""
    if onset is None or onset.empty or onset.error_ms.notna().sum() == 0:
        return _empty("no onset could be measured")
    per = (onset.assign(abs_ms=onset.error_ms.abs()).groupby(["condition", by, "subject"]).abs_ms.median().reset_index())
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        for c, d in per.groupby("condition"):
            g = d.groupby(by).abs_ms
            ax.errorbar(g.mean().index, g.mean().values, yerr=g.apply(_sem).values, marker="o", capsize=3,
                        color=CONDITION_COLORS.get(c, P.INK), label=c)
        ax.set_xlabel(by.replace("_", " ")); ax.set_ylabel("median |onset error| (ms)")
        ax.set_title("when does the predicted muscle activity start, compared with the real one")
        ax.legend()
        fig.tight_layout()
    return fig


# ---- steps 5 and 6 ---------------------------------------------------------------------------------------------------

def training_curves(hist: pd.DataFrame):
    """Validation loss of E, guide weight and the guide's share of the gradient, per epoch, every run."""
    if hist is None or hist.empty:
        return _empty("no training history yet")
    with P.style():
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        for (cond, scale, seed, subj), d in hist.groupby(["condition", "scale", "seed", "subject"]):
            kw = dict(color=CONDITION_COLORS.get(cond, P.INK), alpha=0.35, linewidth=0.9)
            ax[0].plot(d.epoch, d.val_e, **kw); ax[1].plot(d.epoch, d.w, **kw); ax[2].plot(d.epoch, d.grad_share, **kw)
        for c in hist.condition.unique():
            ax[0].plot([], [], color=CONDITION_COLORS.get(c, P.INK), label=c)
        ax[0].legend()
        ax[0].set_title("validation loss of E (model selection uses only this)"); ax[1].set_title("guide weight")
        ax[2].set_title("guide's share of the gradient"); [a.set_xlabel("epoch") for a in ax]
        fig.tight_layout()
    return fig


# ---- step 7 ----------------------------------------------------------------------------------------------------------

def student_accuracy(standard: pd.DataFrame):
    """Mean r of E0 and every Eguided scale (mean over seeds), scored at 500 Hz and at 50 Hz, one dot per participant."""
    if standard is None or standard.empty:
        return _empty("no student evaluated yet")
    per = standard.groupby(["rate", "condition", "scale", "subject"]).r.mean().reset_index()
    groups = sorted(per[["condition", "scale"]].drop_duplicates().itertuples(index=False), key=lambda g: g.scale)
    with P.style():
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        for a, rate in zip(ax, (500, 50)):
            for i, g in enumerate(groups):
                d = per[(per.rate == rate) & (per.condition == g.condition) & (per.scale == g.scale)]
                a.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)), d.r, s=12,
                          color=CONDITION_COLORS[g.condition], alpha=0.5)
                a.errorbar(i, d.r.mean(), yerr=_sem(d.r), fmt="o", color=P.INK, capsize=3)
            a.set_xticks(range(len(groups)), [f"{g.condition}\nscale {g.scale}" for g in groups])
            a.set_title(f"scored at {rate} Hz" + (" (upsampled, as v2)" if rate == 500 else " (no upsampling)"))
        ax[0].set_ylabel("mean r over muscles")
        fig.tight_layout()
    return fig


def swap_scatter(swap: pd.DataFrame, phase_swap: pd.DataFrame):
    """r with the real EEG against r with another trial's EEG, one point per participant and condition/scale."""
    with P.style():
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for a, (name, df) in zip(ax, (("cue-aligned swap", swap), ("phase-matched swap", phase_swap))):
            if df is None or df.empty:
                a.axis("off"); a.text(0.5, 0.5, f"no {name} yet", ha="center", color=P.MUTED)
                continue
            per = df.groupby(["condition", "scale", "subject"])[["r_true", "r_swap"]].mean().reset_index()
            for (c, sc), d in per.groupby(["condition", "scale"]):
                a.scatter(d.r_swap, d.r_true, s=16, color=CONDITION_COLORS.get(c, P.INK), label=f"{c} {sc}", alpha=0.7)
            lim = [per[["r_true", "r_swap"]].min().min() - 0.02, per[["r_true", "r_swap"]].max().max() + 0.02]
            a.plot(lim, lim, color=P.MUTED, linewidth=1)
            a.set_xlabel("r with the EEG of another trial"); a.set_ylabel("r with the real EEG")
            a.set_title(f"{name}: above the diagonal = trial-specific EEG"); a.legend(fontsize=6)
        fig.tight_layout()
    return fig


def residual_phases(residual: pd.DataFrame, timing: str = "cue_causal"):
    """Skill of the student's tokens on the residual of one timing model, per task phase, per condition/scale."""
    r = residual[residual.timing == timing] if residual is not None and not residual.empty else pd.DataFrame()
    if r.empty:
        return _empty(f"no residual over {timing} yet")
    per = r.groupby(["condition", "scale", "segment", "subject"]).skill.mean().reset_index()
    groups = sorted(per[["condition", "scale"]].drop_duplicates().itertuples(index=False), key=lambda g: g.scale)
    x = np.arange(len(PHASES))
    w = 0.8 / max(len(groups), 1)
    with P.style():
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for i, g in enumerate(groups):
            d = per[(per.condition == g.condition) & (per.scale == g.scale)].groupby("segment").skill
            m, e = d.mean().reindex(PHASES), d.apply(_sem).reindex(PHASES)
            ax.bar(x + (i - (len(groups) - 1) / 2) * w, m, w, yerr=e, capsize=2, color=CONDITION_COLORS[g.condition],
                   alpha=0.5 + 0.5 * (i + 1) / len(groups), label=f"{g.condition} scale {g.scale}")
        ax.axhline(0, color=P.MUTED, linewidth=1)
        ax.set_xticks(x, PHASES); ax.set_ylabel("skill over the timing model")
        ax.set_title(f"what the student explains beyond {timing}, per task phase"); ax.legend(fontsize=6)
        fig.tight_layout()
    return fig


def template_skill(summary: pd.DataFrame):
    """Skill of predicting each trial's deviation from the average EMG template, per muscle and EEG source (all tokens)."""
    if summary is None or summary.empty:
        return _empty("no template test yet")
    s = summary[summary.segment == "all"]
    scales = sorted(s.scale.unique()) if "scale" in s else [None]
    with P.style():
        fig, axes = plt.subplots(1, len(scales), figsize=(4.2 * len(scales), 3.4), sharey=True, squeeze=False)
        for a, sc in zip(axes[0], scales):
            d = s if sc is None else s[s.scale == sc]
            m = d.pivot_table(index="muscle", columns="source", values="skill_mean")
            colors = [SOURCE_COLORS.get(c, P.INK) for c in m.columns]
            m = m.rename(columns=lambda c: SOURCE_LABELS.get(c, c))
            m.plot.bar(ax=a, color=colors, width=0.8, legend=a is axes[0][0])
            a.axhline(0, color=P.MUTED, linewidth=1); a.tick_params(axis="x", rotation=0)
            a.set_title("E0" if sc == 0 else f"Eguided scale {sc}" if sc is not None else "students")
            a.set_xlabel("")
        axes[0][0].set_ylabel("skill on the deviation from the template")
        fig.suptitle("does the EEG predict how each trial differs from the average trial?")
        fig.tight_layout()
    return fig


# ---- step 8 ----------------------------------------------------------------------------------------------------------

def verdict_overview(verdict: pd.DataFrame):
    """Per horizon: eeg_skill with the real EEG and with the phase-matched swap, and the flags of the verdict."""
    if verdict is None or verdict.empty:
        return _empty("no verdict yet")
    v = verdict.sort_values("horizon_ms")
    x = np.arange(len(v))
    with P.style():
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(x - 0.2, v.eeg_skill, 0.4, yerr=v.eeg_skill_sem, capsize=3, color=SOURCE_COLORS["eeg"], label="F, real EEG")
        ax.bar(x + 0.2, v.phase_swap_skill, 0.4, color=SOURCE_COLORS["phase_swap"], label="F, phase-matched swap")
        ax.axhline(0, color=P.MUTED, linewidth=1)
        top = np.nanmax(np.r_[v.eeg_skill.values, v.phase_swap_skill.values, 0.0])
        for i, r in enumerate(v.itertuples()):
            flags = "".join(s for s, ok in (("B", r.F_beats_B), ("S", r.F_survives_swap), ("P", r.F_not_peripheral)) if ok)
            ax.text(i, top * 1.1 + 1e-4, flags or "-", ha="center", fontsize=9, color=P.INK)
        ax.set_xticks(x, [f"{h} ms" for h in v.horizon_ms]); ax.set_ylabel("eeg_skill")
        ax.set_title("verdict per horizon (B: F beats B, S: survives the swap, P: not peripheral)")
        ax.legend()
        fig.tight_layout()
    return fig
