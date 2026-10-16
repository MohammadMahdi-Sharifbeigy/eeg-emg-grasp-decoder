"""Figures of the EEG controls, phase 1 (parts W weight check, R phase ridge, conditions). Every function returns
the Figure; none saves or shows it.

Same look as eeg_controls.plots: E blue, K orange, K-free timing baselines grey/purple, E calibrated by one gain per
muscle (E_cal) a lighter dashed blue. Lift weights (CurW 1, 2, 4 = light, medium, heavy) use one ramp, grey-blue
to dark blue, in every weight figure.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from v2.interp import plots as P

E_COLOR, K_COLOR, ECAL_COLOR, TARGET_COLOR = "#2a78d6", "#eb6834", "#86b6ef", "#0b0b0b"
WEIGHT_COLORS = {1: "#b4c7dc", 2: "#5f86b3", 4: "#16365c"}
WEIGHT_NAMES = {1: "light", 2: "medium", 4: "heavy"}
FEATURE_COLORS = {"none": "#b9b6ab", "eeg": E_COLOR, "events": "#5f5d57", "both": "#7b4fb5"}
FEATURE_LABELS = {"none": "intercept only", "eeg": "EEG", "events": "past events", "both": "EEG + past events"}
PHASE_CAVEAT = ("one fit per phase knows the phase (labels at the target time), so its skill includes phase knowledge: "
                "compare EEG + events with events, and each set with intercept only, not with 0")
FIT_STYLES = {"pooled": "-", "per_phase": "--"}
FIT_LABELS = {"pooled": "one fit for all phases", "per_phase": "one fit per phase"}
PHASES = ["pre_move", "reach", "load", "hold", "release", "other", "all"]
CONDITION_COLORS = {"all": "#0b0b0b", "all_frozen": "#8a877f", "central": "#1baf7a", "edge": "#eda100",
                    "eye_clean": "#7b4fb5"}
CONDITION_LABELS = {"all": "all 32, joint K", "all_frozen": "all 32, K frozen", "central": "central 7",
                    "edge": "edge 15", "eye_clean": "eye-clean"}
HORIZON_COLORS = {0: "#9a9890", 200: "#5f86b3", 500: "#16365c"}


def _sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _finite_top(values, default=1.0):
    """Largest finite |value|, or `default` when there is none (all NaN) or it is 0."""
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    top = float(v.max()) if v.size else 0.0
    return top if top > 0 else default


def _note(title, text):
    """A one-panel figure that only says why there is nothing to plot."""
    with P.style():
        fig, ax = plt.subplots(figsize=(7.5, 2.0), layout="constrained")
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, color=P.MUTED, transform=ax.transAxes)
    return P._title(fig, title)


def _has_participant(df):
    return df if "participant" in df else df.assign(participant=0)


def _wlabel(w):
    return WEIGHT_NAMES.get(int(w), str(w))


def _weight_legend(fig, weights):
    handles = [Patch(facecolor=WEIGHT_COLORS[w], label=_wlabel(w)) for w in weights if w in WEIGHT_COLORS]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, title="lifted weight",
               title_fontsize=7.5)


# =============================================================================== part W: weight check
_W1_STYLE = {"target": ("target (EMG)", TARGET_COLOR, "-", "s"), "E": ("E", E_COLOR, "-", "o"),
             "E_cal": ("E, one gain per muscle and window", ECAL_COLOR, "--", "o"), "K": ("K", K_COLOR, "-", "^")}


def weight_levels(stats: pd.DataFrame, window: str = "hold"):
    """W1: mean level per weight, one panel per muscle; each participant divided by the target's medium level."""
    d = _has_participant(stats[stats.window == window])
    muscles = list(dict.fromkeys(d.muscle))
    outs = [o for o in _W1_STYLE if o in set(d.output)]
    cols = [f"level_w{w}" for w in (1, 2, 4)]
    with P.style():
        fig, axes = plt.subplots(1, len(muscles), figsize=(max(7.5, 1.7 * len(muscles)), 3.0), sharey=True,
                                 layout="constrained", squeeze=False)
        axes = axes[0]
        for ax, m in zip(axes, muscles):
            dm = d[d.muscle == m]
            ref = dm[dm.output == "target"].groupby("participant")["level_w2"].mean()
            for o in outs:
                lv = dm[dm.output == o].groupby("participant")[cols].mean()
                norm = lv.div(ref.reindex(lv.index), axis=0).replace([np.inf, -np.inf], np.nan)
                mean = norm.mean()
                sem = norm.apply(_sem)
                lbl, color, ls, mk = _W1_STYLE[o]
                ax.errorbar(range(3), mean, yerr=sem, color=color, linestyle=ls, marker=mk, ms=4.5, capsize=2.5,
                            linewidth=1.3, label=lbl)
            ax.set_xticks(range(3), [_wlabel(w) for w in (1, 2, 4)])
            ax.set_xlim(-0.35, 2.35)
            ax.set_title(m, loc="left")
        axes[0].set_ylabel("level / target level at the medium weight")
        P._legend(fig, axes[0])
    return P._title(fig, f"Does the level of each output follow the lifted weight? ({window}, mean +- SEM)")


def weight_scatter(levels: pd.DataFrame, window: str = "hold"):
    """W2: E (and E_cal) level against the target level, one dot per lift and muscle, coloured by weight."""
    d = _has_participant(levels[levels.window == window]).copy()
    outs = [o for o in ("E", "E_cal") if o in d]
    mean_t = d.groupby(["participant", "muscle"])["target"].transform("mean")
    d["target_n"] = d.target / mean_t
    with P.style():
        fig, axes = plt.subplots(1, len(outs), figsize=(7.5, 3.5), layout="constrained", squeeze=False)
        axes = axes[0]
        hi = float(np.nanpercentile(np.concatenate([d.target_n, *[d[o] / mean_t for o in outs]]), 99.5))
        hi = max(hi, 1.0) * 1.05
        for ax, o in zip(axes, outs):
            y = d[o] / mean_t
            for w in (4, 2, 1):
                sel = d.weight == w
                ax.scatter(d.target_n[sel], y[sel], s=9, color=WEIGHT_COLORS[w], alpha=0.8, linewidths=0)
            ax.plot([0, hi], [0, hi], color=P.INK, linewidth=0.8)
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_aspect("equal")
            ax.set_xlabel("target level / mean target level")
            ax.set_title(_W1_STYLE[o][0], loc="left")
        axes[0].set_ylabel("output level / mean target level")
        _weight_legend(fig, (1, 2, 4))
    return P._title(fig, f"Lift by lift: is the level of E on the line of the target? ({window}, line: identity)")


def gain_from_levels(levels: pd.DataFrame, window: str = "hold") -> dict:
    """Per-muscle gain that `weight_check.calibrate` applied in `window`, read back from a levels frame (E_cal / E)."""
    d = levels[levels.E.abs() > 0]
    if "window" in d:
        d = d[d.window == window]
    return (d.E_cal / d.E).groupby(d.muscle).median().to_dict()


def _pick_stretch(lifts, length_s, fs):
    """Start (s) of the stretch of length_s with the most distinct weights, ties to the most lifts."""
    ok = [lf for lf in lifts if lf.get("CurW") is not None and lf.get("lift_off") is not None
          and lf.get("replace") is not None]
    best, best_key = 0.0, (-1, -1)
    for lf in ok:
        s = max(lf["lift_off"] / fs - 2.0, 0.0)
        inside = [x for x in ok if x["lift_off"] / fs >= s and x["replace"] / fs <= s + length_s]
        key = (len({int(x["CurW"]) for x in inside}), len(inside))
        if key > best_key:
            best, best_key = s, key
    return best


def weight_example(example: dict, names=None, start_s=None, length_s: float = 60.0, gain=None, fs: float = 500.0):
    """W3: target, E and E_cal of one test series, one panel per muscle, lifts shaded and labelled by weight.

    `example` is `weight_check.run_participant(..., keep_example=True)[2]`. E_cal is `example["E_cal"]` if present,
    else E times `gain` (dict by muscle name or array per muscle, see `gain_from_levels`), else not drawn.
    start_s=None picks the stretch of length_s with the most distinct weights.
    """
    names = list(names if names is not None else example["names"])
    target, E = np.asarray(example["target"], float), np.asarray(example["E"], float)
    E_cal = example.get("E_cal")
    if E_cal is None and gain is not None:
        g = np.array([gain[n] for n in names], float) if isinstance(gain, dict) else np.asarray(gain, float)
        E_cal = E * g
    lifts = example.get("lifts", [])
    if start_s is None:
        start_s = _pick_stretch(lifts, length_s, fs)
    a, b = int(start_s * fs), min(int((start_s + length_s) * fs), len(target))
    t = np.arange(a, b) / fs
    with P.style():
        fig, axes = plt.subplots(len(names), 1, figsize=(7.5, 1.15 * len(names) + 1.2), sharex=True,
                                 layout="constrained", squeeze=False)
        axes = axes[:, 0]
        for c, (ax, name) in enumerate(zip(axes, names)):
            ax.fill_between(t, target[a:b, c], color=P.GRID, linewidth=0)
            ax.plot(t, target[a:b, c], color=TARGET_COLOR, linewidth=0.7, label="target (EMG envelope)")
            ax.plot(t, E[a:b, c], color=E_COLOR, linewidth=1.1, label="E")
            if E_cal is not None:
                ax.plot(t, np.asarray(E_cal)[a:b, c], color=ECAL_COLOR, linewidth=1.1, linestyle="--",
                        label="E, one gain per muscle (hold)")
            ax.set_ylabel(name, rotation=0, ha="right", va="center")
            ax.tick_params(labelleft=False)
            ax.set_ylim(bottom=0)
            ax.set_xlim(t[0], t[-1])
            for lf in lifts:
                w, x0, x1 = lf.get("CurW"), lf.get("lift_off"), lf.get("replace")
                if w is None or x0 is None or x1 is None or int(w) not in WEIGHT_COLORS:
                    continue
                x0, x1 = x0 / fs, x1 / fs
                if x1 < t[0] or x0 > t[-1]:
                    continue
                ax.axvspan(max(x0, t[0]), min(x1, t[-1]), color=WEIGHT_COLORS[int(w)], alpha=0.22, linewidth=0)
                if c == 0:
                    ax.text((max(x0, t[0]) + min(x1, t[-1])) / 2, 1.04, _wlabel(w), transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=7, color=P.INK)
        axes[-1].set_xlabel("time in series (s); shaded: lift off to replace")
        P._legend(fig, axes[0])
    return P._title(fig, "One test series: target and E (EEG only) lift by lift, weights labelled above")


def weight_participants(stats: pd.DataFrame, window: str = "hold"):
    """W4: per participant, Spearman rho with the weight (left) and slope against the target (right)."""
    d = _has_participant(stats[stats.window == window])
    per = d.groupby(["participant", "output"])[["rho_weight", "slope"]].mean().reset_index()
    panels = [("rho_weight", "rho between level and weight", ["target", "E", "E_cal", "K"]),
              ("slope", "slope of level on target level", ["E", "E_cal", "K"])]
    with P.style():
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), layout="constrained")
        for ax, (col, lbl, outs) in zip(axes, panels):
            outs = [o for o in outs if o in set(per.output)]
            wide = per.pivot(index="participant", columns="output", values=col).reindex(columns=outs)
            for _, row in wide.iterrows():
                ax.plot(range(len(outs)), row.to_numpy(), color=P.GRID, linewidth=0.9, zorder=1)
            for i, o in enumerate(outs):
                v = wide[o].to_numpy()
                color = TARGET_COLOR if o == "target" else _W1_STYLE[o][1]
                ax.scatter(np.full(len(v), i), v, s=14, color=color, alpha=0.6, zorder=2)
                ax.errorbar(i, np.nanmean(v), yerr=_sem(v), fmt="D", ms=6, color=color, capsize=3, zorder=3)
            ax.axhline(0, color=P.INK, linewidth=0.8)
            if col == "slope":
                ax.axhline(1, color=P.MUTED, linewidth=0.8, linestyle=":")
            ax.set_xticks(range(len(outs)), [_W1_STYLE[o][0].replace(", one gain per muscle and window", "\n(1 gain)") for o in outs])
            ax.set_xlim(-0.5, len(outs) - 0.5)
            ax.set_title(lbl, loc="left")
    return P._title(fig, f"Per participant: does E follow the weight? ({window}; big marker: mean +- SEM)")


# =============================================================================== part R: ridge on phase and past events
def _phases(df, col="segment", drop_all=False):
    have = set(df[col])
    return [s for s in PHASES if s in have and not (drop_all and s == "all")] or sorted(have)


def ridge_heatmaps(summary: pd.DataFrame):
    """R1: phase x horizon mean skill, one panel per feature set (columns) and fit (rows); x/n significant in each cell.

    The per-phase row knows the phase (labels at the target time): read it against `intercept only` (none) and
    `past events`, not against 0 (PHASE_CAVEAT, printed under the panels).
    """
    feats = [f for f in FEATURE_COLORS if f in set(summary.features)]
    fits = [f for f in FIT_STYLES if f in set(summary.fit)]
    phases = _phases(summary)
    hs = sorted(summary.horizon_ms.unique())
    top = _finite_top(summary.skill_mean)
    with P.style(axes_grid=False):
        fig, axes = plt.subplots(len(fits), len(feats), figsize=(3.0 * len(feats) + 0.6, 1.3 + 0.5 * len(phases) * len(fits)),
                                 sharex=True, sharey=True, layout="constrained", squeeze=False)
        im = None
        for i, fit in enumerate(fits):
            for j, f in enumerate(feats):
                ax = axes[i, j]
                d = summary[(summary.features == f) & (summary.fit == fit)]
                mean = d.pivot_table(index="segment", columns="horizon_ms", values="skill_mean").reindex(index=phases, columns=hs)
                nsig = d.pivot_table(index="segment", columns="horizon_ms", values="n_sig").reindex(index=phases, columns=hs)
                n = d.pivot_table(index="segment", columns="horizon_ms", values="n").reindex(index=phases, columns=hs)
                im = ax.imshow(mean.to_numpy(float), cmap=P.DIVERGING, vmin=-top, vmax=top, aspect="auto")
                for r in range(len(phases)):
                    for c in range(len(hs)):
                        v = mean.iat[r, c]
                        if np.isfinite(v):
                            dark = abs(v) / top > 0.55                       # saturated cell: white text
                            ax.text(c, r, f"{v:+.2f}\n{int(nsig.iat[r, c])}/{int(n.iat[r, c])} sig.", ha="center",
                                    va="center", fontsize=7, color="white" if dark else P.INK)
                ax.set_xticks(range(len(hs)), [f"{int(h)}" for h in hs])
                ax.set_yticks(range(len(phases)), phases)
                ax.grid(False)
                ax.set_title(f"{FEATURE_LABELS[f]}, {FIT_LABELS[fit]}", loc="left", fontsize=7.5)
        fig.supxlabel("forecast horizon (ms)\n" + PHASE_CAVEAT, fontsize=8)
        fig.colorbar(im, ax=axes, label="skill = 1 - MSE(K + ridge) / MSE(K)", shrink=0.6)
    return P._title(fig, "Ridge on EEG, past events or both: skill per task phase (red: better than K alone)")


def ridge_lines(summary: pd.DataFrame, phases=("pre_move", "reach")):
    """R2: skill against the horizon in pre_move and reach; one line per feature set, solid pooled, dashed per phase,
    band = SEM over participants (column skill_sem of phase_ridge.summarise, if present). The per-phase lines include
    phase knowledge (PHASE_CAVEAT): compare them with each other, not with 0."""
    feats = [f for f in FEATURE_COLORS if f in set(summary.features)]
    fits = [f for f in FIT_STYLES if f in set(summary.fit)]
    with P.style():
        fig, axes = plt.subplots(1, len(phases), figsize=(7.5, 3.2), sharey=True, layout="constrained", squeeze=False)
        axes = axes[0]
        for ax, ph in zip(axes, phases):
            for f in feats:
                for fit in fits:
                    d = summary[(summary.segment == ph) & (summary.features == f) & (summary.fit == fit)].sort_values("horizon_ms")
                    ax.plot(d.horizon_ms, d.skill_mean, color=FEATURE_COLORS[f], linestyle=FIT_STYLES[fit], marker="o", ms=4)
                    if "skill_sem" in d:
                        sem = d.skill_sem.to_numpy(float)
                        ax.fill_between(d.horizon_ms, d.skill_mean - sem, d.skill_mean + sem, color=FEATURE_COLORS[f],
                                        alpha=0.12, linewidth=0)
            ax.axhline(0, color=P.INK, linewidth=0.8)
            ax.set_xticks(sorted(summary.horizon_ms.unique()))
            ax.set_xlabel("forecast horizon (ms)")
            ax.set_title(ph, loc="left")
        axes[0].set_ylabel("mean skill over participants")
        handles = [Line2D([], [], color=FEATURE_COLORS[f], label=FEATURE_LABELS[f]) for f in feats]
        handles += [Line2D([], [], color="black", linestyle=FIT_STYLES[f], label=FIT_LABELS[f]) for f in fits]
        caption = ("band: SEM over participants. One fit per phase knows the phase (labels at the target time), so its\n"
                   "skill includes phase knowledge: compare EEG + events with events, and each set with intercept only.")
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, title=caption,
                   title_fontsize=7)
    return P._title(fig, "Does the EEG add to the past events? Skill against the forecast horizon")


def ridge_gain(phase_ridge: pd.DataFrame, fit: str = "per_phase"):
    """R3: per participant, skill(both) - skill(events) per phase, one panel per horizon."""
    d = phase_ridge[phase_ridge.fit == fit]
    wide = d.pivot_table(index=["subject", "horizon_ms", "segment"], columns="features", values="skill").reset_index()
    wide["gain"] = wide["both"] - wide["events"]
    phases = _phases(wide, drop_all=True)
    hs = sorted(wide.horizon_ms.unique())
    with P.style():
        fig, axes = plt.subplots(1, len(hs), figsize=(max(7.5, 2.6 * len(hs)), 3.2), sharey=True,
                                 layout="constrained", squeeze=False)
        axes = axes[0]
        for ax, h in zip(axes, hs):
            for i, ph in enumerate(phases):
                v = wide[(wide.horizon_ms == h) & (wide.segment == ph)]["gain"].to_numpy()
                jit = (np.arange(len(v)) - (len(v) - 1) / 2) * 0.06
                ax.scatter(np.full(len(v), i) + jit, v, s=10, color=E_COLOR, alpha=0.5, linewidths=0)
                if np.isfinite(v).any():
                    ax.errorbar(i, np.nanmean(v), yerr=_sem(v), fmt="D", ms=5, color=P.INK, capsize=2.5)
            ax.axhline(0, color=P.INK, linewidth=0.8)
            ax.set_xticks(range(len(phases)), phases, rotation=45, ha="right")
            ax.set_title(f"{int(h)} ms", loc="left")
        axes[0].set_ylabel("skill(EEG + events) - skill(events)")
    return P._title(fig, f"What the EEG adds to the past events, per participant ({FIT_LABELS[fit]})")


# =============================================================================== conditions: electrode subsets, eye
def _condition_lines(summary, panels, title):
    conds = [c for c in CONDITION_COLORS if c in set(summary.condition)]
    if not conds:
        return _note(title, "no condition runs yet")
    hs = sorted(summary.horizon_ms.unique())
    with P.style():
        fig, axes = plt.subplots(1, len(panels), figsize=(7.5, 3.2), layout="constrained", squeeze=False)
        axes = axes[0]
        step = 14.0
        for ax, (col, lbl) in zip(axes, panels):
            for k, c in enumerate(conds):
                g = summary[summary.condition == c].groupby("horizon_ms")[col]
                x = np.array(hs, float) + (k - (len(conds) - 1) / 2) * step
                ax.errorbar(x, g.mean().reindex(hs), yerr=g.apply(_sem).reindex(hs), color=CONDITION_COLORS[c],
                            marker="o", ms=4.5, capsize=2.5, linewidth=1.2, label=CONDITION_LABELS[c])
            ax.axhline(0, color=P.INK, linewidth=0.8)
            ax.set_xticks(hs)
            ax.set_xlabel("forecast horizon (ms)")
            ax.set_title(lbl, loc="left")
        P._legend(fig, axes[0])
    return P._title(fig, title)


def conditions_overview(summary: pd.DataFrame):
    """C1: E r and F - K per condition and horizon (mean +- SEM over participants), all_frozen next to all."""
    return _condition_lines(summary, [("E_r", "E: r with the EMG"), ("F_minus_K_vaf", "F - K (VAF points)")],
                            "Fewer electrodes, or no eye channels: E and the gain of F over K (mean +- SEM)")


def conditions_phases(summary: pd.DataFrame):
    """C2: pre_move and reach residual r per condition and horizon (pooled over the phase's samples)."""
    return _condition_lines(summary, [("pre_move_r", "before the hand moves: residual r"),
                                      ("reach_r", "during the reach: residual r")],
                            "Residual r of the EEG in two phases, per electrode condition (mean +- SEM)")


def conditions_participants(summary: pd.DataFrame, reference: str = "all_frozen", column: str = "E_r"):
    """C3: E r of each condition against the reference condition, one dot per participant and horizon.

    The default reference is `all_frozen` (all 32 electrodes trained like the subsets, K frozen); `all` is then not
    drawn. With reference="all" every other condition, all_frozen included, is drawn against the joint-trained runs.
    If no `all_frozen` run has finished yet, `all` is used instead; the title always names the reference used.
    Without the reference or any other condition the figure only carries a note.
    """
    have = set(summary.condition) if "condition" in summary else set()
    if reference == "all_frozen" and reference not in have and "all" in have:
        reference = "all"
    ref_label = CONDITION_LABELS.get(reference, reference)
    title = f"E r of each condition against {ref_label} (above the line: the condition is better)"
    skip = {reference, "all"} if reference == "all_frozen" else {reference}
    conds = [c for c in CONDITION_COLORS if c in have and c not in skip]
    if not conds or reference not in have:
        return _note(title, "no condition runs yet" if not conds else f"no {ref_label} runs yet")
    ref = summary[summary.condition == reference].set_index(["subject", "horizon_ms"])[column]
    hs = sorted(summary.horizon_ms.unique())
    allv = summary[column].to_numpy(float)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = 0.05 * (hi - lo or 1.0)
    with P.style():
        fig, axes = plt.subplots(1, len(conds), figsize=(max(7.5, 3.0 * len(conds)), 3.5), sharex=True, sharey=True,
                                 layout="constrained", squeeze=False)
        axes = axes[0]
        for ax, c in zip(axes, conds):
            d = summary[summary.condition == c].set_index(["subject", "horizon_ms"])[column]
            for h in hs:
                idx = [i for i in d.index if i[1] == h and i in ref.index]
                ax.scatter(ref.loc[idx], d.loc[idx], s=16, color=HORIZON_COLORS.get(int(h), P.MUTED), alpha=0.85,
                           linewidths=0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=P.INK, linewidth=0.8)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")
            ax.set_xlabel(f"E r, {ref_label}")
            ax.set_title(CONDITION_LABELS[c], loc="left")
        axes[0].set_ylabel("E r, the condition")
        handles = [Line2D([], [], marker="o", linestyle="", color=HORIZON_COLORS.get(int(h), P.MUTED), label=f"{int(h)} ms")
                   for h in hs]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(hs), frameon=False, title="horizon",
                   title_fontsize=7.5)
    return P._title(fig, title)
