"""main/interp_plots.py — figures for main/interp.py (Phase D interpretability).

    plot_interp_window   a stretch of a series (10 s in the notebook): prediction,
                         exact path decomposition, muscle-graph attention
    plot_event_locked    all lifts aligned to a task event: prediction, decomposition,
                         and EEG evidence — or a plain note when the EEG has no effect

Colours (fixed by role): actual = near-black, prediction = blue, EEG/graph path =
purple, kinematic path = orange, force effect = red, velocity effect = grey.
Neither function saves or shows the figure.
"""

from __future__ import annotations

import textwrap
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .interp import EVENT_LABELS

__all__ = ["plot_interp_window", "plot_event_locked", "INTERP_COLORS", "EVENT_STYLES"]

INTERP_COLORS = {
    "actual": "#1a1a2e",
    "pred": "#2a78d6",
    "graph": "#7b4fb5",
    "kin": "#eb6834",
    "force": "#d62728",
    "velocity": "#8c8c8c",
}

# event key -> (colour, line style); drawn as thin vertical lines
EVENT_STYLES = {
    "led_on": ("#c99700", ":"),
    "first_touch": ("#0e9aa7", "--"),
    "lift_off": ("#2e8b57", "-"),
    "replace": ("#8b5a2b", "-"),
    "release": ("#c2185b", "-."),
}
_GATE_COLOR = "#f2c94c"

_RC = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.5,
    "axes.facecolor": "white", "figure.facecolor": "white", "axes.edgecolor": "#444444",
    "font.size": 8, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7, "legend.fontsize": 7.5,
}


def _lowpass(x: np.ndarray, fs: float, hz: float | None) -> np.ndarray:
    if hz is None or hz <= 0:
        return x
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, hz / (fs / 2.0), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 1e-20 else float("nan")


def _legend_above(ax, handles, ncol: int):
    """Legend in a band directly above the axes (never over the data)."""
    return ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
                     ncol=ncol, frameon=False, borderaxespad=0.0, handlelength=1.8,
                     columnspacing=1.2, handletextpad=0.5)


def _title_above(ax, text: str, legend_rows: int = 0) -> None:
    ax.set_title(text, loc="left", fontweight="bold", pad=6 + 13 * legend_rows)


def _wrap(text: str, width: int = 58) -> str:
    """Wrap every line of a title so it stays inside its own column."""
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def _name(names: Sequence[str], i: int) -> str:
    return names[i] if i < len(names) else f"ch{i}"


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: a stretch of a series
# ════════════════════════════════════════════════════════════════════════════

def plot_interp_window(
    pred: np.ndarray,
    target: np.ndarray,
    parts: Mapping[str, np.ndarray],
    muscle_names: Sequence[str],
    fs: float = 500.0,
    *,
    start: int = 0,
    gat_attn: np.ndarray | None = None,
    attn_baseline: np.ndarray | None = None,
    baseline_label: str | None = None,
    events: Sequence[Mapping] | None = None,
    event_keys: Sequence[str] = ("led_on", "first_touch", "lift_off", "replace", "release"),
    gate: np.ndarray | None = None,
    smooth_hz: float | None = None,
    model_window: int | None = None,
    title: str | None = None,
    figsize: tuple | None = None,
) -> plt.Figure:
    """Three panels for one stretch of a series (all arrays cover the same T samples).

    Panel 1  actual vs predicted EMG envelope per muscle.  Both get the SAME filtering
             (none by default; ``smooth_hz`` low-passes both for display); r and RMSE are
             computed on the UNFILTERED signals.  Vertical lines = task events, shading =
             hand near the object (gate > 0).
    Panel 2  exact split of the model output per muscle: EEG/graph path + kinematic
             skip path (their sum = the model output, thin blue), in normalised units.
             Dashed = graph path with the kinematic edge bias of the GAT set to 0
             (an occlusion; it shows the part of the graph path that needs no kinematics).
    Panel 3  muscle-graph attention AFTER softmax, heads averaged, as the change from
             its rest-time mean: one small heat map per receiving muscle, rows = source
             muscle.  It shows how the graph mixes the muscle nodes; it is not EEG
             attention and not an amplitude contribution.

    Args:
        pred, target: (T, C) in the same units (original EMG units in the notebook).
        parts:        ``main.interp.decompose_series`` output cut to the stretch:
                      "graph", "kin" (T, C) required; "graph_no_kin_bias", "total" optional.
        muscle_names: C names in data order (``main.dataloader.emg_channel_names``).
        start:        series sample index of row 0 (time axis = series time).
        gat_attn:     (T, N, N) [receiving, source] attention after softmax, heads averaged
                      (``parts["gat_attn"]``); None -> Panel 3 says so.
        attn_baseline: (N, N) mean attention at rest (gate == 0) over the whole series;
                      None -> rest samples of this stretch, or the stretch mean if there
                      are none.
        events:       ``main.interp.lift_events`` dicts (absolute series samples).
        gate:         (T,) hand-near-object gate in [0, 1]; None -> no shading.
        smooth_hz:    optional display low-pass for BOTH prediction and target.
        model_window: model window in samples -> faint ticks at the window borders in Panel 2.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    T, C = pred.shape
    time = (int(start) + np.arange(T)) / float(fs)
    graph = np.asarray(parts["graph"], dtype=np.float64)
    kin = np.asarray(parts["kin"], dtype=np.float64)
    graph0 = parts.get("graph_no_kin_bias")
    total = parts.get("total")
    if gate is not None:
        gate = np.asarray(gate, dtype=np.float64).reshape(-1)[:T]
    N = gat_attn.shape[-1] if gat_attn is not None else C

    r_val = [_pearson(pred[:, c], target[:, c]) for c in range(C)]
    rmse = [float(np.sqrt(np.mean((pred[:, c] - target[:, c]) ** 2))) for c in range(C)]
    pred_d, target_d = _lowpass(pred, fs, smooth_hz), _lowpass(target, fs, smooth_hz)

    # events inside the stretch: key -> list of times (s)
    ev_times: dict[str, list[float]] = {}
    for e in events or []:
        for k in event_keys:
            s = e.get(k)
            if s is not None and start <= s < start + T:
                ev_times.setdefault(k, []).append(s / float(fs))

    def _mark(ax, heat=False):
        for k, ts in ev_times.items():
            col, ls = EVENT_STYLES.get(k, ("#555555", "-"))
            for t in ts:
                ax.axvline(t, color="#333333" if heat else col, ls=ls, lw=0.7 if heat else 0.9,
                           alpha=0.55 if heat else 0.9, zorder=1.5)

    def _shade(ax):
        if gate is not None and np.any(gate > 0):
            ax.fill_between(time, 0, 1, where=gate > 0, transform=ax.get_xaxis_transform(),
                            color=_GATE_COLOR, alpha=0.22, lw=0, zorder=0)

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=figsize or (15, 2.2 + 1.0 * C + 0.95 * C + 0.8 * N),
                         layout="constrained")
        fig.get_layout_engine().set(h_pad=0.015, hspace=0.0, w_pad=0.02, wspace=0.0)
        gs = fig.add_gridspec(2 * C + N, 2, width_ratios=[1.0, 0.085],
                              height_ratios=[1.0] * C + [0.95] * C + [0.8] * N)

        # ── Panel 1 ──────────────────────────────────────────────────────────
        ax0 = None
        p1_axes = []
        for c in range(C):
            ax = fig.add_subplot(gs[c, 0], sharex=ax0)
            ax0 = ax0 or ax
            p1_axes.append(ax)
            _shade(ax)
            ax.plot(time, target_d[:, c], color=INTERP_COLORS["actual"], lw=1.3, zorder=3)
            ax.plot(time, pred_d[:, c], color=INTERP_COLORS["pred"], lw=1.1, zorder=4)
            _mark(ax)
            ax.set_ylabel(_name(muscle_names, c), rotation=0, ha="right", va="center", fontweight="bold")
            ax.tick_params(axis="x", labelbottom=(c == C - 1))
            side = fig.add_subplot(gs[c, 1])
            side.axis("off")
            side.text(0.0, 0.5, f"r = {r_val[c]:.3f}\nRMSE = {rmse[c]:.3g}",
                      ha="left", va="center", fontsize=7.5, transform=side.transAxes)
        filt = f"both low-passed at {smooth_hz:g} Hz for display" if smooth_hz else "no filtering"
        handles = [Line2D([], [], color=INTERP_COLORS["actual"], lw=1.3, label="actual"),
                   Line2D([], [], color=INTERP_COLORS["pred"], lw=1.1, label="predicted")]
        handles += [Line2D([], [], color=EVENT_STYLES[k][0], ls=EVENT_STYLES[k][1], lw=1.0,
                           label=EVENT_LABELS.get(k, k)) for k in event_keys if k in ev_times]
        if gate is not None and np.any(gate > 0):
            handles.append(Patch(facecolor=_GATE_COLOR, alpha=0.4, label="hand near object (gate > 0)"))
        _legend_above(p1_axes[0], handles, ncol=len(handles))
        _title_above(p1_axes[0], f"Panel 1 — EMG envelope: actual vs model prediction ({filt}; "
                                 "r and RMSE on the unfiltered signals)", legend_rows=1)

        # ── Panel 2 ──────────────────────────────────────────────────────────
        p2_axes = []
        for c in range(C):
            ax = fig.add_subplot(gs[C + c, 0], sharex=ax0)
            p2_axes.append(ax)
            _shade(ax)
            if model_window:
                first = int(np.ceil(start / model_window)) * model_window
                for s in range(first, start + T + 1, int(model_window)):
                    ax.axvline(s / float(fs), color="#000000", lw=0.4, alpha=0.12, zorder=0.5)
            ax.axhline(0.0, color="#666666", lw=0.6, zorder=1)
            if total is not None:
                ax.plot(time, np.asarray(total)[:, c], color=INTERP_COLORS["pred"], lw=0.8, alpha=0.6, zorder=2)
            ax.plot(time, graph[:, c], color=INTERP_COLORS["graph"], lw=1.2, zorder=4)
            ax.plot(time, kin[:, c], color=INTERP_COLORS["kin"], lw=1.2, zorder=3)
            if graph0 is not None:
                ax.plot(time, np.asarray(graph0)[:, c], color=INTERP_COLORS["graph"], lw=0.9,
                        ls=(0, (4, 2)), alpha=0.85, zorder=5)
            _mark(ax)
            ax.set_ylabel(_name(muscle_names, c), rotation=0, ha="right", va="center", fontweight="bold")
            ax.tick_params(axis="x", labelbottom=(c == C - 1))
        handles = [Line2D([], [], color=INTERP_COLORS["graph"], lw=1.2,
                          label="EEG / graph path (incl. decoder bias)"),
                   Line2D([], [], color=INTERP_COLORS["kin"], lw=1.2, label="kinematic skip path")]
        if total is not None:
            handles.append(Line2D([], [], color=INTERP_COLORS["pred"], lw=0.8, alpha=0.6,
                                  label="their sum = model output"))
        if graph0 is not None:
            handles.append(Line2D([], [], color=INTERP_COLORS["graph"], lw=0.9, ls=(0, (4, 2)),
                                  label="graph path, kinematic edge bias zeroed (occlusion)"))
        _legend_above(p2_axes[0], handles, ncol=len(handles))
        _title_above(p2_axes[0], "Panel 2 — Exact split of the model output into its two additive paths "
                                 "(normalised units, per 1 s model window)", legend_rows=1)

        # ── Panel 3 ──────────────────────────────────────────────────────────
        if gat_attn is None:
            ax = fig.add_subplot(gs[2 * C:, 0], sharex=ax0)
            ax.set_yticks([])
            ax.grid(False)
            ax.text(0.5, 0.5, "This model exposes no muscle-graph attention.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=10)
            _title_above(ax, "Panel 3 — Muscle-graph attention (not available)")
            last_ax = ax
        else:
            attn = np.asarray(gat_attn, dtype=np.float64)
            if attn_baseline is not None:
                base = np.asarray(attn_baseline, dtype=np.float64)
                label = baseline_label or "its rest mean over the series (hand-near-object gate = 0)"
            elif gate is not None and int((gate == 0).sum()) >= 50:
                base = attn[gate == 0].mean(axis=0)
                label = baseline_label or "its rest mean in this stretch (gate = 0)"
            else:
                base = attn.mean(axis=0)
                label = baseline_label or "its mean over this stretch (no rest samples available)"
            dev = attn - base
            vmax = float(np.percentile(np.abs(dev), 99.5)) if dev.size else 0.0
            vmax = vmax if vmax > 1e-9 else 1e-3
            im = None
            for i in range(N):
                ax = fig.add_subplot(gs[2 * C + i, 0], sharex=ax0)
                ax.grid(False)
                im = ax.imshow(dev[:, i, :].T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               extent=[time[0], time[-1] + 1.0 / fs, N - 0.5, -0.5],
                               interpolation="nearest", zorder=1)
                _mark(ax, heat=True)
                ax.set_yticks(range(N))
                ax.set_yticklabels([_name(muscle_names, j) for j in range(N)], fontsize=6)
                ax.set_ylabel(f"{_name(muscle_names, i)}  ←", rotation=0, ha="right", va="center",
                              fontweight="bold")
                ax.tick_params(axis="x", labelbottom=(i == N - 1))
                if i == 0:
                    _title_above(ax, "Panel 3 — Muscle-graph attention after softmax (GAT heads averaged), "
                                     f"change from {label}.\n"
                                     "Block = receiving muscle, rows = source muscle. It shows how the graph "
                                     "mixes the muscle nodes; it is not EEG attention.")
                last_ax = ax
            side = fig.add_subplot(gs[2 * C:, 1])
            side.axis("off")
            cax = side.inset_axes([0.05, 0.08, 0.16, 0.84])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("Δ attention weight", fontsize=7.5)
            cb.ax.tick_params(labelsize=6.5)
        last_ax.set_xlabel("Time in series (s)")
        ax0.set_xlim(time[0], time[-1])
        if title:
            fig.suptitle(title, fontsize=11.5, fontweight="bold")
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: all lifts aligned to a task event
# ════════════════════════════════════════════════════════════════════════════

NO_EEG_EFFECT_NOTE = ("EEG input has no measurable effect on this model's output "
                      "(effect size = {effect}); attention is not shown as an explanation")

_GROUP_LS = ["-", "--", ":", "-."]


def plot_event_locked(
    summary: Mapping,
    muscle_names: Sequence[str],
    fs: float = 500.0,
    *,
    event_label: str = "event",
    eeg_effect: float | None = None,
    effect_threshold: float = 0.02,
    eeg_names: Sequence[str] | None = None,
    by_group: bool = False,
    group_label: str = "weight",
    time_bin_ms: float | None = None,
    title: str | None = None,
    figsize: tuple | None = None,
) -> plt.Figure:
    """All lifts aligned to one task event: rows = muscles, three columns.

    (i)   actual vs predicted, mean ± SD over the lifts (``by_group=True`` and a grouped
          summary: one line per group, e.g. per weight, without SD bands).
    (ii)  mean decomposition of the output in normalised units: EEG/graph path and
          kinematic skip path (exact, ± SD), plus three occlusion effects — force,
          pose velocity, and the graph path with the kinematic edge bias zeroed.
    (iii) EEG evidence.  If ``eeg_effect`` < ``effect_threshold`` nothing is drawn but
          the note "EEG input has no measurable effect on this model's output (effect
          size = x); attention is not shown as an explanation".  Otherwise: the
          attention-rollout row of the query at the event sample (mean ± SD), the EEG
          occlusion attribution over time per muscle, and the occlusion attribution per
          electrode; the deletion-curve faithfulness score is in the column title.

    Args:
        summary:     ``main.interp.event_locked_summary`` output.
        muscle_names: C names in data order.
        eeg_effect:  ``main.interp.eeg_effect_size`` of the model (required to decide
                     what column iii shows; None counts as "not measured" -> note).
        eeg_names:   electrode names (``load_hs(...)["eeg_names"]``); default E1, E2, ….
    """
    t = np.asarray(summary["t"], dtype=np.float64) / float(fs)
    pm = np.asarray(summary["pred_mean"], dtype=np.float64)
    C = pm.shape[-1]
    n = int(summary.get("n", 0))
    units = summary.get("units") or "units of the prediction"
    groups = summary.get("groups") or {}
    use_groups = bool(by_group and groups)
    deletion = summary.get("deletion")
    show_eeg = eeg_effect is not None and float(eeg_effect) >= float(effect_threshold)
    has = lambda k: (k + "_mean") in summary                                    # noqa: E731

    def _band(ax, key, c, color, lw=1.3, ls="-", alpha=0.15, band=True, zorder=3):
        m = np.asarray(summary[key + "_mean"], dtype=np.float64)[:, c]
        ax.plot(t, m, color=color, lw=lw, ls=ls, zorder=zorder)
        if band and n > 1:
            s = np.asarray(summary[key + "_sd"], dtype=np.float64)[:, c]
            ax.fill_between(t, m - s, m + s, color=color, alpha=alpha, lw=0, zorder=zorder - 1)

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=figsize or (17, 2.6 + 1.75 * C), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.02, hspace=0.0, w_pad=0.04, wspace=0.02)
        gs = fig.add_gridspec(C, 3, width_ratios=[1.0, 1.0, 1.15])

        # ── (i) actual vs predicted ──────────────────────────────────────────
        col0 = []
        for c in range(C):
            ax = fig.add_subplot(gs[c, 0], sharex=col0[0] if col0 else None)
            col0.append(ax)
            if use_groups:
                for gi, (g, gs_) in enumerate(groups.items()):
                    ls = _GROUP_LS[gi % len(_GROUP_LS)]
                    ax.plot(t, np.asarray(gs_["target_mean"])[:, c], color=INTERP_COLORS["actual"], lw=1.2, ls=ls)
                    ax.plot(t, np.asarray(gs_["pred_mean"])[:, c], color=INTERP_COLORS["pred"], lw=1.2, ls=ls)
            else:
                _band(ax, "target", c, INTERP_COLORS["actual"])
                _band(ax, "pred", c, INTERP_COLORS["pred"], alpha=0.2, zorder=5)
            ax.axvline(0.0, color="#444444", lw=0.9, ls="--", zorder=1)
            ax.set_ylabel(_name(muscle_names, c), rotation=0, ha="right", va="center", fontweight="bold")
            ax.tick_params(axis="x", labelbottom=(c == C - 1))
        if use_groups:
            handles = [Line2D([], [], color=INTERP_COLORS["actual"], lw=1.2, label="actual"),
                       Line2D([], [], color=INTERP_COLORS["pred"], lw=1.2, label="predicted")]
            handles += [Line2D([], [], color="#555555", lw=1.2, ls=_GROUP_LS[gi % len(_GROUP_LS)],
                               label=f"{group_label} {g} (n = {gs_['n']})")
                        for gi, (g, gs_) in enumerate(groups.items())]
            stat = f"mean per {group_label}"
        else:
            handles = [Line2D([], [], color=INTERP_COLORS["actual"], lw=1.3, label="actual"),
                       Line2D([], [], color=INTERP_COLORS["pred"], lw=1.3, label="predicted")]
            stat = "mean ± SD"
        handles.append(Line2D([], [], color="#444444", lw=0.9, ls="--", label=event_label))
        _legend_above(col0[0], handles, ncol=min(len(handles), 5))
        rows = int(np.ceil(len(handles) / 5))
        _title_above(col0[0], _wrap(f"(i) Actual vs predicted, {stat} over {n} lifts\n({units})"),
                     legend_rows=rows)
        col0[-1].set_xlabel(f"Time from {event_label} (s)")
        col0[0].set_xlim(t[0], t[-1])

        # ── (ii) decomposition ───────────────────────────────────────────────
        col1 = []
        for c in range(C):
            ax = fig.add_subplot(gs[c, 1], sharex=col0[0])
            col1.append(ax)
            ax.axhline(0.0, color="#666666", lw=0.6, zorder=1)
            ax.axvline(0.0, color="#444444", lw=0.9, ls="--", zorder=1)
            if has("graph") and has("kin"):
                if has("kin_velocity_effect"):
                    _band(ax, "kin_velocity_effect", c, INTERP_COLORS["velocity"], lw=1.0, band=False, zorder=2)
                if has("kin_force_effect"):
                    _band(ax, "kin_force_effect", c, INTERP_COLORS["force"], lw=1.0, band=False, zorder=2)
                _band(ax, "kin", c, INTERP_COLORS["kin"], alpha=0.14, zorder=4)
                _band(ax, "graph", c, INTERP_COLORS["graph"], alpha=0.14, zorder=5)
                if has("graph_no_kin_bias"):
                    _band(ax, "graph_no_kin_bias", c, INTERP_COLORS["graph"], lw=0.9, ls=(0, (4, 2)),
                          band=False, zorder=6)
            ax.tick_params(axis="x", labelbottom=(c == C - 1))
        if has("graph") and has("kin"):
            handles = [
                Line2D([], [], color=INTERP_COLORS["graph"], lw=1.3, label="EEG / graph path"),
                Line2D([], [], color=INTERP_COLORS["kin"], lw=1.3, label="kinematic skip path"),
                Line2D([], [], color=INTERP_COLORS["graph"], lw=0.9, ls=(0, (4, 2)),
                       label="graph path, kin edge bias zeroed*"),
                Line2D([], [], color=INTERP_COLORS["force"], lw=1.0, label="force effect in kin path*"),
                Line2D([], [], color=INTERP_COLORS["velocity"], lw=1.0, label="pose-velocity effect in kin path*"),
            ]
            _legend_above(col1[0], handles, ncol=2)
            _title_above(col1[0], _wrap("(ii) Exact split of the output: EEG/graph path + kinematic path "
                                        "(normalised units, mean ± SD)\n* occlusion effects, not exact shares"),
                         legend_rows=3)
        else:
            col1[0].set_title("(ii) Decomposition not available for this model", loc="left", fontweight="bold")
        col1[-1].set_xlabel(f"Time from {event_label} (s)")

        # ── (iii) EEG evidence ───────────────────────────────────────────────
        eff_txt = "not measured" if eeg_effect is None else f"{float(eeg_effect):.3g}"
        score_txt = "" if not deletion else f"; deletion-curve faithfulness = {deletion['score']:.3f}"
        if not show_eeg:
            ax = fig.add_subplot(gs[:, 2])
            ax.axis("off")
            note = NO_EEG_EFFECT_NOTE.format(effect=eff_txt)
            detail = ("Effect size = mean |Δ output| / SD of the output when the EEG is time-shuffled "
                      f"within each model window; shown only if ≥ {effect_threshold:g}.")
            if deletion:
                detail += f" Deletion-curve faithfulness of the EEG occlusion map = {deletion['score']:.3f}."
            ax.text(0.5, 0.56, textwrap.fill(note + ".", 44), transform=ax.transAxes, ha="center",
                    va="center", fontsize=12, fontweight="bold", color="#222222", linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.9", fc="#f6f4fa", ec=INTERP_COLORS["graph"], lw=1.2))
            ax.text(0.5, 0.30, textwrap.fill(detail, 62), transform=ax.transAxes, ha="center",
                    va="top", fontsize=8.5, color="#444444", linespacing=1.4)
            ax.set_title("(iii) EEG evidence", loc="left", fontweight="bold")
        else:
            sub = gs[:, 2].subgridspec(3, 1, height_ratios=[1.0, 1.0, 1.25])
            # rollout row
            ax = fig.add_subplot(sub[0])
            if has("rollout"):
                m = np.asarray(summary["rollout_mean"], dtype=np.float64)
                s = np.asarray(summary["rollout_sd"], dtype=np.float64)
                ax.plot(t, m, color=INTERP_COLORS["graph"], lw=1.2, zorder=3)
                if n > 1:
                    ax.fill_between(t, m - s, m + s, color=INTERP_COLORS["graph"], alpha=0.18, lw=0, zorder=2)
                ax.axhline(1.0 / len(t), color="#888888", lw=0.8, ls=":", zorder=1)
                ax.axvline(0.0, color="#444444", lw=0.9, ls="--", zorder=1)
                ax.set_ylabel("rollout weight")
                ax.set_xlim(t[0], t[-1])
                ax.set_xlabel(f"EEG (key) time from {event_label} (s)")
                sub_title = ("Attention rollout of the EEG encoder, query = event sample (mean ± SD; "
                             "dotted = uniform)")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "No encoder attention available for this model.",
                        transform=ax.transAxes, ha="center", va="center")
                sub_title = "Attention rollout"
            ax.set_title(_wrap(f"(iii) EEG evidence — effect size = {eff_txt}{score_txt}\n{sub_title}", 66),
                         loc="left", fontweight="bold")
            # occlusion over time
            ax = fig.add_subplot(sub[1])
            if has("occ_time_abs") and summary.get("bin_centers") is not None:
                occ = np.asarray(summary["occ_time_abs_mean"], dtype=np.float64)       # (n_bins, C)
                centers = np.asarray(summary["bin_centers"], dtype=np.float64)
                width = float(np.median(np.diff(centers))) if len(centers) > 1 else float(len(t))
                bin_ms = time_bin_ms if time_bin_ms is not None else 1000.0 * width / float(fs)
                ax.grid(False)
                im = ax.imshow(occ.T, aspect="auto", cmap="Purples", vmin=0.0,
                               extent=[(centers[0] - width / 2) / fs, (centers[-1] + width / 2) / fs, C - 0.5, -0.5],
                               interpolation="nearest")
                ax.axvline(0.0, color="#444444", lw=0.9, ls="--")
                ax.set_yticks(range(C))
                ax.set_yticklabels([_name(muscle_names, c) for c in range(C)])
                ax.set_xlabel(f"Occluded EEG time from {event_label} (s)")
                cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=12)
                cb.set_label("mean |Δ output| (z)", fontsize=7.5)
                cb.ax.tick_params(labelsize=6.5)
                ax.set_title(_wrap(f"EEG occlusion: output change when a {bin_ms:.0f} ms EEG bin is replaced "
                                   "by its window mean", 66), loc="left", fontweight="bold")
            else:
                ax.axis("off")
            # electrodes
            ax = fig.add_subplot(sub[2])
            if has("occ_electrode_abs"):
                em = np.asarray(summary["occ_electrode_abs_mean"], dtype=np.float64).mean(axis=1)
                es = np.asarray(summary["occ_electrode_abs_sd"], dtype=np.float64).mean(axis=1)
                E = len(em)
                labels = summary.get("electrode_labels")
                if eeg_names is not None and len(eeg_names) == E:
                    names = list(eeg_names)
                elif labels is not None and len(labels) == E and not all(isinstance(x, (int, np.integer)) for x in labels):
                    names = [str(x) for x in labels]
                else:
                    names = [f"E{i + 1}" for i in range(E)]
                ax.bar(np.arange(E), em, yerr=es if n > 1 else None, color=INTERP_COLORS["graph"], alpha=0.85,
                       error_kw=dict(lw=0.7, ecolor="#555555", capsize=1.5))
                ax.set_xticks(np.arange(E))
                ax.set_xticklabels(names, rotation=90, fontsize=6.5)
                ax.set_xlim(-0.7, E - 0.3)
                ax.set_ylim(bottom=0.0)
                ax.grid(axis="x", visible=False)
                ax.set_ylabel("mean |Δ output| (z)")
                ax.set_title(_wrap("EEG occlusion per electrode (whole window replaced by its mean; mean over "
                                   "muscles, bars ± SD over lifts)", 66), loc="left", fontweight="bold")
            else:
                ax.axis("off")
        if title:
            fig.suptitle(title, fontsize=11.5, fontweight="bold")
    return fig
