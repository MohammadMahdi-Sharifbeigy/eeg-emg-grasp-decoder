"""Figures that show what the features of the task-timing baseline (EEG control test A) look like.

Uses the real lift events of one test series of P1 (cache and event files are only read; no model is run) and
writes v2-report/figures/ctrl_features_bumps.png and ctrl_features_matrix.png.

Run from the repository root:  env/Scripts/python.exe v2-report/analysis/eeg_controls_features.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eeg_controls.common import EVENTS                                      # noqa: E402
from eeg_controls.phase_baseline import (BANKS, BOXES, FS_TOK, WEIGHTS, design, lift_weight,  # noqa: E402
                                         phase_features, series_arrays)
from v2.tokens import FS_IN                                                  # noqa: E402
from v2.interp import plots as P                                             # noqa: E402

FIG = ROOT / "v2-report" / "figures"
PARTICIPANT, SERIES = 1, 4
EV_LABEL = {"led_on": "LED on", "hand_start": "hand start", "first_touch": "first touch", "both_touch": "both touch",
            "lift_off": "lift-off", "replace": "replace", "release": "release", "hand_stop": "hand stop"}
BOX_LABEL = {"moving": "hand moving", "touching": "fingers on object", "holding": "object in the air"}
W_LABEL = {1: "light", 2: "medium", 4: "heavy"}
INK, GREY = "#0b0b0b", "#9a9890"

s = next(x for x in series_arrays(PARTICIPANT, "test") if x["series"] == SERIES)
lifts, n_tok = s["lifts"], len(s["emg"])
t = np.arange(n_tok) / FS_TOK
env = s["emg"]


def events_in(a, b):
    return [(ev, lift[ev] / FS_IN) for lift in lifts for ev in EVENTS
            if lift.get(ev) is not None and a <= lift[ev] / FS_IN <= b]


def mark_events(ax, a, b, labels=None, min_gap=0.3):
    """Dotted line at every event; with `labels` (one lift), its events are named. Names closer than `min_gap`
    seconds are pushed apart and joined to their line by a short leader, so events a few ms apart stay readable."""
    own = []
    for ev, te in events_in(a, b):
        ax.axvline(te, color=GREY, linewidth=0.7, linestyle=":", zorder=0)
        if labels is not None and labels.get(ev) is not None and abs(te - labels[ev] / FS_IN) < 1e-9:
            own.append((te, ev))
    xl_prev = -np.inf
    tr = ax.get_xaxis_transform()
    for te, ev in sorted(own):
        xl = max(te, xl_prev + min_gap)
        xl_prev = xl
        ax.plot([te, te, xl], [1.0, 1.03, 1.07], transform=tr, color=GREY, linewidth=0.6, clip_on=False)
        ax.text(xl, 1.08, EV_LABEL[ev], transform=tr, rotation=60, fontsize=6.5, ha="left", va="bottom",
                color="#5f5d57")


# ---------------------------------------------------------------- figure 1: how one bank of bumps is built
a, b = 34.6, 42.7
lift = next(x for x in lifts if a < x["led_on"] / FS_IN < b)
m = (t >= a) & (t <= b)
with P.style():
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.1, 1.1, 1.1, 0.8], "hspace": 0.55})

    ax = axes[0]
    for name, col, c in (("AD", 0, "#2a78d6"), ("FDI", 4, "#eb6834")):
        ax.plot(t[m], env[m, col] / env[m, col].max(), color=c, linewidth=1.1, label=f"EMG envelope {name}")
    mark_events(ax, a, b, labels=lift)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title("one lift of P1 and its eight task events (dotted: events of the lifts before and after)", loc="left", fontsize=8.5, pad=46)

    # cue: every bump of the LED-on bank
    ax = axes[1]
    events, step, width, _ = BANKS["cue"]
    lags = np.arange(-2.0, 3.0 + 1e-9, step)
    x = phase_features([lift], n_tok, events, step, width, boxes=False)
    te = lift["led_on"] / FS_IN
    cmap = P.SEQUENTIAL
    for j, lag in enumerate(lags):
        strong = lag in (-1.0, 0.0, 1.0, 2.0)
        ax.plot(t[m], x[m, j], color=cmap(0.25 + 0.7 * j / len(lags)), linewidth=1.6 if strong else 0.6,
                alpha=1.0 if strong else 0.55)
        if strong:
            ax.text(te + lag, 1.05, f"{lag:+.0f} s" if lag else "0 s", ha="center", fontsize=6.5, color=INK)
    ax.axvline(te, color=INK, linewidth=0.9)
    ax.set_ylim(0, 1.25); ax.set_yticks([])
    ax.set_title(f"cue: {len(lags)} bumps locked to LED on (every {step} s from -2 to +3 s, width {width} s)",
                 loc="left", fontsize=8.5)

    # coarse against phase for one event: wide sparse bumps against narrow dense ones
    for ax, var in ((axes[2], "coarse"), (axes[3], "phase")):
        events, step, width, _ = BANKS[var]
        lags = np.arange(-2.0, 3.0 + 1e-9, step)
        k = EVENTS.index("first_touch")
        x = phase_features([lift], n_tok, events, step, width, boxes=False)[:, k * len(lags):(k + 1) * len(lags)]
        te = lift["first_touch"] / FS_IN
        for j in range(len(lags)):
            ax.plot(t[m], x[m, j], color=cmap(0.25 + 0.7 * j / len(lags)), linewidth=0.9)
        ax.axvline(te, color=INK, linewidth=0.9)
        ax.set_ylim(0, 1.1); ax.set_yticks([])
        ax.set_title(f"{var}: the {len(lags)} bumps of first touch (every {step} s, width {width} s); "
                     f"the same for each of the 8 events", loc="left", fontsize=8.5)

    # phase boxcars
    ax = axes[4]
    xb = phase_features([lift], n_tok, EVENTS, 0.25, 0.25, boxes=True)[:, -len(BOXES):]
    for j, (name, e0, e1) in enumerate(BOXES):
        ax.fill_between(t[m], j, j + 0.8 * xb[m, j], step="mid", color="#5f5d57", alpha=0.75, linewidth=0)
        ax.text(a + 0.05, j + 0.35, BOX_LABEL[name], fontsize=7, va="center")
    ax.set_ylim(-0.1, len(BOXES)); ax.set_yticks([])
    ax.set_title("phase also gets three on/off features for the stages of the lift", loc="left", fontsize=8.5)
    ax.set_xlabel("time in series (s)")
    for ax in axes[1:]:
        mark_events(ax, a, b)
    fig.savefig(FIG / "ctrl_features_bumps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- figure 2: the whole design matrix of each variant
a, b = 29.8, 56.0
m = (t >= a) & (t <= b)
w = lift_weight(lifts, n_tok)
with P.style():
    variants = ["cue", "coarse", "phase", "phase_weight"]
    sizes = [design(lifts, n_tok, v).shape[1] for v in variants]
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [0.45, 0.6, 1.0, 1.3, 2.2], "hspace": 0.5})
    ax = axes[0]
    ax.plot(t[m], env[m, 0] / env[m, 0].max(), color="#2a78d6", linewidth=0.9)
    for lf in lifts:
        if a < lf["led_on"] / FS_IN < b:
            ax.text(lf["led_on"] / FS_IN, 1.08, f"{W_LABEL[lf['CurW']]} object", fontsize=7, color=INK)
    ax.set_ylim(0, 1.3); ax.set_yticks([])
    ax.set_title("four lifts of P1 (EMG envelope of AD) and the weight of each object", loc="left", fontsize=8.5)
    for ax, v, nf in zip(axes[1:], variants, sizes):
        x = design(lifts, n_tok, v)[m]
        ax.grid(False)
        ax.imshow(x.T, aspect="auto", origin="lower", cmap=P.SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest",
                  extent=(t[m][0], t[m][-1], 0, nf))
        events, step, _, boxes = BANKS[v]
        per = len(np.arange(-2.0, 3.0 + 1e-9, step))
        block = len(events) * per + (len(BOXES) if boxes else 0)
        ticks, labels = [], []
        copies = len(WEIGHTS) if v == "phase_weight" else 1
        for c in range(copies):
            base = c * block
            if len(events) > 1 and copies == 1:
                for k, ev in enumerate(events):
                    ticks.append(base + (k + 0.5) * per); labels.append(EV_LABEL[ev])
                    ax.axhline(base + k * per, color="white", linewidth=0.4)
                if boxes:
                    ticks.append(base + len(events) * per + 1.5); labels.append("stages")
            elif copies > 1:
                ticks.append(base + block / 2); labels.append(f"copy for {W_LABEL[WEIGHTS[c]]}")
                ax.axhline(base, color=INK, linewidth=0.8)
            else:
                ticks.append(per / 2); labels.append("LED on")
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_title(f"{v.replace('_', ' + ')}: {nf} features", loc="left", fontsize=8.5)
    axes[-1].set_xlabel("time in series (s)")
    fig.savefig(FIG / "ctrl_features_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

print("wrote", FIG / "ctrl_features_bumps.png", "and", FIG / "ctrl_features_matrix.png", "| sizes", dict(zip(variants, sizes)))
