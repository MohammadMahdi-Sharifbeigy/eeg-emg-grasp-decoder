"""Triptych for a stretch of one test series (the v2 counterpart of main_1stage's interpretability figure).

Panel 1  actual EMG envelope against the three outputs K, E and F.
Panel 2  the exact parts of the outputs: K_base, K = K_base x gain, and the EEG correction F_eeg inside F.
Panel 3  muscle-graph attention of one stream as the change from its rest mean.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..attention import recording
from ..data import from_model_domain, normalise
from ..infer import window_plan
from ..tokens import FS_TOK, Q
from . import plots as P
from .rollout import layer_matrix, rollout

PARTS = ("K", "E", "F", "K_base", "gain", "F_eeg")
COLORS = {"actual": "#1a1a2e", "K": "#eb6834", "E": "#2a78d6", "F": "#1baf7a", "F_eeg": "#7b4fb5", "K_base": "#898781"}
EVENT_STYLES = {"led_on": ("#c99700", ":"), "first_touch": ("#0e9aa7", "--"), "lift_off": ("#2e8b57", "-"),
                "replace": ("#8b5a2b", "-"), "release": ("#c2185b", "-.")}
_HOLD = "#f2c94c"


@torch.no_grad()
def series_internals(model, tok, norm, device, window=400, hop=200, clip=5.0, residual=0.5) -> dict:
    """Stitched token-rate outputs, exact parts and muscle-graph attention of one whole series.

    Returns K, E, F, K_base, gain, F_eeg as (Tk, 5) in the model domain, and muscle_kin, muscle_eeg as
    (Tk, 5, 5): rollout of the stream's muscle-graph layers, row = receiving muscle, column = source muscle.
    Stitching is the one of v2.infer (centre of each window kept).
    """
    kin_z, eeg_z = normalise(tok, norm, clip)
    n = kin_z.shape[0]
    layers = model.attention_layers()
    out = {k: np.zeros((n, 5), dtype=np.float32) for k in PARTS}
    out.update({k: np.zeros((n, 5, 5), dtype=np.float32) for k in ("muscle_kin", "muscle_eeg")})
    model.eval()
    for start, lo, hi in window_plan(n, window, hop):
        sl = slice(start, min(start + window, n))
        x = [torch.from_numpy(np.ascontiguousarray(a[sl]))[None].to(device) for a in (kin_z, tok["hist"], eeg_z)]
        with recording(model) as tape:
            res = model(*x)
        keep = slice(lo - start, hi - start)
        for k in PARTS:
            out[k][lo:hi] = res[k][0, keep].float().cpu().numpy()
        for key, stream in (("muscle_kin", "kin"), ("muscle_eeg", "eeg")):
            graph = rollout([layer_matrix(tape.attn[name], residual) for name in layers[stream]["muscle"]])
            out[key][lo:hi] = graph[0, keep].cpu().numpy()
    return out


def tile_starts(tok, length_s=10.0) -> list:
    """Token index of every complete, non-overlapping stretch of length_s over the whole series."""
    length = int(round(length_s * FS_TOK))
    return list(range(0, len(tok["hold_id"]) - length + 1, length))


def stretch_starts(tok, n=2, length_s=10.0, lead_s=3.0) -> list:
    """Token index where n stretches start: lead_s before a detected hold, holds spread over the series."""
    hold, length = np.asarray(tok["hold_id"]), int(round(length_s * FS_TOK))
    ids = [k for k in np.unique(hold) if k > 0]
    if not ids or len(hold) <= length:
        return [0]
    picked = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, min(n, len(ids))).round()]
    starts = [int(np.flatnonzero(hold == k)[0] - lead_s * FS_TOK) for k in picked]
    return sorted({int(np.clip(s, 0, len(hold) - length)) for s in starts})


def _r(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 1e-20 else float("nan")


def plot_triptych(internals, tok, norm, names, start, length_s=10.0, events=None, stream="kin", title=None):
    """Triptych of one stretch. `start` is a token index; `events` are main.interp.lift_events dicts
    (sample indices at 500 Hz) or None; `stream` picks whose muscle graph Panel 3 shows ("kin" or "eeg")."""
    n = int(round(length_s * FS_TOK))
    sl = slice(int(start), min(int(start) + n, internals["K"].shape[0]))
    t = np.arange(sl.start, sl.stop) / FS_TOK
    scale = norm["emg_scale"]
    target = np.asarray(tok["emg"][sl], dtype=np.float64)
    pred = {k: np.asarray(from_model_domain(internals[k][sl], scale), dtype=np.float64) for k in ("K", "E", "F")}
    hold = np.asarray(tok["hold_id"][sl]) > 0
    C = target.shape[1]

    marks = {}
    for e in events or []:
        for k in EVENT_STYLES:
            s = e.get(k)
            if s is not None and sl.start <= s / Q < sl.stop:
                marks.setdefault(k, []).append(s / Q / FS_TOK)

    def decorate(ax, heat=False):
        if hold.any() and not heat:
            ax.fill_between(t, 0, 1, where=hold, transform=ax.get_xaxis_transform(), color=_HOLD, alpha=0.2, linewidth=0, zorder=0)
        for k, ts in marks.items():
            col, ls = EVENT_STYLES[k]
            for x in ts:
                ax.axvline(x, color="#333333" if heat else col, linestyle=ls, linewidth=0.8, alpha=0.6 if heat else 0.9, zorder=1.5)

    rest = np.asarray(tok["hold_id"]) == 0
    graph_all = internals[f"muscle_{stream}"]
    base = graph_all[rest].mean(0) if rest.sum() >= 50 else graph_all.mean(0)
    dev = graph_all[sl] - base
    vmax = max(float(np.percentile(np.abs(dev), 99.5)), 1e-4)

    with P.style(axes_grid=True):
        fig = plt.figure(figsize=(15, 2.4 + 2.75 * C), layout="constrained")
        grid = fig.add_gridspec(3 * C, 2, width_ratios=[1.0, 0.11], height_ratios=[1.0] * C + [0.95] * C + [0.8] * C)
        first = None
        for c in range(C):                                              # Panel 1: actual against K, E, F
            ax = fig.add_subplot(grid[c, 0], sharex=first)
            first = first or ax
            decorate(ax)
            ax.plot(t, target[:, c], color=COLORS["actual"], linewidth=1.4, zorder=3)
            ax.plot(t, pred["E"][:, c], color=COLORS["E"], linewidth=1.0, zorder=4)
            ax.plot(t, pred["K"][:, c], color=COLORS["K"], linewidth=1.3, zorder=5)
            ax.plot(t, pred["F"][:, c], color=COLORS["F"], linewidth=0.9, linestyle=(0, (4, 2)), zorder=6)
            ax.set_ylabel(names[c], rotation=0, ha="right", va="center", fontweight="bold")
            ax.tick_params(labelbottom=False)
            side = fig.add_subplot(grid[c, 1])
            side.axis("off")
            side.text(0.0, 0.5, "\n".join(f"{k}: r = {_r(pred[k][:, c], target[:, c]):.3f}" for k in ("K", "E", "F")),
                      ha="left", va="center", fontsize=7, color=P.INK, transform=side.transAxes)
            if c == 0:
                handles = [Line2D([], [], color=COLORS["actual"], linewidth=1.4, label="actual"),
                           Line2D([], [], color=COLORS["K"], linewidth=1.3, label="K (kinematics-led)"),
                           Line2D([], [], color=COLORS["E"], linewidth=1.0, label="E (EEG only)"),
                           Line2D([], [], color=COLORS["F"], linewidth=0.9, linestyle=(0, (4, 2)), label="F (fused)")]
                handles += [Line2D([], [], color=EVENT_STYLES[k][0], linestyle=EVENT_STYLES[k][1], linewidth=1.0,
                                   label=k.replace("_", " ")) for k in marks]
                if hold.any():
                    handles.append(Patch(facecolor=_HOLD, alpha=0.4, label="detected hold"))
                ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=len(handles), frameon=False)
                ax.set_title("Panel 1: EMG envelope, actual against the three outputs (original units, 50 Hz tokens; r on this stretch)",
                             loc="left", fontweight="bold", pad=20)
        for c in range(C):                                              # Panel 2: exact parts, model units
            ax = fig.add_subplot(grid[C + c, 0], sharex=first)
            decorate(ax)
            ax.axhline(0.0, color=P.MUTED, linewidth=0.6, zorder=1)
            ax.plot(t, internals["K_base"][sl, c], color=COLORS["K_base"], linewidth=1.1, zorder=3)
            ax.plot(t, internals["K"][sl, c], color=COLORS["K"], linewidth=1.3, zorder=4)
            ax.plot(t, internals["F_eeg"][sl, c], color=COLORS["F_eeg"], linewidth=1.2, zorder=5)
            ax.set_ylabel(names[c], rotation=0, ha="right", va="center", fontweight="bold")
            ax.tick_params(labelbottom=False)
            gain, corr = internals["gain"][sl, c], internals["F_eeg"][sl, c]
            share = np.abs(corr).mean() / max(float(np.abs(internals["K"][sl, c]).mean()), 1e-12)
            side = fig.add_subplot(grid[C + c, 1])
            side.axis("off")
            side.text(0.0, 0.5, f"gain {gain.min():.2f}-{gain.max():.2f}\n|F_eeg| / |K| = {100 * share:.1f}%",
                      ha="left", va="center", fontsize=7, color=P.INK, transform=side.transAxes)
            if c == 0:
                handles = [Line2D([], [], color=COLORS["K_base"], linewidth=1.1, label="K_base (before the history gain)"),
                           Line2D([], [], color=COLORS["K"], linewidth=1.3, label="K = K_base x gain"),
                           Line2D([], [], color=COLORS["F_eeg"], linewidth=1.2, label="F_eeg = gate x EEG correction (F = relu(K + F_eeg))")]
                ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False)
                ax.set_title("Panel 2: exact parts of the outputs (model units: square root of the scaled envelope)",
                             loc="left", fontweight="bold", pad=20)
        im = None
        for i in range(C):                                              # Panel 3: muscle graph, change from rest
            ax = fig.add_subplot(grid[2 * C + i, 0], sharex=first)
            ax.grid(False)
            im = ax.imshow(dev[:, i, :].T, aspect="auto", cmap=P.DIVERGING, vmin=-vmax, vmax=vmax, interpolation="nearest",
                           extent=[t[0], t[-1] + 1.0 / FS_TOK, C - 0.5, -0.5], zorder=1)
            decorate(ax, heat=True)
            ax.set_yticks(range(C), list(names), fontsize=6)
            ax.set_ylabel(f"{names[i]}  <-", rotation=0, ha="right", va="center", fontweight="bold")
            ax.tick_params(labelbottom=(i == C - 1))
            if i == 0:
                who = "kinematic" if stream == "kin" else "EEG"
                ax.set_title(f"Panel 3: muscle-graph attention of the {who} stream (rollout over its graph layers), change from "
                             "its rest mean over the series.\nBlock = receiving muscle, rows = source muscle. It shows how the "
                             "graph mixes the muscle nodes, not an amplitude.", loc="left", fontweight="bold")
        ax.set_xlabel("time in series (s)")
        side = fig.add_subplot(grid[2 * C:, 1])
        side.axis("off")
        bar = fig.colorbar(im, cax=side.inset_axes([0.05, 0.08, 0.14, 0.84]))
        bar.set_label("change of attention share", fontsize=7)
        bar.ax.tick_params(labelsize=6.5)
        first.set_xlim(t[0], t[-1])
        P._pin(fig.axes)
    return P._title(fig, title) if title else fig
