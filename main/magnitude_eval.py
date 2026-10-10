"""Magnitude metrics per test series and per condition (W / S / B).

Pearson r alone does not show whether bursts have the right SIZE or whether the
prediction returns to baseline. These metrics do, in original EMG units:

    height_ratio  Σ(pred · active) / Σ(target · active), active = target above its
                  own `active_quantile` (default 70th percentile). 1 = right size,
                  < 1 under-predicted bursts, > 1 over-predicted.
    rest_pred     mean prediction where the target is below its own
                  `rest_quantile` (default 20th percentile) ...
    rest_target   ... next to the mean target there.
    over_area     Σ relu(pred − target) / Σ target
    under_area    Σ relu(target − pred) / Σ target
plus r, CCC, RMSE and VAF% (same definitions as main.training.compute_metrics).

Usage in the notebook (per subject):
    series = {s: predict_series(model, eeg, kin, emg, prepare_batch, emg_mean, emg_std,
                                window=cfg["dataset"]["window_size"], device=device)
              for s, (eeg, kin, emg) in zip(test_series_ids, unique_series)}
    df = magnitude_report_by_series(series, series_conditions(p_id, data_dir), fs=500)
    print(format_magnitude_table(df))
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .training import compute_metrics

MAGNITUDE_COLUMNS = [
    "r", "ccc", "rmse", "vaf",
    "height_ratio", "rest_pred", "rest_target", "over_area", "under_area",
]

COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_GREY = "#8c8c8c"


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    den = np.asarray(den, dtype=np.float64)
    return np.where(np.abs(den) > 1e-15, num / np.where(den == 0, 1.0, den), np.nan)


def magnitude_report(
    pred: np.ndarray,
    target: np.ndarray,
    fs: float = 500.0,
    rest_quantile: float = 0.2,
    active_quantile: float = 0.7,
    channel_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-muscle magnitude metrics for one (stitched) series in original EMG units.

    Args:
        pred, target:     (T, C) arrays, original (denormalised) EMG units.
        fs:               sampling rate (Hz), only used for the `duration_s` column.
        rest_quantile:    target below its own quantile → "rest".
        active_quantile:  target above its own quantile → "active".
        channel_names:    C names (default ch0..).

    Returns:
        DataFrame with one row per muscle plus a "MEAN" row; columns
        ["muscle", *MAGNITUDE_COLUMNS, "duration_s"].
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"pred and target must be the same (T, C) shape, got {pred.shape} / {target.shape}")
    C = target.shape[1]
    names = list(channel_names) if channel_names is not None else [f"ch{c}" for c in range(C)]

    m = compute_metrics(pred, target, channel_names=names)

    q_rest = np.quantile(target, rest_quantile, axis=0)       # (C,)
    q_act = np.quantile(target, active_quantile, axis=0)
    active = target > q_act
    rest = target < q_rest

    height = _ratio((pred * active).sum(0), (target * active).sum(0))
    n_rest = rest.sum(0)
    rest_pred = _ratio((pred * rest).sum(0), n_rest)
    rest_target = _ratio((target * rest).sum(0), n_rest)
    t_sum = target.sum(0)
    over = _ratio(np.maximum(pred - target, 0.0).sum(0), t_sum)
    under = _ratio(np.maximum(target - pred, 0.0).sum(0), t_sum)

    df = pd.DataFrame({
        "muscle": names,
        "r": m.pearson, "ccc": m.ccc, "rmse": m.rmse, "vaf": m.vaf,
        "height_ratio": height, "rest_pred": rest_pred, "rest_target": rest_target,
        "over_area": over, "under_area": under,
    })
    df["duration_s"] = target.shape[0] / float(fs)
    mean = df[MAGNITUDE_COLUMNS + ["duration_s"]].mean().to_dict()
    mean["muscle"] = "MEAN"
    return pd.concat([df, pd.DataFrame([mean])], ignore_index=True)


def magnitude_report_by_series(
    series: Mapping[int, tuple[np.ndarray, np.ndarray]],
    conditions: Mapping[int, str],
    fs: float = 500.0,
    rest_quantile: float = 0.2,
    active_quantile: float = 0.7,
    channel_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Magnitude report per test series, per condition and over all test series.

    Args:
        series:     {series_id: (pred, target)} in original EMG units, in the order
                    the rows should appear.
        conditions: {series_id: "W" | "S" | "B"} (main.dataloader.series_conditions).

    Rows ("group" column), each a full magnitude_report block:
        "series 4 (W)"   one per series
        "W: series 2+4"  one per condition that has ≥ 2 test series (stitched)
        "all test"       all series stitched
    Stitching = concatenation in time, so quantiles are taken over the whole group.
    """
    kw = dict(fs=fs, rest_quantile=rest_quantile, active_quantile=active_quantile,
              channel_names=channel_names)
    blocks = []

    def _block(label, cond, ids):
        p = np.concatenate([series[s][0] for s in ids], axis=0)
        t = np.concatenate([series[s][1] for s in ids], axis=0)
        b = magnitude_report(p, t, **kw)
        b.insert(0, "condition", cond)
        b.insert(0, "group", label)
        blocks.append(b)

    ids = list(series)
    for s in ids:
        _block(f"series {s} ({conditions.get(s, '?')})", conditions.get(s, "?"), [s])
    for c in ("W", "S", "B"):
        members = [s for s in ids if conditions.get(s) == c]
        if len(members) >= 2:
            _block(f"{c}: series " + "+".join(str(s) for s in members), c, members)
    _block("all test", "all", ids)
    return pd.concat(blocks, ignore_index=True)


def format_magnitude_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Compact text table: one line per (group, muscle)."""
    cols = [c for c in ("group", "muscle") if c in df.columns]
    show = df[cols + MAGNITUDE_COLUMNS].copy()
    show = show.rename(columns={
        "height_ratio": "height", "rest_pred": "rest_pred", "rest_target": "rest_tgt",
        "over_area": "over", "under_area": "under", "vaf": "VAF%",
    })
    if "group" in show.columns:     # print each group label once
        dup = show["group"].eq(show["group"].shift())
        show.loc[dup, "group"] = ""
    fmt = {c: (lambda v, d=digits: f"{v:.{d}g}") for c in ("rmse", "rest_pred", "rest_tgt")}
    for c in ("r", "ccc", "height", "over", "under"):
        fmt[c] = lambda v, d=digits: f"{v:.{d}f}"
    fmt["VAF%"] = lambda v: f"{v:.1f}"
    return show.to_string(index=False, formatters=fmt)


@torch.no_grad()
def predict_series(
    model: torch.nn.Module,
    eeg: np.ndarray,
    kin: np.ndarray,
    emg: np.ndarray,
    prepare_batch: Callable,
    emg_mean,
    emg_std,
    window: int,
    device: torch.device,
    use_amp: bool = False,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over one full series in consecutive non-overlapping windows
    (the incomplete tail is dropped, as in the test loader with stride = window)
    and return (pred, target) stitched and denormalised (x * emg_std + emg_mean),
    i.e. the same code path as the notebook's interpretability cell.

    Returns two (n_windows * window, C) float64 arrays.
    """
    model.eval()
    n_win = eeg.shape[0] // window
    if n_win == 0:
        raise ValueError(f"series of {eeg.shape[0]} samples is shorter than one window ({window})")
    mean = torch.as_tensor(emg_mean).detach().float().cpu().numpy()
    std = torch.as_tensor(emg_std).detach().float().cpu().numpy()
    amp = use_amp and device.type == "cuda"

    preds, targets = [], []
    for b0 in range(0, n_win, batch_size):
        idx = range(b0, min(b0 + batch_size, n_win))
        e = torch.stack([torch.as_tensor(eeg[i * window:(i + 1) * window]) for i in idx])
        k = torch.stack([torch.as_tensor(kin[i * window:(i + 1) * window]) for i in idx])
        m = torch.stack([torch.as_tensor(emg[i * window:(i + 1) * window]) for i in idx])
        inputs, y = prepare_batch(e, k, m)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=amp):
            p = model(inputs["eeg"], inputs["kin"])
        c = y.shape[-1]
        preds.append(p.float().cpu().numpy().reshape(-1, c))
        targets.append(y.float().cpu().numpy().reshape(-1, c))
    pred = np.concatenate(preds).astype(np.float64) * std + mean
    target = np.concatenate(targets).astype(np.float64) * std + mean
    return pred, target


def plot_magnitude_report(
    reports: Mapping[str, pd.DataFrame],
    muscle_names: Sequence[str],
    title: str | None = None,
):
    """One row per subject: (left) height ratio per muscle, one bar per test
    series (+ grey "all test"), line at 1.0; (right) rest level per muscle over
    all test series, prediction vs target.

    Args:
        reports: {subject: magnitude_report_by_series() DataFrame}.
    """
    import matplotlib.pyplot as plt

    subjects = list(reports)
    n = len(subjects)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.6 * n), squeeze=False, facecolor="white",
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    x = np.arange(len(muscle_names))
    series_colors = [COLOR_BLUE, COLOR_ORANGE, "#6a5acd", "#2e8b57"]
    for row, subj in enumerate(subjects):
        df = reports[subj]
        df = df[df["muscle"] != "MEAN"]
        groups = [g for g in dict.fromkeys(df["group"]) if g.startswith("series ")] + ["all test"]
        ax = axes[row, 0]
        w = 0.8 / len(groups)
        for k, g in enumerate(groups):
            vals = df[df["group"] == g].set_index("muscle").loc[list(muscle_names), "height_ratio"]
            color = COLOR_GREY if g == "all test" else series_colors[k % len(series_colors)]
            ax.bar(x + (k - (len(groups) - 1) / 2) * w, vals.values, w * 0.92, color=color, label=g)
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="right size (1.0)")
        ax.set_xticks(x)
        ax.set_xticklabels(muscle_names)
        ax.set_ylabel("height ratio (pred / target)")
        ax.set_title(f"{subj}: burst height per test series", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))

        ax = axes[row, 1]
        allt = df[df["group"] == "all test"].set_index("muscle").loc[list(muscle_names)]
        ax.bar(x - 0.2, allt["rest_target"].values, 0.38, color=COLOR_GREY, label="target")
        ax.bar(x + 0.2, allt["rest_pred"].values, 0.38, color=COLOR_BLUE, label="prediction")
        ax.set_xticks(x)
        ax.set_xticklabels(muscle_names)
        ax.set_ylabel("mean level at rest (EMG units)")
        ax.set_title(f"{subj}: rest level (target < its 20th pct), all test", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================================
# Magnitude per weight / per surface over the HOLD phase of each lift
# ============================================================================

LOAD_COLUMNS = [
    "grouping", "value", "muscle", "n_lifts",
    "target_hold", "pred_hold", "height_ratio", "r_lift",
]
_LOAD_GROUPS = ("CurW", "CurS")
_LOAD_LABELS = {"CurW": "weight", "CurS": "surface"}


def lift_hold_segments(
    participant: int,
    series: int,
    fs: float = 500.0,
    data_dir=None,
    n_samples: int | None = None,
    margin_s: float = 0.3,
    min_samples: int = 100,
) -> list[dict]:
    """Hold phase of every lift of one series, from ``P{p}_AllLifts.mat``.

    Recipe (series time, sample indices at ``fs``):
        t0    = StartTime − LEDOn
        start = int((t0 + tLiftOff + margin_s) · fs)
        stop  = int((t0 + tReplace − margin_s) · fs)
    Lifts with stop − start < ``min_samples`` are skipped, and so are lifts with
    stop > ``n_samples`` when the series length is given.

    Args:
        participant, series: e.g. 1, 4.
        data_dir: raw data root holding ``P{p}/P{p}_AllLifts.mat``
                  (default: repo data/way-eeg/raw).

    Returns:
        [{"lift", "start", "stop", "CurW", "CurS"}, ...] in AllLifts order;
        CurW ∈ {1, 2, 4} (weight), CurS ∈ {1, 2, 3} (surface).
    """
    from .dataloader import _alllifts_path, _mat_load, _names_to_list

    path = _alllifts_path(participant, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"AllLifts file not found: {path}")
    P = _mat_load(path)["P"]
    cols = _names_to_list(P.ColNames)
    lifts = np.atleast_2d(np.asarray(P.AllLifts, dtype=float))
    col = {n: cols.index(n) for n in
           ("Run", "Lift", "CurW", "CurS", "StartTime", "LEDOn", "tLiftOff", "tReplace")}

    out = []
    for row in lifts[lifts[:, col["Run"]].astype(int) == int(series)]:
        t0 = row[col["StartTime"]] - row[col["LEDOn"]]
        lo = t0 + row[col["tLiftOff"]] + margin_s
        hi = t0 + row[col["tReplace"]] - margin_s
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        a, b = int(lo * fs), int(hi * fs)
        if b - a < min_samples or (n_samples is not None and b > n_samples):
            continue
        out.append({"lift": int(row[col["Lift"]]), "start": a, "stop": b,
                    "CurW": int(row[col["CurW"]]), "CurS": int(row[col["CurS"]])})
    return out


def magnitude_by_load(
    pred: np.ndarray,
    target: np.ndarray,
    segments,
    group: str = "CurW",
    channel_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Hold-phase magnitude per weight (``group="CurW"``) or surface (``"CurS"``).

    Args:
        pred, target: (T, C) stitched series in original EMG units (sample i =
                      series sample i, as returned by predict_series).
        segments:     lift_hold_segments() records (list of dicts or a DataFrame);
                      segments that do not fit inside ``pred`` are skipped.

    Returns:
        DataFrame with LOAD_COLUMNS, one row per (group value, muscle) plus the
        value "all" over every lift:
            n_lifts       lifts in the group
            target_hold   mean over lifts of the per-lift mean target
            pred_hold     same for the prediction
            height_ratio  Σ pred / Σ target over all hold samples of the group
            r_lift        Pearson r between per-lift predicted and target means
                          (NaN with < 4 lifts or when one side is constant)
    """
    if group not in _LOAD_GROUPS:
        raise ValueError(f"group must be one of {_LOAD_GROUPS}, got {group!r}")
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"pred and target must be the same (T, C) shape, got {pred.shape} / {target.shape}")
    T, C = target.shape
    names = list(channel_names) if channel_names is not None else [f"ch{c}" for c in range(C)]
    recs = segments.to_dict("records") if isinstance(segments, pd.DataFrame) else list(segments)
    recs = [s for s in recs if 0 <= int(s["start"]) < int(s["stop"]) <= T]

    def _per_lift(x: np.ndarray, fn) -> np.ndarray:
        return np.array([fn(x[int(s["start"]):int(s["stop"])], axis=0) for s in recs]).reshape(-1, C)

    lift_p, lift_t = _per_lift(pred, np.mean), _per_lift(target, np.mean)
    sum_p, sum_t = _per_lift(pred, np.sum), _per_lift(target, np.sum)
    keys = np.array([s[group] for s in recs])

    def _r(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    rows = []
    for v in sorted(set(keys.tolist())) + ["all"]:
        sel = np.ones(len(recs), bool) if v == "all" else keys == v
        n = int(sel.sum())
        for c, m in enumerate(names):
            den = sum_t[sel, c].sum()
            rows.append({
                "grouping": group, "value": v, "muscle": m, "n_lifts": n,
                "target_hold": float(lift_t[sel, c].mean()) if n else float("nan"),
                "pred_hold": float(lift_p[sel, c].mean()) if n else float("nan"),
                "height_ratio": float(sum_p[sel, c].sum() / den) if n and abs(den) > 1e-15 else float("nan"),
                "r_lift": _r(lift_p[sel, c], lift_t[sel, c]),
            })
    return pd.DataFrame(rows, columns=LOAD_COLUMNS)


def format_magnitude_by_load(df: pd.DataFrame, digits: int = 3) -> str:
    """Compact text table per block (subject / series / grouping, when present):
    one line per muscle; per group value the height ratio, target|pred hold
    level and per-lift r."""
    keys = [k for k in ("subject", "series", "condition", "grouping") if k in df.columns]
    blocks = df.groupby(keys, sort=False) if keys else [((), df)]
    lines = []
    for key, block in blocks:
        key = key if isinstance(key, tuple) else (key,)
        label = _LOAD_LABELS.get(block["grouping"].iloc[0], "group")
        lines.append("  ".join(f"{k}={v}" for k, v in zip(keys, key)))
        table = {}
        for v in dict.fromkeys(block["value"]):
            sub = block[block["value"] == v].set_index("muscle")
            name = "all" if v == "all" else f"{label} {v}"
            name += f" (n={int(sub['n_lifts'].iloc[0])})"
            table[(name, "height")] = sub["height_ratio"].map(lambda x: f"{x:.{digits}f}")
            table[(name, "tgt|pred")] = [f"{t:.{digits}g}|{p:.{digits}g}"
                                         for t, p in zip(sub["target_hold"], sub["pred_hold"])]
            table[(name, "r")] = sub["r_lift"].map(lambda x: "nan" if np.isnan(x) else f"{x:.2f}")
        out = pd.DataFrame(table, index=list(dict.fromkeys(block["muscle"])))
        lines.append(out.to_string())
        lines.append("")
    return "\n".join(lines).rstrip("\n")


def plot_magnitude_by_load(
    df: pd.DataFrame,
    muscle_names: Sequence[str],
    group: str = "CurW",
    title: str | None = None,
):
    """Hold height ratio (Σpred / Σtarget over hold samples) per muscle, one bar
    per weight (or surface) value, one panel per (subject, series); dashed line
    at 1.0.

    Args:
        df: concatenated magnitude_by_load() frames, optionally with "subject" /
            "series" columns (rows of the other grouping and "all" are ignored).
    """
    import matplotlib.pyplot as plt

    d = df[(df["grouping"] == group) & (df["value"] != "all")]
    keys = [k for k in ("subject", "series") if k in d.columns]
    panels = list(dict.fromkeys(map(tuple, d[keys].to_numpy().tolist()))) if keys else [()]
    n = max(1, len(panels))
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.4 * nrows), squeeze=False,
                             facecolor="white", sharey=True)
    colors = [COLOR_BLUE, COLOR_ORANGE, "#6a5acd", "#2e8b57"]
    label = _LOAD_LABELS.get(group, group)
    x = np.arange(len(muscle_names))
    for ax, pk in zip(axes.flat, panels):
        sub = d
        for k, v in zip(keys, pk):
            sub = sub[sub[k] == v]
        values = sorted(sub["value"].unique().tolist())
        w = 0.8 / max(1, len(values))
        for j, v in enumerate(values):
            s = sub[sub["value"] == v].set_index("muscle").reindex(list(muscle_names))
            ax.bar(x + (j - (len(values) - 1) / 2) * w, s["height_ratio"].to_numpy(dtype=float),
                   w * 0.92, color=colors[j % len(colors)],
                   label=f"{label} {v} (n={int(s['n_lifts'].max())})")
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(muscle_names)
        ax.set_title("  ".join(f"{k} {v}" for k, v in zip(keys, pk)) or "all", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("hold height ratio (pred / target)")
    fig.suptitle(title or f"Hold-phase magnitude per {label}", fontsize=12)
    fig.tight_layout()
    return fig
