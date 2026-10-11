"""main/residual_eval.py — does EEG explain what kinematics cannot? (Package 2, Phase C)

Two trained models are evaluated together on the test series:

    kin_model   kinematics-only model (Stage K, frozen):   kin_pred_z
    eeg_model   EEG-only model trained on the residual     res_pred_z ~ emg_z - kin_pred_z

Every prediction is made in consecutive non-overlapping windows of the training window
size through the notebook's ``prepare_batch`` (``main.inference``; the tail shorter than
one window is dropped, exactly like the test loader).

Per muscle (and the mean over muscles), per test series and over all test series:

    kin_r / kin_vaf / kin_rmse      kinematics-only prediction
    full_r / full_vaf / full_rmse   kinematics + EEG residual prediction (kin_pred + res_pred)
    d_r / d_vaf / d_rmse            full - kin   (d_vaf in percentage points)
    res_r                           Pearson r between predicted and true residual (z units)
    res_fve                         1 - sum (res_true - res_pred)^2 / sum (res_true - mean)^2
                                    (fraction of residual variance explained; can be negative)
    VAF and RMSE use original EMG units (z * emg_std + emg_mean), as main.training.compute_metrics.

Null distributions (``n_null`` draws, seeded) — the SAME trained EEG model, fed EEG that is

    "shuffle"   time-shuffled within each window (one seeded permutation per draw). The
                window keeps its values, so its mean level survives: this null only removes
                the timing inside the window.
    "swap"      taken from a DIFFERENT test window: a random derangement of all test windows
                (pooled over the test series), so the EEG statistics stay realistic but the
                EEG has nothing to do with this window's EMG.

For each draw d_vaf and res_r are recomputed; the report holds mean and SD of the null and the
empirical p = (1 + #null >= observed) / (1 + n_null).

Per lift (hold phase from ``main.magnitude_eval.lift_hold_segments``): mean true residual and
mean predicted residual over the hold phase, then Pearson r across lifts (``lift_r``; NaN with
fewer than 4 lifts or a constant side), per series and pooled, with the same for the "swap"
null. ``lift_r_within`` is the pooled r after removing each series' mean from both sides: a
between-series offset alone (two test series = two points) cannot produce it.

Verdict per muscle (``residual_verdict``): "EEG adds unique information" only if
    d_vaf > 0 over all test series, AND p <= alpha (0.05) for d_vaf against the "swap" null,
    AND the residual r is > 0 in EVERY test series, per lift or per sample;
otherwise "no evidence".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .ablation import pearson_per_channel, shuffle_time
from .inference import predict_series_stitched, window_batch
from .training import compute_metrics

__all__ = [
    "NULL_KINDS", "NULL_LABELS", "VERDICT_YES", "VERDICT_NO", "RESIDUAL_COLUMNS",
    "ResidualPredictions", "random_derangement", "predict_residuals", "residual_report",
    "residual_verdict", "format_residual_report", "plot_residual_delta_vaf",
    "plot_residual_per_lift",
]

NULL_KINDS: tuple[str, ...] = ("shuffle", "swap")
NULL_LABELS: dict[str, str] = {
    "shuffle": "EEG time-shuffled within the window",
    "swap": "EEG of a different test window",
}
VERDICT_YES = "EEG adds unique information"
VERDICT_NO = "no evidence"
ALL_TEST = "all test"

COLOR_KIN = "#eb6834"      # kinematics only
COLOR_FULL = "#2a78d6"     # kinematics + EEG residual
COLOR_NULL = "#8c8c8c"     # null

_NULL_STATS = ("d_vaf", "res_r")
RESIDUAL_COLUMNS = (
    ["group", "muscle", "n_samples",
     "kin_r", "kin_vaf", "kin_rmse", "full_r", "full_vaf", "full_rmse",
     "d_r", "d_vaf", "d_rmse", "res_r", "res_fve", "n_null"]
    + [f"{s}_{suffix}" for k in NULL_KINDS for s in _NULL_STATS
       for suffix in (f"null_{k}_mean", f"null_{k}_sd", f"p_{k}")]
    + ["n_lifts", "lift_r", "lift_r_within", "lift_r_null_swap_mean", "lift_r_null_swap_sd", "lift_r_p_swap"]
)
PER_LIFT_COLUMNS = ["series", "lift", "CurW", "CurS", "start", "stop", "n_samples", "muscle",
                    "true_res", "pred_res", "null_swap_res", "null_swap_res_mean"]


# ════════════════════════════════════════════════════════════════════════════
# Predictions and nulls
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ResidualPredictions:
    """Stitched predictions of the test series, normalised (z) units.

    target / kin / res: {series_id: (T', C)} — EMG, kinematics-only prediction, predicted
    residual; T' = whole windows only. null: {kind: {series_id: (n_null, T', C)}} predicted
    residual under each null. derangements: (n_null, n_windows) — for the "swap" null,
    window w (all test windows in series order) was fed the EEG of window derangements[i, w].
    """

    series_ids: list
    window: int
    n_null: int
    seed: int
    target: dict = field(default_factory=dict)
    kin: dict = field(default_factory=dict)
    res: dict = field(default_factory=dict)
    null: dict = field(default_factory=dict)
    derangements: np.ndarray | None = None

    def true_residual(self, sid) -> np.ndarray:
        return self.target[sid] - self.kin[sid]


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of 0..n-1 without a fixed point (rejection sampling, ~e tries)."""
    n = int(n)
    if n < 2:
        raise ValueError(f"a derangement needs at least 2 items, got {n}")
    idx = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == idx):
            return perm


@torch.no_grad()
def _forward_windows(model: torch.nn.Module, eeg: torch.Tensor, kin: torch.Tensor | None,
                     device, batch_size: int) -> np.ndarray:
    """(n_w, W, F) normalised EEG windows (+ their kin windows) -> (n_w, W, C) float32."""
    model.eval()
    out = []
    for b0 in range(0, eeg.shape[0], int(batch_size)):
        inputs = {"eeg": eeg[b0:b0 + int(batch_size)].to(device)}
        if kin is not None:
            inputs["kin"] = kin[b0:b0 + int(batch_size)].to(device)
        out.append(model(**inputs).float().cpu().numpy())
    for layer in list(getattr(getattr(model, "encoder", None), "layers", []) or []):
        if getattr(getattr(layer, "mhsa", None), "last_attn_weights", None) is not None:
            layer.mhsa.last_attn_weights = None
    return np.concatenate(out)


def predict_residuals(
    kin_model: torch.nn.Module,
    eeg_model: torch.nn.Module,
    series: Mapping,
    prepare_batch: Callable,
    window: int,
    device: torch.device | None = None,
    n_null: int = 20,
    seed: int = 42,
    batch_size: int = 32,
) -> ResidualPredictions:
    """Kinematics-only prediction, predicted residual and the null residuals of the test series.

    Args:
        kin_model:     frozen kinematics-only model, called as ``model(eeg=…, kin=…)``.
        eeg_model:     residual model, called the same way (an ``EEGOnly`` wrapper ignores kin).
        series:        {series_id: (eeg, kin, emg)} full un-normalised series, as in the
                       dataset (``test_ds._windows``), in the order of the report.
        prepare_batch: the notebook's normalisation closure (NOT the residual one).
        window:        ``cfg["dataset"]["window_size"]``.
        n_null, seed:  draws per null and their seed (see the module docstring).

    Both models are put in eval mode; float32, no autocast.
    """
    window = int(window)
    n_null = int(n_null)
    ids = list(series)
    preds = ResidualPredictions(series_ids=ids, window=window, n_null=n_null, seed=int(seed))
    eeg_w, kin_w, counts = [], [], []
    in_device = None
    for sid in ids:
        eeg, kin, emg = series[sid]
        kin_z, tgt_z = predict_series_stitched(
            kin_model, eeg, kin, prepare_batch, window=window, device=device,
            batch_size=batch_size, emg=emg, tail="drop")
        res_z = predict_series_stitched(
            eeg_model, eeg, kin, prepare_batch, window=window, device=device,
            batch_size=batch_size, tail="drop")
        preds.target[sid], preds.kin[sid], preds.res[sid] = tgt_z, kin_z, res_z
        if n_null > 0:
            inputs, _ = window_batch(eeg, kin, prepare_batch, window)
            if not isinstance(inputs, dict) or "eeg" not in inputs:
                raise TypeError("prepare_batch must return a dict with an 'eeg' entry")
            in_device = inputs["eeg"].device
            eeg_w.append(inputs["eeg"].detach().cpu())
            kin_w.append(inputs["kin"].detach().cpu() if inputs.get("kin") is not None else None)
        counts.append(len(tgt_z) // window)

    preds.null = {k: {} for k in NULL_KINDS}
    if n_null == 0:
        preds.derangements = np.empty((0, int(sum(counts))), dtype=np.int64)
        return preds
    total = int(sum(counts))
    if total < 2:
        raise ValueError("the window-swap null needs at least 2 test windows")
    E = torch.cat(eeg_w)
    K = None if any(k is None for k in kin_w) else torch.cat(kin_w)
    rng = np.random.default_rng(int(seed))
    draws = {k: [] for k in NULL_KINDS}
    derangements = []
    for i in range(n_null):
        d = random_derangement(total, rng)
        derangements.append(d)
        draws["swap"].append(_forward_windows(eeg_model, E[torch.as_tensor(d)], K, in_device, batch_size))
        draws["shuffle"].append(_forward_windows(
            eeg_model, shuffle_time(E, int(seed) + 1 + i), K, in_device, batch_size))
    preds.derangements = np.stack(derangements).astype(np.int64)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    for kind in NULL_KINDS:
        stack = np.stack(draws[kind])                                  # (n_null, total, W, C)
        for j, sid in enumerate(ids):
            part = stack[:, bounds[j]:bounds[j + 1]]
            preds.null[kind][sid] = part.reshape(n_null, -1, part.shape[-1])
    return preds


# ════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════

def _with_mean(x: np.ndarray) -> np.ndarray:
    """Append the mean over the last axis (muscles): (..., C) -> (..., C + 1)."""
    x = np.asarray(x, dtype=np.float64)
    return np.concatenate([x, x.mean(axis=-1, keepdims=True)], axis=-1)


def _null_summary(obs: np.ndarray, null: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """obs (M,), null (n_null, M) -> mean, sd of the null and p = (1 + #null >= obs) / (1 + n_null)."""
    nan = np.full(obs.shape, np.nan)
    if null is None or len(null) == 0:
        return nan, nan, nan
    n = null.shape[0]
    valid = np.isfinite(null)
    cnt = valid.sum(0)
    safe = np.where(valid, null, 0.0)
    mean = np.where(cnt > 0, safe.sum(0) / np.maximum(cnt, 1), np.nan)
    dev = np.where(valid, null - mean, 0.0)
    sd = np.where(cnt > 1, np.sqrt((dev ** 2).sum(0) / np.maximum(cnt - 1, 1)), np.where(cnt == 1, 0.0, np.nan))
    ge = (valid & (null >= obs)).sum(0)
    p = np.where(np.isfinite(obs), (1.0 + ge) / (1.0 + n), np.nan)
    return mean, sd, p


def _lift_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column Pearson r across lifts of (L, C) arrays; NaN with < 4 lifts or a constant side."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = a.shape[-1]
    if a.shape[0] < 4:
        return np.full(C, np.nan)
    ac, bc = a - a.mean(0), b - b.mean(0)
    den = np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum(0))
    ok = (ac.std(0) > 1e-12) & (bc.std(0) > 1e-12)
    return np.where(ok, (ac * bc).sum(0) / np.where(ok, den, 1.0), np.nan)


def _hold_means(x: np.ndarray, recs: list[dict]) -> np.ndarray:
    """(…, T, C) -> (…, L, C): mean over the hold phase of every lift."""
    if not recs:
        return np.empty(x.shape[:-2] + (0, x.shape[-1]))
    return np.stack([x[..., int(s["start"]):int(s["stop"]), :].mean(axis=-2) for s in recs], axis=-2)


def residual_report(
    preds: ResidualPredictions,
    emg_mean,
    emg_std,
    channel_names: Sequence[str] | None = None,
    segments: Mapping | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report of a ``predict_residuals`` result.

    Args:
        emg_mean, emg_std: (C,) training statistics (arrays or tensors) -> original EMG units.
        channel_names:     C muscle names (default ch0..).
        segments:          {series_id: lift_hold_segments(...)} for the per-lift analysis
                           (None = no per-lift columns); lifts that do not fit inside the
                           stitched series are skipped.

    Returns:
        (report, per_lift).  report: RESIDUAL_COLUMNS, one row per muscle plus "MEAN" for
        every group "series <id>" and "all test" (see the module docstring; in the MEAN row
        the null statistics and p-values are those of the mean over muscles).
        per_lift: PER_LIFT_COLUMNS, one row per (series, lift, muscle): mean true / predicted
        residual over the hold phase (z units), the same for the first "swap" null draw
        (null_swap_res) and averaged over all draws (null_swap_res_mean).
    """
    mean = torch.as_tensor(emg_mean).detach().double().cpu().numpy().reshape(-1)
    std = torch.as_tensor(emg_std).detach().double().cpu().numpy().reshape(-1)
    ids = list(preds.series_ids)
    C = preds.target[ids[0]].shape[1]
    names = list(channel_names) if channel_names is not None else [f"ch{c}" for c in range(C)]
    if len(names) != C:
        raise ValueError(f"{len(names)} channel names for {C} channels")
    n_null = int(preds.n_null)
    have_null = n_null > 0

    # hold-phase means per series
    lifts: dict = {}
    lift_rows = []
    for sid in ids:
        T = len(preds.target[sid])
        recs = [dict(s) for s in ((segments or {}).get(sid) or [])
                if 0 <= int(s["start"]) < int(s["stop"]) <= T]
        true = _hold_means(preds.true_residual(sid).astype(np.float64), recs)
        pred = _hold_means(preds.res[sid].astype(np.float64), recs)
        null = _hold_means(preds.null["swap"][sid].astype(np.float64), recs) if have_null else None
        lifts[sid] = (true, pred, null)
        for j, s in enumerate(recs):
            for c, m in enumerate(names):
                lift_rows.append({
                    "series": sid, "lift": int(s.get("lift", j + 1)),
                    "CurW": s.get("CurW"), "CurS": s.get("CurS"),
                    "start": int(s["start"]), "stop": int(s["stop"]),
                    "n_samples": int(s["stop"]) - int(s["start"]), "muscle": m,
                    "true_res": float(true[j, c]), "pred_res": float(pred[j, c]),
                    "null_swap_res": float(null[0, j, c]) if have_null else np.nan,
                    "null_swap_res_mean": float(null[:, j, c].mean()) if have_null else np.nan,
                })
    per_lift = pd.DataFrame(lift_rows, columns=PER_LIFT_COLUMNS)

    def _block(label: str, members: list) -> pd.DataFrame:
        T = np.concatenate([preds.target[s] for s in members]).astype(np.float64)
        K = np.concatenate([preds.kin[s] for s in members]).astype(np.float64)
        R = np.concatenate([preds.res[s] for s in members]).astype(np.float64)
        true_res = T - K
        m_kin = compute_metrics(K * std + mean, T * std + mean, channel_names=names)
        m_full = compute_metrics((K + R) * std + mean, T * std + mean, channel_names=names)
        res_r = pearson_per_channel(R, true_res)
        ss_tot = ((true_res - true_res.mean(0)) ** 2).sum(0)
        res_fve = np.where(ss_tot > 1e-12, 1.0 - ((true_res - R) ** 2).sum(0) / np.where(ss_tot > 1e-12, ss_tot, 1.0), np.nan)
        cols = {
            "kin_r": m_kin.pearson, "kin_vaf": m_kin.vaf, "kin_rmse": m_kin.rmse,
            "full_r": m_full.pearson, "full_vaf": m_full.vaf, "full_rmse": m_full.rmse,
            "d_r": m_full.pearson - m_kin.pearson, "d_vaf": m_full.vaf - m_kin.vaf,
            "d_rmse": m_full.rmse - m_kin.rmse, "res_r": res_r, "res_fve": res_fve,
        }
        out = {k: _with_mean(v) for k, v in cols.items()}

        # nulls: VAF does not depend on the units, so z units are used
        var_t = T.var(axis=0)
        var_res = true_res.var(axis=0)
        for kind in NULL_KINDS:
            null_stats = {s: None for s in _NULL_STATS}
            if have_null:
                N = np.concatenate([preds.null[kind][s] for s in members], axis=1).astype(np.float64)
                d_vaf = np.stack([
                    np.where(var_t > 1e-12, 100.0 * (var_res - (n - true_res).var(axis=0)) / np.where(var_t > 1e-12, var_t, 1.0), 0.0)
                    for n in N])
                null_stats = {"d_vaf": _with_mean(d_vaf),
                              "res_r": _with_mean(np.stack([pearson_per_channel(n, true_res) for n in N]))}
            for s in _NULL_STATS:
                mu, sd, p = _null_summary(out[s], null_stats[s])
                out[f"{s}_null_{kind}_mean"], out[f"{s}_null_{kind}_sd"], out[f"{s}_p_{kind}"] = mu, sd, p

        # per lift
        true_l = np.concatenate([lifts[s][0] for s in members])
        pred_l = np.concatenate([lifts[s][1] for s in members])
        out["lift_r"] = _with_mean(_lift_r(pred_l, true_l))
        true_w = np.concatenate([lifts[s][0] - lifts[s][0].mean(0) if len(lifts[s][0]) else lifts[s][0] for s in members])
        pred_w = np.concatenate([lifts[s][1] - lifts[s][1].mean(0) if len(lifts[s][1]) else lifts[s][1] for s in members])
        out["lift_r_within"] = _with_mean(_lift_r(pred_w, true_w))
        null_l = None
        if have_null and len(true_l):
            stack = np.concatenate([lifts[s][2] for s in members], axis=1)      # (n_null, L, C)
            null_l = _with_mean(np.stack([_lift_r(n, true_l) for n in stack]))
        (out["lift_r_null_swap_mean"], out["lift_r_null_swap_sd"],
         out["lift_r_p_swap"]) = _null_summary(out["lift_r"], null_l)

        df = pd.DataFrame(out)
        df.insert(0, "n_lifts", int(len(true_l)))
        df.insert(0, "n_null", n_null)
        df.insert(0, "n_samples", int(len(T)))
        df.insert(0, "muscle", names + ["MEAN"])
        df.insert(0, "group", label)
        return df[RESIDUAL_COLUMNS]

    blocks = [_block(f"series {s}", [s]) for s in ids] + [_block(ALL_TEST, ids)]
    return pd.concat(blocks, ignore_index=True), per_lift


def residual_verdict(report: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One verdict per muscle (and for the MEAN row) from a ``residual_report`` frame.

    VERDICT_YES ("EEG adds unique information") only if all three hold, else VERDICT_NO:
      1. d_vaf > 0 in the "all test" row;
      2. d_vaf_p_swap <= alpha in the "all test" row (window-swap null; NaN = not met);
      3. the residual r is > 0 in EVERY "series …" row — per lift (lift_r) in all of them,
         or per sample (res_r) in all of them (NaN = not met).

    Returns a frame [muscle, d_vaf, p_swap, res_r_by_series, lift_r_by_series, verdict, why].
    """
    series_groups = [g for g in dict.fromkeys(report["group"]) if str(g).startswith("series ")]
    allt = report[report["group"] == ALL_TEST].set_index("muscle")
    rows = []
    for muscle in allt.index:
        d_vaf = float(allt.loc[muscle, "d_vaf"])
        p = float(allt.loc[muscle, "d_vaf_p_swap"])
        per = report[report["muscle"] == muscle].set_index("group")
        lift = [float(per.loc[g, "lift_r"]) for g in series_groups]
        samp = [float(per.loc[g, "res_r"]) for g in series_groups]
        c1 = d_vaf > 0
        c2 = bool(np.isfinite(p) and p <= alpha)
        lift_ok = bool(series_groups) and all(np.isfinite(v) and v > 0 for v in lift)
        samp_ok = bool(series_groups) and all(np.isfinite(v) and v > 0 for v in samp)
        c3 = lift_ok or samp_ok
        failed = []
        if not c1:
            failed.append(f"ΔVAF {d_vaf:+.2f} is not > 0")
        if not c2:
            failed.append("no null" if not np.isfinite(p) else f"p = {p:.3f} > {alpha:g} against the window-swap null")
        if not c3:
            failed.append("residual r is not > 0 in every test series (neither per lift nor per sample)")
        ok = c1 and c2 and c3
        why = (f"ΔVAF {d_vaf:+.2f} > 0; p = {p:.3f} <= {alpha:g}; residual r > 0 in every test series "
               f"({'per lift' if lift_ok else ''}{' and ' if lift_ok and samp_ok else ''}{'per sample' if samp_ok else ''})"
               if ok else "; ".join(failed))
        rows.append({
            "muscle": muscle, "d_vaf": d_vaf, "p_swap": p,
            "res_r_by_series": ", ".join(f"{v:+.3f}" for v in samp),
            "lift_r_by_series": ", ".join("nan" if not np.isfinite(v) else f"{v:+.2f}" for v in lift),
            "verdict": VERDICT_YES if ok else VERDICT_NO, "why": why,
        })
    return pd.DataFrame(rows, columns=["muscle", "d_vaf", "p_swap", "res_r_by_series",
                                       "lift_r_by_series", "verdict", "why"])


def format_residual_report(report: pd.DataFrame) -> str:
    """Two text tables: sample-level metrics, then nulls and the per-lift analysis."""
    def _table(cols: dict, fmt: dict) -> str:
        show = report[["group", "muscle"] + list(cols)].rename(columns=cols)
        show.loc[show["group"].eq(show["group"].shift()), "group"] = ""
        return show.to_string(index=False, formatters={cols.get(k, k): v for k, v in fmt.items()}, na_rep="nan")

    f3 = lambda v: f"{v:.3f}"            # noqa: E731
    s3 = lambda v: f"{v:+.3f}"           # noqa: E731
    f1 = lambda v: f"{v:.1f}"            # noqa: E731
    s2 = lambda v: f"{v:+.2f}"           # noqa: E731
    g3 = lambda v: f"{v:.3g}"            # noqa: E731
    a = _table(
        {"kin_r": "kin r", "kin_vaf": "kin VAF%", "kin_rmse": "kin RMSE", "full_r": "kin+EEG r",
         "full_vaf": "kin+EEG VAF%", "full_rmse": "kin+EEG RMSE", "d_r": "Δr", "d_vaf": "ΔVAF",
         "res_r": "res r", "res_fve": "res FVE"},
        {"kin_r": f3, "kin_vaf": f1, "kin_rmse": g3, "full_r": f3, "full_vaf": f1, "full_rmse": g3,
         "d_r": s3, "d_vaf": s2, "res_r": s3, "res_fve": s3})
    b = _table(
        {"d_vaf": "ΔVAF", "d_vaf_null_swap_mean": "swap mean", "d_vaf_null_swap_sd": "swap sd",
         "d_vaf_p_swap": "p swap", "d_vaf_null_shuffle_mean": "shuf mean", "d_vaf_null_shuffle_sd": "shuf sd",
         "d_vaf_p_shuffle": "p shuf", "res_r": "res r", "res_r_null_swap_mean": "r swap mean",
         "res_r_null_swap_sd": "r swap sd", "res_r_p_swap": "r p swap", "n_lifts": "lifts",
         "lift_r": "lift r", "lift_r_within": "lift r within", "lift_r_null_swap_mean": "lift swap mean",
         "lift_r_null_swap_sd": "lift swap sd", "lift_r_p_swap": "lift p swap"},
        {"d_vaf": s2, "d_vaf_null_swap_mean": s2, "d_vaf_null_swap_sd": f"{{:.2f}}".format,
         "d_vaf_p_swap": f3, "d_vaf_null_shuffle_mean": s2, "d_vaf_null_shuffle_sd": f"{{:.2f}}".format,
         "d_vaf_p_shuffle": f3, "res_r": s3, "res_r_null_swap_mean": s3, "res_r_null_swap_sd": f3,
         "res_r_p_swap": f3, "lift_r": s2, "lift_r_within": s2, "lift_r_null_swap_mean": s2,
         "lift_r_null_swap_sd": f"{{:.2f}}".format, "lift_r_p_swap": f3})
    n_null = int(report["n_null"].iloc[0]) if len(report) else 0
    return ("Kinematics only vs kinematics + EEG residual (VAF, RMSE: original EMG units; "
            "res r, res FVE: z units)\n" + a
            + f"\n\nNull distributions ({n_null} draws; swap = EEG of a different test window, "
              "shuf = EEG time-shuffled within the window; p = (1 + #null >= observed) / (1 + n)) "
              "and per-lift hold-phase residual r\n" + b)


# ════════════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════════════

def _by_subject(obj) -> dict:
    if isinstance(obj, pd.DataFrame):
        if "subject" in obj.columns:
            return {s: obj[obj["subject"] == s] for s in dict.fromkeys(obj["subject"])}
        return {"": obj}
    return dict(obj)


def plot_residual_delta_vaf(reports, title: str | None = None):
    """One row per subject, all test series: (left) VAF% per muscle, kinematics only (orange)
    vs kinematics + EEG residual (blue); (right) ΔVAF per muscle (blue bars) on top of the
    window-swap null, mean ± 2 SD (grey band, line = null mean), with the empirical p.

    Args:
        reports: {subject: residual_report frame}, or one frame (optionally with a
                 "subject" column).
    """
    import matplotlib.pyplot as plt

    reports = _by_subject(reports)
    n = len(reports)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.9 * n), squeeze=False, facecolor="white",
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    for row, (subj, rep) in enumerate(reports.items()):
        d = rep[rep["group"] == ALL_TEST]
        muscles = list(d["muscle"])
        x = np.arange(len(muscles))
        n_null = int(d["n_null"].iloc[0])
        prefix = f"{subj}: " if subj else ""

        ax = axes[row, 0]
        ax.bar(x - 0.2, d["kin_vaf"].to_numpy(float), 0.38, color=COLOR_KIN, label="kinematics only")
        ax.bar(x + 0.2, d["full_vaf"].to_numpy(float), 0.38, color=COLOR_FULL, label="kinematics + EEG residual")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(muscles)
        ax.set_ylabel("VAF % (all test series)")
        ax.set_title(f"{prefix}variance accounted for", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=2)

        ax = axes[row, 1]
        obs = d["d_vaf"].to_numpy(float)
        mu = d["d_vaf_null_swap_mean"].to_numpy(float)
        sd = d["d_vaf_null_swap_sd"].to_numpy(float)
        p = d["d_vaf_p_swap"].to_numpy(float)
        has_null = np.isfinite(mu) & np.isfinite(sd)
        tops = np.where(has_null, np.maximum(obs, mu + 2 * sd), obs)
        lows = np.where(has_null, np.minimum(obs, mu - 2 * sd), obs)
        ax.bar(x, obs, 0.44, color=COLOR_FULL, label="ΔVAF (kin + EEG residual − kin only)", zorder=2)
        if has_null.any():       # drawn over the bars, half transparent, so it stays visible
            ax.bar(x[has_null], 4 * sd[has_null], 0.8, bottom=(mu - 2 * sd)[has_null], color=COLOR_NULL,
                   alpha=0.55, linewidth=0, zorder=3,
                   label=f"null: EEG of a different window, mean ± 2 SD (n={n_null})")
            ax.hlines(mu[has_null], x[has_null] - 0.4, x[has_null] + 0.4, color="#4a4a4a", linewidth=1.4,
                      zorder=4)
        ax.axhline(0.0, color="black", linewidth=0.8)
        span = float(max(tops.max(), 0.0) - min(lows.min(), 0.0)) or 1.0
        for xi, top, pi in zip(x, tops, p):
            if np.isfinite(pi):
                ax.text(xi, max(top, 0.0) + 0.03 * span, f"p={pi:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(min(lows.min(), 0.0) - 0.06 * span, max(tops.max(), 0.0) + 0.16 * span)
        ax.set_xticks(x)
        ax.set_xticklabels(muscles)
        ax.set_ylabel("ΔVAF (percentage points)")
        ax.set_title(f"{prefix}what the EEG residual model adds", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=1)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def _r_text(a, b) -> str:
    r = _lift_r(np.asarray(a, float).reshape(-1, 1), np.asarray(b, float).reshape(-1, 1))[0]
    return "n/a" if not np.isfinite(r) else f"{r:+.2f}"


def plot_residual_per_lift(per_lift: pd.DataFrame, title: str | None = None):
    """Per muscle: mean TRUE residual of every test lift's hold phase (x) against the mean
    PREDICTED residual (y), z units. Blue = EEG residual model (one marker per test series),
    grey = one window-swap null draw. Title: Pearson r across lifts, pooled and per series.

    Args:
        per_lift: the per-lift frame of ``residual_report`` (optionally with a "subject"
                  column: one row of panels per subject).
    """
    import matplotlib.pyplot as plt

    if per_lift is None or len(per_lift) == 0:
        raise ValueError("per_lift is empty: no lift hold segments were given to residual_report")
    subjects = _by_subject(per_lift)
    muscles = list(dict.fromkeys(per_lift["muscle"]))
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(len(subjects), len(muscles), squeeze=False, facecolor="white",
                             figsize=(3.3 * len(muscles) + 2.4, 3.7 * len(subjects)))
    for row, (subj, df) in enumerate(subjects.items()):
        series = list(dict.fromkeys(df["series"]))
        for col, m in enumerate(muscles):
            ax = axes[row, col]
            dm = df[df["muscle"] == m]
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
            ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
            if dm["null_swap_res"].notna().any():
                ax.scatter(dm["true_res"], dm["null_swap_res"], s=12, color=COLOR_NULL, alpha=0.7,
                           linewidths=0, label="null (EEG of a different window)")
            parts = []
            for k, sid in enumerate(series):
                ds = dm[dm["series"] == sid]
                ax.scatter(ds["true_res"], ds["pred_res"], s=26, color=COLOR_FULL, alpha=0.85,
                           marker=markers[k % len(markers)], edgecolors="white", linewidths=0.4,
                           label=f"series {sid} (n={len(ds)})")
                parts.append(f"S{sid} {_r_text(ds['pred_res'], ds['true_res'])}")
            head = f"{subj} {m}".strip()
            ax.set_title(f"{head}: r = {_r_text(dm['pred_res'], dm['true_res'])}\n({', '.join(parts)})",
                         fontsize=9)
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel("predicted hold residual (z)")
            if row == len(subjects) - 1:
                ax.set_xlabel("true hold residual (z)")
        handles, labels = axes[row, 0].get_legend_handles_labels()
        axes[row, -1].legend(handles, labels, fontsize=8, frameon=False, loc="upper left",
                             bbox_to_anchor=(1.02, 1.0))      # right of the row, outside the panels
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
