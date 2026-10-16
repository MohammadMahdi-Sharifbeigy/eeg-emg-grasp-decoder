"""EEG controls, phase 1, part R: per-phase ridge and a causal timing baseline.

Question: in the causal horizon models (h = 0, 200, 500 ms), does the EEG predict what the kinematic
stream K misses, beyond what the timing of the task alone predicts, and does that differ per task phase?

Residual of K in the model domain at the token rate, fitted on the validation series and scored on the
test series, as eeg_controls.run_horizon.segments. Feature sets at input token t (all causal):
  eeg     the EEG tokens of the trained E stream
  events  past-event bumps: for each of the 8 task events that has happened at or before t, Gaussian bumps
          at lags 0 .. +3 s after it (step 0.25 s, width 0.25 s); a bump is exactly 0 before its event, so
          nothing depends on an event later than t. The causal timing baseline.
  both    eeg and events together
Fits:
  pooled     one ridge over all phases
  per_phase  one ridge per task phase (pre_move, reach, load, hold, release, other) on the validation tokens
             of that phase only (a phase with fewer than 10 of them gives a NaN row). Phase labels come from
             the recorded events at the target time; they are an analysis split, not a model input.
Scores per phase: r with the residual (mean over muscles), skill = 1 - MSE(K + pred) / MSE(K) (prediction
applied as max(K + pred, 0)), and a time-shift null (the prediction shifted circularly by whole hops).
The ridge does not extrapolate: features nearly constant on the fit tokens are dropped and new features are
clipped to the fit range (see _fit).

Output (default eeg_controls/results/phase1/phase_ridge/): phase_ridge.csv (one row per participant,
horizon, feature set, fit and phase) and summary.csv (per horizon, phase, feature set and fit: mean skill,
number of participants with p <= 0.05 and skill > 0, and the Wilcoxon p of skill(both) - skill(events) > 0).
Each participant and horizon is also kept in runs/ and reused unless --overwrite.

Run from the repository root (no training, about 3 min per participant and horizon):
    env/Scripts/python.exe -m eeg_controls.phase_ridge --participants all --horizons-ms 0,200,500 \
        --checkpoints outputs/checkpoints_v2_horizon --out eeg_controls/results/phase1/phase_ridge
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v2.config import CONFIG_V2
from v2.evaluate import _r
from v2.model import build_model
from v2.residual import _apply, fit_ridge
from v2.tokens import FS_IN, Q
from v2.train import load_best

from .common import EVENTS, ROOT, SEGMENTS, parse_subjects, wilcoxon_p
from .run_horizon import collect, horizon_config, null_shifts, subject_data

FS_TOK = FS_IN / Q                                                   # 50 tokens per second
LAGS = np.arange(0.0, 3.0 + 1e-9, 0.25)                              # s after the event
WIDTH = 0.25                                                         # s
PHASES = tuple(s[0] for s in SEGMENTS) + ("other",)
FEATURES = ("eeg", "events", "both")
FITS = ("pooled", "per_phase")
MIN_TOKENS = 10
MIN_STD = 1e-3                                                       # feature SD below this on the fit tokens: dropped
COLUMNS = ["segment", "fit", "n_fit", "n_test", "r", "skill", "skill_null_mean", "p"]


def past_event_features(lift_list, n_tok: int, offset_tok: int = 0) -> np.ndarray:
    """(n_tok, 8 * 13) bumps after each past task event; row t is series token t + offset_tok.

    Row t depends only on events at sample <= (t + offset_tok) * Q (500 Hz samples of the series): every
    bump is exactly 0 before its event, so the lag-0 bump peaks at the event token and has no value before it.
    Missing events (None) are skipped; several lifts add up.
    """
    tok = np.arange(int(n_tok)) + int(offset_tok)
    t = tok / FS_TOK
    out = np.zeros((int(n_tok), len(EVENTS), len(LAGS)))
    for j, ev in enumerate(EVENTS):
        for lift in lift_list:
            s = lift.get(ev)
            if s is None:
                continue
            past = s <= tok * Q
            dt = t[past] - s / FS_IN
            out[past, j] += np.exp(-0.5 * ((dt[:, None] - LAGS[None, :]) / WIDTH) ** 2)
    return out.reshape(int(n_tok), -1)


def _fit(x, y):
    """Ridge (alpha by cross-validation) on the columns of x that vary; returns a predictor.

    No extrapolation: columns with a standard deviation below MIN_STD on the fit tokens are dropped, and
    new features are clipped to the range seen in the fit. Without this a past-event bump that barely
    touches one phase's validation tokens (standardised by a tiny SD) can blow the test prediction up.
    """
    x = np.asarray(x, np.float64)
    keep = x.std(0) > MIN_STD
    if not keep.any():
        mean = np.asarray(y, np.float64).mean(0)
        return lambda z: np.broadcast_to(mean, (len(z), len(mean)))
    lo, hi = x[:, keep].min(0), x[:, keep].max(0)
    ridge, _ = fit_ridge(x[:, keep], y)
    return lambda z: _apply(ridge, np.clip(np.asarray(z, np.float64)[:, keep], lo, hi))


def fit_and_score(feats_val, res_val, lab_val, feats_test, res_test, lab_test, k_test, y_test, fit: str,
                  n_null: int, seed: int, window: int = CONFIG_V2["tokens"]["window"],
                  hop: int = CONFIG_V2["tokens"]["hop"]) -> pd.DataFrame:
    """One row per phase (and "all") for one feature set and one fit ("pooled" or "per_phase").

    Model domain, token rate. With "per_phase", test tokens of a phase without a fit (fewer than 10
    validation tokens) are predicted as 0 (K alone) in the "all" row, and that phase's row is NaN. A phase
    with fewer than 10 test tokens is NaN too. The null shifts the whole test prediction by whole hops.
    """
    if fit not in FITS:
        raise ValueError(f"unknown fit {fit!r}; choose from {FITS}")
    lab_val, lab_test = np.asarray(lab_val, object), np.asarray(lab_test, object)
    res_val, res_test = np.asarray(res_val, np.float64), np.asarray(res_test, np.float64)
    k_test, y_test = np.asarray(k_test, np.float64), np.asarray(y_test, np.float64)
    pred = np.zeros_like(res_test)
    n_fit = {}
    if fit == "pooled":
        pred[:] = _fit(feats_val, res_val)(feats_test)
        n_fit = {ph: len(res_val) for ph in PHASES + ("all",)}
    else:
        for ph in PHASES:
            mv, mt = lab_val == ph, lab_test == ph
            n_fit[ph] = int(mv.sum())
            if n_fit[ph] >= MIN_TOKENS and mt.any():
                pred[mt] = _fit(feats_val[mv], res_val[mv])(feats_test[mt])
        n_fit["all"] = int(sum(v for v in n_fit.values() if v >= MIN_TOKENS))
    mse = lambda q: ((np.maximum(k_test + q, 0.0) - y_test) ** 2).mean(1)    # per token, mean over muscles
    e_k, e_p = ((k_test - y_test) ** 2).mean(1), mse(pred)
    shifts = null_shifts(len(res_test), window, hop)
    rng = np.random.default_rng(seed)
    e_null = [mse(np.roll(pred, int(rng.choice(shifts)), axis=0)) for _ in range(n_null)] if len(shifts) else []
    rows = []
    for ph in PHASES + ("all",):
        m = np.ones(len(res_test), bool) if ph == "all" else (lab_test == ph)
        row = {"segment": ph, "fit": fit, "n_fit": n_fit[ph], "n_test": int(m.sum()),
               "r": np.nan, "skill": np.nan, "skill_null_mean": np.nan, "p": np.nan}
        if m.sum() >= MIN_TOKENS and n_fit[ph] >= MIN_TOKENS:
            sk = lambda e: float(1.0 - e[m].mean() / max(e_k[m].mean(), 1e-12))
            skill, null = sk(e_p), [sk(e) for e in e_null]
            row.update(r=float(np.mean(_r(pred[m], res_test[m]))), skill=skill,
                       skill_null_mean=float(np.mean(null)) if null else np.nan,
                       p=float((1 + sum(v >= skill for v in null)) / (1 + len(null))) if null else np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def _events(c) -> np.ndarray:
    """Past-event features of the collected series; input token t is series token t (offset 0)."""
    return np.concatenate([past_event_features(l, n, 0) for l, n in zip(c["lifts"], c["n_tok"])])


def run_participant(p: int, h_ms: int, ckpt_root: Path, n_null: int = 99) -> pd.DataFrame:
    """All feature sets and fits for one participant and horizon, from the causal horizon checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = horizon_config(h_ms)
    sd = subject_data(p, cfg)
    model = build_model(cfg, sd["prior"]).to(device)
    load_best(model, Path(ckpt_root) / f"h{int(h_ms)}ms" / f"P{p}", device)
    return score_model(model, sd, cfg, device, p, h_ms, n_null=n_null)


def score_model(model, sd, cfg, device, p: int, h_ms: int, n_null: int = 99) -> pd.DataFrame:
    """All feature sets and fits for one trained model and its participant data (run_horizon.subject_data)."""
    v = collect(model, sd["tokens"]["val"], sd["norm"], cfg, device, p)
    t = collect(model, sd["tokens"]["test"], sd["norm"], cfg, device, p)
    ev_v, ev_t = _events(v), _events(t)
    feats = {"eeg": (v["x"], t["x"]), "events": (ev_v, ev_t),
             "both": (np.concatenate([v["x"], ev_v], 1), np.concatenate([t["x"], ev_t], 1))}
    tc, seed = cfg["tokens"], int(cfg["evaluation"]["seed"])
    out = []
    for name in FEATURES:
        fv, ft = feats[name]
        for fit in FITS:
            df = fit_and_score(fv, v["y"] - v["K"], v["labels"], ft, t["y"] - t["K"], t["labels"], t["K"], t["y"],
                               fit=fit, n_null=n_null, seed=seed, window=tc["window"], hop=tc["hop"])
            out.append(df.assign(features=name))
    df = pd.concat(out, ignore_index=True).assign(subject=f"P{p}", horizon_ms=int(h_ms))
    return df[["subject", "horizon_ms", "features"] + COLUMNS]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per horizon, phase, feature set and fit: mean skill, n significant, Wilcoxon p of both - events > 0."""
    keys = ["horizon_ms", "segment", "features", "fit"]
    g = df.groupby(keys, sort=False)
    out = g.agg(n=("subject", "nunique"), skill_mean=("skill", "mean"), r_mean=("r", "mean")).reset_index()
    out["n_sig"] = g.apply(lambda d: int(((d.p <= 0.05) & (d.skill > 0)).sum()), include_groups=False).to_numpy()
    wide = df.pivot_table(index=["horizon_ms", "segment", "fit", "subject"], columns="features", values="skill",
                          dropna=False)
    if {"both", "events"} <= set(wide.columns):
        gain = (wide["both"] - wide["events"]).groupby(level=["horizon_ms", "segment", "fit"])
        test = pd.DataFrame({"gain_both_minus_events": gain.mean(), "p_both_gt_events": gain.apply(wilcoxon_p)})
        out = out.merge(test.reset_index(), on=["horizon_ms", "segment", "fit"], how="left")
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--horizons-ms", default="0,200,500")
    ap.add_argument("--checkpoints", default=str(ROOT / "outputs" / "checkpoints_v2_horizon"))
    ap.add_argument("--out", default=str(ROOT / "eeg_controls" / "results" / "phase1" / "phase_ridge"))
    ap.add_argument("--n-null", type=int, default=int(CONFIG_V2["evaluation"]["n_null"]))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    tables = []
    for h in (int(x) for x in a.horizons_ms.split(",")):
        for p in parse_subjects(a.participants):
            f = out / "runs" / f"h{h}ms_P{p}.csv"
            if f.exists() and not a.overwrite:
                df = pd.read_csv(f)
            else:
                df = run_participant(p, h, Path(a.checkpoints), n_null=a.n_null)
                df.to_csv(f, index=False)
            s = df[df.segment.isin(["pre_move", "reach", "all"])].pivot_table(
                index="features", columns=["fit", "segment"], values="skill")
            print(f"P{p} h={h} ms skill:\n{s.round(3).to_string()}", flush=True)
            tables.append(df)
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out / "phase_ridge.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / "summary.csv", index=False)
    print("\nmean over participants:")
    print(summ.round(3).to_string(index=False))
    return summ


if __name__ == "__main__":
    main()
