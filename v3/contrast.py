"""Rows C and F of the paper's model table: does EEG history add to a causal motion + cue model?

All causal, at horizons 0, 200, 500 and 1000 ms (input token t, target t + h), from the causal v2 horizon models
(eeg_controls.run_horizon, checkpoints outputs/checkpoints_v2_horizon/h{h}ms/P{p}):
  C   K + ridge(past task events -> residual of K). K is the kinematic stream (past hand kinematics and force), the
      events are eeg_controls.phase_ridge.past_event_features (bumps after the LED cue and every past task event).
      This is phase_ridge's `events` feature set.
  F   K + ridge(EEG history and past task events -> residual of K): phase_ridge's `both`.
The primary contrast is delta = skill(F) - skill(C) per participant, task phase and horizon, with
skill = 1 - MSE(max(K + ridge, 0)) / MSE(K) in the model domain at the token rate (as phase_ridge, pooled fit: fitted on
the validation series over all phases, scored per phase of the test series; per-phase fits read the phase at the target
time and are not causal, so they are not used here).

The time-shift null of delta: the ridge is linear, so F's prediction is intercept + EEG term + event term. The null
moves only the EEG term in time by whole hops (eeg_controls.run_horizon.null_shifts) and keeps the event term, so it
asks whether the EEG term is aligned with this test series' target, not whether F beats a shifted F.

EEG sources of F (G-style negative controls next to the real one):
  eeg         the EEG tokens of the trained causal model (the real F)
  swap        the same tokens computed from cue-aligned swapped EEG (v3.nulls.swap_series, validation and test series
              each swapped among their own trials): same task timing, EEG of another trial. A real F must beat it.
  bandpower   the trailing 1 s band power of all channels (v2.residual.eeg_features, no learned weights)
  peripheral  the same band power on the `peripheral` preset channels only (frontal pole, temporal, occipital: eye and
              muscle artefacts); compare it with `bandpower`.

Run from the repository root (no training; it reads the horizon checkpoints):
    env/Scripts/python.exe -m v3.contrast --participants all --horizons-ms 0,200,500,1000 \
        --checkpoints outputs/checkpoints_v2_horizon --out results/v3/contrast
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eeg_controls.common import ROOT, parse_subjects, wilcoxon_p
from eeg_controls.phase_ridge import MIN_STD, MIN_TOKENS, PHASES
from eeg_controls.phase_ridge import _events as event_features
from eeg_controls.phase_ridge import _fit as _ridge_fit
from eeg_controls.run_horizon import collect, horizon_config, null_shifts, subject_data
from v2.electrodes import channel_names
from v2.evaluate import _r
from v2.model import build_model
from v2.residual import _apply, eeg_features, fit_ridge
from v2.train import load_best

from .config import CONFIG_V3, PERIPHERAL
from .evaluate import holm
from .nulls import swap_series

SOURCES = ("eeg", "swap", "bandpower", "peripheral")
COLUMNS = ["segment", "n_fit", "n_test", "skill_C", "skill_F", "delta", "r_C", "r_F", "delta_null_mean", "p"]


def _fit_terms(x, y, n_eeg: int):
    """phase_ridge._fit (no extrapolation) on [EEG | events]; the predictor returns (prediction, EEG term).

    The EEG term is the part of the linear prediction carried by the first n_eeg columns, so
    prediction = intercept + EEG term + event term.
    """
    x = np.asarray(x, np.float64)
    keep = x.std(0) > MIN_STD
    if not keep.any():
        mean = np.asarray(y, np.float64).mean(0)
        return lambda z: (np.broadcast_to(mean, (len(z), len(mean))).copy(), np.zeros((len(z), len(mean))))
    lo, hi = x[:, keep].min(0), x[:, keep].max(0)
    ridge, _ = fit_ridge(x[:, keep], y)
    mu, sd, w, _ = ridge
    is_eeg = (np.arange(x.shape[1]) < int(n_eeg))[keep]

    def predict(z):
        zk = np.clip(np.asarray(z, np.float64)[:, keep], lo, hi)
        return _apply(ridge, zk), (((zk - mu) / sd)[:, is_eeg]) @ w[is_eeg]
    return predict


def contrast_scores(eeg_val, ev_val, res_val, eeg_test, ev_test, res_test, lab_test, k_test, y_test, n_null: int,
                    seed: int, window: int = CONFIG_V3["tokens"]["window"],
                    hop: int = CONFIG_V3["tokens"]["hop"]) -> pd.DataFrame:
    """skill(C), skill(F) and delta = skill(F) - skill(C), one row per phase and "all".

    Pooled ridges fitted on the validation tokens (C: events; F: EEG and events) to the residual of K, scored on the test
    tokens of every phase (labels at the target time, only to group tokens). p: delta against the null in which the
    EEG term of F is shifted circularly by whole hops (the event term stays). NaN rows where a phase has fewer than 10
    test tokens.
    """
    lab_test = np.asarray(lab_test, object)
    res_val, res_test = np.asarray(res_val, np.float64), np.asarray(res_test, np.float64)
    k_test, y_test = np.asarray(k_test, np.float64), np.asarray(y_test, np.float64)
    eeg_val, eeg_test = np.asarray(eeg_val, np.float64), np.asarray(eeg_test, np.float64)
    pred_c = _ridge_fit(ev_val, res_val)(ev_test)
    pred_f, term = _fit_terms(np.concatenate([eeg_val, ev_val], 1), res_val, eeg_val.shape[1])(
        np.concatenate([eeg_test, ev_test], 1))
    mse = lambda q: ((np.maximum(k_test + q, 0.0) - y_test) ** 2).mean(1)        # per token, mean over muscles
    e_k, e_c, e_f = ((k_test - y_test) ** 2).mean(1), mse(pred_c), mse(pred_f)
    shifts = null_shifts(len(res_test), window, hop)
    rng = np.random.default_rng(seed)
    e_null = [mse(pred_f - term + np.roll(term, int(rng.choice(shifts)), axis=0)) for _ in range(n_null)] \
        if len(shifts) else []
    rows = []
    for ph in PHASES + ("all",):
        m = np.ones(len(res_test), bool) if ph == "all" else (lab_test == ph)
        row = dict.fromkeys(COLUMNS, np.nan)
        row.update(segment=ph, n_fit=len(res_val), n_test=int(m.sum()))
        if m.sum() >= MIN_TOKENS and len(res_val) >= MIN_TOKENS:
            sk = lambda e: float(1.0 - e[m].mean() / max(e_k[m].mean(), 1e-12))
            s_c, s_f = sk(e_c), sk(e_f)
            null = [sk(e) - s_c for e in e_null]
            row.update(skill_C=s_c, skill_F=s_f, delta=s_f - s_c, r_C=float(np.mean(_r(pred_c[m], res_test[m]))),
                       r_F=float(np.mean(_r(pred_f[m], res_test[m]))),
                       delta_null_mean=float(np.mean(null)) if null else np.nan,
                       p=float((1 + sum(v >= s_f - s_c for v in null)) / (1 + len(null))) if null else np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def channel_columns(cfg, channels, n_bands: int = None) -> np.ndarray:
    """Columns of the flattened (channel, band) band power that belong to `channels` (channel-major, as v2.data)."""
    names = channel_names(cfg)
    n_bands = len(cfg["eeg"]["bands"]) if n_bands is None else int(n_bands)
    missing = [c for c in channels if c not in names]
    if missing:
        raise ValueError(f"channels {missing} are not among the model's electrodes {names}")
    idx = [names.index(c) for c in channels]
    return np.asarray([i * n_bands + b for i in idx for b in range(n_bands)], dtype=int)


def _features(model, toks, norm, cfg, device, eeg=None) -> dict:
    """EEG tokens and trailing band power of a list of series, concatenated (v2.residual.eeg_features).

    `eeg` replaces each series' raw (not normalised) EEG token array.
    """
    tc = cfg["tokens"]
    kw = dict(window=tc["window"], hop=tc["hop"], clip=cfg["eeg"]["clip"])
    parts = [eeg_features(model, t if eeg is None else dict(t, eeg=e), norm, device, **kw)
             for t, e in zip(toks, [None] * len(toks) if eeg is None else eeg)]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("tokens", "bandpower")}


def eeg_sources(model, toks_val, toks_test, lifts_val, lifts_test, norm, cfg, device, sources=SOURCES,
                rng=None) -> dict:
    """source -> (validation features, test features) at the input tokens, for contrast_scores.

    swap: validation and test EEG are swapped among their own trials (cue-aligned, v3.nulls.swap_series), then passed
    through the same model. peripheral: band power columns of v3.config.PERIPHERAL.
    """
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown EEG sources {sorted(unknown)}; choose from {SOURCES}")
    rng = np.random.default_rng(cfg["evaluation"]["seed"]) if rng is None else rng
    real = [_features(model, t, norm, cfg, device) for t in (toks_val, toks_test)]
    out = {}
    for s in sources:
        if s == "eeg":
            out[s] = (real[0]["tokens"], real[1]["tokens"])
        elif s == "bandpower":
            out[s] = (real[0]["bandpower"], real[1]["bandpower"])
        elif s == "peripheral":
            cols = channel_columns(cfg, PERIPHERAL)
            out[s] = (real[0]["bandpower"][:, cols], real[1]["bandpower"][:, cols])
        else:
            swapped = []
            for toks, lifts in ((toks_val, lifts_val), (toks_test, lifts_test)):
                alt, _ = swap_series([t["eeg"] for t in toks], lifts, rng)
                swapped.append(_features(model, toks, norm, cfg, device, eeg=alt)["tokens"])
            out[s] = tuple(swapped)
    return out


def score_contrast(model, sd, cfg, device, p: int, h_ms: int, n_null: int = 99, sources=SOURCES) -> pd.DataFrame:
    """All EEG sources for one trained horizon model and its participant data (run_horizon.subject_data)."""
    v = collect(model, sd["tokens"]["val"], sd["norm"], cfg, device, p)
    t = collect(model, sd["tokens"]["test"], sd["norm"], cfg, device, p)
    ev_v, ev_t = event_features(v), event_features(t)
    feats = eeg_sources(model, sd["tokens"]["val"], sd["tokens"]["test"], v["lifts"], t["lifts"], sd["norm"], cfg,
                        device, sources)
    tc, seed = cfg["tokens"], int(cfg["evaluation"]["seed"])
    out = []
    for name in sources:
        fv, ft = feats[name]
        df = contrast_scores(fv, ev_v, v["y"] - v["K"], ft, ev_t, t["y"] - t["K"], t["labels"], t["K"], t["y"],
                             n_null=n_null, seed=seed, window=tc["window"], hop=tc["hop"])
        out.append(df.assign(source=name))
    df = pd.concat(out, ignore_index=True).assign(subject=f"P{p}", horizon_ms=int(h_ms))
    return df[["subject", "horizon_ms", "source"] + COLUMNS]


def run_participant(p: int, h_ms: int, ckpt_root, n_null: int = 99, sources=SOURCES) -> pd.DataFrame:
    """score_contrast from the causal horizon checkpoint ckpt_root/h{h}ms/P{p} (as phase_ridge.run_participant)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = horizon_config(h_ms)
    sd = subject_data(p, cfg)
    model = build_model(cfg, sd["prior"]).to(device)
    load_best(model, Path(ckpt_root) / f"h{int(h_ms)}ms" / f"P{p}", device)
    return score_contrast(model, sd, cfg, device, p, h_ms, n_null=n_null, sources=sources)


def contrast_group_test(df: pd.DataFrame, primary: str = "eeg") -> pd.DataFrame:
    """Per horizon, phase and EEG source over participants: mean skill(C), skill(F) and delta, n significant, and the
    Wilcoxon p of delta > 0. For a control source also gap = delta(primary) - delta(source) and its p (> 0).

    Holm: p_delta_holm and p_gap_holm correct over phases and horizons within one source (the phase rows), and over
    horizons for the "all" rows.
    """
    keys = ["horizon_ms", "segment", "source"]
    rows = []
    wide = df.pivot_table(index=["horizon_ms", "segment", "subject"], columns="source", values="delta", dropna=False)
    for (h, seg, src), d in df.groupby(keys, sort=False):
        row = {"horizon_ms": h, "segment": seg, "source": src, "n_subjects": d.subject.nunique(),
               "mean_skill_C": d.skill_C.mean(), "mean_skill_F": d.skill_F.mean(), "mean_delta": d.delta.mean(),
               "sem_delta": d.delta.sem(), "n_sig": int(((d.p <= 0.05) & (d.delta > 0)).sum()),
               "p_delta": wilcoxon_p(d.delta.values), "mean_gap": np.nan, "p_gap": np.nan}
        if src != primary and primary in wide.columns:
            gap = (wide[primary] - wide[src]).xs((h, seg), level=["horizon_ms", "segment"]).dropna()
            row.update(mean_gap=gap.mean() if len(gap) else np.nan, p_gap=wilcoxon_p(gap.values))
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ("p_delta", "p_gap"):
        out[col + "_holm"] = np.nan
        for _, idx in out.groupby([out.source, out.segment == "all"]).groups.items():
            out.loc[idx, col + "_holm"] = holm(out.loc[idx, col])
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


def main(argv=None):
    cc = CONFIG_V3["contrast"]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--horizons-ms", default=",".join(str(h) for h in cc["horizons_ms"]))
    ap.add_argument("--checkpoints", default=str(ROOT / cc["checkpoint_dir"]))
    ap.add_argument("--out", default=str(ROOT / cc["results_dir"]))
    ap.add_argument("--sources", default=",".join(cc["sources"]))
    ap.add_argument("--n-null", type=int, default=int(CONFIG_V3["evaluation"]["n_null"]))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out, sources = Path(a.out), tuple(s.strip() for s in a.sources.split(","))
    (out / "runs").mkdir(parents=True, exist_ok=True)
    tables = []
    for h in (int(x) for x in a.horizons_ms.split(",")):
        for p in parse_subjects(a.participants):
            f = out / "runs" / f"h{h}ms_P{p}.csv"
            if f.exists() and not a.overwrite:
                df = pd.read_csv(f)
            else:
                df = run_participant(p, h, a.checkpoints, n_null=a.n_null, sources=sources)
                df.to_csv(f, index=False)
            s = df[df.segment.isin(["pre_move", "reach", "all"])].pivot_table(index="source", columns="segment",
                                                                              values="delta")
            print(f"P{p} h={h} ms delta = skill(F) - skill(C):\n{s.round(4).to_string()}", flush=True)
            tables.append(df)
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out / "contrast.csv", index=False)
    summ = contrast_group_test(df)
    summ.to_csv(out / "summary.csv", index=False)
    print("\nmean over participants:")
    print(summ.round(4).to_string(index=False))
    return summ


if __name__ == "__main__":
    main()
