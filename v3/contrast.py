"""Critical experiment II, B vs F: what does EEG history add beyond a causal motion + task-event model?

Conditions (proposal names), all causal: features up to input token t, target EMG at t + h, horizons 0, 200, 500 and
1000 ms, from the causal v2 horizon models (eeg_controls.run_horizon, checkpoints outputs/checkpoints_v2_horizon/h{h}ms/P{p}):
  causal_cue  past LED onsets only: v3.timing.causal_features(..., "cue_causal") -> ridge to the target (reference)
  K           the kinematic stream of the horizon model: past hand kinematics and force (reference)
  B           K + ridge(past task events -> residual of K); the primary no-EEG baseline. The events are
              eeg_controls.phase_ridge.past_event_features (bumps after the LED cue and every past task event),
              phase_ridge's `events` feature set.
  F           K + ridge(EEG history and past task events -> residual of K); phase_ridge's `both` feature set.
B and F are one model family with the same inputs except the EEG, the same target domain (model domain u at the token
rate), the same held-out test series and the same calibration: every ridge (causal_cue, B, F) is fitted on the
validation series (K was trained on the training series), pooled over all phases, and every prediction is clipped at 0.
Per-phase fits are not used: they read the phase at the target time and are not causal.

Metrics per participant, horizon, phase and EEG source (score_b_vs_f):
  eeg_skill   1 - MSE(F) / MSE(B), primary; p against a time-shift null that moves only the EEG term of F by whole
              hops (the ridge is linear: F = intercept + EEG term + event term), so the event term stays aligned
  vaf_gain    VAF(F) - VAF(B) in VAF points; ccc_B, ccc_F (both mean over muscles, model domain, token rate)
  mse_B, mse_F, mse_K, mse_causal_cue   absolute MSE (per token, mean over muscles)

EEG sources of F (the real one and G-style negative controls):
  eeg         the EEG tokens of the trained causal model (the primary F)
  phase_swap  the same, with the phase-matched swap (v3.nulls.phase_matched_swap): the donor of the same weight and
              surface is time-warped so all its task events land on the recipient's; the primary trial-specific control
  swap        the same tokens computed from cue-aligned swapped EEG (v3.nulls.swap_series, donors with the same weight
              and surface by default); validation and test series are each swapped among their own trials and the ridge
              is fitted on the swapped EEG: same task timing, EEG of another trial
  bandpower   trailing 1 s band power of all channels (v2.residual.eeg_features, no learned weights)
  peripheral  the same band power on the `peripheral` preset channels only; compare it with `bandpower`

Run from the repository root (no training; it reads the horizon checkpoints):
    env/Scripts/python.exe -m v3.contrast --participants all --horizons-ms 0,200,500,1000 \
        --checkpoints outputs/checkpoints_v2_horizon --out results/v3
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
from v2.evaluate import standard_metrics
from v2.model import build_model
from v2.residual import _apply, eeg_features, fit_ridge
from v2.train import load_best

from .config import CONFIG_V3, PERIPHERAL
from .evaluate import holm
from .nulls import phase_matched_swap, swap_series
from .timing import causal_features

SOURCES = ("eeg", "swap", "phase_swap", "bandpower", "peripheral")
COLUMNS = ["segment", "n_fit", "n_test", "eeg_skill", "eeg_skill_null_mean", "p", "vaf_gain", "ccc_B", "ccc_F",
           "mse_B", "mse_F", "mse_K", "mse_causal_cue"]
METRICS = ["eeg_skill", "vaf_gain", "ccc_B", "ccc_F", "mse_B", "mse_F", "mse_K", "mse_causal_cue"]


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


def b_vs_f_scores(eeg_val, ev_val, res_val, eeg_test, ev_test, res_test, lab_test, k_test, y_test, n_null: int,
                  seed: int, cue_val=None, y_val=None, cue_test=None, window: int = CONFIG_V3["tokens"]["window"],
                  hop: int = CONFIG_V3["tokens"]["hop"]) -> pd.DataFrame:
    """The B vs F metrics (module docstring), one row per phase and "all"; arrays at the token rate, model domain.

    res = y - K. B: pooled ridge from the events, F: from EEG and events, both fitted on the validation tokens to the
    residual of K and applied as max(K + ridge, 0). The causal_cue reference (cue_val, y_val, cue_test given) is a
    ridge from the past-LED features to y, fitted on the same validation tokens. Phase labels (target time) only group
    the test tokens. A phase with fewer than 10 test tokens gives a row of NaN.
    """
    lab_test = np.asarray(lab_test, object)
    res_val, res_test = np.asarray(res_val, np.float64), np.asarray(res_test, np.float64)
    k_test, y_test = np.asarray(k_test, np.float64), np.asarray(y_test, np.float64)
    eeg_val, eeg_test = np.asarray(eeg_val, np.float64), np.asarray(eeg_test, np.float64)
    pred_b = np.maximum(k_test + _ridge_fit(ev_val, res_val)(ev_test), 0.0)
    corr_f, term = _fit_terms(np.concatenate([eeg_val, ev_val], 1), res_val, eeg_val.shape[1])(
        np.concatenate([eeg_test, ev_test], 1))
    pred_f = np.maximum(k_test + corr_f, 0.0)
    pred_cue = None
    if cue_val is not None:
        pred_cue = np.maximum(_ridge_fit(cue_val, np.asarray(y_val, np.float64))(cue_test), 0.0)
    err = lambda pred: ((pred - y_test) ** 2).mean(1)                        # per token, mean over muscles
    e_k, e_b, e_f = err(k_test), err(pred_b), err(pred_f)
    e_cue = err(pred_cue) if pred_cue is not None else np.full(len(y_test), np.nan)
    shifts = null_shifts(len(res_test), window, hop)
    rng = np.random.default_rng(seed)
    e_null = [err(np.maximum(k_test + corr_f - term + np.roll(term, int(rng.choice(shifts)), axis=0), 0.0))
              for _ in range(n_null)] if len(shifts) else []
    names = [str(c) for c in range(y_test.shape[1])]
    rows = []
    for ph in PHASES + ("all",):
        m = np.ones(len(res_test), bool) if ph == "all" else (lab_test == ph)
        row = dict.fromkeys(COLUMNS, np.nan)
        row.update(segment=ph, n_fit=len(res_val), n_test=int(m.sum()))
        if m.sum() >= MIN_TOKENS and len(res_val) >= MIN_TOKENS:
            sk = lambda e: float(1.0 - e[m].mean() / max(e_b[m].mean(), 1e-12))
            skill, null = sk(e_f), [sk(e) for e in e_null]
            mb, mf = standard_metrics(pred_b[m], y_test[m], names), standard_metrics(pred_f[m], y_test[m], names)
            row.update(eeg_skill=skill, eeg_skill_null_mean=float(np.mean(null)) if null else np.nan,
                       p=float((1 + sum(v >= skill for v in null)) / (1 + len(null))) if null else np.nan,
                       vaf_gain=float(mf.vaf.mean() - mb.vaf.mean()), ccc_B=float(mb.ccc.mean()),
                       ccc_F=float(mf.ccc.mean()), mse_B=float(e_b[m].mean()), mse_F=float(e_f[m].mean()),
                       mse_K=float(e_k[m].mean()), mse_causal_cue=float(e_cue[m].mean()))
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


def eeg_sources(model, toks_val, toks_test, lifts_val, lifts_test, norm, cfg, device, sources=SOURCES, rng=None,
                match_condition: bool = True) -> dict:
    """source -> (validation features, test features) at the input tokens, for b_vs_f_scores.

    swap: validation and test EEG are swapped among their own trials (cue-aligned, v3.nulls.swap_series; with
    match_condition donors have the same weight and surface), then passed through the same model. peripheral: band
    power columns of v3.config.PERIPHERAL.
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
                raw = [t["eeg"] for t in toks]
                alt = (phase_matched_swap(raw, lifts, rng)[0] if s == "phase_swap"
                       else swap_series(raw, lifts, rng, match_condition=match_condition)[0])
                swapped.append(_features(model, toks, norm, cfg, device, eeg=alt)["tokens"])
            out[s] = tuple(swapped)
    return out


def _cue_features(c) -> np.ndarray:
    """causal_cue features of the collected series; input token t is series token t."""
    return np.concatenate([causal_features(l, n, "cue_causal") for l, n in zip(c["lifts"], c["n_tok"])])


def score_b_vs_f(model, sd, cfg, device, participant: int, horizon_ms: int, n_null: int = 99, sources=SOURCES,
                 match_condition: bool = True) -> pd.DataFrame:
    """b_vs_f_scores for every EEG source, for one trained horizon model and its data (run_horizon.subject_data).

    Columns: subject, horizon_ms, source, then COLUMNS.
    """
    p = int(participant)
    v = collect(model, sd["tokens"]["val"], sd["norm"], cfg, device, p)
    t = collect(model, sd["tokens"]["test"], sd["norm"], cfg, device, p)
    ev_v, ev_t = event_features(v), event_features(t)
    cue_v, cue_t = _cue_features(v), _cue_features(t)
    feats = eeg_sources(model, sd["tokens"]["val"], sd["tokens"]["test"], v["lifts"], t["lifts"], sd["norm"], cfg,
                        device, sources, match_condition=match_condition)
    tc, seed = cfg["tokens"], int(cfg["evaluation"]["seed"])
    out = []
    for name in sources:
        fv, ft = feats[name]
        df = b_vs_f_scores(fv, ev_v, v["y"] - v["K"], ft, ev_t, t["y"] - t["K"], t["labels"], t["K"], t["y"],
                           n_null=n_null, seed=seed, cue_val=cue_v, y_val=v["y"], cue_test=cue_t,
                           window=tc["window"], hop=tc["hop"])
        out.append(df.assign(source=name))
    df = pd.concat(out, ignore_index=True).assign(subject=f"P{p}", horizon_ms=int(horizon_ms))
    return df[["subject", "horizon_ms", "source"] + COLUMNS]


def run_b_vs_f(participant: int, horizon_ms: int, ckpt_root, n_null: int = 99, sources=SOURCES,
               match_condition: bool = True) -> pd.DataFrame:
    """score_b_vs_f from the causal horizon checkpoint ckpt_root/h{h}ms/P{p} (as phase_ridge.run_participant)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = horizon_config(horizon_ms)
    sd = subject_data(int(participant), cfg)
    model = build_model(cfg, sd["prior"]).to(device)
    load_best(model, Path(ckpt_root) / f"h{int(horizon_ms)}ms" / f"P{int(participant)}", device)
    return score_b_vs_f(model, sd, cfg, device, participant, horizon_ms, n_null=n_null, sources=sources,
                        match_condition=match_condition)


def summarise_b_vs_f(df: pd.DataFrame, primary: str = "eeg") -> pd.DataFrame:
    """Per horizon, phase and EEG source over participants.

    {metric}_mean and {metric}_sem for every metric in METRICS; n_sig (participants with p <= 0.05 and eeg_skill > 0);
    p_eeg_skill, the one-sided Wilcoxon p of eeg_skill > 0; for a control source also gap = eeg_skill(primary) -
    eeg_skill(source) and p_gap (> 0). Holm (v3.evaluate.holm), within one source: *_holm_phases over the six phases
    of one horizon, *_holm_horizons over the horizons of one phase ("all" included).
    """
    keys = ["horizon_ms", "segment", "source"]
    wide = df.pivot_table(index=["horizon_ms", "segment", "subject"], columns="source", values="eeg_skill",
                          dropna=False)
    rows = []
    for (h, seg, src), d in df.groupby(keys, sort=False):
        row = {"horizon_ms": h, "segment": seg, "source": src, "n_subjects": d.subject.nunique()}
        for col in METRICS:
            row.update({f"{col}_mean": d[col].mean(), f"{col}_sem": d[col].sem()})
        row.update(n_sig=int(((d.p <= 0.05) & (d.eeg_skill > 0)).sum()), p_eeg_skill=wilcoxon_p(d.eeg_skill.values),
                   gap_mean=np.nan, p_gap=np.nan)
        if src != primary and primary in wide.columns:
            gap = (wide[primary] - wide[src]).xs((h, seg), level=["horizon_ms", "segment"]).dropna()
            row.update(gap_mean=gap.mean() if len(gap) else np.nan, p_gap=wilcoxon_p(gap.values))
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ("p_eeg_skill", "p_gap"):
        out[col + "_holm_phases"] = np.nan
        for _, idx in out[out.segment != "all"].groupby(["source", "horizon_ms"]).groups.items():
            out.loc[idx, col + "_holm_phases"] = holm(out.loc[idx, col])
        out[col + "_holm_horizons"] = np.nan
        for _, idx in out.groupby(["source", "segment"]).groups.items():
            out.loc[idx, col + "_holm_horizons"] = holm(out.loc[idx, col])
    return out.sort_values(keys, kind="stable").reset_index(drop=True)


def main(argv=None):
    cc = CONFIG_V3["b_vs_f"]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--horizons-ms", default=",".join(str(h) for h in cc["horizons_ms"]))
    ap.add_argument("--checkpoints", default=str(ROOT / cc["checkpoint_dir"]))
    ap.add_argument("--out", default=str(ROOT / cc["results_dir"]))
    ap.add_argument("--sources", default=",".join(cc["sources"]))
    ap.add_argument("--n-null", type=int, default=int(CONFIG_V3["evaluation"]["n_null"]))
    ap.add_argument("--any-condition-swap", action="store_true", help="swap donors from any weight and surface")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out, sources = Path(a.out), tuple(s.strip() for s in a.sources.split(","))
    (out / "b_vs_f_runs").mkdir(parents=True, exist_ok=True)
    tables = []
    for h in (int(x) for x in a.horizons_ms.split(",")):
        for p in parse_subjects(a.participants):
            f = out / "b_vs_f_runs" / f"h{h}ms_P{p}.csv"
            if f.exists() and not a.overwrite:
                df = pd.read_csv(f)
            else:
                df = run_b_vs_f(p, h, a.checkpoints, n_null=a.n_null, sources=sources,
                                match_condition=not a.any_condition_swap)
                df.to_csv(f, index=False)
            s = df[df.segment.isin(["pre_move", "reach", "all"])].pivot_table(index="source", columns="segment",
                                                                              values="eeg_skill")
            print(f"P{p} h={h} ms eeg_skill = 1 - MSE(F) / MSE(B):\n{s.round(4).to_string()}", flush=True)
            tables.append(df)
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out / "b_vs_f.csv", index=False)
    summ = summarise_b_vs_f(df)
    summ.to_csv(out / "b_vs_f_summary.csv", index=False)
    show = summ[summ.segment == "all"][["horizon_ms", "source", "n_subjects", "eeg_skill_mean", "eeg_skill_sem",
                                        "p_eeg_skill", "p_eeg_skill_holm_horizons", "vaf_gain_mean"]]
    print("\nall tokens, mean over participants:")
    print(show.round(4).to_string(index=False))
    return summ


if __name__ == "__main__":
    main()
