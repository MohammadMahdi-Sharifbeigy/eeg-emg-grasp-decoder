"""Template deviation test: does the EEG predict how each trial's EMG deviates from the event-locked average trial?

A model that knows the task events can predict the average EMG of every task moment. What it cannot predict is how one
trial differs from that average: a stronger grip, an earlier burst, a different muscle balance. If the EEG carries
motor information beyond the task clock, it should predict these deviations.

Template (fit_emg_templates, per participant and muscle, TRAINING series only). Every token of a trial (one LED onset
to the next, v3.nulls.trial_bounds) gets a bin of event-warped time (template_bins):
  between two consecutive task events (eeg_controls.common.EVENTS, led_on .. hand_stop; 7 intervals), the time is
  warped piecewise-linearly and cut into n_bins equal bins, so a bin is "the same fraction of the way from first_touch
  to lift_off" in every trial whatever its duration;
  after hand_stop: bins of post_bin_s seconds up to post_s seconds; later tokens and tokens before the first LED of a
  series share one "rest" bin;
  a trial with a missing event or events out of order gets no bin (-1): it is left out of the template and the test.
The template of a bin is the mean model-domain envelope u = sqrt(y / s_c) of its training tokens (bins never seen in
training take the mean of all binned training tokens). The deviation of a test token is u - template[bin].

The template reads the event times of the whole trial, future ones included, as the `phase` timing model does: it is a
description of the average trial that is subtracted from the target, not a competitor, and the EEG never sees it.
Because of the warping, the deviation also excludes the trial's own event timing, so what is left is amplitude and shape.

Test (template_deviation_test): a ridge from EEG features to the deviation of the five muscles, cross-validated over
trials (v3.evaluate._cv_ridge: `folds` groups of consecutive trials, fit on the others). Per phase and muscle:
skill = 1 - MSE(ridge) / MSE(fold mean) (the baseline is each fold's training mean, so a mean offset is no skill),
r, and p against the prediction shifted in time by whole hops (run_horizon.null_shifts). Row "MEAN" is the mean over
muscles. Sources (student_template_deviation): the student's final tokens, the same tokens from cue-aligned swapped EEG
(v3.nulls.swap_series, donors with the same weight and surface by default), the trailing 1 s band power of all
channels, and of the `peripheral` preset channels only (dropped when the student does not see them).
"""
import warnings

import numpy as np
import pandas as pd

from eeg_controls.common import EVENTS, segment_labels, wilcoxon_p
from eeg_controls.phase_ridge import MIN_TOKENS, PHASES
from eeg_controls.run_horizon import null_shifts
from v2.data import to_model_domain
from v2.evaluate import _r
from v2.tokens import FS_IN, Q

from .config import CONFIG_V3, PERIPHERAL
from .contrast import channel_columns
from .evaluate import _cv_ridge, _trial_folds, holm
from .infer import normalise_eeg, predict_eeg_outputs
from .nulls import swap_series, trial_bounds

FS_TOK = FS_IN / Q
SOURCES = ("student", "swap", "bandpower", "peripheral")
COLUMNS = ["segment", "muscle", "n_test", "skill", "r", "skill_null_mean", "p"]
TC = CONFIG_V3["templates"]


def n_template_bins(n_bins: int = TC["n_bins"], post_s: float = TC["post_s"],
                    post_bin_s: float = TC["post_bin_s"]) -> int:
    """Total number of bins: 7 event intervals x n_bins, the post-hand_stop bins, and the rest bin (the last)."""
    return (len(EVENTS) - 1) * int(n_bins) + int(round(post_s / post_bin_s)) + 1


def template_bins(lift_list, n_tok: int, n_bins: int = TC["n_bins"], post_s: float = TC["post_s"],
                  post_bin_s: float = TC["post_bin_s"]) -> np.ndarray:
    """(n_tok,) bin of event-warped time per token (module docstring); -1 for trials with missing or unordered events."""
    n_tok, n_bins = int(n_tok), int(n_bins)
    n_post = int(round(post_s / post_bin_s))
    rest = (len(EVENTS) - 1) * n_bins + n_post
    out = np.full(n_tok, rest, dtype=int)
    led_lift = {int(l["led_on"]) // Q: l for l in lift_list if l.get("led_on") is not None}
    for a, b, led in trial_bounds(lift_list, n_tok):
        lift = led_lift.get(led, {})
        ev = [lift.get(e) for e in EVENTS]
        if any(e is None for e in ev) or np.any(np.diff(ev) < 0):
            out[a:b] = -1
            continue
        ev = np.asarray(ev, np.float64) / Q                                   # token positions
        pos = np.arange(a, b, dtype=np.float64)
        inside = (pos >= ev[0]) & (pos < ev[-1])
        k = np.searchsorted(ev, pos[inside], side="right") - 1               # zero-length intervals are never chosen
        frac = (pos[inside] - ev[k]) / (ev[k + 1] - ev[k])
        out[a:b][inside] = k * n_bins + np.minimum((frac * n_bins).astype(int), n_bins - 1)
        after = pos >= ev[-1]
        dt = (pos[after] - ev[-1]) / FS_TOK
        out[a:b][after] = np.where(dt < post_s, (len(EVENTS) - 1) * n_bins + (dt / post_bin_s).astype(int), rest)
    return out


def fit_emg_templates(train_tokens, train_lifts, norm, n_bins: int = TC["n_bins"], post_s: float = TC["post_s"],
                      post_bin_s: float = TC["post_bin_s"]) -> dict:
    """Event-locked average envelope per bin and muscle from the training series (model domain u).

    Returns template (n_bins_total, 5), count (training tokens per bin) and the binning settings.
    """
    n_total = n_template_bins(n_bins, post_s, post_bin_s)
    sums, count = None, np.zeros(n_total)
    for tok, lifts in zip(train_tokens, train_lifts):
        u = to_model_domain(np.asarray(tok["emg"], np.float32), norm["emg_scale"]).astype(np.float64)
        bins = template_bins(lifts, len(u), n_bins, post_s, post_bin_s)
        ok = bins >= 0
        if sums is None:
            sums = np.zeros((n_total, u.shape[1]))
        np.add.at(sums, bins[ok], u[ok])
        count += np.bincount(bins[ok], minlength=n_total)
    if sums is None or count.sum() == 0:
        raise ValueError("no training token has a template bin: every trial lacks an event")
    template = np.where(count[:, None] > 0, sums / np.maximum(count, 1)[:, None], sums.sum(0) / count.sum())
    return {"template": template, "count": count, "n_bins": int(n_bins), "post_s": float(post_s),
            "post_bin_s": float(post_bin_s)}


def template_deviation(templates: dict, tok, lifts, norm):
    """(deviation u - template (n_tok, 5) with NaN rows where a token has no bin, bins) of one series."""
    u = to_model_domain(np.asarray(tok["emg"], np.float32), norm["emg_scale"]).astype(np.float64)
    bins = template_bins(lifts, len(u), templates["n_bins"], templates["post_s"], templates["post_bin_s"])
    dev = np.full_like(u, np.nan)
    dev[bins >= 0] = u[bins >= 0] - templates["template"][bins[bins >= 0]]
    return dev, bins


def template_deviation_test(feats, deviation, labels, trial, names, n_null: int, seed: int, folds: int = 5,
                            window: int = CONFIG_V3["tokens"]["window"],
                            hop: int = CONFIG_V3["tokens"]["hop"]) -> pd.DataFrame:
    """Cross-validated ridge from `feats` to `deviation`, scored per phase and muscle (module docstring).

    All arrays cover the tokens to test only (tokens without a template bin removed). `trial` holds consecutive
    trial ids 0 .. n_trials - 1. One row per phase ("all" last) and muscle, plus "MEAN" per phase.
    """
    x, dev = np.asarray(feats, np.float64), np.asarray(deviation, np.float64)
    labels, trial = np.asarray(labels, object), np.asarray(trial)
    n_trials = int(trial.max()) + 1 if len(trial) else 0
    fold = _trial_folds(trial, n_trials, folds)
    pred = _cv_ridge(x, dev, fold)
    base = np.zeros_like(dev)
    for f in np.unique(fold):
        held = fold == f
        base[held] = dev[~held].mean(0) if (~held).any() else 0.0
    e_base = (base - dev) ** 2
    shifts = null_shifts(len(dev), window, hop)
    rng = np.random.default_rng(seed)
    e_null = [(np.roll(pred, int(rng.choice(shifts)), axis=0) - dev) ** 2 for _ in range(n_null)] if len(shifts) else []
    e_pred = (pred - dev) ** 2
    rows = []
    for ph in PHASES + ("all",):
        m = np.ones(len(dev), bool) if ph == "all" else (labels == ph)
        ok = m.sum() >= MIN_TOKENS
        if ok:
            eb = np.maximum(e_base[m].mean(0), 1e-12)
            sk = 1.0 - e_pred[m].mean(0) / eb
            null = np.asarray([1.0 - e[m].mean(0) / eb for e in e_null]).reshape(len(e_null), len(names))
            r = _r(pred[m], dev[m])
        for c, mus in enumerate(list(names) + ["MEAN"]):
            row = {"segment": ph, "muscle": mus, "n_test": int(m.sum()), "skill": np.nan, "r": np.nan,
                   "skill_null_mean": np.nan, "p": np.nan}
            if ok:
                v, nv = (sk[c], null[:, c]) if mus != "MEAN" else (sk.mean(), null.mean(1))
                row.update(skill=float(v), r=float(r[c]) if mus != "MEAN" else float(np.mean(r)),
                           skill_null_mean=float(nv.mean()) if len(nv) else np.nan,
                           p=float((1 + (nv >= v).sum()) / (1 + len(nv))) if len(nv) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def trailing_band_power(eeg_z, smooth: int = 50) -> np.ndarray:
    """(n, C * B) mean of the normalised band power over the past `smooth` tokens (1 s), as v2.residual for a causal model."""
    n = len(eeg_z)
    flat = np.asarray(eeg_z, np.float64).reshape(n, -1)
    csum = np.concatenate([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    a, b = np.clip(np.arange(n) - smooth + 1, 0, n), np.arange(n) + 1
    return (csum[b] - csum[a]) / (b - a)[:, None]


def student_template_deviation(model, test_tokens, test_lifts, templates: dict, norm, cfg, device, names,
                               sources=SOURCES, n_null=None, folds=None, match_condition=None,
                               rng=None) -> pd.DataFrame:
    """template_deviation_test of the student on the test series, for every feasible EEG source.

    Columns: source, then COLUMNS, then var_template (share of the test envelope's variance, summed over muscles, that
    the template explains) and n_trials. `peripheral` is dropped with a warning when the student lacks those channels.
    """
    ec, tc = cfg["evaluation"], cfg["tokens"]
    n_null = int(ec["n_null"] if n_null is None else n_null)
    folds = int(ec.get("residual_folds", 5) if folds is None else folds)
    match_condition = bool(cfg.get("templates", TC).get("swap_match_condition", True)
                           if match_condition is None else match_condition)
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown EEG sources {sorted(unknown)}; choose from {SOURCES}")
    if "peripheral" in sources:
        try:
            cols = channel_columns(cfg, PERIPHERAL)
        except ValueError:
            warnings.warn("the student does not see the peripheral channels: the peripheral source is skipped")
            sources = tuple(s for s in sources if s != "peripheral")
    rng = np.random.default_rng(ec["seed"]) if rng is None else rng
    kw = dict(window=tc["window"], hop=tc["hop"], use_amp=bool(cfg["training"]["use_amp"]), clip=cfg["eeg"]["clip"])
    dev, lab, trial, keep, u = [], [], [], [], []
    n_trials = 0
    for tok, lifts in zip(test_tokens, test_lifts):
        d, bins = template_deviation(templates, tok, lifts, norm)
        n = len(d)
        ids = np.full(n, n_trials)
        for k, (a, b, _) in enumerate(trial_bounds(lifts, n)):
            ids[a:b] = n_trials + k
        n_trials = int(ids.max()) + 1
        dev.append(d); lab.append(segment_labels(lifts, n, FS_TOK)); trial.append(ids); keep.append(bins >= 0)
        u.append(to_model_domain(np.asarray(tok["emg"], np.float32), norm["emg_scale"]).astype(np.float64))
    keep = np.concatenate(keep)
    dev, lab, u = np.concatenate(dev)[keep], np.concatenate(lab)[keep], np.concatenate(u)[keep]
    trial = np.unique(np.concatenate(trial)[keep], return_inverse=True)[1]          # consecutive ids of kept trials
    var_template = float(1.0 - dev.var(0).sum() / max(u.var(0).sum(), 1e-12))
    eeg_z = [normalise_eeg(t, norm, cfg["eeg"]["clip"]) for t in test_tokens]
    feats = {}
    for s in sources:
        if s == "student":
            feats[s] = np.concatenate([predict_eeg_outputs(model, t, norm, device, **kw)["fin"] for t in test_tokens])
        elif s == "swap":
            alt, _ = swap_series(eeg_z, test_lifts, rng, match_condition=match_condition, clip=cfg["eeg"]["clip"])
            feats[s] = np.concatenate([predict_eeg_outputs(model, t, norm, device, eeg=a, **kw)["fin"]
                                       for t, a in zip(test_tokens, alt)])
        elif s == "bandpower":
            feats[s] = np.concatenate([trailing_band_power(e) for e in eeg_z])
        else:
            feats[s] = np.concatenate([trailing_band_power(e) for e in eeg_z])[:, cols]
    out = [template_deviation_test(feats[s][keep], dev, lab, trial, names, n_null, int(ec["seed"]), folds,
                                   tc["window"], tc["hop"]).assign(source=s) for s in sources]
    df = pd.concat(out, ignore_index=True).assign(var_template=var_template, n_trials=int(trial.max()) + 1)
    return df[["source"] + COLUMNS + ["var_template", "n_trials"]]


def summarise_template_deviation(df: pd.DataFrame, primary: str = "student") -> pd.DataFrame:
    """Per (scale,) source, phase and muscle over participants (mean over seeds first): mean and SEM of skill, the
    one-sided Wilcoxon p of skill > 0, and for a control source gap = skill(primary) - skill(source) with p_gap (> 0).

    Holm (v3.evaluate.holm) over the muscles (not "MEAN") and the six phases of one source (and scale), and separately
    over the muscles of the "all" rows.
    """
    by = (["scale"] if "scale" in df else []) + ["source", "segment", "muscle"]
    per = df.groupby(by + ["subject"], as_index=False)[["skill", "r"]].mean()
    wide = per.pivot_table(index=[c for c in by if c != "source"] + ["subject"], columns="source", values="skill")
    rows = []
    for key, d in per.groupby(by, sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(n_subjects=len(d), skill_mean=d.skill.mean(), skill_sem=d.skill.sem(), r_mean=d.r.mean(),
                   p_skill=wilcoxon_p(d.skill.values), gap_mean=np.nan, p_gap=np.nan)
        if row["source"] != primary and primary in wide.columns:
            sel = tuple(row[c] for c in by if c != "source")
            gap = (wide[primary] - wide[row["source"]]).xs(sel, level=[c for c in by if c != "source"]).dropna()
            row.update(gap_mean=gap.mean() if len(gap) else np.nan, p_gap=wilcoxon_p(gap.values))
        rows.append(row)
    out = pd.DataFrame(rows)
    fam = [c for c in ("scale", "source") if c in out]
    for col in ("p_skill", "p_gap"):
        out[col + "_holm"] = np.nan
        sub = out[out.muscle != "MEAN"]
        for _, idx in sub.groupby(fam + [sub.segment == "all"]).groups.items():
            out.loc[idx, col + "_holm"] = holm(out.loc[idx, col])
    return out.reset_index(drop=True)
