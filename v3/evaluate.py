"""Evaluation of the EEG-only student, and the criteria a paper can rest on.

The question is not "does the guide raise r" but "does any EEG model explain EMG beyond the task clock?". Matching
a model that knows only the task event times is the null result, not the goal. The criteria (`criteria`):
  beats_control   the guided student beats the scale-0 control (seed-averaged, Wilcoxon over participants, Holm)
  beyond_timing   the student explains part of the residual of a causal timing model (residual_over_timing)
  survives_swap   the student loses accuracy when its EEG comes from another, cue-aligned trial (v3.nulls), and
                  the guide's gain shows in that loss too
  timing_level    information only: the student's mean r is within a margin of the `phase` timing model
Two analyses sit in their own modules: v3.contrast (B vs F, EEG history over a causal motion + task-event model) and
v3.templates (does the EEG predict how each trial deviates from the event-locked average EMG?).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eeg_controls.common import segment_labels, wilcoxon_p
from eeg_controls.phase_ridge import MIN_TOKENS, PHASES
from eeg_controls.phase_ridge import _fit as _ridge_fit
from eeg_controls.run_horizon import null_shifts
from v2.data import from_model_domain, to_model_domain
from v2.evaluate import _r, block_shuffle, block_swap, standard_metrics, swap_p
from v2.tokens import FS_IN, Q, to_samples

from .infer import normalise_eeg, predict_eeg_outputs, predict_eeg_series
from .nulls import trial_bounds

FS_TOK = FS_IN / Q
OUTCOMES = {"a": "the guide helps and the gain is beyond timing",
            "b": "the guide helps, but only through timing (the gain is gone under the swap or over the timing model)",
            "c": "no help", "d": "worse than the control", "control": "control (scale 0)"}


def evaluate_student(model, test_tokens, norm, cfg, device, names) -> dict:
    """Standard metrics of E on the test series plus the shuffle and swap nulls on its EEG input.

    `standard` has one row per muscle and rate: 500 (the prediction upsampled by cubic spline and scored against the
    original 500 Hz envelope, as v2) and 50 (no upsampling: the model's tokens against the envelope at the token rate).
    """
    n_null = int(cfg["evaluation"]["n_null"])
    if n_null < 1:
        raise ValueError(f"evaluation.n_null must be at least 1, got {n_null}")
    win, hop = cfg["tokens"]["window"], cfg["tokens"]["hop"]
    kw = dict(window=win, hop=hop, use_amp=bool(cfg["training"]["use_amp"]), clip=cfg["eeg"]["clip"])
    tok_pred = [from_model_domain(predict_eeg_outputs(model, tok, norm, device, **kw)["E"], norm["emg_scale"])
                for tok in test_tokens]
    preds = {tok["series"] if tok.get("series") is not None else i: np.maximum(to_samples(p, tok["n_samples"]), 0.0)
             for i, (tok, p) in enumerate(zip(test_tokens, tok_pred))}
    target = np.concatenate([tok["emg_full"] for tok in test_tokens])
    pred = np.concatenate(list(preds.values()))
    standard = pd.concat([standard_metrics(pred, target, names).assign(rate=500),
                          standard_metrics(np.maximum(np.concatenate(tok_pred), 0.0),
                                           np.concatenate([tok["emg"] for tok in test_tokens]), names).assign(rate=50)],
                         ignore_index=True)

    rng = np.random.default_rng(cfg["evaluation"]["seed"])
    eeg_z = [normalise_eeg(t, norm, cfg["eeg"]["clip"]) for t in test_tokens]
    lens = np.cumsum([0] + [len(e) for e in eeg_z])
    all_eeg = np.concatenate(eeg_z)
    n_blocks = len(all_eeg) // win
    r_e = _r(pred, target)
    null = {"shuffle": [], "swap": []}
    kinds = (("shuffle", block_shuffle), ("swap", block_swap))
    if n_blocks < 2:                                         # the swap cannot change anything
        kinds = kinds[:1]
        null["swap"] = np.full((n_null, len(names)), np.nan)
    for _ in range(n_null):
        for kind, fn in kinds:
            alt = fn(all_eeg, win, rng)
            parts = [predict_eeg_series(model, tok, norm, device, eeg=alt[lens[i]:lens[i + 1]], **kw)
                     for i, tok in enumerate(test_tokens)]
            null[kind].append(_r(np.concatenate(parts), target))
    null = {k: np.asarray(v) for k, v in null.items()}
    nulls = pd.DataFrame({
        "muscle": list(names), "E_r": r_e,
        "E_r_shuffle_mean": null["shuffle"].mean(0), "E_r_shuffle_sd": null["shuffle"].std(0),
        "E_r_swap_mean": null["swap"].mean(0), "E_r_swap_sd": null["swap"].std(0),
        "E_p_swap": [swap_p(r_e[c], null["swap"][:, c], n_blocks) for c in range(len(names))]})
    return {"standard": standard, "nulls": nulls, "predictions": preds}


def _mean_r_vaf(df: pd.DataFrame, by) -> pd.DataFrame:
    return df.groupby(by)[["r", "vaf"]].mean().reset_index()


def subject_means(v3_standard: pd.DataFrame, v2_standard_csv, phase_csv) -> pd.DataFrame:
    """One row per (subject, scale): mean over muscles of r and vaf for E_guided, and the references.

    v3_standard: columns muscle, r, vaf, subject, scale, and optionally seed and rate (only rate 500 is used, the
    scoring of the references). With seeds, E_guided_* is the mean over seeds of the per-seed muscle means and
    E_guided_*_seed_sd their standard deviation (NaN for one seed). v2_standard_csv: results/v2/standard.csv
    (column `output`). phase_csv: eeg_controls/results/phase/phase_metrics.csv (column `variant`).
    """
    std = v3_standard
    if "rate" in std:
        std = std[std["rate"] == 500]
    if "seed" not in std:
        std = std.assign(seed=0)
    per_seed = _mean_r_vaf(std, ["subject", "scale", "seed"])
    g = per_seed.groupby(["subject", "scale"])
    out = g[["r", "vaf"]].mean().rename(columns={"r": "E_guided_r", "vaf": "E_guided_vaf"})
    sd = g[["r", "vaf"]].std(ddof=1).rename(columns={"r": "E_guided_r_seed_sd", "vaf": "E_guided_vaf_seed_sd"})
    out = out.join(sd).join(g.size().rename("n_seeds")).reset_index()
    v2 = pd.read_csv(v2_standard_csv)
    ph = pd.read_csv(phase_csv)
    for label, df, col, key in (("E_saved", v2, "output", "E"), ("K", v2, "output", "K"),
                                ("phase", ph, "variant", "phase"), ("cue", ph, "variant", "cue")):
        ref = _mean_r_vaf(df[df[col] == key], "subject").rename(columns={"r": f"{label}_r", "vaf": f"{label}_vaf"})
        out = out.merge(ref, on="subject", how="left")
    return out.sort_values(["scale", "subject"]).reset_index(drop=True)


def group_tests(means: pd.DataFrame, control_scale: float = 0.0, close_margin: float = 0.05) -> pd.DataFrame:
    """One row per guided scale: group means and one-sided Wilcoxon tests over participants (descriptive table).

    Its flags (beats_E, close_to_timing, beats_timing) are the first version's and are kept for the old tables;
    the criteria of the paper are in `criteria`, where reaching the timing model is information only.
    """
    ctrl = means[means.scale == control_scale].set_index("subject")
    rows = []
    for scale in sorted(s for s in means.scale.unique() if s != control_scale):
        cur = means[means.scale == scale].set_index("subject")
        subj = cur.index.intersection(ctrl.index)
        g, c, e = cur.loc[subj, "E_guided_r"], ctrl.loc[subj, "E_guided_r"], cur.loc[subj, "E_saved_r"]
        p_ctrl = float(wilcoxon(g.values, c.values, alternative="greater").pvalue)
        p_saved = float(wilcoxon(g.values, e.values, alternative="greater").pvalue)
        mean_g, phase = float(g.mean()), float(cur.loc[subj, "phase_r"].mean())
        rows.append({
            "scale": scale, "n_subjects": len(subj), "mean_r": mean_g, "mean_control_r": float(c.mean()),
            "mean_E_saved_r": float(e.mean()), "mean_phase_r": phase, "mean_K_r": float(cur.loc[subj, "K_r"].mean()),
            "p_vs_control": p_ctrl, "p_vs_E_saved": p_saved, "n_better_than_control": int((g > c).sum()),
            "beats_E": bool(mean_g > c.mean() and mean_g > e.mean() and p_ctrl < 0.05 and p_saved < 0.05),
            "close_to_timing": bool(mean_g >= phase - close_margin), "beats_timing": bool(mean_g > phase)})
    return pd.DataFrame(rows)


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN stays NaN and is not counted)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def _student_over_timing(model, test_tokens, timing_u, lifts_by_series, norm, cfg, device) -> dict:
    """E, final tokens x, target y and timing t (model domain, token rate), phase labels and trial ids, concatenated."""
    tc = cfg["tokens"]
    kw = dict(window=tc["window"], hop=tc["hop"], use_amp=bool(cfg["training"]["use_amp"]), clip=cfg["eeg"]["clip"])
    e, x, y, t, lab, trial = [], [], [], [], [], []
    n_trials = 0
    for tok, tu, lifts in zip(test_tokens, timing_u, lifts_by_series):
        out = predict_eeg_outputs(model, tok, norm, device, **kw)
        n = len(out["E"])
        if len(tu) != n:
            raise ValueError(f"timing prediction of series {tok.get('series')} has {len(tu)} tokens, the series {n}")
        if out["fin"] is None:
            raise ValueError("the student returns no final tokens ('fin')")
        e.append(out["E"]); x.append(out["fin"]); t.append(np.asarray(tu, np.float64))
        y.append(to_model_domain(np.asarray(tok["emg"], np.float32), norm["emg_scale"]).astype(np.float64))
        lab.append(segment_labels(lifts, n, FS_TOK))
        ids = np.full(n, n_trials)
        for k, (a, b, _) in enumerate(trial_bounds(lifts, n)):
            ids[a:b] = n_trials + k
        n_trials = int(ids.max()) + 1
        trial.append(ids)
    e, x, y, t, lab, trial = (np.concatenate(v) for v in (e, x, y, t, lab, trial))
    return {"E": e, "x": x, "y": y, "t": t, "labels": lab, "trial": trial, "n_trials": n_trials}


def _trial_folds(trial, n_trials: int, folds: int) -> np.ndarray:
    """Fold of every token: `folds` groups of consecutive trials."""
    return (np.asarray(trial) * min(int(folds), n_trials)) // max(n_trials, 1)


def _cv_ridge(x, target, fold) -> np.ndarray:
    """Held-out ridge prediction of `target` from `x` (eeg_controls.phase_ridge._fit), one fit per fold."""
    pred = np.zeros_like(target, dtype=np.float64)
    for f in np.unique(fold):
        held = fold == f
        if (~held).sum() >= MIN_TOKENS:
            pred[held] = _ridge_fit(x[~held], target[~held])(x[held])
    return pred


def residual_over_timing(model, test_tokens, timing_u, lifts_by_series, norm, cfg, device, names,
                         n_null=None, folds=None) -> pd.DataFrame:
    """How much of the residual of a timing model (y - y_timing) does the student explain? One row per task phase.

    Model domain u at the token rate. `timing_u`: one (n_tok, 5) array per test series, the timing model's u
    (v3.timing, horizon 0). Columns:
      r_direct   correlation of E - y_timing with y - y_timing (mean over muscles)
      r_ridge    correlation of a ridge prediction of the residual from the student's final tokens with the residual;
                 the ridge is cross-validated over trials (`folds` groups of consecutive trials, fit on the others).
                 Each fold's intercept is the mean residual of the other folds, so with few trials per fold an
                 uninformative feature set gives a slightly negative r_ridge and skill: the bias is against the student
      skill      1 - MSE(max(timing + ridge, 0)) / MSE(timing), as the v2 K-residual tests
      p          against the same prediction shifted in time by whole hops (eeg_controls.run_horizon.null_shifts)
    Phases: eeg_controls.common.SEGMENTS of the recorded events, "other" = rest, and "all".
    """
    ec, tc = cfg["evaluation"], cfg["tokens"]
    n_null = int(ec["n_null"] if n_null is None else n_null)
    folds = int(ec.get("residual_folds", 5) if folds is None else folds)
    d = _student_over_timing(model, test_tokens, timing_u, lifts_by_series, norm, cfg, device)
    e, x, y, t, lab, n_trials = d["E"], d["x"], d["y"], d["t"], d["labels"], d["n_trials"]
    res = y - t
    fold = _trial_folds(d["trial"], n_trials, folds)
    pred = _cv_ridge(x, res, fold)
    mse = lambda q: ((np.maximum(t + q, 0.0) - y) ** 2).mean(1)
    e_t, e_p = ((t - y) ** 2).mean(1), mse(pred)
    shifts = null_shifts(len(res), tc["window"], tc["hop"])
    rng = np.random.default_rng(ec["seed"])
    e_null = [mse(np.roll(pred, int(rng.choice(shifts)), axis=0)) for _ in range(n_null)] if len(shifts) else []
    rows = []
    for ph in PHASES + ("all",):
        m = np.ones(len(res), bool) if ph == "all" else (lab == ph)
        row = {"segment": ph, "n_test": int(m.sum()), "n_trials": n_trials, "r_direct": np.nan, "r_ridge": np.nan,
               "skill": np.nan, "skill_null_mean": np.nan, "p": np.nan}
        if m.sum() >= MIN_TOKENS:
            sk = lambda err: float(1.0 - err[m].mean() / max(e_t[m].mean(), 1e-12))
            skill, null = sk(e_p), [sk(err) for err in e_null]
            row.update(r_direct=float(np.mean(_r(e[m] - t[m], res[m]))), r_ridge=float(np.mean(_r(pred[m], res[m]))),
                       skill=skill, skill_null_mean=float(np.mean(null)) if null else np.nan,
                       p=float((1 + sum(v >= skill for v in null)) / (1 + len(null))) if null else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def residual_group_test(residual: pd.DataFrame, control_scale: float = 0.0) -> pd.DataFrame:
    """Per timing model, phase and scale over participants (mean over seeds first): skill > 0, and skill above the
    control's. p_holm_phases corrects p_skill over the phases (not "all") of one timing model and scale."""
    keys = ["timing", "segment", "scale"]
    per = residual.groupby(keys + ["subject"], as_index=False)[["skill", "r_ridge", "r_direct"]].mean()
    rows = []
    for (timing, seg, scale), d in per.groupby(keys):
        d = d.set_index("subject")
        c = per[(per.timing == timing) & (per.segment == seg) & (per.scale == control_scale)].set_index("subject").skill
        gain = (d.skill - c.reindex(d.index)).dropna() if scale != control_scale else pd.Series(dtype=float)
        rows.append({"timing": timing, "segment": seg, "scale": scale, "n_subjects": len(d),
                     "mean_skill": d.skill.mean(), "sem_skill": d.skill.sem(), "mean_r_ridge": d.r_ridge.mean(),
                     "mean_r_direct": d.r_direct.mean(), "p_skill": wilcoxon_p(d.skill.values),
                     "mean_skill_gain": gain.mean() if len(gain) else np.nan,
                     "p_skill_gain": wilcoxon_p(gain.values) if len(gain) else np.nan})
    out = pd.DataFrame(rows)
    out["p_holm_phases"] = np.nan
    for _, idx in out[out.segment != "all"].groupby(["timing", "scale"]).groups.items():
        out.loc[idx, "p_holm_phases"] = holm(out.loc[idx, "p_skill"])
    return out


def criteria(means: pd.DataFrame, swap_tests: pd.DataFrame, residual_tests: pd.DataFrame, primary: str = "cue_causal",
             control_scale: float = 0.0, close_margin: float = 0.05, alpha: float = 0.05) -> pd.DataFrame:
    """The flags and the outcome per guided scale; every p-value is Holm-corrected over the guided scales.

    means: subject_means (seed-averaged); swap_tests: v3.nulls.swap_group_test; residual_tests: residual_group_test.
    Outcome: (a) beats_control, survives_swap, beyond_timing and a residual skill above the control's;
    (b) beats_control without the rest; (d) significantly below the control; (c) otherwise.
    timing_level (mean r within close_margin of the phase model) is information only: matching a model that knows
    only the task clock is the null result, not the goal.
    """
    ctrl = means[means.scale == control_scale].set_index("subject").E_guided_r
    res = residual_tests[(residual_tests.timing == primary) & (residual_tests.segment == "all")].set_index("scale")
    sw = swap_tests.set_index("scale")
    get = lambda df, col, scale: float(df[col].get(scale, np.nan)) if col in df else np.nan
    rows = []
    for scale in sorted(s for s in means.scale.unique() if s != control_scale):
        cur = means[means.scale == scale].set_index("subject")
        d = (cur.E_guided_r - ctrl.reindex(cur.index)).dropna()
        rows.append({"scale": scale, "n_subjects": len(d), "mean_r": cur.E_guided_r.mean(),
                     "mean_r_seed_sd": cur["E_guided_r_seed_sd"].mean() if "E_guided_r_seed_sd" in cur else np.nan,
                     "mean_control_r": ctrl.mean(), "mean_gain_r": d.mean(),
                     "p_gain": wilcoxon_p(d.values), "p_worse": wilcoxon_p(d.values, "less"),
                     "mean_phase_r": cur.phase_r.mean(), "mean_K_r": cur.K_r.mean(),
                     "residual_skill": get(res, "mean_skill", scale), "p_residual": get(res, "p_skill", scale),
                     "residual_skill_gain": get(res, "mean_skill_gain", scale),
                     "p_residual_gain": get(res, "p_skill_gain", scale),
                     "mean_delta": get(sw, "mean_delta", scale), "p_delta": get(sw, "p_delta", scale),
                     "mean_delta_gain": get(sw, "mean_delta_gain", scale), "p_delta_gain": get(sw, "p_delta_gain", scale)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("p_gain", "p_worse", "p_residual", "p_residual_gain", "p_delta", "p_delta_gain"):
        out[col + "_holm"] = holm(out[col])
    sig = lambda col: out[col + "_holm"] < alpha
    out["beats_control"] = (out.mean_gain_r > 0) & sig("p_gain")
    out["worse"] = (out.mean_gain_r < 0) & sig("p_worse")
    out["beyond_timing"] = (out.residual_skill > 0) & sig("p_residual")
    out["survives_swap"] = (out.mean_delta > 0) & sig("p_delta") & (out.mean_delta_gain > 0) & sig("p_delta_gain")
    out["residual_gain"] = (out.residual_skill_gain > 0) & sig("p_residual_gain")
    out["timing_level"] = out.mean_r >= out.mean_phase_r - close_margin          # information only, not a goal
    a = out.beats_control & out.beyond_timing & out.survives_swap & out.residual_gain
    out["outcome"] = np.where(a, "a", np.where(out.beats_control, "b", np.where(out.worse, "d", "c")))
    out["verdict"] = out.outcome.map(OUTCOMES)
    return out
