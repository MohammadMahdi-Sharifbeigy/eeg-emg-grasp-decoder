"""Evaluation of the EEG-only student at 500 Hz, and its comparison with E, K and the timing baseline."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from v2.evaluate import _r, block_shuffle, block_swap, standard_metrics, swap_p

from .infer import normalise_eeg, predict_eeg_series


def evaluate_student(model, test_tokens, norm, cfg, device, names) -> dict:
    """Standard metrics of E on the test series plus the shuffle and swap nulls on its EEG input."""
    n_null = int(cfg["evaluation"]["n_null"])
    if n_null < 1:
        raise ValueError(f"evaluation.n_null must be at least 1, got {n_null}")
    win, hop = cfg["tokens"]["window"], cfg["tokens"]["hop"]
    kw = dict(window=win, hop=hop, use_amp=bool(cfg["training"]["use_amp"]), clip=cfg["eeg"]["clip"])
    preds = {tok["series"] if tok.get("series") is not None else i: predict_eeg_series(model, tok, norm, device, **kw)
             for i, tok in enumerate(test_tokens)}
    target = np.concatenate([tok["emg_full"] for tok in test_tokens])
    pred = np.concatenate(list(preds.values()))
    standard = standard_metrics(pred, target, names)

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

    v3_standard: columns muscle, r, vaf, subject, scale. v2_standard_csv: results/v2/standard.csv (column `output`).
    phase_csv: eeg_controls/results/phase/phase_metrics.csv (column `variant`).
    """
    out = _mean_r_vaf(v3_standard, ["subject", "scale"]).rename(columns={"r": "E_guided_r", "vaf": "E_guided_vaf"})
    v2 = pd.read_csv(v2_standard_csv)
    ph = pd.read_csv(phase_csv)
    for label, df, col, key in (("E_saved", v2, "output", "E"), ("K", v2, "output", "K"),
                                ("phase", ph, "variant", "phase"), ("cue", ph, "variant", "cue")):
        ref = _mean_r_vaf(df[df[col] == key], "subject").rename(columns={"r": f"{label}_r", "vaf": f"{label}_vaf"})
        out = out.merge(ref, on="subject", how="left")
    return out.sort_values(["scale", "subject"]).reset_index(drop=True)


def group_tests(means: pd.DataFrame, control_scale: float = 0.0, close_margin: float = 0.05) -> pd.DataFrame:
    """One row per guided scale: group means, one-sided Wilcoxon tests over participants, and the success flags."""
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
