"""The verdict of Critical experiment II: what does EEG add beyond timing?

Primary question, per horizon: does F (K + past events + EEG) beat B (K + past events)?
  F_beats_B         eeg_skill = 1 - MSE(F) / MSE(B) above 0 over all tokens, Holm over the horizons, p < 0.05
  F_survives_swap   F's skill with the real EEG is above F's skill with phase-matched swapped EEG (gap > 0, Holm over
                    horizons), and the swapped EEG alone gives no significant skill: the gain is trial-specific
  F_not_peripheral  F from peripheral-channel band power does not do the job of all-channel band power: its skill
                    is not significant (Holm over horizons) or it is below the all-channel skill; unknown (NaN) fails
A positive claim needs all three. F_beats_B without the other two reads "task-locked or artifact-related".
Secondary question: does the kinematic guide help the EEG-only student (Eguided against E0, seed-averaged, Holm over
the guided scales)? The interpretation matrix of the proposal combines the two:
  F beats B (with controls)   Eguided beats E0   interpretation
  yes                         yes                EEG adds information beyond motion and timing; the guide also helps
  yes                         no                 EEG adds information beyond motion and timing; the guide does not help
  no                          yes                the guide taught a better movement template (representation learning),
                                                 not evidence of muscle-specific commands
  no                          no                 no evidence that EEG adds information beyond motion and timing
A pooled correlation alone is never enough: every row also carries the absolute MSE of each condition, the VAF gain,
CCC and the median onset error of B and F.
"""
import numpy as np
import pandas as pd

from eeg_controls.common import wilcoxon_p

from .evaluate import holm

ALPHA = 0.05
INTERPRETATION = {
    (True, True): "EEG adds information beyond motion and timing; the kinematic guide also helps the EEG-only model",
    (True, False): "EEG adds information beyond motion and timing; the kinematic guide does not help",
    (False, True): "the kinematic guide taught a better movement template (representation learning), "
                   "not evidence of muscle-specific commands",
    (False, False): "no evidence that EEG adds information beyond motion and timing",
}
TASK_LOCKED = "F beats B, but the gain is task-locked or artifact-related (it fails the swap or the peripheral control)"


def eguided_vs_e0(means: pd.DataFrame, control_scale: float = 0.0) -> pd.DataFrame:
    """Per guided scale: E0 and Eguided mean r (v3.evaluate.subject_means, seed-averaged), the gain, its one-sided
    Wilcoxon p over participants, Holm over the guided scales, and Eguided_beats_E0."""
    e0 = means[means.scale == control_scale].set_index("subject").student_r
    rows = []
    for scale in sorted(s for s in means.scale.unique() if s != control_scale):
        cur = means[means.scale == scale].set_index("subject").student_r
        d = (cur - e0.reindex(cur.index)).dropna()
        rows.append({"scale": scale, "n_subjects": len(d), "E0_r": e0.mean(), "Eguided_r": cur.mean(),
                     "gain_r": d.mean(), "p_gain": wilcoxon_p(d.values)})
    out = pd.DataFrame(rows, columns=["scale", "n_subjects", "E0_r", "Eguided_r", "gain_r", "p_gain"])
    out["p_gain_holm"] = holm(out.p_gain) if len(out) else []
    out["Eguided_beats_E0"] = (out.gain_r > 0) & (out.p_gain_holm < ALPHA)
    return out


def _row(summary: pd.DataFrame, horizon, source: str, segment: str = "all") -> pd.Series:
    sel = summary[(summary.horizon_ms == horizon) & (summary.source == source) & (summary.segment == segment)]
    return sel.iloc[0] if len(sel) else pd.Series(dtype=float)


def _onset(onset: pd.DataFrame, horizon, condition: str) -> float:
    if onset is None or onset.empty:
        return np.nan
    e = onset[(onset.horizon_ms == horizon) & (onset.condition == condition)].error_ms
    return float(np.nanmedian(np.abs(e))) if e.notna().any() else np.nan


def proposal_verdict(b_vs_f_summary: pd.DataFrame, student: pd.DataFrame = None, onset: pd.DataFrame = None,
                     template_summary: pd.DataFrame = None, primary: str = "eeg") -> pd.DataFrame:
    """One row per horizon: the flags, the interpretation and the metrics behind them.

    b_vs_f_summary: v3.contrast.summarise_b_vs_f. student: eguided_vs_e0 (None: the secondary question is open).
    onset: rows of v3.contrast.onset_b_vs_f (subject, horizon_ms, condition, ..., error_ms). template_summary:
    v3.templates.summarise_template_deviation, used for E0 (scale 0) and the best guided scale.
    """
    eguided = bool(student.Eguided_beats_E0.any()) if student is not None and len(student) else None
    best_scale = (student.sort_values("gain_r", ascending=False).scale.iloc[0]
                  if student is not None and len(student) else np.nan)
    rows = []
    for h in sorted(b_vs_f_summary.horizon_ms.unique()):
        f, ps = _row(b_vs_f_summary, h, primary), _row(b_vs_f_summary, h, "phase_swap")
        bp, per = _row(b_vs_f_summary, h, "bandpower"), _row(b_vs_f_summary, h, "peripheral")
        get = lambda r, c: float(r[c]) if c in r and pd.notna(r[c]) else np.nan
        beats = get(f, "eeg_skill_mean") > 0 and get(f, "p_eeg_skill_holm_horizons") < ALPHA
        survives = (get(ps, "gap_mean") > 0 and get(ps, "p_gap_holm_horizons") < ALPHA
                    and not get(ps, "p_eeg_skill_holm_horizons") < ALPHA)
        per_skill, bp_skill = get(per, "eeg_skill_mean"), get(bp, "eeg_skill_mean")
        not_peripheral = bool(np.isfinite(per_skill) and np.isfinite(bp_skill)
                              and (not get(per, "p_eeg_skill_holm_horizons") < ALPHA or per_skill < bp_skill))
        neural = bool(beats and survives and not_peripheral)
        if beats and not neural:
            text = TASK_LOCKED
        elif eguided is None:
            text = INTERPRETATION[(neural, False)].split(";")[0] + "; the guide has not been evaluated yet"
        else:
            text = INTERPRETATION[(neural, eguided)]
        rows.append({
            "horizon_ms": h, "n_subjects": get(f, "n_subjects"),
            "eeg_skill": get(f, "eeg_skill_mean"), "eeg_skill_sem": get(f, "eeg_skill_sem"),
            "p_eeg_skill_holm": get(f, "p_eeg_skill_holm_horizons"),
            "phase_swap_skill": get(ps, "eeg_skill_mean"), "swap_gap": get(ps, "gap_mean"),
            "p_swap_gap_holm": get(ps, "p_gap_holm_horizons"),
            "bandpower_skill": get(bp, "eeg_skill_mean"), "peripheral_skill": get(per, "eeg_skill_mean"),
            "vaf_gain": get(f, "vaf_gain_mean"), "ccc_B": get(f, "ccc_B_mean"), "ccc_F": get(f, "ccc_F_mean"),
            "mse_causal_cue": get(f, "mse_causal_cue_mean"), "mse_K": get(f, "mse_K_mean"),
            "mse_B": get(f, "mse_B_mean"), "mse_F": get(f, "mse_F_mean"),
            "onset_abs_ms_B": _onset(onset, h, "B"), "onset_abs_ms_F": _onset(onset, h, "F"),
            "F_beats_B": bool(beats), "F_survives_swap": bool(survives), "F_not_peripheral": not_peripheral,
            "Eguided_beats_E0": eguided, "best_guided_scale": best_scale,
            "templates_E0": _template_flag(template_summary, 0.0),
            "templates_Eguided": _template_flag(template_summary, best_scale),
            "interpretation": text})
    return pd.DataFrame(rows)


def _template_flag(summary: pd.DataFrame, scale):
    """Does the student's EEG predict deviations from the EMG template in some muscle (all tokens, Holm over muscles)?"""
    if summary is None or summary.empty or (isinstance(scale, float) and np.isnan(scale)):
        return None
    s = summary[(summary.source == "student") & (summary.segment == "all") & (summary.muscle != "MEAN")]
    if "scale" in s:
        s = s[s.scale == scale]
    if s.empty:
        return None
    return bool(((s.skill_mean > 0) & (s.p_skill_holm < ALPHA)).any())
