"""Onset timing: when does each muscle switch on in each lift, and how far off is a prediction?

The proposal asks for onset timing next to MSE, VAF and CCC, because a model can score well on the pooled envelope
while switching muscles on at the wrong time. Rule (fixed before any result was seen):
  baseline  median of the envelope in [hand_start - 1 s, hand_start) (led_on when hand_start is missing)
  peak      maximum in [led_on, led_on + 3 s)
  onset     first sample in that search window where the envelope reaches baseline + 20 % of (peak - baseline)
A lift without an LED onset is skipped; a muscle that never rises above its baseline in a lift gets NaN, never 0.
The same rule is applied to the prediction (its own baseline and peak) and to the target.
Event times are 500 Hz samples of the series (eeg_controls.common.lifts); `fs` is the rate of the envelope.
"""
import numpy as np
import pandas as pd

FRAC = 0.2
BASELINE_S = (-1.0, 0.0)
SEARCH_S = (0.0, 3.0)


def onset_times(env, lifts, fs: float = 500.0, frac: float = FRAC, baseline_s=BASELINE_S, search_s=SEARCH_S,
                names=None) -> pd.DataFrame:
    """One row per lift and muscle: lift, muscle, onset_s (seconds after the LED onset, NaN when there is none)."""
    env = np.asarray(env, dtype=np.float64)
    n, c = env.shape
    names = list(names) if names is not None else list(range(c))
    scale = fs / 500.0
    rows = []
    for k, lift in enumerate(lifts):
        if lift.get("led_on") is None:
            continue
        led = int(round(lift["led_on"] * scale))
        anchor = lift.get("hand_start")
        anchor = led if anchor is None else int(round(anchor * scale))
        b0, b1 = max(anchor + int(round(baseline_s[0] * fs)), 0), max(anchor + int(round(baseline_s[1] * fs)), 0)
        s0, s1 = min(led + int(round(search_s[0] * fs)), n), min(led + int(round(search_s[1] * fs)), n)
        for m in range(c):
            onset = np.nan
            if b1 > b0 and s1 > s0:
                base = float(np.median(env[b0:b1, m]))
                seg = env[s0:s1, m]
                peak = float(seg.max())
                if peak - base > 1e-12 * max(abs(peak), 1.0):
                    hit = np.flatnonzero(seg >= base + frac * (peak - base))
                    if len(hit):
                        onset = (s0 + hit[0] - led) / fs
            rows.append({"lift": lift.get("lift", k), "muscle": names[m], "onset_s": onset})
    return pd.DataFrame(rows, columns=["lift", "muscle", "onset_s"])


def onset_error(pred, y, lifts, fs: float = 500.0, names=None, **kw) -> pd.DataFrame:
    """Per lift and muscle: onset_true_s, onset_pred_s and error_ms = (predicted - true onset) in milliseconds."""
    t = onset_times(y, lifts, fs, names=names, **kw).rename(columns={"onset_s": "onset_true_s"})
    p = onset_times(pred, lifts, fs, names=names, **kw).rename(columns={"onset_s": "onset_pred_s"})
    out = t.merge(p, on=["lift", "muscle"])
    out["error_ms"] = 1000.0 * (out.onset_pred_s - out.onset_true_s)
    return out


def onset_summary(errors: pd.DataFrame, by=("muscle",)) -> pd.DataFrame:
    """Median absolute and median signed onset error (ms), number of lifts and number without an onset, per group."""
    g = errors.groupby(list(by))
    return pd.DataFrame({
        "median_abs_error_ms": g.error_ms.apply(lambda e: float(np.nanmedian(np.abs(e))) if e.notna().any() else np.nan),
        "median_error_ms": g.error_ms.apply(lambda e: float(np.nanmedian(e)) if e.notna().any() else np.nan),
        "n_lifts": g.size(), "n_no_onset": g.error_ms.apply(lambda e: int(e.isna().sum())),
    }).reset_index()
