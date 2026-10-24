"""Timing-preserving null: the EEG of each test trial is replaced by the EEG of another trial, aligned on the LED cue.

The shuffle and swap nulls of v3.evaluate destroy the task timing together with the EEG, so they cannot tell
"the EEG of this trial" from "an EEG response that follows every cue". Here every trial keeps its own EMG target and
its own timing, but gets the cue-aligned EEG of a different trial of the same participant. A model that only reads
the task clock from the EEG keeps its score (r_swap close to r_true); delta = r_true - r_swap is what is specific
to the trial.

A trial runs from one lift's LED onset to the next lift's (the first trial starts at token 0, the last ends with
its series). Where the donor trial is shorter than the place it fills, its end values are held (numpy "edge"). A
trial ends just before the next cue, at rest, so the held value is rest-like. Reading on into the donor's next trial
would put that trial's cue response where the recipient has none; the odd reflection of v2.tokens extrapolates
trends over long pads, and a mirror would copy the cue response into the padding. The result is clipped to the
input clip again.
"""
import numpy as np
import pandas as pd

from eeg_controls.common import wilcoxon_p
from v2.evaluate import _r
from v2.tokens import Q

from .infer import normalise_eeg, predict_eeg_series


def trial_bounds(lift_list, n_tok: int) -> list[tuple[int, int, int]]:
    """(start, end, led) token indices of every trial of one series; lifts without an LED onset are skipped."""
    leds = sorted({int(l["led_on"]) // Q for l in lift_list if l.get("led_on") is not None and int(l["led_on"]) // Q < n_tok})
    if not leds:
        return []
    starts = [0] + leds[1:]
    ends = leds[1:] + [int(n_tok)]
    return list(zip(starts, ends, leds))


def derangement(n: int, rng, groups=None) -> np.ndarray:
    """donor[i] != i for every i: a rotation of a random order (inside each group when `groups` is given).

    A group with a single trial takes its donor from all other trials.
    """
    if n < 2:
        raise ValueError(f"a swap needs at least 2 trials, got {n}")
    donor = np.full(n, -1)
    if groups is None:
        labels = np.zeros(n, dtype=int)
    else:
        ids = {}
        labels = np.array([ids.setdefault(g, len(ids)) for g in groups])
    for g in np.unique(labels):
        idx = np.flatnonzero(labels == g)
        if len(idx) == 1:
            others = np.flatnonzero(np.arange(n) != idx[0])
            donor[idx[0]] = int(rng.choice(others))
            continue
        order = rng.permutation(idx)
        donor[order] = np.roll(order, -1)
    return donor


def _aligned(src: np.ndarray, src_led: int, length: int, led: int) -> np.ndarray:
    """`length` tokens of `src` with its LED token `src_led` placed at `led`; the end values held outside src."""
    idx = np.arange(length) - led + src_led
    left, right = max(0, -int(idx.min())), max(0, int(idx.max()) - (len(src) - 1))
    if left or right:
        pad = [(left, right)] + [(0, 0)] * (src.ndim - 1)
        src = np.pad(src, pad, mode="edge")
    return src[idx + left]


def cue_aligned_swap(eeg_trials, led_onsets, rng, match_condition=None, clip=None, return_donors=False):
    """The EEG of every trial replaced by the cue-aligned EEG of another trial (list, one array per trial).

    `led_onsets[i]` is the LED token inside trial i; `match_condition` (one label per trial, e.g. (CurW, CurS))
    restricts donors to trials with the same label; with `return_donors` also the donor index of every trial.
    """
    donors = derangement(len(eeg_trials), rng, match_condition)
    out = []
    for i, j in enumerate(donors):
        x = _aligned(np.asarray(eeg_trials[j]), int(led_onsets[j]), len(eeg_trials[i]), int(led_onsets[i]))
        out.append(np.clip(x, -clip, clip) if clip is not None else x)
    return (out, donors) if return_donors else out


def swap_series(eeg_by_series, lifts_by_series, rng, match_condition: bool = False, clip=None):
    """Swap the trials of all test series of one participant; returns one EEG array per series and the donors."""
    trials, leds, where, cond = [], [], [], []
    for s, (eeg, lifts) in enumerate(zip(eeg_by_series, lifts_by_series)):
        led_lift = {int(l["led_on"]) // Q: l for l in lifts if l.get("led_on") is not None}
        for a, b, led in trial_bounds(lifts, len(eeg)):
            trials.append(eeg[a:b]); leds.append(led - a); where.append((s, a, b))
            lift = led_lift.get(led, {})
            cond.append((lift.get("CurW"), lift.get("CurS")))
    swapped, donors = cue_aligned_swap(trials, leds, rng, cond if match_condition else None, clip, return_donors=True)
    out = [np.array(e, copy=True) for e in eeg_by_series]
    for (s, a, b), x in zip(where, swapped):
        out[s][a:b] = x
    return out, donors


def evaluate_swap(model, test_tokens, lifts_by_series, norm, cfg, device, names) -> pd.DataFrame:
    """Per muscle: r_true, r_swap (mean over draws, 500 Hz scoring as v3.evaluate), r_swap_sd and delta."""
    ec = cfg["evaluation"]
    n_draws, clip = int(ec.get("n_swap", 10)), cfg["eeg"]["clip"]
    kw = dict(window=cfg["tokens"]["window"], hop=cfg["tokens"]["hop"], use_amp=bool(cfg["training"]["use_amp"]),
              clip=clip)
    target = np.concatenate([t["emg_full"] for t in test_tokens])
    r_true = _r(np.concatenate([predict_eeg_series(model, t, norm, device, **kw) for t in test_tokens]), target)
    eeg_z = [normalise_eeg(t, norm, clip) for t in test_tokens]
    rng = np.random.default_rng(ec["seed"])
    draws = []
    for _ in range(n_draws):
        alt, _ = swap_series(eeg_z, lifts_by_series, rng, bool(ec.get("swap_match_condition", False)), clip)
        pred = [predict_eeg_series(model, t, norm, device, eeg=a, **kw) for t, a in zip(test_tokens, alt)]
        draws.append(_r(np.concatenate(pred), target))
    draws = np.asarray(draws)
    return pd.DataFrame({"muscle": list(names), "r_true": r_true, "r_swap": draws.mean(0), "r_swap_sd": draws.std(0),
                         "delta": r_true - draws.mean(0), "n_draws": n_draws})


def swap_group_test(swap: pd.DataFrame, control_scale: float = 0.0) -> pd.DataFrame:
    """Per scale over participants (mean over muscles and seeds first): delta > 0, and delta above the control's."""
    per = swap.groupby(["subject", "scale"], as_index=False)[["r_true", "r_swap", "delta"]].mean()
    ctrl = per[per.scale == control_scale].set_index("subject").delta
    rows = []
    for scale, d in per.groupby("scale"):
        d = d.set_index("subject")
        gain = (d.delta - ctrl.reindex(d.index)).dropna() if scale != control_scale else pd.Series(dtype=float)
        rows.append({"scale": scale, "n_subjects": len(d), "mean_r_true": d.r_true.mean(), "mean_r_swap": d.r_swap.mean(),
                     "mean_delta": d.delta.mean(), "p_delta": wilcoxon_p(d.delta.values),
                     "mean_delta_gain": gain.mean() if len(gain) else np.nan,
                     "p_delta_gain": wilcoxon_p(gain.values) if len(gain) else np.nan})
    return pd.DataFrame(rows)
