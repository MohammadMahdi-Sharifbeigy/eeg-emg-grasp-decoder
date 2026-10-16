"""Experiment 2: how much of E can be explained by the task phase alone?

E (the EEG-only output of KG-GT v2) reaches r = 0.645. If E mostly reads WHERE in the lift the
participant is (reach, grasp, lift, hold, release, rest), then a model that knows only the timing of
the task events, and nothing about the EEG, should come close. This script fits that model.

Phase model, per participant (ridge regression, no neural network):
  features  a bank of Gaussian bumps locked to each task event of each lift (LED on, hand start, first
            touch, both touch, lift-off, replace, release, hand stop) at lags -2 s ... +3 s every 0.25 s
            (width 0.25 s), plus boxcars for "moving", "touching the object" and "holding it up", plus
            an intercept. Like E, it may use the whole lift (past and future events).
  variant   "phase"         timing only
            "phase_weight"  the same features once per object weight (CurW), so the level can depend on
                            the load; an upper reference, since E is never told the weight
            "coarse"        the same events, wide bumps (0.75 s) every 0.5 s and no boxcars: only a rough
                            idea of where in the lift a moment is
            "cue"           only the LED-on cue (the one event the EEG sees directly, as a visual stimulus);
                            everything after it must be inferred from the typical timing of a lift
  The event times come from the kinematic and force recordings, so "phase" is a timing-only summary of
  the kinematics: E below "phase" means E holds less than the timing of the lift.
  target    u = sqrt(y / s_c) at 50 Hz, the domain v2 is trained in (s_c = 95th percentile of the
            training series)
  fit       ridge on the training series, alpha chosen on the validation series, scored on the test
            series at 500 Hz against the original envelope, with the metrics of v2.evaluate.

Output (default eeg_controls/results/phase/): phase_metrics.csv (one row per participant, variant, muscle) and
phase_vs_E.csv, which adds E and K from results/v2/standard.csv (read only) when that file exists.

Run from the repository root:
    env/Scripts/python.exe -m eeg_controls.phase_baseline --participants all
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from main.dataloader import emg_channel_names
from v2.data import load_split
from v2.evaluate import standard_metrics
from v2.tokens import FS_IN, Q, to_samples, to_tokens

from .common import EVENTS, ROOT, cache_dir, data_dir, lifts, parse_subjects

FS_TOK = FS_IN / Q
LAGS = np.arange(-2.0, 3.0 + 1e-9, 0.25)
WIDTH = 0.25
BOXES = (("moving", "hand_start", "hand_stop"), ("touching", "first_touch", "release"), ("holding", "lift_off", "replace"))
ALPHAS = tuple(10.0 ** np.arange(-6.0, 4.0))
VARIANTS = ("phase", "phase_weight", "coarse", "cue")
# variant -> (events, lag step s, bump width s, with the phase boxcars)
BANKS = {"phase": (EVENTS, 0.25, WIDTH, True), "phase_weight": (EVENTS, 0.25, WIDTH, True),
         "coarse": (EVENTS, 0.5, 0.75, False), "cue": (("led_on",), 0.25, WIDTH, False)}
WEIGHTS = (1, 2, 4)


def phase_features(lift_list, n_tok: int, events=EVENTS, step: float = 0.25, width: float = WIDTH,
                   boxes: bool = True) -> np.ndarray:
    """(n_tok, n_features) event-locked bumps (lags -2 .. +3 s every `step`) and, optionally, phase boxcars."""
    t = np.arange(n_tok) / FS_TOK
    lags = np.arange(-2.0, 3.0 + 1e-9, step)
    cols = []
    for ev in events:
        bank = np.zeros((n_tok, len(lags)))
        for lift in lift_list:
            if lift.get(ev) is None:
                continue
            te = lift[ev] / FS_IN
            bank += np.exp(-0.5 * ((t[:, None] - te - lags[None, :]) / width) ** 2)
        cols.append(bank)
    if not boxes:
        return np.concatenate(cols, axis=1)
    box = np.zeros((n_tok, len(BOXES)))
    for j, (_, a, b) in enumerate(BOXES):
        for lift in lift_list:
            if lift.get(a) is not None and lift.get(b) is not None:
                box[int(lift[a] // Q):int(lift[b] // Q), j] = 1.0
    cols.append(box)
    return np.concatenate(cols, axis=1)


def lift_weight(lift_list, n_tok: int) -> np.ndarray:
    """CurW of the lift each token belongs to (from one lift's LED on to the next one's), 0 before the first."""
    w = np.zeros(n_tok, dtype=int)
    starts = [(lift.get("led_on") if lift.get("led_on") is not None else lift.get("hand_start"), lift["CurW"])
              for lift in lift_list]
    starts = sorted((int(s // Q), int(cw)) for s, cw in starts if s is not None)
    for i, (s, cw) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else n_tok
        w[s:e] = cw
    return w


def design(lift_list, n_tok: int, variant: str) -> np.ndarray:
    if variant not in BANKS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    events, step, width, boxes = BANKS[variant]
    x = phase_features(lift_list, n_tok, events, step, width, boxes)
    if variant != "phase_weight":
        return x
    w = lift_weight(lift_list, n_tok)
    return np.concatenate([x * (w == cw)[:, None] for cw in WEIGHTS], axis=1)


def series_arrays(participant: int, split: str) -> list[dict]:
    """Envelope at 500 Hz and at the token rate, plus the lift events, for every series of a split."""
    out = []
    for raw in load_split(cache_dir(), participant, split, data_dir()):
        emg = raw["emg"]
        n = len(emg)
        out.append({"series": raw["series"], "emg_full": emg, "emg": np.maximum(to_tokens(emg), 0.0),
                    "n_samples": n, "lifts": lifts(participant, raw["series"], n)})
    return out


def _fit(x, y, alpha):
    mu, sd = x.mean(0), x.std(0) + 1e-8
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + alpha * len(z) * np.eye(z.shape[1]), z.T @ (y - y.mean(0)))
    return lambda xn: ((xn - mu) / sd) @ w + y.mean(0)


def run_participant(participant: int, variant: str = "phase") -> tuple[pd.DataFrame, dict]:
    sp = {s: series_arrays(participant, s) for s in ("train", "val", "test")}
    scale = np.maximum(np.percentile(np.concatenate([s["emg"] for s in sp["train"]]), 95, axis=0), 1e-8)
    to_u = lambda y: np.sqrt(np.maximum(y, 0.0) / scale)
    xy = {k: (np.concatenate([design(s["lifts"], len(s["emg"]), variant) for s in v]),
              np.concatenate([to_u(s["emg"]) for s in v])) for k, v in sp.items()}
    best = None
    for a in ALPHAS:                                              # alpha by validation error in the model domain
        pred = _fit(*xy["train"], a)(xy["val"][0])
        err = float(((pred - xy["val"][1]) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, a)
    alpha = best[1]
    model = _fit(np.concatenate([xy["train"][0], xy["val"][0]]), np.concatenate([xy["train"][1], xy["val"][1]]), alpha)
    preds, targets = [], []
    for s in sp["test"]:
        u = model(design(s["lifts"], len(s["emg"]), variant))
        y = scale * np.maximum(u, 0.0) ** 2
        preds.append(np.maximum(to_samples(y, s["n_samples"]), 0.0))
        targets.append(s["emg_full"])
    names = emg_channel_names(participant, data_dir())
    m = standard_metrics(np.concatenate(preds), np.concatenate(targets), names)
    m = m.assign(subject=f"P{participant}", variant=variant, alpha=alpha,
                 n_features=xy["train"][0].shape[1], n_test_samples=sum(len(t) for t in targets))
    return m, {"pred": preds, "target": targets}


def compare_with_v2(phase: pd.DataFrame, standard_csv: Path) -> pd.DataFrame | None:
    """Phase metrics next to E and K of results/v2/standard.csv, one row per participant and muscle."""
    if not Path(standard_csv).exists():
        return None
    v2 = pd.read_csv(standard_csv)
    wide = v2[v2["output"].isin(["E", "K"])].pivot_table(index=["subject", "muscle"], columns="output",
                                                         values=["r", "vaf"])
    wide.columns = [f"{o}_{m}" for m, o in wide.columns]
    ph = phase.pivot_table(index=["subject", "muscle"], columns="variant", values=["r", "vaf"])
    ph.columns = [f"{v}_{m}" for m, v in ph.columns]
    out = ph.join(wide, how="inner").reset_index()
    if "phase_r" in out and "E_r" in out:
        out["E_minus_phase_r"] = out["E_r"] - out["phase_r"]
    return out


def summary(cmp: pd.DataFrame) -> str:
    from scipy.stats import wilcoxon
    lines = []
    per_subj = cmp.groupby("subject").mean(numeric_only=True)
    for col in [c for c in ("cue_r", "coarse_r", "phase_r", "phase_weight_r", "E_r", "K_r") if c in per_subj]:
        lines.append(f"mean r {col[:-2]:>13}: {per_subj[col].mean():.3f} (participants {per_subj[col].min():.3f} .. {per_subj[col].max():.3f})")
    if {"phase_r", "E_r"} <= set(per_subj):
        d = per_subj["E_r"] - per_subj["phase_r"]
        p = wilcoxon(d).pvalue if len(d) >= 6 else float("nan")
        lines.append(f"E - phase, mean over muscles: {d.mean():+.3f}; E better in {int((d > 0).sum())} of {len(d)} "
                     f"participants (Wilcoxon p = {p:.3g})")
        lines.append(f"ratio r_phase / r_E: {(per_subj['phase_r'] / per_subj['E_r']).mean():.2f}")
        by_m = cmp.groupby("muscle")[["phase_r", "E_r"]].mean()
        lines.append("per muscle (phase_r, E_r): " + ", ".join(f"{m} {a:.2f}/{b:.2f}" for m, (a, b) in by_m.iterrows()))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default=str(ROOT / "eeg_controls" / "results" / "phase"))
    ap.add_argument("--v2-standard", default=str(ROOT / "results" / "v2" / "standard.csv"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in parse_subjects(a.participants):
        for v in a.variants.split(","):
            m, _ = run_participant(p, v.strip())
            rows.append(m)
            print(f"P{p} {v:>13}: mean r {m['r'].mean():.3f}, mean VAF {m['vaf'].mean():.1f} (alpha {m['alpha'].iloc[0]:g})", flush=True)
    phase = pd.concat(rows, ignore_index=True)
    phase.to_csv(out / "phase_metrics.csv", index=False)
    cmp = compare_with_v2(phase, Path(a.v2_standard))
    if cmp is not None:
        cmp.to_csv(out / "phase_vs_E.csv", index=False)
        text = summary(cmp)
        (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
        print("\n" + text)
    print("saved", out)


if __name__ == "__main__":
    main()
