"""EEG controls, phase 1, part W: does the EEG-only output E give different levels for light, medium and heavy lifts?

Question: the EEG-only stream E of the (non-causal) KG-GT v2 model is known to sit low. Is E only too low
(one gain would fix it) or flat across the lifted weights (the EEG does not tell the model how hard the lift is)?

For every lift of a test series (weights CurW 1, 2, 4 = light, medium, heavy) the mean envelope of the target
and of the outputs K, E, F (500 Hz, muscle units) is taken in two windows:
  hold  lift_off + 0.5 s to replace - 0.2 s (main window: the object is carried, no onset transient)
  load  first_touch to lift_off (the force builds up)
A lift whose window events or weight are missing, or whose window is empty, is skipped for that window.
One least-squares gain per muscle and window (hold, load), through the origin, is fitted on the validation lifts
and applied to the test lifts of the same window (E_cal), so that the size of E is taken out and only its
dependence on the weight is left.
Per muscle, window and output (target, K, E, E_cal, F) the statistics are: level_w1, level_w2, level_w4 (mean
level per weight), ratio_heavy_light, rho_weight (Spearman of the lift level with the weight, over lifts),
slope (of the output level on the target level over lifts) and size_ratio (mean output / mean target).
Group tests over participants (one-sided Wilcoxon, per window, muscles averaged within a participant):
  rho_E_gt_0          rho_weight of E > 0
  ratio_E_lt_target   heavy/light ratio of E smaller than that of the target (test on target - E)

Output (default eeg_controls/results/phase1/weight/): levels.csv (one row per lift, muscle and window),
stats.csv, group.csv. No training: it only loads the checkpoint of the non-causal model.

Run from the repository root (no --smoke: this CLI trains nothing; pass --out to a scratch folder for a check):
    env/Scripts/python.exe -m eeg_controls.weight_check --participants all --checkpoints outputs/checkpoints_v2 \
        --out eeg_controls/results/phase1/weight
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from v2.config import CONFIG_V2
from v2.infer import predict_series
from v2.model import build_model
from v2.train import load_best

from .common import ROOT, lifts as task_lifts, parse_subjects, wilcoxon_p
from .run_horizon import subject_data

OUTPUTS = ("K", "E", "F")
HOLD_AFTER, HOLD_BEFORE = 0.5, 0.2          # s after lift_off and before replace


def _window(lift: dict, name: str, fs: float):
    """(start, stop) sample of a window of one lift, or None when an event is missing or the window is empty."""
    if name == "hold":
        a, b = lift.get("lift_off"), lift.get("replace")
        if a is None or b is None:
            return None
        a, b = int(a + round(HOLD_AFTER * fs)), int(b - round(HOLD_BEFORE * fs))
    else:
        a, b = lift.get("first_touch"), lift.get("lift_off")
        if a is None or b is None:
            return None
        a, b = int(a), int(b)
    return (a, b) if b > a else None


def lift_levels(pred: dict, target: np.ndarray, lift_list, fs: float = 500.0, names=None) -> pd.DataFrame:
    """Mean envelope of the target and of K, E, F per lift, muscle and window (hold, load)."""
    target = np.asarray(target, float)
    n, n_mus = target.shape
    rows = []
    for lf in lift_list:
        w = lf.get("CurW")
        if w is None or not np.isfinite(w):
            continue
        for win in ("hold", "load"):
            span = _window(lf, win, fs)
            if span is None:
                continue
            a, b = max(span[0], 0), min(span[1], n)
            if b <= a:
                continue
            tgt = target[a:b].mean(0)
            out = {o: np.asarray(pred[o], float)[a:b].mean(0) for o in OUTPUTS}
            for m in range(n_mus):
                rows.append({"lift": lf.get("lift"), "weight": int(w), "window": win,
                             "muscle": names[m] if names is not None else m,
                             "target": tgt[m], **{o: out[o][m] for o in OUTPUTS}})
    cols = ["lift", "weight", "window", "muscle", "target", *OUTPUTS]
    return pd.DataFrame(rows, columns=cols)


def calibrate(levels_val: pd.DataFrame, levels_test: pd.DataFrame) -> pd.DataFrame:
    """Add E_cal = gain * E to the test levels; one gain per muscle and window = sum(E * target) / sum(E ** 2) on the
    validation lifts of that muscle and window (nan where the validation has none)."""
    gain = {}
    for key, d in levels_val.groupby(["window", "muscle"], sort=False):
        den = float((d.E ** 2).sum())
        gain[key] = float((d.E * d.target).sum()) / den if den > 0 else np.nan
    out = levels_test.copy()
    g = [gain.get((w, m), np.nan) for w, m in zip(out.window, out.muscle)]
    out["E_cal"] = out.E.to_numpy(float) * np.asarray(g, float)
    return out


def _stats_one(weight: np.ndarray, target: np.ndarray, y: np.ndarray) -> dict:
    row = {f"level_w{w}": float(y[weight == w].mean()) if (weight == w).any() else np.nan for w in (1, 2, 4)}
    row["ratio_heavy_light"] = row["level_w4"] / row["level_w1"] if row["level_w1"] > 0 else np.nan
    if len(y) >= 3 and np.ptp(weight) > 0 and np.ptp(y) > 0:
        row["rho_weight"] = float(spearmanr(weight, y)[0])
    else:
        row["rho_weight"] = np.nan
    row["slope"] = float(np.polyfit(target, y, 1)[0]) if len(y) >= 2 and np.ptp(target) > 0 else np.nan
    row["size_ratio"] = float(y.mean() / target.mean()) if target.mean() > 0 else np.nan
    return row


def weight_stats(levels: pd.DataFrame) -> pd.DataFrame:
    """Per window, muscle and output: level per weight, heavy/light ratio, rho with the weight, slope, size ratio."""
    outs = [o for o in ("target", "K", "E", "E_cal", "F") if o in levels.columns]
    rows = []
    for (win, mus), d in levels.groupby(["window", "muscle"], sort=False):
        w, t = d.weight.to_numpy(), d.target.to_numpy(float)
        for o in outs:
            rows.append({"window": win, "muscle": mus, "output": o, **_stats_one(w, t, d[o].to_numpy(float))})
    return pd.DataFrame(rows)


def group_tests(stats: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon over participants, per window (needs a `participant` column in `stats`)."""
    rows = []
    for win, d in stats.groupby("window", sort=False):
        e = d[d.output == "E"].groupby("participant")
        t = d[d.output == "target"].groupby("participant")
        rho = e.rho_weight.mean().dropna()
        diff = (t.ratio_heavy_light.mean() - e.ratio_heavy_light.mean()).dropna()
        for name, v in (("rho_E_gt_0", rho), ("ratio_E_lt_target", diff)):
            rows.append({"window": win, "test": name, "n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan,
                         "p": wilcoxon_p(v.to_numpy(), "greater")})
    return pd.DataFrame(rows, columns=["window", "test", "n", "mean", "p"])


def run_participant(p: int, ckpt_root: Path = Path("outputs/checkpoints_v2"), keep_example: bool = False):
    """Levels and stats of one participant from the non-causal checkpoint (`<ckpt_root>/P{p}/best.pt`).

    Returns (levels, stats); with keep_example=True also `example`, a dict with target, K, E, F (arrays (T, muscles)
    at 500 Hz of the first test series), lifts (its lift list) and names (muscle names).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = copy.deepcopy(CONFIG_V2)
    sd = subject_data(p, cfg)
    model = build_model(cfg, sd["prior"]).to(device)
    load_best(model, Path(ckpt_root) / f"P{p}", device)
    tc = cfg["tokens"]
    kw = dict(window=tc["window"], hop=tc["hop"], clip=cfg["eeg"]["clip"])

    def per_series(split):
        frames, example = [], None
        for tok in sd["tokens"][split]:
            pred = predict_series(model, tok, sd["norm"], device, **kw)
            target = np.asarray(tok["emg_full"], float)
            ll = task_lifts(p, tok["series"], tok["n_samples"])
            frames.append(lift_levels(pred, target, ll, names=sd["names"]).assign(series=tok["series"]))
            if example is None:
                example = dict(target=target, K=pred["K"], E=pred["E"], F=pred["F"], lifts=ll, names=list(sd["names"]))
        return pd.concat(frames, ignore_index=True), example

    val, _ = per_series("val")
    test, example = per_series("test")
    levels = calibrate(val, test).assign(participant=p)
    stats = weight_stats(levels).assign(participant=p)
    return (levels, stats, example) if keep_example else (levels, stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--checkpoints", default=str(ROOT / "outputs" / "checkpoints_v2"))
    ap.add_argument("--out", default=str(ROOT / "eeg_controls" / "results" / "phase1" / "weight"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lv, st = [], []
    for p in parse_subjects(a.participants):
        levels, stats = run_participant(p, Path(a.checkpoints))
        lv.append(levels); st.append(stats)
        s = stats[(stats.window == "hold") & stats.output.isin(["target", "E", "E_cal"])]
        m = s.groupby("output")[["ratio_heavy_light", "rho_weight", "size_ratio"]].mean()
        print(f"P{p} hold, mean over muscles:\n{m.round(3).to_string()}", flush=True)
    levels, stats = pd.concat(lv, ignore_index=True), pd.concat(st, ignore_index=True)
    group = group_tests(stats)
    levels.to_csv(out / "levels.csv", index=False)
    stats.to_csv(out / "stats.csv", index=False)
    group.to_csv(out / "group.csv", index=False)
    print("\ngroup tests:\n" + group.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
