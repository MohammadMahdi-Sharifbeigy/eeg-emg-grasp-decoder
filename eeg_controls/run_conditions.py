"""EEG controls, phase 1, parts S and E: does the EEG gain come from motor cortex, from the scalp edge, or from the eyes?

Each condition is a causal horizon model (as eeg_controls.run_horizon) trained with a different EEG input:
  central    C3, Cz, C4, CP1, CP2, CP5, CP6 only (over the motor cortex)
  edge       Fp1, Fp2, F7, F8, T7, T8, TP9, TP10, P7, P8, PO9, PO10, O1, Oz, O2 only (eye and muscle artefacts)
  eye_clean  all channels after least-squares regression on Fp1, Fp2, F7, F8 (then dropped: 28 channels)
K is loaded from the matching all-electrode horizon checkpoint ({kin-checkpoints}/h{h}ms/P{p}/best.pt) and kept
frozen from the first epoch (cfg["training"]["freeze_kin"]); only E and the fusion train. So K is identical to the
all-electrode run and every difference between conditions comes from the EEG.

Per condition, horizon and participant ({results}/{condition}/h{h}ms/P{p}/):
  standard.csv, residual.csv, segments.csv   as eeg_controls.run_horizon
  phase_ridge.csv                            the per-phase ridge of eeg_controls.phase_ridge on this model
  history.json                               training summary, with the loaded K checkpoint (kin_checkpoint)
Over all runs ({results}/):
  summary.csv   one row per condition (`all` = the 32-electrode horizon runs in --baseline), horizon and participant
  vs_all.csv    per condition, horizon and measure: mean difference to `all` and the two-sided Wilcoxon p, paired
                by participant

Finished result folders are skipped unless --overwrite; a finished checkpoint is reused (resume).

Run from the repository root, for example:
    env/Scripts/python.exe -m eeg_controls.run_conditions --participants all --conditions central,edge,eye_clean \
        --horizons-ms 0,200,500
A short check that saves nothing where real runs go:
    env/Scripts/python.exe -m eeg_controls.run_conditions --participants 1 --conditions central --horizons-ms 0 \
        --smoke --kin-checkpoints <scratch>/hz_ckpt --checkpoints <scratch>/ckpt --results <scratch>/res
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v2.model import build_model
from v2.residual import residual_test
from v2.train import fit, load_kin

from .common import ROOT, parse_subjects, wilcoxon_p
from .phase_ridge import score_model
from .run_horizon import horizon_config, segments, standard, subject_data

CONDITIONS = {
    "central": {"channels": ["C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6"]},
    "edge": {"channels": ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "PO10",
                          "O1", "Oz", "O2"]},
    "eye_clean": {"eye_regress": ["Fp1", "Fp2", "F7", "F8"]},
}
FILES = ("standard.csv", "residual.csv", "segments.csv", "phase_ridge.csv", "history.json")
MEASURES = ["E_r", "F_minus_K_vaf", "res_r_tokens", "pre_move_r", "reach_r", "pre_move_skill", "reach_skill"]


def condition_config(name: str, h_ms: int, smoke: bool = False, use_amp: bool = False) -> dict:
    """horizon_config + the condition's eeg keys + training.freeze_kin = True."""
    if name not in CONDITIONS:
        raise ValueError(f"unknown condition {name!r}; choose from {', '.join(CONDITIONS)}")
    cfg = horizon_config(h_ms, smoke, use_amp)
    cfg["eeg"].update(copy.deepcopy(CONDITIONS[name]))
    cfg["training"]["freeze_kin"] = True
    return cfg


def kin_checkpoint(kin_root, p: int, h_ms: int) -> Path:
    return Path(kin_root) / f"h{int(h_ms)}ms" / f"P{p}" / "best.pt"


def run_one(p: int, name: str, h_ms: int, args, device) -> Path:
    """Train E + fusion of one condition against the loaded, frozen K, and write the five result files."""
    h_ms = int(h_ms)
    cfg = condition_config(name, h_ms, args.smoke, args.amp)
    out = Path(args.results) / name / f"h{h_ms}ms" / f"P{p}"
    if all((out / f).exists() for f in FILES) and not args.overwrite:
        print(f"{name} P{p} h={h_ms} ms: results exist, skipped ({out})")
        return out
    kin_path = kin_checkpoint(args.kin_checkpoints, p, h_ms)
    if not kin_path.is_file():                                       # before the (slow) data loading
        raise FileNotFoundError(f"K checkpoint not found: {kin_path} (train the horizon model first: "
                                f"eeg_controls.run_horizon --horizons-ms {h_ms} --participants {p})")
    t0 = time.time()
    sd = subject_data(p, cfg)
    torch.manual_seed(cfg["training"]["seed"])
    model = build_model(cfg, sd["prior"]).to(device)
    n_kin = load_kin(model, kin_path)                     # always before fit: freeze_kin keeps what is loaded,
    hist = fit(model, sd["train_ds"], sd["val_ds"], cfg, device,  # and a resumed run restores its own state
               Path(args.checkpoints) / name / f"h{h_ms}ms" / f"P{p}", log=print)
    out.mkdir(parents=True, exist_ok=True)
    tag = dict(condition=name, subject=f"P{p}", horizon_ms=h_ms)
    std = standard(model, sd, cfg, device).assign(**tag)
    std.to_csv(out / "standard.csv", index=False)
    n_null = int(cfg["evaluation"]["n_null"])
    res = residual_test(model, sd["tokens"]["val"], sd["tokens"]["test"], sd["norm"], cfg, device, sd["names"],
                        n_null=n_null).assign(**tag)
    res.to_csv(out / "residual.csv", index=False)
    seg = segments(model, sd, cfg, device, p, n_null, cfg["evaluation"]["seed"]).assign(**tag)
    seg.to_csv(out / "segments.csv", index=False)
    pr = score_model(model, sd, cfg, device, p, h_ms, n_null=n_null)
    pr.insert(0, "condition", name)
    pr.to_csv(out / "phase_ridge.csv", index=False)
    (out / "history.json").write_text(json.dumps({
        "condition": name, "eeg": CONDITIONS[name], "kin_checkpoint": str(kin_path), "kin_tensors_loaded": n_kin,
        "epochs_run": hist["epochs_run"], "best_epoch_kin": hist["best_epoch_kin"], "best_epoch_eeg": hist["best_epoch_eeg"],
        "frozen_epoch": hist.get("frozen_epoch"), "minutes": (time.time() - t0) / 60,
        "gate": model.fusion.gate.detach().cpu().tolist(), "config_tokens": cfg["tokens"]}, indent=1), encoding="utf-8")
    piv = std.pivot(index="muscle", columns="output", values="vaf")
    mean = res[res.muscle == "MEAN"].set_index("features")
    s = seg.set_index("segment")
    print(f"{name} P{p} h={h_ms} ms: VAF K {piv['K'].mean():.1f}, F - K {piv['F'].mean() - piv['K'].mean():+.2f}, "
          f"E r {std[std.output == 'E']['r'].mean():.3f}; residual r tokens {mean.loc['tokens', 'r']:.3f} "
          f"(p {mean.loc['tokens', 'p']:.2f}); pre_move skill ridge {s['skill_ridge'].get('pre_move', np.nan):+.3f}, "
          f"F {s['skill_F'].get('pre_move', np.nan):+.3f}; {(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def _row(d: Path, condition: str) -> dict:
    """The comparison measures of one finished run folder (standard, residual, segments csv)."""
    std, res, seg = (pd.read_csv(d / f) for f in ("standard.csv", "residual.csv", "segments.csv"))
    v = std.pivot(index="muscle", columns="output", values="vaf").mean()
    r = std.pivot(index="muscle", columns="output", values="r").mean()
    m = res[res.muscle == "MEAN"].set_index("features")
    s = seg.set_index("segment")
    get = lambda col, ph: float(s[col].get(ph, np.nan))
    return {"condition": condition, "horizon_ms": int(round(float(std.horizon_ms.iloc[0]))),
            "subject": std.subject.iloc[0], "E_r": r["E"], "F_minus_K_vaf": v["F"] - v["K"],
            "res_r_tokens": m.loc["tokens", "r"],
            "pre_move_r": get("r_residual_ridge", "pre_move"), "reach_r": get("r_residual_ridge", "reach"),
            "pre_move_skill": get("skill_ridge", "pre_move"), "reach_skill": get("skill_ridge", "reach")}


def _runs(root: Path, condition: str) -> list[dict]:
    return [_row(d, condition) for d in sorted(Path(root).glob("h*ms/P*")) if (d / "segments.csv").exists()]


def summarise(results: Path, baseline: Path = Path("results/v2_horizon")) -> pd.DataFrame:
    """One row per condition (`all` from `baseline`), horizon and participant; writes summary.csv and vs_all.csv."""
    results, baseline = Path(results), Path(baseline)
    if not baseline.is_absolute() and not baseline.exists():
        baseline = ROOT / baseline
    rows = [r for name in CONDITIONS for r in _runs(results / name, name)]
    horizons = {r["horizon_ms"] for r in rows}
    rows = [r for r in _runs(baseline, "all") if not horizons or r["horizon_ms"] in horizons] + rows
    df = pd.DataFrame(rows, columns=["condition", "horizon_ms", "subject"] + MEASURES)
    if df.empty:
        print(f"no finished runs in {results} or {baseline}")
        return df
    order = {c: i for i, c in enumerate(["all", *CONDITIONS])}
    df = df.sort_values(["condition", "horizon_ms", "subject"], key=lambda c: c.map(order) if c.name == "condition"
                        else c).reset_index(drop=True)
    results.mkdir(parents=True, exist_ok=True)
    df.to_csv(results / "summary.csv", index=False)
    base = df[df.condition == "all"].set_index(["horizon_ms", "subject"])
    tests = []
    for (name, h), d in df[df.condition != "all"].groupby(["condition", "horizon_ms"], sort=False):
        paired = d.set_index(["horizon_ms", "subject"]).join(base[MEASURES], rsuffix="_all", how="inner")
        for meas in MEASURES:
            diff = paired[meas] - paired[f"{meas}_all"]
            tests.append({"condition": name, "horizon_ms": h, "metric": meas, "n": int(diff.notna().sum()),
                          "mean_condition": paired[meas].mean(), "mean_all": paired[f"{meas}_all"].mean(),
                          "mean_diff": diff.mean(), "p": wilcoxon_p(diff, alternative="two-sided")})
    vs = pd.DataFrame(tests, columns=["condition", "horizon_ms", "metric", "n", "mean_condition", "mean_all",
                                      "mean_diff", "p"])
    vs.to_csv(results / "vs_all.csv", index=False)
    g = df.groupby(["condition", "horizon_ms"], sort=False)
    print("\nmean over participants, per condition and horizon:")
    print(g[MEASURES].mean().assign(n=g.size()).round(3).to_string())
    if not vs.empty:
        print("\ncondition - all (two-sided Wilcoxon, paired by participant):")
        print(vs.pivot_table(index=["condition", "horizon_ms"], columns="metric", values=["mean_diff", "p"],
                             sort=False).round(3).to_string())
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--horizons-ms", default="0,200,500")
    ap.add_argument("--results", default=str(ROOT / "results" / "v2_controls"))
    ap.add_argument("--checkpoints", default=str(ROOT / "outputs" / "checkpoints_v2_controls"))
    ap.add_argument("--kin-checkpoints", default=str(ROOT / "outputs" / "checkpoints_v2_horizon"),
                    help="the all-electrode horizon runs whose K is loaded: {dir}/h{h}ms/P{p}/best.pt")
    ap.add_argument("--baseline", default=str(ROOT / "results" / "v2_horizon"),
                    help="results of the all-electrode horizon runs (condition `all` in the summary)")
    ap.add_argument("--amp", action="store_true", help="mixed precision (slower on a GTX 1660 Ti)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch and 5 null draws: a code check, not a result")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    names = [c.strip() for c in a.conditions.split(",") if c.strip()]
    for n in names:
        if n not in CONDITIONS:
            raise SystemExit(f"unknown condition {n!r}; choose from {', '.join(CONDITIONS)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not a.summary_only:
        print("device:", device)
        for h in (int(x) for x in a.horizons_ms.split(",")):
            for name in names:
                for p in parse_subjects(a.participants):
                    run_one(p, name, h, a, device)
    return summarise(Path(a.results), Path(a.baseline))


if __name__ == "__main__":
    main()
