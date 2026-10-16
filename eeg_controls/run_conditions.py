"""EEG controls, phase 1, parts S and E: does the EEG gain come from motor cortex, from the scalp edge, or from the eyes?

Each condition is a causal horizon model (as eeg_controls.run_horizon) trained with a different EEG input:
  all_frozen all 32 electrodes, trained exactly like the subsets (K loaded and frozen, E + fusion from scratch):
             the fair reference, since `all` (the joint-trained horizon runs) also trained K together with E
  central    C3, Cz, C4, CP1, CP2, CP5, CP6 only (over the motor cortex)
  edge       Fp1, Fp2, F7, F8, T7, T8, TP9, TP10, P7, P8, PO9, PO10, O1, Oz, O2 only (eye and muscle artefacts)
  eye_clean  all channels after least-squares regression on Fp1, Fp2, F7, F8 (then dropped: 28 channels)
K is loaded from the matching all-electrode horizon checkpoint ({kin-checkpoints}/h{h}ms/P{p}/best.pt) and kept
frozen from the first epoch (cfg["training"]["freeze_kin"]); only E and the fusion train. So K is identical to the
all-electrode run and every difference between conditions comes from the EEG.

Before the first run every needed K checkpoint must exist and its stored config must agree with the condition config
in the blocks that define K (model, tokens, history, loss); otherwise nothing trains and every problem is listed.
After each fit the `kin.*` tensors are compared with the loaded ones (RuntimeError naming the first that differs).
A run that fails is logged with its traceback to {results}/failures.log and the loop goes on with the next run.

Per condition, horizon and participant ({results}/{condition}/h{h}ms/P{p}/):
  standard.csv, residual.csv, segments.csv   as eeg_controls.run_horizon
  phase_ridge.csv                            the per-phase ridge of eeg_controls.phase_ridge on this model
  history.json                               training summary, with the loaded K checkpoint (kin_checkpoint)
Over all runs ({results}/):
  summary.csv   one row per condition (`all` = the 32-electrode horizon runs in --baseline), horizon and participant
  vs_all.csv    per condition, reference, horizon and measure: mean difference to the reference and the two-sided
                Wilcoxon p, paired by participant. Every subset condition is compared with `all_frozen` (primary,
                same training procedure) and with `all`; `all_frozen` itself only with `all`.

Finished result folders are skipped unless --overwrite; a finished checkpoint is reused (resume).
With --smoke, a --results or --checkpoints left at its default gets `_smoke` appended (results/v2_controls_smoke).

Run from the repository root, for example:
    env/Scripts/python.exe -m eeg_controls.run_conditions --participants all --conditions all_frozen,central,edge,eye_clean \
        --horizons-ms 0,200,500
A short check that saves nothing where real runs go:
    env/Scripts/python.exe -m eeg_controls.run_conditions --participants 1 --conditions central --horizons-ms 0 \
        --smoke --kin-checkpoints <scratch>/hz_ckpt --checkpoints <scratch>/ckpt --results <scratch>/res
"""
import argparse
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v2.model import build_model, stream_of
from v2.residual import residual_test
from v2.train import IDENTITY_BLOCKS, fit, load_kin

from .common import ROOT, parse_subjects, wilcoxon_p
from .phase_ridge import score_model
from .run_horizon import horizon_config, segments, standard, subject_data

CONDITIONS = {
    "all_frozen": {},                                                # all 32 electrodes, K frozen like the others
    "central": {"channels": ["C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6"]},
    "edge": {"channels": ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "PO10",
                          "O1", "Oz", "O2"]},
    "eye_clean": {"eye_regress": ["Fp1", "Fp2", "F7", "F8"]},
}
FILES = ("standard.csv", "residual.csv", "segments.csv", "phase_ridge.csv", "history.json")
KIN_BLOCKS = tuple(b for b in IDENTITY_BLOCKS if b != "eeg")      # the identity blocks that define K
REFERENCES = ("all_frozen", "all")                                  # vs_all.csv; all_frozen is the primary one
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


def _finished(args, name: str, h_ms: int, p: int) -> bool:
    out = Path(args.results) / name / f"h{int(h_ms)}ms" / f"P{p}"
    return all((out / f).exists() for f in FILES) and not args.overwrite


def check_kin_checkpoints(names, horizons, participants, args) -> None:
    """Before any training: the K checkpoint of every unfinished run exists and was trained with the same K config.

    Raises FileNotFoundError listing every missing path, or RuntimeError listing every checkpoint whose stored config
    differs from the condition config in a block that defines K (KIN_BLOCKS: v2.train's identity blocks without
    `eeg`, which differs between conditions by design). A checkpoint without a stored config passes with a warning.
    """
    need = {}
    for h in horizons:
        for name in names:
            for p in participants:
                if not _finished(args, name, h, p):
                    need.setdefault((int(h), int(p)), name)
    paths = {k: kin_checkpoint(args.kin_checkpoints, k[1], k[0]) for k in sorted(need)}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} K checkpoint(s) not found; nothing was trained (train the horizon "
                                "models first: eeg_controls.run_horizon):\n  " + "\n  ".join(missing))
    differ = []
    for (h, p), path in paths.items():
        stored = torch.load(path, map_location="cpu", weights_only=False).get("config")
        if stored is None:
            print(f"WARNING: {path} has no config record; its K config is not checked")
            continue
        cfg = condition_config(need[(h, p)], h, args.smoke, args.amp)
        bad = [b for b in KIN_BLOCKS if stored.get(b) != cfg.get(b)]
        if bad:
            differ.append(f"{path}: block(s) {', '.join(bad)} differ")
    if differ:
        raise RuntimeError("K checkpoint(s) trained with another K config; nothing was trained:\n  "
                           + "\n  ".join(differ))


def _kin_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if stream_of(k) == "kin"}


def check_kin_equal(model, loaded: dict, kin_path) -> None:
    """RuntimeError naming the first `kin.*` tensor of `model` that differs from `loaded` (K as loaded before fit)."""
    now = _kin_state(model)
    for k, v in loaded.items():
        if k not in now or now[k].shape != v.shape or not torch.equal(now[k], v):
            raise RuntimeError(f"K changed during training: {k} differs from the loaded checkpoint {kin_path}")


def run_one(p: int, name: str, h_ms: int, args, device) -> Path:
    """Train E + fusion of one condition against the loaded, frozen K, and write the five result files."""
    h_ms = int(h_ms)
    cfg = condition_config(name, h_ms, args.smoke, args.amp)
    out = Path(args.results) / name / f"h{h_ms}ms" / f"P{p}"
    if _finished(args, name, h_ms, p):
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
    loaded = _kin_state(model)                            # and a resumed run restores its own state
    hist = fit(model, sd["train_ds"], sd["val_ds"], cfg, device,
               Path(args.checkpoints) / name / f"h{h_ms}ms" / f"P{p}", log=print)
    check_kin_equal(model, loaded, kin_path)              # K must be exactly the loaded one
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


def run_all(names, horizons, participants, args, device) -> int:
    """check_kin_checkpoints, then every run (horizon, condition, participant). A failing run is logged with its
    traceback to {results}/failures.log and the loop goes on. Returns the number of failed runs."""
    horizons, participants = [int(h) for h in horizons], [int(p) for p in participants]
    check_kin_checkpoints(names, horizons, participants, args)
    log = Path(args.results) / "failures.log"
    failed = 0
    for h in horizons:
        for name in names:
            for p in participants:
                try:
                    run_one(p, name, h, args, device)
                except Exception as e:                                # one failure must not stop a 27-hour loop
                    failed += 1
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a", encoding="utf-8") as f:
                        f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {name} h={h} ms P{p}\n"
                                f"{traceback.format_exc()}\n")
                    print(f"FAILED {name} P{p} h={h} ms: {type(e).__name__}: {e} (traceback in {log})", flush=True)
    total = len(horizons) * len(names) * len(participants)
    print(f"{failed} of {total} runs failed" + (f"; see {log}" if failed else ""), flush=True)
    return failed


def summarise(results: Path, baseline: Path = Path("results/v2_horizon")) -> pd.DataFrame:
    """One row per condition (`all` from `baseline`), horizon and participant; writes summary.csv and vs_all.csv.

    vs_all.csv compares every subset condition with `all_frozen` (the primary reference: same training procedure)
    and with `all`, and `all_frozen` with `all` only (column `reference`)."""
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
    tests = []
    for ref in REFERENCES:
        base = df[df.condition == ref].set_index(["horizon_ms", "subject"])
        compared = df[~df.condition.isin(("all", "all_frozen"))] if ref == "all_frozen" else df[df.condition != "all"]
        for (name, h), d in compared.groupby(["condition", "horizon_ms"], sort=False):
            paired = d.set_index(["horizon_ms", "subject"]).join(base[MEASURES], rsuffix="_ref", how="inner")
            if paired.empty:
                continue
            for meas in MEASURES:
                diff = paired[meas] - paired[f"{meas}_ref"]
                tests.append({"condition": name, "reference": ref, "horizon_ms": h, "metric": meas,
                              "n": int(diff.notna().sum()), "mean_condition": paired[meas].mean(),
                              "mean_reference": paired[f"{meas}_ref"].mean(), "mean_diff": diff.mean(),
                              "p": wilcoxon_p(diff, alternative="two-sided")})
    vs = pd.DataFrame(tests, columns=["condition", "reference", "horizon_ms", "metric", "n", "mean_condition",
                                      "mean_reference", "mean_diff", "p"])
    vs.to_csv(results / "vs_all.csv", index=False)
    g = df.groupby(["condition", "horizon_ms"], sort=False)
    print("\nmean over participants, per condition and horizon:")
    print(g[MEASURES].mean().assign(n=g.size()).round(3).to_string())
    if not vs.empty:
        print("\ncondition - reference (two-sided Wilcoxon, paired by participant; all_frozen is the fair reference):")
        print(vs.pivot_table(index=["reference", "condition", "horizon_ms"], columns="metric",
                             values=["mean_diff", "p"], sort=False).round(3).to_string())
    return df


def smoke_paths(a, defaults: dict) -> None:
    """--smoke must not write where real runs go: each path argument still at its default gets `_smoke` appended."""
    for key, default in defaults.items():
        cur = Path(getattr(a, key))
        if cur.resolve() == Path(default).resolve():
            cur = cur.with_name(cur.name + "_smoke")
            setattr(a, key, str(cur))
        print(f"smoke: --{key.replace('_', '-')} -> {cur}")


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
    if a.smoke:
        smoke_paths(a, {k: ap.get_default(k) for k in ("results", "checkpoints")})
    names = [c.strip() for c in a.conditions.split(",") if c.strip()]
    for n in names:
        if n not in CONDITIONS:
            raise SystemExit(f"unknown condition {n!r}; choose from {', '.join(CONDITIONS)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not a.summary_only:
        print("device:", device)
        try:
            run_all(names, [int(x) for x in a.horizons_ms.split(",")], parse_subjects(a.participants), a, device)
        except (FileNotFoundError, RuntimeError) as e:          # only the up-front K check gets here
            raise SystemExit(str(e))
    return summarise(Path(a.results), Path(a.baseline))


if __name__ == "__main__":
    main()
