"""Experiment 1: does the EEG help to predict muscle activity AHEAD of time?

Hypothesis: in the concurrent, non-causal setting of KG-GT v2 the EEG adds nothing to the kinematics
(F = K). Movement-related EEG changes (beta/mu desynchronisation) start about 1 s before the hand moves,
while the kinematics are still flat. If the EEG carries anything the kinematics do not, it should show
when the model must forecast the envelope h ms ahead using only the past.

For every horizon h and participant this script trains KG-GT v2 with
    cfg["model"]["causal"] = True     each output token sees inputs up to its own time only
    cfg["tokens"]["causal"] = True    the inputs themselves are built from past samples only (v2.causal)
    cfg["tokens"]["horizon"] = h / 20 ms     input token t is paired with the target at t + h
and scores it on the test series:
  standard.csv   r / CCC / RMSE / VAF of K, E and F at 500 Hz (as v2.evaluate)
  residual.csv   v2.residual.residual_test: can the EEG predict what K misses (ridge, time-shift null)?
  segments.csv   the same question per task phase (pre_move = the last second before the hand starts,
                 reach, load, hold, release, other): skill = 1 - MSE(K + EEG) / MSE(K) for F and for the
                 ridge prediction, with a time-shift null for the ridge
  history.json   training summary
and writes summary.csv over all participants and horizons.

The prediction is that the EEG gain (F - K, residual r, pre_move skill) grows with h. If it stays at zero,
the EEG holds nothing beyond the kinematics even where the kinematics cannot know what is coming.

Training is long (about 20-30 min per participant and horizon on a GTX 1660 Ti). Finished runs are reused
(resume) and finished result folders are skipped unless --overwrite.

Run from the repository root, for example:
    env/Scripts/python.exe -m eeg_controls.run_horizon --participants all --horizons-ms 0,200,500,1000
A short check that saves nothing where real runs go:
    env/Scripts/python.exe -m eeg_controls.run_horizon --participants 1 --horizons-ms 500 --smoke \
        --results <scratch>/res --checkpoints <scratch>/ckpt
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main.dataloader import emg_channel_names
from main.preprocessing_emg_kin import compute_muscle_edge_prior
from v2 import data as D
from v2.config import CONFIG_V2
from v2.data import from_model_domain, to_model_domain
from v2.evaluate import _r, standard_metrics
from v2.infer import predict_series, predict_tokens
from v2.model import build_model
from v2.residual import _apply, eeg_features, fit_ridge, residual_test
from v2.tokens import FS_IN, Q
from v2.train import fit

from .common import ROOT, data_dir, lifts, parse_subjects, segment_labels

TOKEN_MS = 1000.0 * Q / FS_IN                                       # 20 ms


def horizon_config(h_ms: float, smoke: bool = False, use_amp: bool = False) -> dict:
    cfg = copy.deepcopy(CONFIG_V2)
    h = int(round(float(h_ms) / TOKEN_MS))
    if abs(h * TOKEN_MS - float(h_ms)) > 1e-6:
        raise ValueError(f"horizon {h_ms} ms is not a multiple of the {TOKEN_MS:g} ms token")
    cfg["model"]["causal"] = True
    cfg["tokens"]["causal"] = True
    cfg["tokens"]["horizon"] = h
    cfg["training"]["use_amp"] = bool(use_amp)
    if smoke:
        cfg["training"].update(max_epochs=1, patience=1, warmup_epochs=1)
        cfg["evaluation"]["n_null"] = 5
    return cfg


def subject_data(p: int, cfg: dict) -> dict:
    cache = ROOT / cfg["data"]["cache_dir"]
    tok = {s: [D.build_tokens(r, cfg) for r in D.load_split(cache, p, s, data_dir())] for s in ("train", "val", "test")}
    norm = D.fit_norm(tok["train"])
    tc = cfg["tokens"]
    return dict(tokens=tok, norm=norm, names=emg_channel_names(p, data_dir()),
                prior=compute_muscle_edge_prior([t["emg_full"] for t in tok["train"]]),
                train_ds=D.TokenWindows(tok["train"], norm, tc["window"], tc["stride"], random_offset=True,
                                        seed=cfg["training"]["seed"]),
                val_ds=D.TokenWindows(tok["val"], norm, tc["window"], tc["stride"]))


def standard(model, sd, cfg, device) -> pd.DataFrame:
    tc = cfg["tokens"]
    kw = dict(window=tc["window"], hop=tc["hop"], use_amp=cfg["training"]["use_amp"], clip=cfg["eeg"]["clip"])
    preds = [predict_series(model, t, sd["norm"], device, **kw) for t in sd["tokens"]["test"]]
    target = np.concatenate([t["emg_full"] for t in sd["tokens"]["test"]])
    return pd.concat([standard_metrics(np.concatenate([p[o] for p in preds]), target, sd["names"]).assign(output=o)
                      for o in ("K", "E", "F")], ignore_index=True)


def segments(model, sd, cfg, device, p: int, n_null: int, seed: int) -> pd.DataFrame:
    """Per task phase: does F, or a ridge from EEG tokens to the residual of K, beat K (model domain, token rate)?"""
    tc, norm = cfg["tokens"], sd["norm"]
    kw = dict(window=tc["window"], hop=tc["hop"], clip=cfg["eeg"]["clip"])
    h = int(tc.get("horizon", 0))

    def collect(toks, with_labels):
        k, f, y, x, lab = [], [], [], [], []
        for t in toks:
            pt = predict_tokens(model, t, norm, device, **kw)
            k.append(pt["K"]); f.append(pt["F"])
            y.append(to_model_domain(np.asarray(t["emg"], np.float32), norm["emg_scale"]))
            x.append(eeg_features(model, t, norm, device, **kw)["tokens"])
            if with_labels:
                n_orig = t["n_samples"] + h * Q
                lab.append(segment_labels(lifts(p, t["series"], n_orig), len(t["emg"]), FS_IN / Q, offset=h))
        cat = np.concatenate
        return cat(k), cat(f), cat(y), cat(x), (cat(lab) if with_labels else None)

    k_v, _, y_v, x_v, _ = collect(sd["tokens"]["val"], False)
    k_t, f_t, y_t, x_t, lab = collect(sd["tokens"]["test"], True)
    ridge, _ = fit_ridge(x_v, y_v - k_v)
    r_pred = _apply(ridge, x_t.astype(np.float64))
    res = y_t - k_t
    mse = lambda pred: ((pred - y_t) ** 2).mean(1)                      # per token, mean over muscles
    e_k, e_f, e_r = mse(k_t), mse(f_t), mse(np.maximum(k_t + r_pred, 0.0))
    hop = int(tc["hop"])
    shifts = np.arange(max(tc["window"] // hop, 1), len(res) // hop - max(tc["window"] // hop, 1) + 1) * hop
    rng = np.random.default_rng(seed)
    null_preds = [np.roll(r_pred, int(rng.choice(shifts)), axis=0) for _ in range(n_null)] if len(shifts) else []
    rows = []
    for name in ["pre_move", "reach", "load", "hold", "release", "other", "all"]:
        m = np.ones(len(res), bool) if name == "all" else (lab == name)
        if m.sum() < 10:
            continue
        sk = lambda e: float(1.0 - e[m].mean() / max(e_k[m].mean(), 1e-12))
        skill_r = sk(e_r)
        null = [sk(mse(np.maximum(k_t + q, 0.0))) for q in null_preds]
        r_res = float(np.mean(_r(r_pred[m], res[m])))
        rows.append({"segment": name, "n_tokens": int(m.sum()), "mse_K": float(e_k[m].mean()),
                     "skill_F": sk(e_f), "skill_ridge": skill_r,
                     "skill_ridge_null_mean": float(np.mean(null)) if null else np.nan,
                     "p_skill_ridge": float((1 + sum(v >= skill_r for v in null)) / (1 + len(null))) if null else np.nan,
                     "r_residual_ridge": r_res})
    return pd.DataFrame(rows)


def run_one(p: int, h_ms: float, args, device) -> Path:
    cfg = horizon_config(h_ms, args.smoke, args.amp)
    out = Path(args.results) / f"h{int(h_ms)}ms" / f"P{p}"
    if (out / "segments.csv").exists() and not args.overwrite:
        print(f"P{p} h={h_ms:g} ms: results exist, skipped ({out})")
        return out
    t0 = time.time()
    sd = subject_data(p, cfg)
    torch.manual_seed(cfg["training"]["seed"])
    model = build_model(cfg, sd["prior"]).to(device)
    hist = fit(model, sd["train_ds"], sd["val_ds"], cfg, device, Path(args.checkpoints) / f"h{int(h_ms)}ms" / f"P{p}",
               log=print)
    out.mkdir(parents=True, exist_ok=True)
    std = standard(model, sd, cfg, device).assign(subject=f"P{p}", horizon_ms=h_ms)
    std.to_csv(out / "standard.csv", index=False)
    n_null = int(cfg["evaluation"]["n_null"])
    res = residual_test(model, sd["tokens"]["val"], sd["tokens"]["test"], sd["norm"], cfg, device, sd["names"],
                        n_null=n_null).assign(subject=f"P{p}", horizon_ms=h_ms)
    res.to_csv(out / "residual.csv", index=False)
    seg = segments(model, sd, cfg, device, p, n_null, cfg["evaluation"]["seed"]).assign(subject=f"P{p}", horizon_ms=h_ms)
    seg.to_csv(out / "segments.csv", index=False)
    (out / "history.json").write_text(json.dumps({
        "epochs_run": hist["epochs_run"], "best_epoch_kin": hist["best_epoch_kin"], "best_epoch_eeg": hist["best_epoch_eeg"],
        "frozen_epoch": hist.get("frozen_epoch"), "minutes": (time.time() - t0) / 60,
        "gate": model.fusion.gate.detach().cpu().tolist(), "config_tokens": cfg["tokens"]}, indent=1), encoding="utf-8")
    piv = std.pivot(index="muscle", columns="output", values="vaf")
    mean = res[res.muscle == "MEAN"].set_index("features")
    pre = seg.set_index("segment")
    print(f"P{p} h={h_ms:g} ms: VAF K {piv['K'].mean():.1f}, F - K {piv['F'].mean() - piv['K'].mean():+.2f}, "
          f"E r {std[std.output == 'E']['r'].mean():.3f}; residual r tokens {mean.loc['tokens', 'r']:.3f} "
          f"(p {mean.loc['tokens', 'p']:.2f}); pre_move skill ridge "
          f"{pre['skill_ridge'].get('pre_move', np.nan):+.3f}, F {pre['skill_F'].get('pre_move', np.nan):+.3f}; "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return out


def summarise(results: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(results.glob("h*ms/P*")):
        if not (d / "segments.csv").exists():
            continue
        std, res, seg = (pd.read_csv(d / f) for f in ("standard.csv", "residual.csv", "segments.csv"))
        v = std.pivot(index="muscle", columns="output", values="vaf").mean()
        r = std.pivot(index="muscle", columns="output", values="r").mean()
        m = res[res.muscle == "MEAN"].set_index("features")
        s = seg.set_index("segment")
        rows.append({"horizon_ms": std.horizon_ms.iloc[0], "subject": std.subject.iloc[0],
                     "K_vaf": v["K"], "F_minus_K_vaf": v["F"] - v["K"], "E_r": r["E"], "K_r": r["K"],
                     "res_r_tokens": m.loc["tokens", "r"], "res_p_tokens": m.loc["tokens", "p"],
                     "res_r_bandpower": m.loc["bandpower", "r"], "res_p_bandpower": m.loc["bandpower", "p"],
                     "pre_move_skill_ridge": s["skill_ridge"].get("pre_move", np.nan),
                     "pre_move_p_ridge": s["p_skill_ridge"].get("pre_move", np.nan),
                     "pre_move_skill_F": s["skill_F"].get("pre_move", np.nan),
                     "all_skill_ridge": s["skill_ridge"].get("all", np.nan)})
    df = pd.DataFrame(rows).sort_values(["horizon_ms", "subject"])
    if not df.empty:
        df.to_csv(results / "summary.csv", index=False)
        g = df.groupby("horizon_ms")
        table = g.mean(numeric_only=True).assign(
            n=g.size(), n_res_sig=g.apply(lambda d: int(((d.res_p_tokens <= 0.05) & (d.res_r_tokens > 0)).sum())),
            n_pre_sig=g.apply(lambda d: int(((d.pre_move_p_ridge <= 0.05) & (d.pre_move_skill_ridge > 0)).sum())))
        print("\nmean over participants, per horizon:")
        print(table[["n", "K_vaf", "F_minus_K_vaf", "E_r", "res_r_tokens", "n_res_sig", "pre_move_skill_ridge",
                     "n_pre_sig", "pre_move_skill_F"]].round(3).to_string())
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--horizons-ms", default="0,200,500,1000")
    ap.add_argument("--results", default=str(ROOT / "results" / "v2_horizon"))
    ap.add_argument("--checkpoints", default=str(ROOT / "outputs" / "checkpoints_v2_horizon"))
    ap.add_argument("--amp", action="store_true", help="mixed precision (slower on a GTX 1660 Ti)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch and 5 null draws: a code check, not a result")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not a.summary_only:
        print("device:", device)
        for h in (float(x) for x in a.horizons_ms.split(",")):
            for p in parse_subjects(a.participants):
                run_one(p, h, a, device)
    summarise(Path(a.results))


if __name__ == "__main__":
    main()
