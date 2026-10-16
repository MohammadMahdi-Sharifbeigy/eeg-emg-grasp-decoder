"""Builds main_v3.ipynb (EEG-only model with a training-only kinematic guide). Run from the repo root:
env/Scripts/python.exe tools/build_main_v3_notebook.py"""
import json
from pathlib import Path

TITLE = """# EEG-base model with a kinematic guide used only in training

**Input: EEG band power only. Output: the EMG envelope of five muscles.** Kinematics are not available at test time.
During training a frozen kinematic teacher (the K stream of the v2 model, loaded from `outputs/checkpoints_v2/P*/best.pt`)
helps the EEG stream a little:

- `align`: the EEG tokens are pulled towards the teacher's tokens through a small linear head,
- `distill`: the EEG output E is pulled towards the teacher's output K (soft target),
- `kin_aux` (off by default): a head predicts a few kinematic features from the EEG tokens.

The help is one number, `guide.scale` (0.0, 0.05, 0.1, 0.3). Scale 0.0 is the control: the same model and seed with no help,
which is the EEG-only result of the v2 report. The guide fades out between epoch 40 and 60, so the end of every run is plain
EEG-to-EMG training. Early stopping and the best epoch use the validation loss of E alone, never a guide term. Saved checkpoints
hold the EEG student only.

**Success (see `docs/superpowers/specs/2026-10-16-eeg-base-kin-guide-design.md`)**

1. Mean r above the saved E of the v2 report (0.645) and above the scale-0.0 control, Wilcoxon p below 0.05 over the participants.
2. Close to the timing baseline `phase` (r about 0.81): within 0.05 of its mean r. Beating it is the stretch goal.
3. Not required: matching K (r 0.872).
4. The EEG shuffle and swap nulls must drop the score.

The timing baseline reads the event times from the whole recording, so it is a ceiling and not a real predictor. Outputs go to
`outputs/checkpoints_v3/` and `results/v3/`; the v2 files are only read.
"""

SETUP = """# V3-SETUP
import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from main.dataloader import emg_channel_names, describe_split
from main.preprocessing_emg_kin import compute_muscle_edge_prior
from v2 import data as D
from v3.config import CONFIG_V3
from v3.evaluate import evaluate_student, group_tests, subject_means
from v3.guide import build_heads
from v3.model import build_student
from v3.smoke import project_runtime
from v3.teacher import load_teacher, teacher_path
from v3.train import fit_guided

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""

CONFIG = """# V3-CONFIG
CONFIG = copy.deepcopy(CONFIG_V3)
subject_list = [1,2,3,4,5,6,7,8,9,10,11,12]                      # participants to run; each needs its cache and a v2 K checkpoint
GUIDE_SCALES = CONFIG["guide"]["scales"]                         # 0.0 is the control (no guide); drop scales to run fewer
SEEDS = [CONFIG["training"]["seed"]]                             # add a second seed to measure run-to-run variation
RESULTS = Path(CONFIG["outputs"]["results_dir"]); RESULTS.mkdir(parents=True, exist_ok=True)
CKPT = Path(CONFIG["outputs"]["checkpoint_dir"])
print(json.dumps({k: CONFIG[k] for k in ("tokens", "model", "guide", "teacher", "training")}, indent=1))
"""

BUDGET = """# V3-BUDGET
# A few guided steps on RANDOM data: peak GPU memory, seconds per step, projected time for max_epochs. Trains nothing real.
budget = project_runtime(CONFIG, device, n_windows=1100)
CONFIG["training"]["batch_size"] = budget["batch_size"]
CONFIG["training"]["gradient_accumulation_steps"] = budget["gradient_accumulation_steps"]
CONFIG["training"]["use_amp"] = budget["use_amp"]
print(pd.DataFrame(budget["candidates"]).round(3).to_string(index=False, na_rep=""))
print(f"chosen: mixed precision {'on' if budget['use_amp'] else 'off'}, batch size {budget['batch_size']}, "
      f"accumulation {budget['gradient_accumulation_steps']}, peak memory {budget['peak_gb']} of {budget['total_gb']} GB, "
      f"{budget['parameters']:,} student parameters")
n_runs = len(subject_list) * len(GUIDE_SCALES) * len(SEEDS)
print(f"projected time for {CONFIG['training']['max_epochs']} epochs: {budget['projected_minutes']:.1f} min per run (upper bound, early "
      f"stopping ends runs sooner); {n_runs} runs = {n_runs * budget['projected_minutes'] / 60:.1f} h at most")
"""

DATA = """# V3-DATA
subject_data = {}
for p_id in subject_list:
    t0 = time.time(); subj = f"P{p_id}"
    cache = CONFIG["data"]["cache_dir"]
    tok = {s: [D.build_tokens(r, CONFIG) for r in D.load_split(cache, p_id, s)] for s in ("train", "val", "test")}
    norm = D.fit_norm(tok["train"])
    prior = compute_muscle_edge_prior([t["emg_full"] for t in tok["train"]])
    tc = CONFIG["tokens"]
    subject_data[subj] = dict(
        p_id=p_id, tokens=tok, norm=norm, muscle_prior=prior, names=emg_channel_names(p_id),
        train_ds=D.TokenWindows(tok["train"], norm, tc["window"], tc["stride"], random_offset=True, seed=CONFIG["training"]["seed"]),
        val_ds=D.TokenWindows(tok["val"], norm, tc["window"], tc["stride"]),
    )
    sd = subject_data[subj]
    print(f"{subj}: train windows {len(sd['train_ds'])}, val windows {len(sd['val_ds'])}, {time.time() - t0:.0f} s")
    print(describe_split(p_id))
"""

TRAIN = """# V3-TRAIN
runs = {}
for subj, sd in subject_data.items():
    ck = teacher_path(CONFIG, sd["p_id"])
    if not ck.is_file():
        print(f"{subj}: no v2 K checkpoint at {ck}, skipped")
        continue
    teacher = load_teacher(CONFIG, sd["muscle_prior"], ck, device)       # frozen; never saved with the student
    for seed in SEEDS:
        for scale in GUIDE_SCALES:
            t0 = time.time()
            cfg = copy.deepcopy(CONFIG)
            cfg["guide"]["scale"] = scale
            cfg["training"]["seed"] = seed
            torch.manual_seed(seed)
            student, heads = build_student(cfg, sd["muscle_prior"]).to(device), build_heads(cfg)
            run_dir = CKPT / (f"g{scale}" if len(SEEDS) == 1 else f"g{scale}_s{seed}") / subj
            print(f"\\n=== {subj}, guide scale {scale}, seed {seed}: {sum(p.numel() for p in student.parameters()):,} student parameters ===")
            hist = fit_guided(student, None if scale == 0 else teacher, heads, sd["train_ds"], sd["val_ds"], cfg, device, run_dir)
            runs[(subj, scale, seed)] = dict(student=student, history=hist, cfg=cfg)
            print(f"{subj}: scale {scale}: {hist['epochs_run']} epochs, best epoch {hist['best_epoch']}, "
                  f"val E {min(hist['val_e']):.4f}, {(time.time() - t0) / 60:.1f} min")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    for (s, scale, seed), run in runs.items():
        if s != subj:
            continue
        h = run["history"]
        ax[0].plot(h["val_e"], label=f"scale {scale}"); ax[1].plot(h["w"]); ax[2].plot(h["grad_share"])
    ax[0].set_title(f"{subj}: validation loss of E"); ax[1].set_title("guide weight"); ax[2].set_title("guide share of the gradient")
    ax[0].legend(); [a.set_xlabel("epoch") for a in ax]
    fig.tight_layout(); fig.savefig(RESULTS / f"history_{subj}.png", dpi=130); plt.show()
"""

EVAL = """# V3-EVAL
std_parts, null_parts = [], []
for (subj, scale, seed), run in runs.items():
    sd = subject_data[subj]
    res = evaluate_student(run["student"], sd["tokens"]["test"], sd["norm"], run["cfg"], device, sd["names"])
    std_parts.append(res["standard"].assign(subject=subj, scale=scale, seed=seed))
    null_parts.append(res["nulls"].assign(subject=subj, scale=scale, seed=seed))
    print(f"{subj} scale {scale} seed {seed}: mean r {res['standard'].r.mean():.3f}, mean VAF {res['standard'].vaf.mean():.1f}, "
          f"swap-null r {res['nulls'].E_r_swap_mean.mean():.3f}, shuffle-null r {res['nulls'].E_r_shuffle_mean.mean():.3f}")
std_all, nulls_all = pd.concat(std_parts, ignore_index=True), pd.concat(null_parts, ignore_index=True)
std_all.to_csv(RESULTS / "standard.csv", index=False)
nulls_all.to_csv(RESULTS / "nulls.csv", index=False)
print("saved tables to", RESULTS)
"""

COMPARE = """# V3-COMPARE
# Average the seeds first, then compare with the saved E, K and the timing baseline (read-only files of the v2 report).
by_seed = std_all.groupby(["muscle", "subject", "scale"], as_index=False)[["r", "vaf"]].mean()
means = subject_means(by_seed, "results/v2/standard.csv", "eeg_controls/results/phase/phase_metrics.csv")
tests = group_tests(means)
means.to_csv(RESULTS / "compare.csv", index=False)
tests.to_csv(RESULTS / "group_tests.csv", index=False)
print(tests.round(3).to_string(index=False))
for _, g in tests.iterrows():
    print(f"scale {g['scale']}: mean r {g['mean_r']:.3f} (control {g['mean_control_r']:.3f}, saved E {g['mean_E_saved_r']:.3f}, "
          f"timing baseline {g['mean_phase_r']:.3f}, K {g['mean_K_r']:.3f}); better than the control in {g['n_better_than_control']} of {g['n_subjects']}")
    print(f"  beats the saved E and the control (p < 0.05): {g['beats_E']}  |  within 0.05 of the timing baseline: {g['close_to_timing']}  |  beats it: {g['beats_timing']}")
fig, ax = plt.subplots(figsize=(6.5, 4))
curve = means.groupby("scale").E_guided_r.mean()
ax.plot(curve.index, curve.values, "o-", color="#2a78d6", label="EEG student")
for col, label in (("E_saved_r", "saved E (v2)"), ("phase_r", "timing baseline"), ("cue_r", "cue baseline"), ("K_r", "K (kinematics)")):
    ax.axhline(means[col].mean(), ls="--", lw=1, color="gray"); ax.text(curve.index.max(), means[col].mean(), " " + label, va="center", fontsize=8)
ax.set_xlabel("guide scale"); ax.set_ylabel("mean r over participants and muscles"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(RESULTS / "guide_strength.png", dpi=130); plt.show()
print("If no scale beats the control, the kinematic guide does not help the EEG stream on this data; that is the result.")
"""


def _cell(kind, src):
    lines = src.rstrip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build() -> dict:
    code = (SETUP, CONFIG, BUDGET, DATA, TRAIN, EVAL, COMPARE)
    cells = [_cell("markdown", TITLE)] + [_cell("code", s) for s in code]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    Path("main_v3.ipynb").write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote main_v3.ipynb")
