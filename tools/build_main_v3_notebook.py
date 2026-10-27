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

The help is one number, `guide.scale` (0.0, 0.05, 0.1, 0.3). Scale 0.0 is the control: the same model with no help (the
EEG-only result of the v2 report reached r 0.645). Every scale is trained with 3 seeds. The guide fades out between epoch 40
and 60. Early stopping and the best epoch use the validation loss of E alone. Saved checkpoints hold the EEG student only.

**The question: does the EEG model explain EMG beyond the task clock?** A linear model that knows only the task event times
(`phase`, r about 0.81) beats E in every participant, and the LED cue alone reaches r about 0.58. The teacher's tokens mostly
encode the task phase, so a guide can raise r by teaching the EEG the task timing better. That is the known confound, not
motor information, and the shuffle and swap nulls cannot see it because they destroy the timing too. The criteria
(`v3.evaluate.criteria`, p-values Holm-corrected over the scales):

1. `beats_control`: the guided student beats the scale-0 control (seed-averaged, Wilcoxon over participants).
2. `beyond_timing` (primary): a ridge from the student's tokens explains part of the residual of the causal cue model
   `cue_causal` (skill above 0 against a time-shift null, group test).
3. `survives_swap`: the student loses accuracy when every test trial gets the cue-aligned EEG of another trial (delta > 0), and
   the guide's gain shows in delta too.
4. `timing_level`: information only. Being within 0.05 of the `phase` model is what a model without a brain already does:
   it is the null result, not the goal.

The verdict per scale: (a) the guide helps and the gain is beyond timing; (b) it helps, but only through timing; (c) no help;
(d) worse. `phase` and `cue` read the event times from the whole recording, so they are a ceiling; `cue_causal` and
`events_causal` use past events only.

Two more tables. `V3-MUSCLE`: does the student explain what is specific to each muscle, once the timing model and the
activation shared by all five muscles are removed? `V3-CONTRAST` (rows C and F of the paper's table, no training): C is the
causal kinematic stream K of the v2 horizon models (past hand kinematics and force) plus past task cues, F is C plus EEG
history; delta = skill(F) - skill(C) per participant, phase and horizon (0, 200, 500, 1000 ms), with the cue-aligned
trial swap and the peripheral channels as negative controls.

**Run order** (cheap parts first): run every cell with `STAGE = "control"` (timing models, then scale 0 with the first seed for
all participants, then the swap and residual cells on it). Check the swap and residual tables, then set `STAGE = "all"` and
run `V3-BUDGET` to the end again: finished runs are skipped. Outputs go to `outputs/checkpoints_v3/` and `results/v3/`; the
v2 files are only read.
"""

SETUP = """# V3-SETUP
import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from eeg_controls.common import lifts
from main.dataloader import emg_channel_names, describe_split
from main.preprocessing_emg_kin import compute_muscle_edge_prior
from v2 import data as D
from v3.config import CONFIG_V3, preset_config
from v3.contrast import contrast_group_test
from v3.contrast import run_participant as contrast_participant
from v3.evaluate import (criteria, evaluate_student, muscle_group_test, muscle_specificity, residual_group_test,
                         residual_over_timing, subject_means)
from v3.guide import build_heads
from v3.model import build_student
from v3.nulls import evaluate_swap, swap_group_test
from v3.smoke import run_budget
from v3.teacher import check_teacher_split, load_teacher, teacher_path
from v3.timing import CAUSAL, cached_timing, score_timing
from v3.train import fit_guided, run_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""

CONFIG = """# V3-CONFIG
PRESET = None                                    # None, or an artifact control: "central", "peripheral", "eye_regressed", "no_gamma"
CONFIG = copy.deepcopy(CONFIG_V3) if PRESET is None else preset_config(PRESET)
subject_list = [1,2,3,4,5,6,7,8,9,10,11,12]      # participants to run; each needs its cache and a v2 K checkpoint
GUIDE_SCALES = CONFIG["guide"]["scales"]         # 0.0 is the control (no guide)
SEEDS = CONFIG["training"]["seeds"]              # [0, 1, 2]: every test averages the seeds per participant
STAGE = "control"                                # "control": scale 0 with the first seed only; "all": every scale and seed
run_scales = [0.0] if STAGE == "control" else GUIDE_SCALES
run_seeds = SEEDS[:1] if STAGE == "control" else SEEDS
RESULTS = Path(CONFIG["outputs"]["results_dir"]) / (PRESET or ""); RESULTS.mkdir(parents=True, exist_ok=True)
CKPT = Path(CONFIG["outputs"]["checkpoint_dir"]) / (PRESET or "")
TIMING_CACHE = RESULTS / "timing_preds"          # per-sample predictions of the timing models
print(f"stage {STAGE}: scales {run_scales}, seeds {run_seeds}; preset {PRESET}")
print(json.dumps({k: CONFIG[k] for k in ("tokens", "model", "guide", "timing", "training")}, indent=1))
"""

BUDGET = """# V3-BUDGET
# A few steps on RANDOM data per kind of run (control: no teacher; guided: with the teacher). Trains nothing real.
budget = run_budget(CONFIG, device, GUIDE_SCALES, SEEDS, len(subject_list), n_windows=1100)
chosen = budget["settings"].get("guided", budget["settings"].get("control"))     # the heavier setting fits both kinds
for key in ("batch_size", "gradient_accumulation_steps", "use_amp"):
    CONFIG["training"][key] = chosen[key]
for kind, st in budget["settings"].items():
    print(f"{kind}: mixed precision {'on' if st['use_amp'] else 'off'}, batch {st['batch_size']}, peak {st['peak_gb']} of "
          f"{st['total_gb']} GB, {st['projected_minutes']:.1f} min per run at {budget['max_epochs']} epochs")
stage_hours = sum(budget["per_run_minutes"][float(s)] for s in run_scales) * len(run_seeds) * len(subject_list) / 60
print(f"this stage ({STAGE}): at most {stage_hours:.1f} GPU hours")
print(f"full plan: {len(GUIDE_SCALES)} scales x {len(SEEDS)} seeds x {len(subject_list)} participants = {budget['n_runs']} runs, "
      f"at most {budget['total_gpu_hours']:.1f} GPU hours (upper bound: early stopping ends most runs sooner, guided runs "
      f"last at least {budget['min_epochs_guided']} epochs)")
"""

DATA = """# V3-DATA
subject_data = {}
for p_id in subject_list:
    t0 = time.time(); subj = f"P{p_id}"
    cache = CONFIG["data"]["cache_dir"]
    tok = {s: [D.build_tokens(r, CONFIG) for r in D.load_split(cache, p_id, s)] for s in ("train", "val", "test")}
    for s in tok:                                      # task events of every lift, for the timing models and the trial swap
        for t in tok[s]:
            t["lifts"] = lifts(p_id, t["series"], t["n_samples"])
    norm = D.fit_norm(tok["train"])
    prior = compute_muscle_edge_prior([t["emg_full"] for t in tok["train"]])
    tc = CONFIG["tokens"]
    subject_data[subj] = dict(
        p_id=p_id, tokens=tok, norm=norm, muscle_prior=prior, names=emg_channel_names(p_id),
        split={s: [t["series"] for t in tok[s]] for s in tok},
        lifts={s: [t["lifts"] for t in tok[s]] for s in tok},
        train_ds=D.TokenWindows(tok["train"], norm, tc["window"], tc["stride"], random_offset=True, seed=CONFIG["training"]["seed"]),
        val_ds=D.TokenWindows(tok["val"], norm, tc["window"], tc["stride"]),
    )
    sd = subject_data[subj]
    print(f"{subj}: train windows {len(sd['train_ds'])}, val windows {len(sd['val_ds'])}, "
          f"test lifts {[len(l) for l in sd['lifts']['test']]}, {time.time() - t0:.0f} s")
    print(describe_split(p_id))
"""

TIMING = """# V3-TIMING
# The task clock without a brain: timing models fitted on train + validation and scored on the test series (no training).
# phase and cue read the whole recording (ceilings); cue_causal and events_causal use only past events, at every horizon.
tcfg, timing, rows = CONFIG["timing"], {}, []
for subj, sd in subject_data.items():
    for variant in tcfg["variants"]:
        for h in (tcfg["horizons_ms"] if variant in CAUSAL else [0]):
            fitted = cached_timing(TIMING_CACHE, sd["p_id"], sd["tokens"], variant, h)
            timing[(subj, variant, h)] = fitted
            rows.append(score_timing(fitted, sd["tokens"]["test"], sd["names"]).assign(subject=subj))
timing_causal = pd.concat(rows, ignore_index=True)
timing_causal.to_csv(RESULTS / "timing_causal.csv", index=False)
summary = timing_causal.groupby(["variant", "horizon_ms", "rate"]).r.mean().unstack("rate").round(3)
print("mean r over participants and muscles (columns: scoring rate in Hz):")
print(summary.to_string())
"""

TRAIN = """# V3-TRAIN
runs = {}
for subj, sd in subject_data.items():
    ck = teacher_path(CONFIG, sd["p_id"])
    if not ck.is_file():
        print(f"{subj}: no v2 K checkpoint at {ck}, skipped")
        continue
    check_teacher_split(ck, sd["p_id"], sd["split"])                # raises if the teacher saw a v3 validation or test series
    teacher = load_teacher(CONFIG, sd["muscle_prior"], ck, device)   # frozen; never saved with the student
    for seed in run_seeds:
        for scale in run_scales:
            t0 = time.time()
            cfg = copy.deepcopy(CONFIG)
            cfg["guide"]["scale"] = scale
            cfg["training"]["seed"] = seed
            torch.manual_seed(seed)
            student, heads = build_student(cfg, sd["muscle_prior"]).to(device), build_heads(cfg)
            folder = run_dir(CKPT, scale, seed, subj)
            print(f"\\n=== {subj}, guide scale {scale}, seed {seed} -> {folder} ===")
            hist = fit_guided(student, None if scale == 0 else teacher, heads, sd["train_ds"], sd["val_ds"], cfg, device, folder)
            runs[(subj, scale, seed)] = dict(student=student, history=hist, cfg=cfg)
            print(f"{subj}: scale {scale}, seed {seed}: {hist['epochs_run']} epochs, best epoch {hist['best_epoch']}, "
                  f"val E {min(hist['val_e']):.4f}, {(time.time() - t0) / 60:.1f} min")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    for (s, scale, seed), run in runs.items():
        if s == subj:
            h = run["history"]
            ax[0].plot(h["val_e"], label=f"scale {scale}, seed {seed}"); ax[1].plot(h["w"]); ax[2].plot(h["grad_share"])
    ax[0].set_title(f"{subj}: validation loss of E"); ax[1].set_title("guide weight"); ax[2].set_title("guide share of the gradient")
    ax[0].legend(fontsize=7); [a.set_xlabel("epoch") for a in ax]
    fig.tight_layout(); fig.savefig(RESULTS / f"history_{subj}.png", dpi=130); plt.show()
"""

EVAL = """# V3-EVAL
std_parts, null_parts = [], []
for (subj, scale, seed), run in runs.items():
    sd = subject_data[subj]
    res = evaluate_student(run["student"], sd["tokens"]["test"], sd["norm"], run["cfg"], device, sd["names"])
    std_parts.append(res["standard"].assign(subject=subj, scale=scale, seed=seed))
    null_parts.append(res["nulls"].assign(subject=subj, scale=scale, seed=seed))
std_all, nulls_all = pd.concat(std_parts, ignore_index=True), pd.concat(null_parts, ignore_index=True)
std_all.to_csv(RESULTS / "standard.csv", index=False)
nulls_all.to_csv(RESULTS / "nulls.csv", index=False)
print("mean r per scale, scored at 500 Hz (upsampled, as v2) and at 50 Hz (the model's own tokens, no upsampling):")
print(std_all.groupby(["scale", "rate"]).r.mean().unstack("rate").round(3).to_string())
print("shuffle and swap nulls (they destroy the task timing too, so they cannot tell timing from trial-specific EEG):")
print(nulls_all.groupby("scale")[["E_r", "E_r_shuffle_mean", "E_r_swap_mean"]].mean().round(3).to_string())
"""

SWAP = """# V3-SWAP
# Timing-preserving null: each test trial keeps its EMG target but gets the EEG of another trial of the same participant,
# aligned on the LED cue (v3.nulls). Set CONFIG["evaluation"]["swap_match_condition"] = True to draw donors only from
# trials with the same weight and surface.
swap_parts = []
for (subj, scale, seed), run in runs.items():
    sd = subject_data[subj]
    df = evaluate_swap(run["student"], sd["tokens"]["test"], sd["lifts"]["test"], sd["norm"], run["cfg"], device, sd["names"])
    swap_parts.append(df.assign(subject=subj, scale=scale, seed=seed))
swap_all = pd.concat(swap_parts, ignore_index=True)
swap_all.to_csv(RESULTS / "swap.csv", index=False)
swap_tests = swap_group_test(swap_all)
print(swap_tests.round(4).to_string(index=False))
print("If r_swap is close to r_true, the model's accuracy is task timing; delta = r_true - r_swap is the trial-specific "
      "EEG information. A guide that helps beyond timing must raise delta, not only r_true.")
"""

RESIDUAL = """# V3-RESIDUAL
# How much of the residual of each timing model (y - y_timing) does the student explain? Ridge from the student's final
# tokens, cross-validated over trials, time-shift null; per task phase ("other" = rest).
res_parts = []
for (subj, scale, seed), run in runs.items():
    sd = subject_data[subj]
    for variant in CONFIG["timing"]["variants"]:
        fitted = timing[(subj, variant, 0)]
        df = residual_over_timing(run["student"], sd["tokens"]["test"], [t["u"] for t in fitted["test"]], sd["lifts"]["test"],
                                  sd["norm"], run["cfg"], device, sd["names"])
        res_parts.append(df.assign(subject=subj, scale=scale, seed=seed, timing=variant))
residual_all = pd.concat(res_parts, ignore_index=True)
residual_all.to_csv(RESULTS / "residual_over_timing.csv", index=False)
residual_tests = residual_group_test(residual_all)
show = residual_tests[residual_tests.segment == "all"][["timing", "scale", "n_subjects", "mean_skill", "p_skill", "mean_r_direct"]]
print(show.round(4).to_string(index=False))
if not CONFIG["model"].get("causal", False):
    print("note: the student is not causal (it sees EEG up to 4 s ahead inside a window), the causal timing models only the past. "
          "For the strictest comparison train the student with model.causal and tokens.causal set to True.")
"""

MUSCLE = """# V3-MUSCLE
# Muscle specificity: the residual of the primary timing model with the activation shared by all five muscles regressed out
# (its first principal component, or the muscle mean). Does a ridge from the student's tokens predict what is left, per
# muscle? Cross-validated over trials, time-shift null; the group test is Holm-corrected over the five muscles.
mus_parts, primary = [], CONFIG["timing"]["primary"]
for (subj, scale, seed), run in runs.items():
    sd = subject_data[subj]
    fitted = timing[(subj, primary, 0)]
    for shared in CONFIG["muscle"]["shared"]:
        df = muscle_specificity(run["student"], sd["tokens"]["test"], [t["u"] for t in fitted["test"]], sd["lifts"]["test"],
                                sd["norm"], run["cfg"], device, sd["names"], shared=shared)
        mus_parts.append(df.assign(subject=subj, scale=scale, seed=seed, timing=primary))
muscle_all = pd.concat(mus_parts, ignore_index=True)
muscle_all.to_csv(RESULTS / "muscle_specificity.csv", index=False)
muscle_tests = muscle_group_test(muscle_all)
muscle_tests.to_csv(RESULTS / "muscle_tests.csv", index=False)
print("mean skill over participants (rows: shared component and scale):")
print(muscle_tests.pivot_table(index=["shared", "scale"], columns="muscle", values="mean_skill").round(4).to_string())
print("Holm p over the five muscles:")
print(muscle_tests.pivot_table(index=["shared", "scale"], columns="muscle", values="p_holm_muscles").round(3).to_string())
print("share of the timing residual that the shared activation removed:", round(float(muscle_all.var_shared.mean()), 3))
"""

VERDICT = """# V3-VERDICT
means = subject_means(std_all, "results/v2/standard.csv", "eeg_controls/results/phase/phase_metrics.csv")
means.to_csv(RESULTS / "compare.csv", index=False)
crit = criteria(means, swap_tests, residual_tests, primary=CONFIG["timing"]["primary"])
crit.to_csv(RESULTS / "verdict.csv", index=False)
print("per participant (mean over seeds, seed SD next to it):")
print(means[["subject", "scale", "E_guided_r", "E_guided_r_seed_sd", "E_saved_r", "phase_r", "K_r"]].round(3).to_string(index=False))
if crit.empty:
    print("no guided scale has been evaluated yet: set STAGE = 'all' and run again")
for _, c in crit.iterrows():
    print(f"\\nscale {c['scale']}: mean r {c['mean_r']:.3f} (seed SD {c['mean_r_seed_sd']:.3f}), control {c['mean_control_r']:.3f}")
    print(f"  beats_control {c['beats_control']} (Holm p {c['p_gain_holm']:.3g}) | beyond_timing {c['beyond_timing']} "
          f"(skill {c['residual_skill']:.4f}, Holm p {c['p_residual_holm']:.3g}) | survives_swap {c['survives_swap']} "
          f"(delta {c['mean_delta']:.3f}, gain {c['mean_delta_gain']:.3f})")
    print(f"  timing_level {c['timing_level']} (information only: matching the phase model {c['mean_phase_r']:.3f} is the null result)")
    print(f"  verdict ({c['outcome']}): {c['verdict']}")
scales = sorted(swap_all.scale.unique())
fig, ax = plt.subplots(1, 1 + len(scales), figsize=(4.5 * (1 + len(scales)), 4), squeeze=False)
per = swap_all.groupby(["subject", "scale"], as_index=False)[["r_true", "r_swap"]].mean()
for scale, d in per.groupby("scale"):
    ax[0, 0].scatter(d.r_swap, d.r_true, label=f"scale {scale}", s=18)
lim = [per[["r_true", "r_swap"]].min().min() - 0.02, per[["r_true", "r_swap"]].max().max() + 0.02]
ax[0, 0].plot(lim, lim, color="gray", lw=1); ax[0, 0].set_xlabel("r_swap (EEG of another trial)"); ax[0, 0].set_ylabel("r_true")
ax[0, 0].set_title("trial swap, one point per participant"); ax[0, 0].legend(fontsize=7)
rt = residual_tests[(residual_tests.timing == CONFIG["timing"]["primary"]) & (residual_tests.segment != "all")]
for i, scale in enumerate(scales, start=1):
    d = rt[rt.scale == scale]
    ax[0, i].bar(d.segment, d.mean_skill, yerr=d.sem_skill, color="#2a78d6"); ax[0, i].axhline(0, color="gray", lw=1)
    ax[0, i].set_title(f"skill over {CONFIG['timing']['primary']}, scale {scale}"); ax[0, i].tick_params(axis="x", rotation=45)
fig.tight_layout(); fig.savefig(RESULTS / "swap_and_residual.png", dpi=130); plt.show()
"""

CONTRAST = """# V3-CONTRAST
# Rows C and F of the paper's table (v3.contrast; no training, it reads the causal v2 horizon models):
#   C = K (past hand kinematics and force) + past task cues;  F = C + EEG history;  delta = skill(F) - skill(C).
# Null: the EEG term of F shifted in time. Controls: EEG of a cue-aligned swapped trial, band power of all channels and of
# the peripheral channels only. Finished participants and horizons are read back from RESULTS / "contrast".
cc = CONFIG["contrast"]
(RESULTS / "contrast").mkdir(exist_ok=True)
con_parts = []
for h in cc["horizons_ms"]:
    for p_id in subject_list:
        ck = Path(cc["checkpoint_dir"]) / f"h{h}ms" / f"P{p_id}" / "best.pt"
        done = RESULTS / "contrast" / f"h{h}ms_P{p_id}.csv"
        if done.exists():
            con_parts.append(pd.read_csv(done)); continue
        if not ck.is_file():
            print(f"P{p_id} h={h} ms: no horizon checkpoint at {ck}, skipped"); continue
        t0 = time.time()
        df = contrast_participant(p_id, h, cc["checkpoint_dir"], n_null=CONFIG["evaluation"]["n_null"], sources=cc["sources"])
        df.to_csv(RESULTS / "contrast" / f"h{h}ms_P{p_id}.csv", index=False)
        con_parts.append(df)
        print(f"P{p_id} h={h} ms: delta (all) {df[(df.segment == 'all') & (df.source == 'eeg')].delta.iloc[0]:+.4f}, "
              f"{(time.time() - t0) / 60:.1f} min")
contrast_all = pd.concat(con_parts, ignore_index=True)
contrast_all.to_csv(RESULTS / "contrast.csv", index=False)
contrast_tests = contrast_group_test(contrast_all)
contrast_tests.to_csv(RESULTS / "contrast_tests.csv", index=False)
allrows = contrast_tests[contrast_tests.segment == "all"]
print("mean skill(C) and delta = skill(F) - skill(C) over participants, all tokens (rows: EEG source, columns: horizon in ms):")
print(allrows.pivot_table(index="source", columns="horizon_ms", values=["mean_skill_C", "mean_delta"]).round(4).to_string())
print("Holm p of delta > 0 (over phases and horizons; the 'all' rows over horizons), per phase:")
print(contrast_tests.pivot_table(index=["source", "segment"], columns="horizon_ms", values="p_delta_holm").round(3).to_string())
print("A real EEG gain needs delta > 0 for 'eeg' AND above the swap control (p_gap): the swap keeps the task timing.")
fig, ax = plt.subplots(figsize=(6, 4))
for src, d in allrows.groupby("source"):
    ax.errorbar(d.horizon_ms, d.mean_delta, yerr=d.sem_delta, marker="o", capsize=3, label=src)
ax.axhline(0, color="gray", lw=1); ax.set_xlabel("horizon (ms)"); ax.set_ylabel("skill(F) - skill(C)")
ax.set_title("EEG history over causal motion + cues"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "contrast.png", dpi=130); plt.show()
"""


def _cell(kind, src):
    lines = src.rstrip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build() -> dict:
    code = (SETUP, CONFIG, BUDGET, DATA, TIMING, TRAIN, EVAL, SWAP, RESIDUAL, MUSCLE, VERDICT, CONTRAST)
    cells = [_cell("markdown", TITLE)] + [_cell("code", s) for s in code]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    Path("main_v3.ipynb").write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote main_v3.ipynb")
