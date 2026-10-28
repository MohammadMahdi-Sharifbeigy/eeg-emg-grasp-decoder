"""Builds main_v3.ipynb: the whole v3 analysis, one section per run step (v3/run.py), each with its explanation, its
tables and its figures. Run from the repo root:
env/Scripts/python.exe tools/build_main_v3_notebook.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v3.run import STEPS  # noqa: E402

TITLE = """# What EEG adds beyond timing (Critical experiment II)

This notebook runs and shows the whole v3 analysis, from the data to the verdict, in nine steps. Each step is one section:
a short explanation of what it does and what to look for, then one cell that runs it and shows its tables and figures.
Every cell can be re-run on its own: finished training runs and saved tables are reused, and each cell re-reads its
step's saved tables, so the notebook also works after a kernel restart.

## Why this analysis

- **v2** had three outputs: K (kinematics only, r 0.872), E (EEG only, r 0.645) and F (K plus a gated EEG correction).
  F stayed equal to K: the gate never opened, and the EEG explained almost nothing of K's error (residual r about 0.07).
- **The EEG controls** showed that a linear model that knows only the times of the task events (`phase`) reaches r 0.81
  and beats E in every participant; the LED cue alone reaches 0.58. Most of what E "knows" is the task clock.
- **The team review** pointed out that the EEG should be the base and kinematics only a guide, and **the team proposal**
  (Critical experiment II) turned the question into one fair comparison, which this notebook runs.

## The question

**Does EEG history improve the prediction of the EMG envelope beyond what past movement and past task events already
explain?** Every condition is evaluated on the same held-out test series of the same 12 participants.

| condition | input available at time t | role |
|---|---|---|
| `causal_cue` | past LED and event times only | measures the task timing |
| `K` | past hand kinematics and force | biomechanical explanation |
| `B` | K + past task events | **primary no-EEG baseline** |
| `F` | K + past task events + EEG history | **primary neural candidate** |
| `E0` | EEG only, trained without a guide | EEG-only reference |
| `Eguided` | EEG only, trained with the frozen K teacher | secondary training intervention |

Kinematics are never an input of `E0` or `Eguided` at test time: the teacher only guides their training.

## How it is measured

- **Primary metric:** `eeg_skill = 1 - MSE(F) / MSE(B)`, the share of B's error that the EEG removes, per participant,
  forecast horizon (0, 200, 500, 1000 ms) and task phase. Group test: one-sided Wilcoxon over participants, Holm over
  horizons (primary) and over phases.
- **Secondary:** VAF gain (VAF(F) - VAF(B)), CCC, the absolute MSE of every condition and the onset timing error. A
  positive pooled correlation alone is not enough. For comparison: matching the timing models is the null result,
  not the goal.
- **Controls:** F with EEG from a **phase-matched swap** (another trial of the same weight and surface, time-warped so all
  task events line up: same task timing, different trial), F from **peripheral-channel band power** (eyes, neck),
  the cue-aligned swap, and the **template test** (does the EEG predict how each trial deviates from the event-locked
  average EMG of the training series?). `phase` and `cue` read the whole recording (ceilings); the causal timing models
  use past events only.

## How to read the verdict (step 8)

| F beats B, survives the phase-matched swap, not peripheral | Eguided beats E0 | interpretation |
|---|---|---|
| yes | yes | EEG adds information beyond motion and timing; the guide also helps |
| yes | no | EEG adds information beyond motion and timing; the guide does not help |
| no | yes | the guide taught a better movement template: representation learning, not muscle-specific commands |
| no | no | no evidence that EEG adds information beyond motion and timing |

F beats B but fails a control: the gain is task-locked or artifact-related, never a positive claim.

## Run order

1. Steps 0 to 3: no training, minutes per participant and horizon. They give the primary result.
2. Step 8 can already run on step 3 alone (it then says the guide has not been evaluated).
3. Step 4 with the GPU free, then steps 5 and 6 (training: hours), then steps 7 and 8.

The same steps run from the command line (`env/Scripts/python.exe -m v3.run --step b_vs_f`); `v3/RUN.md` describes every
output column. Each step writes only into `results/v3/step{n}_{name}/`; checkpoints go to `outputs/checkpoints_v3/E0/`
and `outputs/checkpoints_v3/Eguided_g{scale}/`. The v2 files and checkpoints are only read.
"""

SETUP = """# STEP 0 setup
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from v3 import run, plots
from v3.conditions import condition_table

pd.set_option("display.max_columns", 40); pd.set_option("display.width", 200)

PARTICIPANTS = "all"          # "all", "1-4" or "1,3,5"
SEEDS = None                  # None: config training.seeds = [0, 1, 2]
SCALES = None                 # None: config guide.scales = [0.0, 0.05, 0.1, 0.3]; 0.0 is E0, the rest Eguided
PRESET = None                 # None, or an artifact control: "central", "peripheral", "eye_regressed", "no_gamma"

def show(fig):
    \"\"\"Show a figure inside this cell and free it.\"\"\"
    display(fig); plt.close(fig)

ctx = run.Context(participants=PARTICIPANTS, seeds=SEEDS, scales=SCALES, preset=PRESET)
run.step_setup(ctx)
display(condition_table()[["condition", "inputs", "role", "uses_eeg", "trained", "source"]])
"""

SECTIONS = {
    "data": ("""Loads, for every participant, the cached 500 Hz series of v2, turns them into 50 Hz tokens (EEG band power in 4
bands x 32 electrodes, kinematics, EMG envelope) and reads the task events of every lift (LED on, hand start, first touch,
lift-off, replace, release, hand stop). The split is v2's: 6 training series, 1 validation series, 2 test series.

The first run takes several minutes (band power and decimation of every series). The result is cached per participant
in `results/v3/step1_data/cache/P{p}.pkl`, so the next session loads in seconds. The cache is rebuilt by itself when the
token settings or the v2 cache files change; `run.Context(..., rebuild_data=True)` forces a rebuild.

**Look for:** every participant has 2 test series and a few dozen test lifts; a participant with no lifts cannot be
used by the swaps or the template test.""",
             """# STEP 1 data
data = run.step_data(ctx) or ctx.data
display(pd.DataFrame([{"participant": s, "train series": d["split"]["train"], "validation series": d["split"]["val"],
                       "test series": d["split"]["test"],
                       "test tokens": sum(len(t["emg"]) for t in d["tokens"]["test"]),
                       "test lifts": sum(len(l) for l in d["lifts"]["test"])} for s, d in data.items()]))"""),
    "timing_baselines": ("""The task clock without a brain. Four ridge models predict the envelope from event times only:
`causal_cue` (time since the last LED onset) and `events_causal` (time since the last occurrence of each task event), at
every horizon, using only the past; and the v2 controls `phase` and `cue`, which read the whole recording and are
ceilings. Fitted on training + validation, scored on the test series at 500 Hz and at 50 Hz.

**Look for:** how high a model with no EEG and no movement signal already gets. `phase` near r 0.81 and `cue` near 0.58
reproduce the v2 controls; the causal models are lower because they cannot see the future events.""",
                         """# STEP 2 timing_baselines
run.step_timing_baselines(ctx)
timing = run.load_table(ctx, "timing_baselines", "timing_baselines.csv")
if timing is not None:
    display(timing.groupby(["variant", "horizon_ms", "rate"]).r.mean().unstack("rate").round(3)
            .rename(columns=lambda c: f"mean r at {c} Hz"))
    show(plots.timing_baselines(timing))"""),
    "b_vs_f": ("""**The primary result.** For every participant and horizon, K is the kinematic stream of the causal v2 horizon model
(past hand kinematics and force). B adds a ridge from past task events to K's residual; F adds a ridge from past
events + EEG history. Both are fitted on the validation series and scored on the test series, so B and F differ only in
the EEG. The EEG of F comes from five sources: `eeg` (the real F) and four controls (`phase_swap`, `swap`, `bandpower`,
`peripheral`). The onset table gives, per lift and muscle, how early or late each condition predicts the muscle to switch
on. No training: this step reads `outputs/checkpoints_v2_horizon`.

**Look for:**
- the `eeg` line above 0 with a star (Holm p < 0.05 over horizons), and growing with the horizon;
- `phase_swap` near 0 and clearly below `eeg` (the gain is trial-specific, not the task clock);
- `peripheral` not as good as `bandpower` (the gain is not from eyes or neck muscles);
- where in the lift the gain sits (the phase heatmap): before movement and during the reach is what cortex would predict;
- the MSE ladder: how much each step (task clock, movement, events, EEG) removes;
- whether the group mean is carried by all participants or by a few.""",
               """# STEP 3 b_vs_f
run.step_b_vs_f(ctx)
b_table = run.load_table(ctx, "b_vs_f", "b_vs_f.csv")
b_summary = run.load_table(ctx, "b_vs_f", "b_vs_f_summary.csv")
b_onset = run.load_table(ctx, "b_vs_f", "onset.csv")
if b_summary is not None:
    main = b_summary[(b_summary.segment == "all") & (b_summary.source == "eeg")]
    display(Markdown("**Primary test: real EEG, all tokens**"))
    display(main[["horizon_ms", "n_subjects", "eeg_skill_mean", "eeg_skill_sem", "p_eeg_skill", "p_eeg_skill_holm_horizons",
                  "n_sig", "vaf_gain_mean", "ccc_B_mean", "ccc_F_mean", "mse_B_mean", "mse_F_mean"]].round(4))
    display(Markdown("**Controls: skill of each EEG source and its gap to the real EEG (all tokens)**"))
    display(b_summary[b_summary.segment == "all"].pivot_table(index="horizon_ms", columns="source",
            values=["eeg_skill_mean", "gap_mean", "p_gap_holm_horizons"]).round(4))
    show(plots.b_vs_f_horizon(b_summary))
    show(plots.b_vs_f_phases(b_summary, "eeg"))
    show(plots.b_vs_f_phases(b_summary, "phase_swap"))
    show(plots.b_vs_f_mse_ladder(b_summary))
if b_table is not None:
    show(plots.b_vs_f_participants(b_table))
if b_onset is not None:
    display(Markdown("**Median absolute onset error (ms) per condition and horizon**"))
    display(b_onset.assign(abs_ms=b_onset.error_ms.abs()).groupby(["horizon_ms", "condition"]).abs_ms.median()
            .unstack("condition").round(1))
    show(plots.onset_errors(b_onset, by="horizon_ms"))
for _, r in (main.iterrows() if b_summary is not None else []):
    ok = r.eeg_skill_mean > 0 and r.p_eeg_skill_holm_horizons < 0.05
    print(f"h = {int(r.horizon_ms)} ms: EEG removes {100 * r.eeg_skill_mean:.2f} % of B's error "
          f"(Holm p {r.p_eeg_skill_holm_horizons:.3g}): " + ("significant" if ok else "not significant"))"""),
    "budget": ("""Measures a few training steps on random data (with the GPU free) and projects the hours of steps 5 and 6: one run per
participant, seed and scale. It also sets the batch size and mixed precision used by the training steps. Each run is
bounded by the epoch limit; early stopping usually ends runs sooner, and guided runs last at least 60 epochs (the guide
fades out between epochs 40 and 60). Step 7's evaluation is not included.""",
               """# STEP 4 budget
budget = run.step_budget(ctx)
if budget:
    display(pd.DataFrame(budget["settings"]).T[["batch_size", "gradient_accumulation_steps", "use_amp", "peak_gb",
                                                  "sec_per_step", "projected_minutes"]])
    display(pd.Series(budget["per_run_minutes"], name="minutes per run at the epoch limit").rename_axis("scale"))"""),
    "train_E0": ("""**Training (long).** The EEG-only student without the guide (`guide.scale` 0), once per seed and participant. Its
input is EEG band power only. Early stopping and the best epoch use the validation loss of E alone. Finished runs are
skipped, so this cell can be re-run after an interruption.

**Look for:** validation curves that flatten (not still falling at the end) and similar curves across seeds.""",
                 """# STEP 5 train_E0
run.step_train_E0(ctx)
hist = run.load_histories(ctx)
e0 = hist[hist.condition == "E0"]
if len(e0):
    display(e0.groupby(["subject", "seed"]).agg(epochs=("epoch", "max"), best_epoch=("best_epoch", "first"),
                                                best_val_loss=("val_e", "min")).round(4))
    show(plots.training_curves(e0))"""),
    "train_Eguided": ("""**Training (long).** The same EEG-only student, now trained with the frozen K teacher: its tokens are pulled towards
the teacher's tokens (`align`) and its output towards K (`distill`), with a weight `guide.scale` that fades to zero
between epochs 40 and 60, so every run ends as plain EEG-to-EMG training. The teacher is never part of the saved model,
and its train / validation / test split is checked against this run first.

**Look for:** the guide weight falling to 0 and the guide's share of the gradient staying small: the guide helps,
it does not drive the training.""",
                      """# STEP 6 train_Eguided
run.step_train_Eguided(ctx)
hist = run.load_histories(ctx)
eg = hist[hist.condition == "Eguided"]
if len(eg):
    display(eg.groupby(["scale", "subject", "seed"]).agg(epochs=("epoch", "max"), best_epoch=("best_epoch", "first"),
            best_val_loss=("val_e", "min"), max_guide_share=("grad_share", "max")).round(4))
    show(plots.training_curves(hist))"""),
    "evaluate_students": ("""E0 and every Eguided scale on the test series, for every seed:
- accuracy (r, VAF, CCC) at 500 Hz (upsampled, as v2) and at 50 Hz (the model's own tokens, no upsampling);
- the shuffle and cue-aligned swap nulls on the EEG input, and the phase-matched swap;
- onset timing error per lift and muscle;
- the residual over each timing model (does the student explain what the task clock does not?);
- the template test (does the student predict how each trial differs from the average trial?).

**Look for:** Eguided above E0 in r (the secondary question), points above the diagonal in the swap plots
(trial-specific EEG), and positive residual and template skill. Similar numbers at 500 and 50 Hz mean the up- and
downsampling does not change the result.""",
                          """# STEP 7 evaluate_students
run.step_evaluate_students(ctx)
load = lambda f: run.load_table(ctx, "evaluate_students", f)
standard, swap, phase_swap = load("standard.csv"), load("swap.csv"), load("phase_swap.csv")
residual, templates, s_onset = load("residual_over_timing.csv"), load("template_deviation_summary.csv"), load("onset.csv")
if standard is not None:
    display(Markdown("**Mean r per condition and scale (mean over seeds, then participants)**"))
    display(standard.groupby(["condition", "scale", "rate", "subject"]).r.mean().groupby(["condition", "scale", "rate"])
            .agg(["mean", "sem"]).unstack("rate").round(3))
    show(plots.student_accuracy(standard))
if swap is not None or phase_swap is not None:
    display(Markdown("**Swaps: r with the real EEG, with another trial's EEG, and the difference (delta)**"))
    display(pd.concat({k: d.groupby(["condition", "scale"])[["r_true", "r_swap", "delta"]].mean()
                       for k, d in (("cue-aligned", swap), ("phase-matched", phase_swap)) if d is not None}, axis=1).round(3))
    show(plots.swap_scatter(swap, phase_swap))
if residual is not None:
    display(Markdown("**Residual over each timing model, all tokens (mean skill)**"))
    display(residual[residual.segment == "all"].groupby(["timing", "condition", "scale"]).skill.mean().unstack("timing").round(4))
    show(plots.residual_phases(residual, "cue_causal"))
    show(plots.residual_phases(residual, "events_causal"))
if templates is not None:
    display(Markdown("**Template test, all tokens: skill per source (mean over muscles)**"))
    display(templates[(templates.segment == "all") & (templates.muscle == "MEAN")]
            [["scale", "source", "n_subjects", "skill_mean", "skill_sem", "p_skill", "gap_mean", "p_gap"]].round(4))
    show(plots.template_skill(templates[templates.muscle != "MEAN"]))
if s_onset is not None:
    show(plots.onset_errors(s_onset, by="scale"))"""),
    "verdict": ("""Combines everything into one row per horizon: F beats B (Holm over horizons), F survives the phase-matched swap,
F is not explained by peripheral channels, Eguided beats E0 (seed-averaged, Holm over scales), the metrics behind them
(absolute MSE of every condition, VAF gain, CCC, onset error of B and F, the template test), and the interpretation of
the proposal. It runs on whatever steps exist: with step 3 alone it answers the primary question.""",
                "SEE_VERDICT_CODE"),
}

VERDICT_CODE = """# STEP 8 verdict
run.step_verdict(ctx)
verdict = run.load_table(ctx, "verdict", "verdict.csv")
students = run.load_table(ctx, "verdict", "eguided_vs_e0.csv")
if verdict is not None:
    display(verdict.set_index("horizon_ms")[["n_subjects", "eeg_skill", "p_eeg_skill_holm", "phase_swap_skill", "swap_gap",
            "bandpower_skill", "peripheral_skill", "vaf_gain", "mse_B", "mse_F", "onset_abs_ms_B", "onset_abs_ms_F",
            "F_beats_B", "F_survives_swap", "F_not_peripheral", "Eguided_beats_E0", "templates_E0",
            "templates_Eguided"]].T)
    show(plots.verdict_overview(verdict))
    for _, v in verdict.iterrows():
        display(Markdown(f"**{int(v.horizon_ms)} ms:** {v.interpretation}"))
if students is not None:
    display(Markdown("**Secondary question: Eguided against E0 (seed-averaged, Holm over scales)**"))
    display(students.round(4))"""

CLOSING = """## Writing it up

Report, for the paper, in this order:
1. The timing baselines (step 2): how much of the envelope the task clock alone explains.
2. B against F (step 3): eeg_skill per horizon with its Holm p, the MSE ladder, the phase heatmap, the per-participant
   plot and the onset errors. Claim an EEG contribution only with the phase-matched swap and peripheral controls passed.
3. E0 against Eguided (steps 7 and 8) as the secondary question, read through the interpretation table.
4. The limits: the timing models use events detected from the recordings; the default students are not causal;
   12 participants bound the smallest p-values; the artifact presets were not trained.
"""


def _cell(kind, src):
    lines = src.rstrip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build() -> dict:
    cells = [_cell("markdown", TITLE), _cell("markdown", "## Step 0: setup\n\nSettings, the run context and the six "
                                                       "conditions. Change the four settings below to run a subset."),
             _cell("code", SETUP)]
    for i, (name, _, trains) in enumerate(STEPS):
        if name == "setup":
            continue
        text, code = SECTIONS[name]
        tag = "trains" if trains else "no training"
        cells.append(_cell("markdown", f"## Step {i}: {name} ({tag})\n\n{text}"))
        cells.append(_cell("code", VERDICT_CODE if code == "SEE_VERDICT_CODE" else code))
    cells.append(_cell("markdown", CLOSING))
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    Path("main_v3.ipynb").write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote main_v3.ipynb")
