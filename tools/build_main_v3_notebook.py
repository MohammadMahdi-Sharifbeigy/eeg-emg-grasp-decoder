"""Builds main_v3.ipynb: the v3 analysis as numbered steps (v3/run.py). Run from the repo root:
env/Scripts/python.exe tools/build_main_v3_notebook.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v3.run import STEPS  # noqa: E402

TITLE = """# What EEG adds beyond timing (Critical experiment II)

**Question: does EEG history improve the prediction of the EMG envelope beyond what past movement and past task events
already explain?** Every condition is evaluated on the same held-out test series of the same 12 participants.

| condition | input available at time t | role |
|---|---|---|
| `causal_cue` | past LED and event times only | measures the task timing |
| `K` | past hand kinematics and force | biomechanical explanation |
| `B` | K + past task events | **primary no-EEG baseline** |
| `F` | K + past task events + EEG history | **primary neural candidate** |
| `E0` | EEG only, trained without a guide | EEG-only reference |
| `Eguided` | EEG only, trained with the frozen K teacher | secondary training intervention |

Kinematics are never an input of `E0` or `Eguided` at test time: the teacher only guides training.

**Primary metric:** `eeg_skill = 1 - MSE(F) / MSE(B)` per participant, horizon (0, 200, 500, 1000 ms) and task phase, with
VAF gain and CCC as secondary metrics, the absolute MSE of every condition and the onset timing error. A positive pooled
correlation alone is not enough. For comparison: a model that knows only the task event times (`phase`) reaches r 0.81 and
the LED cue alone 0.58, while the v2 EEG-only model reached 0.645. Matching the timing models is the null result, not the goal.

**Controls:** F with EEG from a phase-matched swap (another trial of the same weight and surface, time-warped so all task
events line up: same task timing, different trial), F from peripheral-channel band power, the cue-aligned swap, and the
test whether EEG predicts each trial's deviation from the event-locked average EMG of the training series. `phase` and `cue`
read the whole recording (ceilings); the causal timing models use past events only.

**Interpretation (step 8):** F beats B and survives the controls -> EEG adds information beyond motion and timing. Only
Eguided beats E0 -> the guide taught a better movement template, not evidence of muscle-specific commands. F beats B but fails
a control -> the gain is task-locked or artifact-related.

**Run order:** steps 0 to 3 first (no training, minutes per participant): they give the primary result. Then step 4 (GPU
budget), steps 5 and 6 (training: you run them), step 7 and step 8. Every step can be re-run: finished training runs and
cached tables are reused. The same steps run from the command line: `env/Scripts/python.exe -m v3.run --step b_vs_f`.
Each step writes only into `results/v3/step{n}_{name}/`; checkpoints go to `outputs/checkpoints_v3/E0/` and
`outputs/checkpoints_v3/Eguided_g{scale}/`. The v2 files are only read.
"""

SETUP = """# STEP 0 setup
from v3 import run
PARTICIPANTS = "all"          # "all", "1-4" or "1,3,5"
SEEDS = None                  # None: config training.seeds = [0, 1, 2]
SCALES = None                 # None: config guide.scales = [0.0, 0.05, 0.1, 0.3]; 0.0 is E0, the rest Eguided
PRESET = None                 # None, or an artifact control: "central", "peripheral", "eye_regressed", "no_gamma"
ctx = run.Context(participants=PARTICIPANTS, seeds=SEEDS, scales=SCALES, preset=PRESET)
run.step_setup(ctx)
"""

CODE = {
    "data": "data = run.step_data(ctx)",
    "timing_baselines": "timing_baselines = run.step_timing_baselines(ctx)",
    "b_vs_f": "b_vs_f = run.step_b_vs_f(ctx)\nb_vs_f.get(\"summary\")",
    "budget": "budget = run.step_budget(ctx)       # sets batch size and mixed precision for steps 5 and 6",
    "train_E0": "run.step_train_E0(ctx)",
    "train_Eguided": "run.step_train_Eguided(ctx)",
    "evaluate_students": "students = run.step_evaluate_students(ctx)",
    "verdict": "verdict = run.step_verdict(ctx)\nverdict.get(\"verdict\")",
}

NOTES = {
    "data": "Tokens (50 Hz), task events and the train / validation / test split of every participant, as in v2.",
    "timing_baselines": "The task clock without a brain: `causal_cue` and `events_causal` at every horizon, and the "
                        "non-causal ceilings `phase` and `cue`. Fitted on train + validation, scored on test.",
    "b_vs_f": "**The primary result.** K from the causal v2 horizon checkpoints; B and F are ridges on K's residual with "
              "past events, or past events + EEG. Sources of F: `eeg` (the real one), `phase_swap`, `swap`, "
              "`bandpower`, `peripheral`. Tables per participant, horizon, phase and source, plus onset timing.",
    "budget": "GPU hours of steps 5 and 6 on random data. Run it with the GPU free; on the CPU it prints CPU hours.",
    "train_E0": "Training (you run it): the EEG-only student without the guide, every seed.",
    "train_Eguided": "Training (you run it): the EEG-only student with the frozen K teacher, every guided scale and "
                     "seed. The teacher's split is checked first.",
    "evaluate_students": "E0 and Eguided on the test series: r, VAF, CCC at 500 and 50 Hz, the shuffle and swap nulls, "
                         "onset timing, the cue-aligned and phase-matched swaps, the residual over each timing model and "
                         "the template deviation test.",
    "verdict": "One row per horizon: F beats B, F survives the phase-matched swap, F is not peripheral, Eguided beats E0, "
               "the metrics behind them and the interpretation.",
}


def _cell(kind, src):
    lines = src.rstrip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build() -> dict:
    cells = [_cell("markdown", TITLE), _cell("code", SETUP)]
    for i, (name, text, trains) in enumerate(STEPS):
        if name == "setup":
            continue
        cells.append(_cell("markdown", f"## Step {i}: {name}\n\n{NOTES[name]}"))
        cells.append(_cell("code", f"# STEP {i} {name}\n{CODE[name]}"))
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    Path("main_v3.ipynb").write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote main_v3.ipynb")
