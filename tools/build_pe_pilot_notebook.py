"""Builds pe_pilot.ipynb: the prediction-error pilot, one section per step of pe_pilot/run.py. Run from the repo root:
env/Scripts/python.exe tools/build_pe_pilot_notebook.py"""
import json
from pathlib import Path

TITLE = """# Prediction-error pilot on WAY-EEG-GAL

The EEG-to-EMG line ended in a null result: once past movement and past task events are known, the EEG adds nothing
(v3, B vs F). This pilot asks a different question of the same data, one the task was designed for: **what does the
EEG show when the object is heavier or lighter than the participant expected?**

In the weight series the weight changes without warning. Each participant has 38 lifts that are heavier and 38 that
are lighter than the lift before. The force data give the exact moment the expectation fails. The dataset paper showed
this effect only for one participant (P3), with no statistical test.

**No training.** Every step reads the raw WS files and `P{p}_AllLifts.mat`. Step 1 builds a cache per participant
(about 30 s each the first time); later steps take seconds to minutes. Each step writes to
`results/pe_pilot/step{n}_{name}/`.

## The key control

A lift of the 660 g object after a lighter one (unexpected) is compared with a lift of the **same 660 g object** after
another 660 g lift (expected). The object is physically the same; only the expectation differs. The difference is
taken within each current weight and then averaged (`pe_pilot.stats.matched_contrast`). Lifts where the surface also
changed are left out. Participants are the unit of every group test.

## Go / no-go, fixed before looking at the results

| analysis | go if | why |
|---|---|---|
| 1 behaviour | heavier: longer load phase and lower grip-force rate in most participants | the labels work (classic Johansson and Westling effect) |
| 2 error | an EEG cluster (p < 0.05) over central channels, ideally starting before the EMG and force clusters | an EEG response to the error that the movement does not explain |
| 3 expectation | `eeg_pre_contact` AUC > 0.5 over participants (p < 0.05), above the EMG and hand-speed decoders, and the pre-cue decoder clearly weaker | the EEG encodes the expected weight before contact, not leftovers from the last lift |
| 4 learning | exploratory only | too few lifts per participant for a firm claim |

If 2 and 3 both fail, the pilot says this question is not worth a paper on this data.
"""

SETUP = """
from IPython.display import Image, display
import pandas as pd

from pe_pilot.config import CONFIG
from pe_pilot.run import run

PARTICIPANTS = CONFIG["participants"]          # e.g. [1, 2, 3] for a quick look
RESULTS = CONFIG["results_dir"]
pd.set_option("display.width", 160)

def figure(step, name):
    display(Image(filename=f"{RESULTS}/{step}/{name}"))"""

SECTIONS = [
    ("""## Step 1: epochs

For every lift: EEG (common average reference; slow potentials 0.5 to 30 Hz; alpha and beta power in dB relative to
rest before the LED), the EMG envelope of the five muscles and the grip and load force, from 0.5 s before to 1.5 s after
both digits touch the object. Also window means before contact for the decoder of step 3.

**Artifacts.** In this data the median lift already has about 300 uV peak-to-peak in two seconds of the central
channels, so a fixed threshold rejects everything. A lift is rejected when it lies above the participant's median plus
3 robust SD. Expect about 10 to 20 rejected lifts per participant.

**Look for:** about 34 heavier, 34 lighter and 179 expected lifts per participant (first lifts of a series and lifts
with a surface change have no label).""",
     """out = run(["epochs"], PARTICIPANTS)
out["counts"]"""),
    ("""## Step 2: behaviour (sanity check)

Weight-matched difference, unexpected minus expected, of the peak grip-force rate (`dGF_Max`), the peak grip force and
the load-phase duration. One-sided expectations from the literature: an unexpectedly heavier object is lifted with a
grip-force rate planned for a lighter one (negative `dGF_Max` difference) and needs a longer load phase; an unexpectedly
lighter object gets a too-high rate.

**Look for:** the signs above in most participants. If they fail, the labels or the lift table are wrong, and nothing
after this step can be read.""",
     """out = run(["behaviour"], PARTICIPANTS)
display(out["behaviour"])
figure("step2_behaviour", "behaviour.png")"""),
    ("""## Step 3: error response

Group-mean matched contrast for the slow EEG, alpha and beta power, the EMG envelopes and the forces, touch at 0 s. The
cluster test flips the sign of each participant's contrast (EEG clusters over neighbouring channels and time; each EMG
and force signal over time alone). `first_s` is where the earliest significant cluster starts.

**Look for:** an EEG cluster over C3, Cz or CP channels (the right hand lifts, so left sensorimotor cortex). Compare its
`first_s` with that of `force` and `emg`: an EEG effect that starts after the force difference may simply follow the
movement correction. A cluster led by Fp, F7/F8, T or O channels points to eye or muscle artifacts.""",
     """out = run(["error"], PARTICIPANTS)
display(out["error"])
figure("step3_error", "error_heavier.png")
figure("step3_error", "error_lighter.png")"""),
    ("""## Step 4: expectation before contact

Leave-one-series-out logistic decoder of the previous weight (165 g vs 660 g) from features in the 0.6 s before contact,
when the hand has not touched the object yet, so nothing physical differs. AUC per participant, with a label-shuffle
null within series; the group test is one-sided Wilcoxon of AUC > 0.5.

Controls: the same decoder on peripheral channels (eye and neck), on the EMG and on hand speed (a heavier last lift could
change the reach), and on the EEG before the LED (`eeg_pre_cue`): if that one decodes as well, the signal is a leftover
of the previous lift rather than a prepared expectation.""",
     """out = run(["expectation"], PARTICIPANTS)
display(out["expectation"])
figure("step4_expectation", "expectation.png")"""),
    ("""## Step 5: error and learning (exploratory)

For unexpected lifts followed by a lift of the same weight: partial Spearman correlation between the size of the EEG
response (beta dB in 0.2 to 0.6 s after touch over left sensorimotor channels, fixed beforehand) and the size of the
change in grip-force rate on the next lift. The current lift's peak grip force, load-phase duration and EMG peak are
partialled out, so a bigger correction movement cannot explain both. The group test is Wilcoxon on Fisher z.

**Look for:** a positive mean. With at most about 30 lifts per participant, treat any result as a hint for a
confirmatory study, not as a finding.""",
     """out = run(["learning"], PARTICIPANTS)
display(out["learning"])
out["learning_per"]"""),
]

CLOSING = """## Decision

Fill in the go / no-go table at the top from steps 2 to 5. A paper is worth writing if step 3 or step 4 passes with its
controls. If both fail, the honest outcome is the audit paper on the EEG-to-EMG line."""


def _cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def build() -> dict:
    cells = [_cell("markdown", TITLE), _cell("code", SETUP)]
    for md, code in SECTIONS:
        cells += [_cell("markdown", md), _cell("code", code)]
    cells.append(_cell("markdown", CLOSING))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    Path("pe_pilot.ipynb").write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
