# How to run the v3 analysis

One analysis, nine numbered steps. The notebook `main_v3.ipynb` and the command line run the same functions
(`v3/run.py`), so they cannot drift apart. Every step prints one header line and the files it wrote.

## The question and the conditions

Does EEG history improve the prediction of the EMG envelope beyond what past movement and past task events already
explain? Each condition below is evaluated on the same held-out test series.

| condition | input at time t | role |
|---|---|---|
| `causal_cue` | past LED and event times only | measures the task timing |
| `K` | past hand kinematics and force | biomechanical explanation |
| `B` | K + past task events | primary no-EEG baseline |
| `F` | K + past task events + EEG history | primary neural candidate |
| `E0` | EEG only, trained without the guide | EEG-only reference |
| `Eguided` | EEG only, trained with the frozen K teacher | secondary training intervention |

The primary result is `eeg_skill = 1 - MSE(F) / MSE(B)`.

## The steps

| # | step | trains? | what it does | output folder |
|---|---|---|---|---|
| 0 | `setup` | no | config, participants, seeds, scales, preset | |
| 1 | `data` | no | 50 Hz tokens, task events, train / validation / test split | |
| 2 | `timing_baselines` | no | `causal_cue`, `events_causal` (0, 200, 500, 1000 ms), `phase`, `cue` | `results/v3/step2_timing_baselines/` |
| 3 | `b_vs_f` | no | K, B, F at every horizon from `outputs/checkpoints_v2_horizon`, the EEG controls, onset timing | `results/v3/step3_b_vs_f/` |
| 4 | `budget` | no | GPU hours of steps 5 and 6, on random data | |
| 5 | `train_E0` | yes | the EEG-only student without the guide, every seed | `outputs/checkpoints_v3/E0/s{seed}/P{p}/` |
| 6 | `train_Eguided` | yes | the EEG-only student with the K teacher, every guided scale and seed | `outputs/checkpoints_v3/Eguided_g{scale}/s{seed}/P{p}/` |
| 7 | `evaluate_students` | no | metrics at 500 and 50 Hz, nulls, onset, swaps, residual over timing, templates | `results/v3/step7_evaluate_students/` |
| 8 | `verdict` | no | the interpretation, one row per horizon | `results/v3/step8_verdict/` |

With a preset (`--preset central`, `peripheral`, `eye_regressed` or `no_gamma`) the folders get the preset's name:
`results/v3/central/step3_b_vs_f/`.

## Order

1. Steps 0 to 3, which need no training (minutes per participant and horizon), give the primary result:
   ```
   env/Scripts/python.exe -m v3.run --step setup,data,timing_baselines,b_vs_f --participants all
   ```
2. Look at `results/v3/step3_b_vs_f/b_vs_f.png` and `b_vs_f_summary.csv`, then run the verdict on them alone:
   ```
   env/Scripts/python.exe -m v3.run --step verdict
   ```
   Without steps 5 to 7 the verdict answers the primary question and says the guide has not been evaluated yet.
3. Step 4 with the GPU free, then steps 5 and 6 (the long part; finished runs are skipped when you start again):
   ```
   env/Scripts/python.exe -m v3.run --step budget,train_E0,train_Eguided
   ```
4. Steps 7 and 8:
   ```
   env/Scripts/python.exe -m v3.run --step evaluate_students,verdict
   ```

To see what a command would do without running or writing anything, add `--dry-run`. Other options: `--participants 1-4`,
`--seeds 0`, `--scales 0.0,0.1`.

## What the main columns mean

`step3_b_vs_f/b_vs_f.csv` has one row per participant, horizon, EEG source and task phase (`pre_move`, `reach`, `load`,
`hold`, `release`, `other` = rest, `all`).

| column | meaning |
|---|---|
| `eeg_skill` | 1 - MSE(F) / MSE(B): the share of B's error that the EEG removes |
| `p` | time-shift null: only F's EEG term is moved in time |
| `vaf_gain` | VAF(F) - VAF(B), in VAF points |
| `ccc_B`, `ccc_F` | concordance of B and of F with the target |
| `mse_causal_cue`, `mse_K`, `mse_B`, `mse_F` | absolute errors of every condition (model domain, 50 Hz) |

The EEG source of F:

| source | what it is |
|---|---|
| `eeg` | the real F |
| `phase_swap` | F with EEG from another trial of the same weight and surface, time-warped so all task events line up |
| `swap` | the same with only the LED cue lined up |
| `bandpower` | F from all-channel band power without the trained network |
| `peripheral` | the same band power on the frontal-pole, temporal and occipital channels only |

`b_vs_f_summary.csv` holds the means over participants and the one-sided Wilcoxon p, Holm-corrected over the phases and,
separately, over the horizons (`p_eeg_skill_holm_horizons` is the primary test). For a control source, `gap_mean` and
`p_gap` compare it with the real `eeg`. `onset.csv` holds the onset error of K, B, F and causal_cue per lift and muscle,
in ms (the onset is where the envelope first reaches 20 % of its baseline-to-peak range after the cue).

## How to read the verdict (`step8_verdict/verdict.csv`)

| F beats B, survives the phase-matched swap, not peripheral | Eguided beats E0 | interpretation |
|---|---|---|
| yes | yes | EEG adds information beyond motion and timing; the guide also helps |
| yes | no | EEG adds information beyond motion and timing; the guide does not help |
| no | yes | the guide taught a better movement template: representation learning, not muscle-specific commands |
| no | no | no evidence that EEG adds information beyond motion and timing |

If F beats B but fails the swap or the peripheral control, the row says the gain is task-locked or artifact-related; that
is never a positive claim. Each row also carries the absolute MSE of every condition, the VAF gain, CCC and the median onset
error of B and F, because a pooled correlation alone is not enough.
