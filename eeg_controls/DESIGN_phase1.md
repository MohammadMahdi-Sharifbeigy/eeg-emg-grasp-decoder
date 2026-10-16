# EEG controls, phase 1: design

Date: 2026-10-16. Branch: KG-GT-v2. Builds on `eeg_controls/` (tests A, B, C of report chapter 13).

## Why

The three control tests showed that E (the EEG-only output of KG-GT v2) mostly knows *when* a lift happens,
that its electrode map is partly sensorimotor (C3, CP1, C4) and partly eye and neck electrodes, and that the
EEG adds a little to K only around reaching when forecasting ahead. The long-term goal of the user is an E that
gives each lift the right **level** (heavy objects higher than light ones), so that fusing E with K adds value.

Phase 1 answers four questions on the **current preprocessing**, before any preprocessing is changed:

1. **Weight check:** does today's E already give different levels for light, medium and heavy lifts, and is
   its level only too low (fixable by a gain) or flat across weights?
2. **Step 3, ridge per phase:** does the EEG information that grows before movement become real error reduction
   when the ridge is fitted per task phase instead of over all phases together?
3. **Step 2, causal timing baseline:** does the EEG add anything beyond knowing the times of past task events
   (LED on, hand start, ...)?
4. **Step 1, electrode subsets** and **step 4, eye control:** how much of E and of the EEG gain survives with
   only central electrodes, with only edge electrodes, and with the eye channels regressed out?

Order of the larger project (decided by the user): phase 1 (this document) -> phase 2 (better EEG
preprocessing, retrain) -> phase 3 (EEG-only again, aiming at per-lift level) -> leave-one-subject-out CV
(the former step 5) after phase 2.

## Goals

- Complete scripts (importable modules with command lines), one generated notebook with a RUN and a PLOT cell
  per part, and dedicated figures for every part. Horizons for every forecasting part: 0, 200, 500 ms.
- Every part runs for all 12 participants and gives group tests over participants (Wilcoxon, sign-flip).
- Parts W and R need no training. Parts S and E train; they reuse what the user already trained where possible.

## Non-goals

- No change to the default behaviour of `v2/` (every new option is opt-in, off by default), no change to
  `main.ipynb`, `main_1stage*.ipynb`, `main_p2p.ipynb`, `main/`, or `main_v2.ipynb`.
- No new EEG preprocessing beyond the simple eye regression of part E (that is phase 2).
- No leave-one-subject-out, no report chapter (the report is updated after the user has run phase 1).
- No 1000 ms horizon.

## Anti-goals

- Any leak of future information into a "causal" feature (the whole point of parts R and E).
- Comparisons where the condition differs in more than the EEG (for example a different K between the
  electrode-subset runs and the all-electrode runs).
- Claude running real training or full evaluation, or loading the user's checkpoints; only smoke tests in the
  scratchpad. The user runs everything.

## Constraints

- Read-only: `data/`, `outputs/`, `results/`. New results go to `results/v2_controls/` and new checkpoints to
  `outputs/checkpoints_v2_controls/` (created by the user's runs), plus `eeg_controls/results/phase1/` for the
  no-training parts and the figures.
- Existing checkpoints read by the user's runs: `outputs/checkpoints_v2/P{p}/best.pt` (non-causal v2, for part W)
  and `outputs/checkpoints_v2_horizon/h{h}ms/P{p}/best.pt` (causal, for parts R, S, E).
- Every runner skips finished runs unless `--overwrite`, and has `--smoke` (1 epoch, 5 null draws).

## The parts

### W. Weight check (no training)

Model: the non-causal v2 of each participant (`outputs/checkpoints_v2`). Test series only.

- Predict K, E and F at 500 Hz (`v2.infer.predict_series`).
- For every lift with a known weight (CurW 1, 2, 4 = light, medium, heavy): the **level** of the target and of
  each output, per muscle, as the mean envelope in two windows: *load* (first touch -> lift-off) and *hold*
  (lift-off + 0.5 s -> replace - 0.2 s). Hold is the main window.
- Per participant and muscle:
  - weight effect: mean level per weight; heavy / light ratio; Spearman rho of level against weight over lifts;
  - amplitude: slope of output level on target level over lifts (1 = right size, below 1 = shrunk) and the
    ratio of mean output level to mean target level;
  - **calibrated E:** E multiplied by one gain per muscle fitted on the validation lifts (least squares of target
    level on E level). It shows how much of the level error a single rescaling fixes, and whether a weight
    difference appears once the size is right.
- Group: Wilcoxon over the 12 participants for rho(E level, weight) > 0, and for E's heavy/light ratio against
  the target's.

Figures: W1 mean level per weight (target, E, calibrated E, K), per muscle; W2 lift-by-lift scatter of E level
against target level, coloured by weight, with the identity line; W3 a stretch of one test series with lifts of
different weights (target, E, calibrated E); W4 per-participant summary (rho and slope for E and K).

### R. Ridge per phase and causal timing baseline (steps 3 and 2, no training)

Models: the causal horizon checkpoints, h = 0, 200, 500 ms. Residual of K in the model domain at token rate,
on the validation series (fit) and the test series (score), exactly as `run_horizon.segments` does now.

Feature sets (all at token t, all causal):

- `eeg`: the EEG tokens of the trained E stream (as now);
- `events`: **past-event bumps**: for each of the 8 task events that has already happened at or before t,
  Gaussian bumps at lags 0 to +3 s after it (step 0.25 s, width 0.25 s). Nothing depends on an event later than
  t. This is the causal timing baseline: it knows everything a timing model could know at t.
- `both`: `eeg` and `events` together.

Fits:

- `pooled`: one ridge over all phases (the current analysis, now for every feature set);
- `per_phase`: one ridge per task phase (pre_move, reach, load, hold, release, other), fitted on the validation
  tokens of that phase only. The phase labels come from the recorded events; this is an analysis split, not a
  model input.

Scores per phase, feature set, fit and horizon: r with the residual, skill = 1 - MSE(K + pred) / MSE(K), and
the time-shift null (as now). The key numbers: skill(`both`) - skill(`events`), the gain of the EEG beyond
timing, with a Wilcoxon test over participants; and skill(`per_phase`) against skill(`pooled`).

Figures: R1 heat map phase x horizon of skill, one panel per feature set and fit; R2 skill against horizon for
pre_move and reach, lines for `eeg`, `events`, `both`; R3 per-participant gain of the EEG beyond timing
(`both` - `events`) per phase.

### S. Electrode subsets (step 1, trains)

New opt-in config key `eeg.channels` (list of electrode names, default `None` = all 32): `build_tokens` keeps
those columns of the raw EEG before band power, and `build_model` builds the EEG stream with that many
electrodes and the electrode prior of those names.

- `central`: C3, Cz, C4, CP1, CP2, CP5, CP6 (7).
- `edge`: Fp1, Fp2, F7, F8, T7, T8, TP9, TP10, P7, P8, PO9, PO10, O1, Oz, O2 (15).

Each subset is trained as a causal horizon model at h = 0, 200, 500 ms. **K is loaded from the matching
horizon checkpoint and frozen from the first epoch; only E and the fusion train.** This keeps K identical to
the all-electrode run (so every difference comes from the EEG) and saves time. Outputs per run as
`run_horizon.run_one` writes them (standard, residual, segments, history), plus part R's per-phase table.

### E. Eye control (step 4, trains)

New opt-in config key `eeg.eye_regress` (list of reference electrodes, default `None`): per series, every other
EEG channel is regressed by least squares on the reference channels (raw 500 Hz signal, with intercept) and
replaced by its residual; the reference channels are then dropped. References: Fp1, Fp2, F7, F8 (vertical and
horizontal eye movement; the montage has no EOG channel). 28 channels remain. Trained exactly like part S
(K loaded and frozen, h = 0, 200, 500 ms), as the condition `eye_clean`.

Note: the regression weights are fitted on the whole series. They are a fixed spatial filter, not a time
filter, so no future value enters a token; this is the usual EOG-regression practice.

### Runner, notebook, tests

- `eeg_controls/weight_check.py` (W), `eeg_controls/phase_ridge.py` (R), `eeg_controls/run_conditions.py`
  (S and E: `--conditions central,edge,eye_clean --horizons-ms 0,200,500`), shared figures in
  `eeg_controls/plots_phase1.py`. Each module has a command line and writes CSV tables.
- `eeg_controls_phase1.ipynb`, generated by `tools/build_eeg_controls_phase1_notebook.py`, sections W, R, S+E;
  per section a markdown guide, a RUN cell and a PLOT cell. The S+E section also prints the summary over
  conditions (all 32 from the existing horizon runs, central, edge, eye_clean).
- Tests: channel selection (shapes, prior size, default unchanged); eye regression (removes a planted frontal
  component, keeps 28 channels); past-event features never depend on a later event; per-phase ridge shapes;
  K loaded and kept frozen (a 1-epoch smoke run in a temporary folder); the notebook matches its generator and
  every cell parses.

### Comparison table over conditions (S + E)

Per condition (`all` from `results/v2_horizon`, `central`, `edge`, `eye_clean`) and horizon: E r, F - K,
residual r, pre_move and reach r and skill (pooled and per phase), with Wilcoxon tests of each condition
against `all`. Figures: C1 E r and F - K per condition and horizon; C2 per-phase residual r per condition;
C3 per-participant E r, condition against `all`.

## Cost (user's GPU)

| Part | Training | Size |
|---|---|---|
| W | none | about 1 min per participant |
| R | none | about 3 min per participant and horizon |
| S | E + fusion only | 2 subsets x 3 horizons x 12 = 72 runs |
| E | E + fusion only | 3 horizons x 12 = 36 runs |

Freezing K roughly halves the time of a horizon run (about 10 to 15 minutes instead of 20 to 30).

## Calls made by Claude (please check)

1. The level of a lift is the mean envelope in the hold window (lift-off + 0.5 s to replace - 0.2 s), with the
   load window as a second view.
2. Calibrated E uses one gain per muscle fitted on validation lifts.
3. `central` excludes FC1 and FC2; `edge` has 15 electrodes against 7 in `central`, so the edge set has more
   channels. This is noted in the results; no extra size-matched set.
4. K is loaded from the existing horizon checkpoints and frozen in parts S and E.
5. Eye references are Fp1, Fp2, F7, F8; regression per series.
6. The per-phase ridge uses the recorded task events to split the phases.
7. The design lives in `eeg_controls/DESIGN_phase1.md` because `/docs` is ignored by git.
