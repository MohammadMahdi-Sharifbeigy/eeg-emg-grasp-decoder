# EEG-base model with a kinematic guide used only in training: design

Date: 2026-10-16. Status: draft, waiting for the user's approval.
Branch: new, cut from `KG-GT-v2` (name proposed: `KG-GT-v3-eeg-guided`). Package: `v3/`, notebook `main_v3.ipynb`.

## 1. Why

The v2 report (`v2-report/`) found that in the v2 model the kinematic stream K does all the work and EEG adds almost nothing:

| Quantity (mean of 12 participants) | Value |
|---|---|
| K (kinematics only) | VAF 75.8, r 0.872 |
| F (K plus gated EEG correction) | VAF 75.9, gate 0.01 to 0.04 |
| E (EEG only) | VAF 39.1, r 0.645 |
| Timing baseline, "phase" ridge (knows only when the task events happen) | r 0.81 |

A teammate's review (2026-10-16, transcript in `alireza-review-transcript.md`) pointed out that this is the opposite of the original idea: EEG was supposed to be the base, with kinematics only guiding it. In v2, K is the base and EEG is a small correction, so the question "what can EEG alone do, if kinematics teach it?" is never asked.

The user's decision: build the model the other way round. **Input is EEG only. Kinematics help the EEG stream a little during training, and are not available at test time. Output is the EMG envelope.** Because the final model cannot read kinematics, it cannot cheat with them; the open question is only how much the training-time help is worth.

## 2. Goals, non-goals, anti-goals

**Goals**
1. A model whose forward pass takes EEG band power only and returns the EMG envelope of the five muscles.
2. A guide that uses the kinematic stream K as a frozen teacher during training and is switched off (and absent from the deployed model) at test time.
3. A measured, small guide: the report must show how much the score depends on the guide strength.
4. Success, per the user (see section 7): beat the current EEG-only result E, and reach at least "close to" the timing baseline. Being below K is accepted.

**Non-goals**
- No kinematics, history features, or fusion/F output in the final model.
- No new preprocessing. The v2 token pipeline (50 Hz tokens, 8 s windows, log band power) is reused as it is.
- No new interpretability work in this phase. The existing recorder is kept working on the EEG stream; the interpretability chapter for the new model comes after the result is known.
- No cross-participant model.
- Claude does not run real training or evaluation (standing rule). Claude writes code and tests and runs smoke tests with random weights.

**Anti-goals** (would make it a failure even if the numbers look good)
- Any path by which kinematics reach the model at test time, including through a saved teacher inside the deployed checkpoint.
- A guide strong enough that the model is effectively K in disguise (see the budget in section 5).
- Early stopping or model selection that favours the guide terms instead of EMG accuracy.

## 3. Constraints (from the project rules)

- Never run real training/evaluation; never load or write the user's checkpoints, `data/`, `outputs/`, `results/` (reading saved results is fine).
- Commits: no Claude attribution, never push, `git add` explicit paths only, do not add `.superpowers/` or other people's work-in-progress. The working copy has uncommitted changes (`main_v2.ipynb`, `v2-report/*`); do not touch them.
- GPU is a 6 GB GTX 1660 Ti; target under 1 hour of training per participant.
- Code style follows `v2/`: pre-norm blocks, every attention module recorded, one test file per concern.

## 4. Design

### 4.1 Model parts

```
EEG band power (B, T, 32, 4)
        |
   EEG student   = v2 EEGStream (embed, electrode graph, pooling, temporal encoder, muscle graph)
        |  \
        |   mid tokens s_mid and final tokens s_fin  (d = 64)
        v        \
   E output       guide heads  (train only)  <---  frozen teacher: v2 KinStream
 (EMG envelope)                                    kin tokens t_mid, t_fin, output K
```

- **Student** is the v2 `EEGStream` unchanged in structure, so the comparison with the report's E is clean.
- **Teacher** is the v2 `KinStream` with weights loaded from the user's existing v2 K checkpoint for the same participant (expected at `outputs/checkpoints_v2/P{p}/best.pt`; `v2.train.load_kin` already copies the `kin.*` tensors and raises a clear error if the file is missing or the shapes differ). If a participant has no K checkpoint, that participant is skipped with a message, not retrained silently. It is set to `eval()`, has `requires_grad=False`, and runs under `torch.no_grad()`.
- **Guide heads** are small linear/MLP projections from the student's tokens to the teacher's token space. They exist only in the training wrapper.
- **Deployed model** (`EEGOnlyModel`) contains the student only. The training wrapper `GuidedTrainer` holds student + teacher + guide heads. Saving a checkpoint saves the student's state dict (plus config); the teacher is never part of it.

### 4.2 Guide terms (all train-only, all against detached teacher outputs)

| Term | What it does | Default |
|---|---|---|
| `align` | Cosine distance between a linear projection of the student's tokens at a chosen encoder layer and the teacher's tokens at the matching stage. Tells the EEG representation to carry the kind of movement-phase information the kinematic stream builds. | on |
| `distill` | Huber loss between student output E and teacher output K (in the v2 square-root amplitude domain). Gives the student a soft target on top of the hard EMG target. | on, smaller weight |
| `kin_aux` | A head predicts a few standardised kinematic features (for example hand-state velocities) from the student's tokens. | off (switchable, for ablation) |

Base weights (my call): `w_align = 1.0`, `w_distill = 0.5`, `w_aux = 0.5`. One scalar `guide_scale` (the grid in section 4.5) multiplies all of them, so the strength of the help is a single number.

Total loss for a batch: `L = L_EMG(E) + g(epoch) * (w_align * L_align + w_distill * L_distill + w_aux * L_aux)`, where `L_EMG` is the existing v2 four-part stream loss applied to E only, and `g(epoch)` is the guide schedule.

### 4.3 Guide schedule ("help in the middle of training")

`g(epoch)` has three modes: `constant`, `fade` and `off`. `fade` ramps up over the warm-up epochs (5), stays at 1 until epoch 40 and falls linearly to 0 by epoch 60; from then on training is pure EEG-to-EMG. Early stopping cannot end a `fade` run before epoch 60. Fixed epochs (not a share of the 150-epoch budget) keep a run near the 1-hour target, because early stopping usually ends runs well before 150. **Default `fade`** (my call, from the user's wording "in the middle of training"); `constant` and `off` are kept for the ablation. `off` with the same seed is the control that reproduces the current E.

### 4.4 Training and model selection

- One optimizer (AdamW, same hyperparameters as v2) over the student and guide heads. The teacher has no optimizer. `fit` from `v2/train.py` is reused or wrapped so resume, per-run config check, best/last checkpoints and the NaN-skip behaviour stay identical.
- **Early stopping and best-epoch selection use the validation EMG loss of E only**, never the guide terms. The guide terms are logged separately.
- The guide's share of the total gradient norm is logged each epoch, so "how small is the help" is measured, not assumed.

### 4.5 The cheating budget

"Help must be very low" is made concrete in three ways:
1. **Structural:** the deployed model takes `(eeg)` only; a test asserts that the forward signature has no kinematic argument and that no teacher/guide parameter is in the saved state dict.
2. **Weight cap:** guide weights are searched on a small grid (`w` scale in {0, 0.05, 0.1, 0.3}, where 0 is the control); the largest scale is 0.3 so the EMG loss always has the larger weight. The grid is a config, so the user can change it.
3. **Reported dependency:** the report shows E-score against guide strength and the gradient-share log. If the best score needs a large guide, that is said plainly.

## 5. Data

Reuse `v2.data` and `v2.tokens` exactly: cached 500 Hz series to 50 Hz tokens, 8 s windows (400 tokens), stride 2 s with random start jitter in training, stitched 4 s centre pieces at test time. The training windows must additionally return the kinematic tokens and history features (for the teacher); the test windows need EEG and target only. Normalisation statistics come from the training series only, as in v2.

## 6. Evaluation

For each of the 12 participants, on the same test series as v2:

- E-guided: r, VAF, CCC per muscle and mean, via the existing `v2.evaluate` metrics on the 500 Hz-restored prediction.
- Controls, same code and seed: **E-control** (guide `off`) and the **report's saved E** (read from `results/v2`).
- References: timing baseline `phase` and `cue` (read from the saved `eeg_controls` results), and K (from `results/v2`).
- Nulls: the existing block-swap and shuffle nulls on the EEG input (the student must lose its score when the EEG is replaced).
- Group statistics: Wilcoxon signed-rank over the 12 participants for guided vs control and guided vs saved E; per-participant table.
- Guide-strength curve (section 4.5).

## 7. Success criteria (the user's, with my numbers marked)

1. **Must:** mean r of E-guided is above the saved E (0.645) and above E-control, with a Wilcoxon p below 0.05 over the 12 participants.
2. **Must:** E-guided is "close to" the timing baseline `phase` (r 0.81). *My number:* within 0.05, that is mean r of at least 0.76. Beating 0.81 is the stretch goal.
3. **Not required:** matching K (r 0.872). The user accepts being below it.
4. **Must:** every anti-goal check passes (structural test, selection on E loss only, null test).
5. Caveat to state in the report: the timing baseline reads the event times from the whole recording, so it is a ceiling and not a causal predictor (the v2 report says so in its limitations). Beating it with EEG alone would be surprising, so criterion 2 is the realistic target.

## 8. Testing (code only; no real training)

- Unit tests for each guide term (shape, zero when student equals teacher, no gradient into the teacher).
- Test that the saved student checkpoint contains no `teacher.*` or `guide.*` keys and that the forward signature takes EEG only.
- Test that `g(epoch)` follows the three schedules.
- Test that model selection uses the E-only validation loss (a run with a guide term made huge must not change the selected epoch logic).
- CPU wiring smoke test with random weights and a random "teacher" on a few windows, and a GPU memory/timing smoke test on random data if the GPU is free.
- The existing v2 tests must still pass unchanged.

## 9. Risks

- The v2 report suggests E mostly encodes task timing; the guide may not add much beyond that. The guide-strength curve and the E-control make this visible instead of hidden.
- The teacher was trained on the same training series. This is fine (no test leakage), but distillation can only pass on what K learned from the training set.
- Per-participant instability (the v2 report notes run-to-run variability was never measured). The plan includes running the control and the guided model with the same seed and, if time allows, a second seed.
- 12 participants x several guide strengths x 2 seeds is a lot of GPU time. The user decides how many to run; the plan lists the minimum set.

## 10. Left to the builder

File layout inside `v3/`, names of config keys, whether `fit` is wrapped or copied, which encoder layer the `align` term taps (default: the output of the first of the two EEG temporal layers vs. the final tokens, a config switch), and the notebook generator.

## 11. My calls (to be checked)

- Teacher = existing v2 K checkpoints, not retrained.
- Default schedule `fade`; default terms `align` + `distill`; `kin_aux` off.
- Grid {0, 0.05, 0.1, 0.3}.
- "Close to timing baseline" = mean r >= 0.76.
- Branch and package names.
