# v3: B vs F and the template deviation test

Branch `KG-GT-v3-eeg-guided`. Code in commit `98cf913` (it builds on `66aae0f`). Not pushed. Nothing has been run on
real data: there are only code changes, unit tests on random data and CPU smoke checks.

## Conditions (proposal names)

All causal: input features up to token t, EMG target at t + h, with h = 0, 200, 500 and 1000 ms.

| Name | Inputs | Role |
|---|---|---|
| `causal_cue` | past LED onsets only (`v3.timing.causal_features(..., "cue_causal")`), ridge to y | reference |
| `K` | kinematic stream of the causal v2 horizon model (`outputs/checkpoints_v2_horizon/h{h}ms/P{p}`): past hand kinematics + force | reference |
| `B` | K + ridge(past task events -> y - K) (`phase_ridge` `events` set) | primary no-EEG baseline |
| `F` | K + ridge(EEG history + past task events -> y - K) (`phase_ridge` `both` set) | primary neural candidate |

B and F share the model family, the inputs apart from the EEG, the target domain (model domain u = sqrt(y / s_c) at
the 50 Hz token rate), the held-out test series and the calibration:

- every ridge (causal_cue, B, F) is fitted on the validation series, because K was trained on the training series;
- every ridge is pooled over phases (per-phase fits read the phase at the target time and are not causal);
- every prediction is clipped at 0.

## API

### `v3/contrast.py`

```python
SOURCES = ("eeg", "swap", "bandpower", "peripheral")
COLUMNS = ["segment", "n_fit", "n_test", "eeg_skill", "eeg_skill_null_mean", "p", "vaf_gain", "ccc_B", "ccc_F",
           "mse_B", "mse_F", "mse_K", "mse_causal_cue"]
METRICS = ["eeg_skill", "vaf_gain", "ccc_B", "ccc_F", "mse_B", "mse_F", "mse_K", "mse_causal_cue"]

b_vs_f_scores(eeg_val, ev_val, res_val, eeg_test, ev_test, res_test, lab_test, k_test, y_test, n_null, seed,
              cue_val=None, y_val=None, cue_test=None, window=400, hop=200) -> DataFrame[COLUMNS]
eeg_sources(model, toks_val, toks_test, lifts_val, lifts_test, norm, cfg, device, sources=SOURCES, rng=None,
            match_condition=True) -> {source: (val_features, test_features)}
score_b_vs_f(model, sd, cfg, device, participant, horizon_ms, n_null=99, sources=SOURCES, match_condition=True)
    -> DataFrame[subject, horizon_ms, source, *COLUMNS]
run_b_vs_f(participant, horizon_ms, ckpt_root, n_null=99, sources=SOURCES, match_condition=True) -> same
summarise_b_vs_f(df, primary="eeg") -> DataFrame
channel_columns(cfg, channels, n_bands=None) -> np.ndarray     # band-power columns of a channel list
main(argv)   # python -m v3.contrast --participants all --horizons-ms 0,200,500,1000 --out results/v3
```

What the columns of `b_vs_f.csv` mean (one row per subject × horizon_ms × source × segment, segments `pre_move`,
`reach`, `load`, `hold`, `release`, `other`, `all`):

- `eeg_skill` = 1 - MSE(F) / MSE(B). The MSE is per token, averaged over muscles, in the model domain.
- `p` comes from a time-shift null that moves only F's EEG term by whole hops. Because the ridge is linear, F = intercept + EEG term + event term, so the event term stays aligned with the target. `eeg_skill_null_mean` is the mean of that null.
- `vaf_gain` = VAF(F) - VAF(B) in VAF points. `ccc_B` and `ccc_F` are CCC values; both metrics come from `v2.evaluate.standard_metrics` on the same tokens and are averaged over muscles.
- `mse_B`, `mse_F`, `mse_K` and `mse_causal_cue` are the absolute MSEs.
- `n_fit` is the number of validation tokens and `n_test` the number of test tokens in the phase. A phase with fewer than 10 test tokens gives a row of NaN.

What the columns of `b_vs_f_summary.csv` mean (one row per horizon_ms × segment × source):

- `n_subjects`, `{metric}_mean` and `{metric}_sem` for every metric in METRICS.
- `n_sig`: the number of participants with p <= 0.05 and eeg_skill > 0.
- `p_eeg_skill`: the one-sided Wilcoxon p of eeg_skill > 0.
- `gap_mean` and `p_gap`, for control sources only: the gap eeg_skill(eeg) - eeg_skill(source) and its one-sided Wilcoxon p.
- `p_eeg_skill_holm_phases` and `p_gap_holm_phases`: Holm over the six phases of one horizon and source (NaN for `all`).
- `p_eeg_skill_holm_horizons` and `p_gap_holm_horizons`: Holm over the horizons of one phase and source (`all` included).

The F variants per EEG source:

- `eeg`: the horizon model's EEG tokens. This is the real F.
- `swap`: the same tokens recomputed from EEG swapped by `v3.nulls.swap_series`, cue-aligned, with `match_condition=True` by default (weight and surface). Validation and test are each swapped among their own trials, and the ridge is fitted on the swapped EEG.
- `bandpower`: the trailing 1 s band power of all channels (`v2.residual.eeg_features`), with no learned weights.
- `peripheral`: the same band power on `v3.config.PERIPHERAL` only. Compare it with `bandpower`, not with `eeg`.

### `v3/templates.py`

```python
SOURCES = ("student", "swap", "bandpower", "peripheral")
COLUMNS = ["segment", "muscle", "n_test", "skill", "r", "skill_null_mean", "p"]

n_template_bins(n_bins=20, post_s=3.0, post_bin_s=0.1) -> int                # 7 * n_bins + 30 + 1 (rest bin last)
template_bins(lift_list, n_tok, n_bins=20, post_s=3.0, post_bin_s=0.1) -> int array (-1 = no bin)
fit_emg_templates(train_tokens, train_lifts, norm, n_bins=20, post_s=3.0, post_bin_s=0.1)
    -> {"template": (n_bins_total, 5), "count", "n_bins", "post_s", "post_bin_s"}
template_deviation(templates, tok, lifts, norm) -> (deviation (n_tok, 5), NaN rows where bin == -1; bins)
template_deviation_test(feats, deviation, labels, trial, names, n_null, seed, folds=5, window=400, hop=200)
    -> DataFrame[COLUMNS]
trailing_band_power(eeg_z, smooth=50) -> (n, C * B)
student_template_deviation(model, test_tokens, test_lifts, templates, norm, cfg, device, names, sources=SOURCES,
                           n_null=None, folds=None, match_condition=None, rng=None)
    -> DataFrame[source, *COLUMNS, var_template, n_trials]
summarise_template_deviation(df, primary="student") -> DataFrame
```

How the template is built, per participant and muscle, from the TRAINING series only:

- A trial runs from one LED onset to the next (`trial_bounds`).
- Between consecutive task events (`EVENTS`, led_on .. hand_stop, 7 intervals), time is warped piecewise-linearly into 20 bins.
- After hand_stop come 0.1 s bins up to 3 s. Later tokens, and the tokens before a series' first LED, share one rest bin.
- A trial with a missing or out-of-order event gets bin -1 and is left out of both the template and the test. Equal consecutive events are allowed.
- The value of a bin is the mean of u over the training tokens in that bin. A bin never seen in training takes the global mean.
- The deviation is u - template[bin].
- The template uses the event times of the whole trial, as the `phase` model does. It is subtracted from the target, never fed to the model, and the EEG never sees it. The warping also removes each trial's own event timing, so the deviation holds amplitude and shape.

How the test works:

- A ridge maps EEG features to the 5-muscle deviation, cross-validated over trials (`v3.evaluate._trial_folds` and `_cv_ridge`, `evaluation.residual_folds` groups).
- Per phase and muscle: skill = 1 - MSE(ridge) / MSE(fold training mean), r, and p from the time-shift null. The `MEAN` row is the mean over muscles.
- `var_template` is the share of the test envelope's variance (summed over muscles) that the template explains.

The four sources:

- `student`: the student's final tokens (`fin`).
- `swap`: the same tokens from swap_series EEG with match_condition=True (from `cfg["templates"]["swap_match_condition"]`).
- `bandpower`: the trailing 1 s band power of all channels.
- `peripheral`: the same band power on the peripheral channels. It is dropped with a UserWarning when the student lacks those channels (for example under the `central` preset).

What the columns of `template_deviation_summary.csv` mean:

- The table is per (scale), source, segment and muscle, after averaging over seeds per subject.
- `n_subjects`, `skill_mean`, `skill_sem`, `r_mean`, and `p_skill` (one-sided Wilcoxon of skill > 0).
- `gap_mean` and `p_gap`: the gap to the `student` source and its p.
- `p_skill_holm` and `p_gap_holm`: Holm over the 5 muscles × 6 phases of one (scale, source), and separately over the muscles of the `all` rows. `MEAN` rows get NaN.

### Config (`v3/config.py`)

```python
CONFIG_V3["b_vs_f"] = {"checkpoint_dir": "outputs/checkpoints_v2_horizon", "horizons_ms": [0, 200, 500, 1000],
                       "sources": ["eeg", "swap", "bandpower", "peripheral"], "results_dir": "results/v3"}
CONFIG_V3["templates"] = {"n_bins": 20, "post_s": 3.0, "post_bin_s": 0.1,
                          "sources": ["student", "swap", "bandpower", "peripheral"], "swap_match_condition": True}
```

### Notebook (`tools/build_main_v3_notebook.py` -> `main_v3.ipynb`, regenerated with no outputs)

Cell order: SETUP, CONFIG, BUDGET, DATA, TIMING, TRAIN, EVAL, SWAP, RESIDUAL, `# V3-TEMPLATES`, VERDICT, `# V3-B-VS-F`.

- `V3-TEMPLATES` writes `RESULTS / "template_deviation.csv"` and `template_deviation_summary.csv`.
- `V3-B-VS-F` caches each participant and horizon in `RESULTS / "b_vs_f_runs"`, then writes `b_vs_f.csv`, `b_vs_f_summary.csv` and `b_vs_f.png`. It skips any horizon whose checkpoint is missing.

## Files changed (98cf913)

- `v3/contrast.py`: rewritten for B/F (C naming dropped; `contrast_scores` / `contrast_group_test` / `run_participant` renamed to `b_vs_f_scores` / `summarise_b_vs_f` / `run_b_vs_f`, plus `score_b_vs_f`).
- `v3/templates.py`: new.
- `v3/evaluate.py`: the PCA muscle test (`remove_shared`, `muscle_specificity`, `muscle_group_test`) was removed, because the template test replaces it. The shared helpers `_student_over_timing`, `_trial_folds` and `_cv_ridge` stay, and `residual_over_timing` uses them unchanged.
- `v3/config.py`: the `contrast` and `muscle` blocks were replaced by `b_vs_f` and `templates`.
- `tools/build_main_v3_notebook.py` and `main_v3.ipynb`: the cells `V3-B-VS-F` and `V3-TEMPLATES`.
- `tests/test_v3_contrast.py` (rewritten), `tests/test_v3_templates.py` (new), `tests/test_v3_muscle.py` (deleted) and `tests/test_v3_notebook.py` (new markers).
- `v3/README_fa.md`: sections 4.14 and 4.15 rewritten with the B/F names, plus the summary table, files, outputs and limitations (persian-writing skill; lint 0 issues).
- `v2/`, `eeg_controls/`, `data/`, `outputs/` and `results/` are unchanged.

## Tests

`tests/test_v3_contrast.py` has 10 tests:

- planted EEG information gives eeg_skill > 0.5, p <= 0.05 and a positive vaf_gain, with eeg_skill = 1 - mse_F/mse_B;
- uninformative EEG gives eeg_skill ≈ 0, and the null mean is ≈ 0 (the event term is kept);
- B and F match `phase_ridge.fit_and_score` `events` and `both`;
- vaf_gain, CCC and mse_causal_cue match `standard_metrics` and the ridge computed by hand;
- `_fit_terms` matches `phase_ridge._fit`, and the EEG term is split off correctly;
- the channel-major `channel_columns`;
- `eeg_sources` (the swap matches `swap_series(match_condition=True)`, and peripheral selects the right columns);
- the `score_b_vs_f` glue, with a monkeypatched `collect`;
- `summarise_b_vs_f` Holm over phases and over horizons, and the control gap;
- the config block.

`tests/test_v3_templates.py` has 8 tests:

- bins in event-warped time (stretched trials, post bins, rest bin);
- missing or unordered events give -1;
- the template recovers the average trial;
- the student predicts deviations (skill > 0.5 for all muscles, p <= 0.05) and the swap does not;
- uninformative EEG gives skill < 0.05;
- peripheral is skipped under the `central` preset;
- the summary's Holm and gap;
- the config block.

Output of `env/Scripts/python.exe -m pytest tests/test_v3_*.py -q`:

```
106 passed, 1 warning in 60.61s (0:01:00)
```

(The warning is an old one in `test_v3_guide.py`.)

Smoke checks, on CPU with random data and random weights:

- A real causal v2 model at h = 1000 ms produced every B/F source with the right shapes (tokens 64, band power 128, peripheral 44), and they went through `b_vs_f_scores` together with the causal_cue reference.
- A real v3 student (`build_student`) went through `fit_emg_templates` and `student_template_deviation` with all four sources.

In the template smoke check the student and swap rows were equal to 4 decimals. I checked this: the swap does change the tokens (mean absolute difference 0.16), but on a pure-noise target the ridge's cross-validation picks a strong alpha, so both predictions stay close to the fold intercept. Not a bug.

## Open issues

1. **The 1000 ms horizon checkpoints probably don't exist.** The v2 horizon runs covered 0, 200 and 500 ms. `eeg_controls.run_horizon --horizons-ms 1000` has to be trained first; the notebook cell skips missing checkpoints.
2. **One swap draw per run.** Both B/F and the template test use one swap draw per run (one full model pass each). With 12 participants that is probably enough at group level, but per-participant swap estimates are noisy.
3. **The `peripheral` source is not a peripheral-trained model.** It is band power with no learned weights. A real artifact control at the model level still needs a student (or horizon model) trained with the `peripheral` preset.
4. **The template test sees a non-causal student.** It runs at horizon 0 on the default student, which is not causal. The template itself is non-causal by design, since it is only subtracted from the target.
5. **How strict the multiple-comparison correction is.** In the template summary, Holm runs over 30 tests (5 muscles × 6 phases); with 12 participants the smallest Holm p possible is about 0.007. In B vs F the floor is about 0.0015 over phases and 0.001 over horizons.
6. **Some task events come from the recordings.** Apart from the LED, events such as hand_start and first_touch are detected from the kinematic and force recordings. The B features respect causality (only past events), but the event detection itself used the whole recording.
7. **Results go to the preset folder.** B vs F reads v2 horizon checkpoints regardless of `PRESET`, but the notebook writes its tables under `RESULTS`, which is `results/v3/<preset>` when a preset is set.
8. **Not done here.** Onset timing, the phase-matched swap, the E0/Eguided verdict and the run-process cleanup are left to the main session.
