# EEG controls, phase 1: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build parts W (weight check), R (per-phase ridge and causal timing baseline), S (electrode subsets) and E (eye regression) of the phase-1 EEG controls, with a runner, dedicated figures, a generated notebook and tests.

**Architecture:** Two opt-in keys in `v2` (`eeg.channels`, `eeg.eye_regress`) and one in training (`training.freeze_kin`) make the new conditions possible without changing defaults. New modules in `eeg_controls/` compute each part from saved checkpoints or train E + fusion against a loaded, frozen K. Figures live in one plotting module; a generated notebook runs everything with a RUN and a PLOT cell per part.

**Tech Stack:** Python 3.13 (`env/Scripts/python.exe`), PyTorch, NumPy, pandas, SciPy, matplotlib, pytest.

**Spec:** `eeg_controls/DESIGN_phase1.md`

## Global Constraints

- Every new option is opt-in; with the keys absent, `v2` behaves exactly as before (existing tests stay green).
- Do not modify `main.ipynb`, `main_1stage*.ipynb`, `main_p2p.ipynb`, `main_v2.ipynb`, `main/`.
- Claude never runs real training or full evaluation and never loads the user's checkpoints (`outputs/checkpoints_v2*`); smoke tests write only to the scratchpad. Never write to `data/`, `outputs/`, `results/`.
- Default output locations (written by the user's runs): `results/v2_controls/{condition}/h{h}ms/P{p}/`, `outputs/checkpoints_v2_controls/{condition}/h{h}ms/P{p}/`, `eeg_controls/results/phase1/{weight,phase_ridge,figures}/`.
- Horizons: 0, 200, 500 ms. Conditions: `central` = C3, Cz, C4, CP1, CP2, CP5, CP6; `edge` = Fp1, Fp2, F7, F8, T7, T8, TP9, TP10, P7, P8, PO9, PO10, O1, Oz, O2; `eye_clean` = regress on Fp1, Fp2, F7, F8 and drop them.
- Lift level window: hold = lift_off + 0.5 s to replace - 0.2 s (main); load = first_touch to lift_off. Weights CurW 1, 2, 4 = light, medium, heavy.
- Past-event bumps: 8 events of `eeg_controls.common.EVENTS`, lags 0 to +3 s after each, step 0.25 s, width 0.25 s, only events at or before the token.
- Tests: `env/Scripts/python.exe -m pytest -q tests -k "v2 or eeg_controls" --deselect tests/test_v2_notebook.py::test_committed_notebook_has_the_generators_cells`.
- Commits: explicit paths, no attribution lines; never add `main_v2.ipynb`, `handoff.md`, `CLAUDE.md`, `.superpowers/`, LaTeX build files.
- No emojis in markdown or notebook text.

## Review Focus

1. An unknown electrode name in `eeg.channels` or `eeg.eye_regress` should raise `ValueError` naming it, not crash later on a shape mismatch. Test in Task 1.
2. A lift with a missing event or no weight should be skipped by the weight check and the event features, not raise. Tests in Tasks 3 and 4.
3. A task phase with fewer than 10 validation tokens should give a NaN row in the per-phase fit, not an exception. Test in Task 3.
4. A missing K checkpoint for a condition run should stop with a message naming the expected path. Test in Task 5.
5. A resumed condition run (checkpoint complete, results missing) should reuse the trained model and keep K identical to the loaded one. Test in Task 2.

## Implementation notes (from the plan review)

1. Task 1: `select_channels(names, available)` returns indices into the *current* channel list (after `eye_regress` has dropped its references), not into the 32 `EEG_NAMES`. A channel in `eeg.channels` that the eye regression already removed raises `ValueError` naming it.
2. Task 2: with `freeze_kin`, `best_state`'s `kin.*` entries are copied from the loaded weights at the start of `fit`, `best_epoch_kin = 0`, and the K modules stay in `eval()` mode for the whole run (no dropout in K, so the fusion always reads fixed K tokens).
3. Task 3: every past-event bump is masked to exactly 0 where `t < t_event` (dt = t - t_event < 0), so a bump at lag 0 has its peak at the event token and no value before it. `test_past_events_are_causal` must include an event moved by less than one bump width.
4. Tasks 3 to 5: one helper `eeg_controls.common.wilcoxon_p(x, alternative="greater") -> float` returns `nan` when fewer than 2 finite non-zero values remain (smoke runs with one participant must not raise).
5. Task 6: figures use `constrained_layout=True` and short subplot titles, so the title-extent test passes.

---

### Task 1: EEG channel selection and eye regression in `v2`

**Files:**
- Modify: `v2/data.py` (`build_tokens`), `v2/model.py` (`build_model`), `v2/electrodes.py`
- Test: `tests/test_v2_channels.py`

**Interfaces:**
- Produces: `v2.electrodes.select_channels(names: list[str] | None) -> list[int]` (indices into `EEG_NAMES`, `None` -> all 32; unknown name -> `ValueError`); `v2.electrodes.eye_regress(eeg: np.ndarray, refs: list[str], names=EEG_NAMES) -> tuple[np.ndarray, list[str]]` (per series least squares with intercept of every non-reference channel on the reference channels; returns residual channels and their names); `v2.electrodes.channel_names(cfg) -> list[str]` (names after `eye_regress` then `channels`); `build_model(cfg, ...)` uses `len(channel_names(cfg))` electrodes and `electrode_prior(channel_names(cfg))`.
- Order inside `build_tokens`: eye regression on the raw 500 Hz EEG first, then channel selection, then band power.

- [ ] **Step 1: Write failing tests**: `test_default_keeps_32_channels` (no keys -> tokens `eeg` shape `(n, 32, 4)`, model `eeg.identity` shape `(32, 32)`); `test_channel_subset_shapes` (`central` list -> `(n, 7, 4)`, model builds and forward runs on a `(1, 400, 7, 4)` input); `test_unknown_channel_raises` (`["C3", "Cx"]` -> `ValueError` whose message contains `"Cx"`); `test_eye_regress_removes_planted_component` (random 32-channel EEG with `3 * Fp1` added to C3: after regression on Fp1, Fp2, F7, F8 the correlation of C3 with Fp1 is below 0.05, 28 channels remain, the names exclude the references); `test_eye_regress_then_subset` (references dropped, then `channels` = central -> 7 channels).
- [ ] **Step 2: Run them** (`pytest tests/test_v2_channels.py -q`): FAIL (functions missing).
- [ ] **Step 3: Implement** the three functions in `v2/electrodes.py` (regression with `np.linalg.lstsq` on `[refs, 1]`), the two keys in `build_tokens` (read from `cfg["eeg"].get(...)`), and `build_model` using `channel_names(cfg)`.
- [ ] **Step 4: Run the new tests and the whole v2 suite**: PASS (175 existing tests unchanged).
- [ ] **Step 5: Commit** `v2/data.py v2/model.py v2/electrodes.py tests/test_v2_channels.py`: "feat(v2): opt-in EEG channel subset and eye-channel regression".

### Task 2: Train E + fusion against a loaded, frozen K

**Files:**
- Modify: `v2/train.py` (`fit`)
- Create: function in `v2/train.py`: `load_kin(model, ckpt_path) -> int`
- Test: `tests/test_v2_train_frozen_kin.py`

**Interfaces:**
- Consumes: `v2.model.stream_of`.
- Produces: `load_kin(model, ckpt_path: Path) -> int` copies every `kin.*` tensor of the checkpoint's best state into `model` and returns the number of tensors copied (raises `FileNotFoundError` with the path if missing, `ValueError` if a `kin.*` shape differs). `cfg["training"]["freeze_kin"] = True` makes `fit` start with K frozen: K never gets an optimizer step, its weights at the end equal the loaded ones, `hist["frozen_epoch"]["kin"]` is `-1` (marks "loaded frozen"), `hist["best_epoch_kin"] == 0`, training ends when E + fusion freeze or at `max_epochs`.

- [ ] **Step 1: Write failing tests** (reuse the `_datasets` and `_cfg` pattern of `tests/test_v2_train.py`): `test_frozen_kin_stays_identical` (train 2 epochs normally, save; new model with `load_kin` + `freeze_kin`, train 2 epochs; every `kin.*` tensor equals the loaded one, at least one `eeg.*` tensor changed); `test_load_kin_reports_missing_file` (path in the message); `test_frozen_kin_resume_keeps_k` (second `fit` call on the same folder skips training and returns the same K); `test_freeze_kin_absent_changes_nothing` (no key -> same history keys and behaviour as before).
- [ ] **Step 2: Run them**: FAIL.
- [ ] **Step 3: Implement** `load_kin` and the `freeze_kin` branch: before the epoch loop, when the key is set and the run starts fresh, `set_trainable("kin", False)` and `frozen["kin"] = -1`; `_is_training_complete` and `update_freezes` treat `-1` as frozen and never reset E's patience for it.
- [ ] **Step 4: Run the new tests and `tests/test_v2_train.py`**: PASS.
- [ ] **Step 5: Commit** `v2/train.py tests/test_v2_train_frozen_kin.py`: "feat(v2): train E and fusion against a loaded, frozen K".

### Task 3: Per-phase ridge and causal timing baseline (part R)

**Files:**
- Create: `eeg_controls/phase_ridge.py`
- Test: `tests/test_eeg_controls_phase_ridge.py`

**Interfaces:**
- Consumes: `eeg_controls.run_horizon.horizon_config, subject_data`; `v2.residual.eeg_features, fit_ridge, _apply`; `v2.infer.predict_tokens`; `eeg_controls.common.EVENTS, lifts, segment_labels`.
- Produces:
  - `past_event_features(lift_list, n_tok: int, offset_tok: int = 0) -> np.ndarray` shape `(n_tok, 8 * 13)`; value at token t depends only on events at sample `<= (t + offset_tok) * Q` (lags 0, 0.25, ..., 3 s).
  - `fit_and_score(feats_val, res_val, lab_val, feats_test, res_test, lab_test, k_test, y_test, fit: str, n_null: int, seed: int) -> pd.DataFrame` with columns `segment, fit, n_fit, n_test, r, skill, skill_null_mean, p` (`fit` in `{"pooled", "per_phase"}`; a phase with fewer than 10 validation tokens -> NaN row).
  - `run_participant(p: int, h_ms: int, ckpt_root: Path, n_null: int = 99) -> pd.DataFrame` with columns `subject, horizon_ms, features, fit, segment, ...` for `features` in `eeg, events, both`.
  - CLI: `python -m eeg_controls.phase_ridge --participants all --horizons-ms 0,200,500 --checkpoints outputs/checkpoints_v2_horizon --out eeg_controls/results/phase1/phase_ridge`, writes `phase_ridge.csv` and `summary.csv` (per horizon, segment, features, fit: mean skill, n significant, Wilcoxon p of `both - events > 0`).

- [ ] **Step 1: Write failing tests**: `test_past_events_are_causal` (moving an event later never changes earlier rows; a lift with a `None` event gives zeros for that event's block); `test_past_events_shape_and_lags` (an event at token 100: block peaks at tokens 100, 112.5, ..., 250 within one token); `test_per_phase_fit_handles_small_phase` (a phase with 5 validation tokens -> NaN row, others finite); `test_pooled_and_per_phase_on_planted_signal` (synthetic residual equal to a feature in one phase only: per-phase skill > pooled skill in that phase).
- [ ] **Step 2: Run them**: FAIL.
- [ ] **Step 3: Implement** the module (labels from `segment_labels(..., offset=h)` as in `run_horizon.segments`; null by circular shift in multiples of the hop, as there).
- [ ] **Step 4: Run tests**: PASS. Smoke: run `run_participant` for P1 at h = 500 against the 1-epoch smoke checkpoint in the scratchpad (`--checkpoints <scratchpad>/hz_ckpt`, `--out <scratchpad>/phase1/phase_ridge`, `n_null=5`).
- [ ] **Step 5: Commit** `eeg_controls/phase_ridge.py tests/test_eeg_controls_phase_ridge.py`: "feat(eeg-controls): per-phase ridge and causal timing baseline".

### Task 4: Weight check (part W)

**Files:**
- Create: `eeg_controls/weight_check.py`
- Test: `tests/test_eeg_controls_weight_check.py`

**Interfaces:**
- Consumes: `v2.config.CONFIG_V2`, `v2.model.build_model`, `v2.train.load_best`, `v2.infer.predict_series`, `eeg_controls.run_horizon.subject_data`, `eeg_controls.common.lifts`.
- Produces:
  - `lift_levels(pred: dict[str, np.ndarray], target: np.ndarray, lift_list, fs=500.0) -> pd.DataFrame`, one row per lift and muscle and window (`hold`, `load`): `lift, weight, window, muscle, target, K, E, F` (mean envelope; lifts with a missing event or weight skipped).
  - `calibrate(levels_val, levels_test) -> pd.DataFrame` adds column `E_cal` (one least-squares gain per muscle from the validation lifts, through the origin).
  - `weight_stats(levels) -> pd.DataFrame` per muscle and output: mean level per weight, `ratio_heavy_light`, `rho_weight` (Spearman over lifts), `slope` (output on target over lifts), `size_ratio` (mean output / mean target).
  - `run_participant(p, ckpt_root=Path("outputs/checkpoints_v2")) -> tuple[pd.DataFrame, pd.DataFrame]` (levels, stats) for the hold and load windows; also returns the predictions of the first test series for the example figure via `run_participant(..., keep_example=True)`.
  - CLI: `python -m eeg_controls.weight_check --participants all --checkpoints outputs/checkpoints_v2 --out eeg_controls/results/phase1/weight` writes `levels.csv`, `stats.csv`, `group.csv` (Wilcoxon over participants of `rho_weight` of E > 0 and of E's heavy/light ratio against the target's).

- [ ] **Step 1: Write failing tests**: `test_lift_levels_windows` (synthetic envelope with known plateaus per lift -> exact hold and load means); `test_missing_event_or_weight_skipped`; `test_calibration_recovers_gain` (E = target / 3 -> `E_cal` equals target); `test_weight_stats_on_planted_weights` (target rising with weight, E flat -> target `rho_weight` near 1, E near 0, E `slope` near 0).
- [ ] **Step 2: Run them**: FAIL.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run tests**: PASS. Smoke: train a 1-epoch non-causal P1 model into the scratchpad (`v2.train.fit` with `max_epochs=1`), run `run_participant(1, ckpt_root=<scratchpad>/w_ckpt)`.
- [ ] **Step 5: Commit** `eeg_controls/weight_check.py tests/test_eeg_controls_weight_check.py`: "feat(eeg-controls): weight check of the E level per lift".

### Task 5: Condition runner (parts S and E)

**Files:**
- Create: `eeg_controls/run_conditions.py`
- Test: `tests/test_eeg_controls_conditions.py`

**Interfaces:**
- Consumes: Task 1 keys, Task 2 `load_kin` and `freeze_kin`, Task 3 `run_participant` internals (`fit_and_score`), `eeg_controls.run_horizon.horizon_config, subject_data, standard, segments`, `v2.residual.residual_test`.
- Produces:
  - `CONDITIONS: dict[str, dict]` with `central`, `edge`, `eye_clean` (the `eeg` keys of each).
  - `condition_config(name: str, h_ms: int, smoke=False, use_amp=False) -> dict` = `horizon_config` + the condition's `eeg` keys + `training.freeze_kin = True`.
  - `run_one(p, name, h_ms, args, device) -> Path`: loads K from `{args.kin_checkpoints}/h{h}ms/P{p}/best.pt` (missing -> `FileNotFoundError` naming it), trains into `{args.checkpoints}/{name}/h{h}ms/P{p}`, writes `standard.csv`, `residual.csv`, `segments.csv`, `phase_ridge.csv`, `history.json` to `{args.results}/{name}/h{h}ms/P{p}/`; skips finished runs unless `--overwrite`.
  - `summarise(results: Path, baseline: Path = Path("results/v2_horizon")) -> pd.DataFrame` one row per condition (`all` read from `baseline`), horizon and participant: `E_r, F_minus_K_vaf, res_r_tokens, pre_move_r, reach_r, pre_move_skill, reach_skill`; writes `summary.csv` and `vs_all.csv` (Wilcoxon of each condition against `all`, paired by participant).
  - CLI: `python -m eeg_controls.run_conditions --participants all --conditions central,edge,eye_clean --horizons-ms 0,200,500 [--smoke] [--overwrite] [--summary-only]`, defaults as in Global Constraints, `--kin-checkpoints outputs/checkpoints_v2_horizon`.

- [ ] **Step 1: Write failing tests**: `test_condition_configs` (each config: causal, horizon tokens = h / 20, `freeze_kin` True, the right `eeg` keys; unknown condition -> `ValueError` listing the valid names); `test_missing_kin_checkpoint_message` (temporary empty folder -> `FileNotFoundError` with the expected path); `test_summarise_reads_all_and_conditions` (small fake result folders -> one row per condition, horizon, participant; `vs_all.csv` has a p column).
- [ ] **Step 2: Run them**: FAIL.
- [ ] **Step 3: Implement** the module (reuse `run_horizon` helpers; no copy of their code).
- [ ] **Step 4: Run tests**: PASS. Smoke in the scratchpad: first a 1-epoch causal P1 horizon model at h = 0 (`run_horizon` with `--smoke`, scratchpad paths), then `run_conditions --participants 1 --conditions central,eye_clean --horizons-ms 0 --smoke` with scratchpad `--kin-checkpoints`, `--checkpoints`, `--results`; check K weights equal and the five files exist.
- [ ] **Step 5: Commit** `eeg_controls/run_conditions.py tests/test_eeg_controls_conditions.py`: "feat(eeg-controls): runner for electrode-subset and eye-regression conditions".

### Task 6: Figures of phase 1

**Files:**
- Create: `eeg_controls/plots_phase1.py`
- Test: `tests/test_eeg_controls_plots_phase1.py`

**Interfaces:**
- Consumes: the CSV frames of Tasks 3, 4, 5; style from `v2.interp.plots` (`P.style()`), colours as in `eeg_controls/plots.py` (E `#2a78d6`, K `#eb6834`).
- Produces (each returns a `Figure`, saves nothing):
  - `weight_levels(stats)` W1: mean hold level per weight, per muscle; target, E, E_cal, K; normalised to the target's medium level.
  - `weight_scatter(levels)` W2: E (and E_cal) level against target level per lift, coloured by weight, identity line.
  - `weight_example(example, names, start_s, length_s)` W3: target, E, E_cal on a stretch with three weights, weight labelled above each lift.
  - `weight_participants(stats)` W4: per participant `rho_weight` and `slope` for E, E_cal, K, with the target's `rho_weight`.
  - `ridge_heatmaps(summary)` R1: phase x horizon skill; one panel per feature set and fit, `x/n sig.` in each cell.
  - `ridge_lines(summary)` R2: skill against horizon for pre_move and reach; lines `eeg`, `events`, `both`; pooled and per_phase.
  - `ridge_gain(phase_ridge)` R3: per participant `both - events` per phase at each horizon.
  - `conditions_overview(summary)` C1: E r and F - K per condition and horizon.
  - `conditions_phases(summary)` C2: pre_move and reach residual r per condition and horizon.
  - `conditions_participants(summary)` C3: each condition's E r against `all`, one dot per participant, identity line.
- [ ] **Step 1: Write failing tests**: one test per function on small synthetic frames with the documented columns: returns a `matplotlib.figure.Figure` with the expected number of axes, and no text overlaps the axes title (check that every title fits in the figure width via `fig.canvas.draw()` and `get_window_extent`).
- [ ] **Step 2: Run them**: FAIL.
- [ ] **Step 3: Implement** the module.
- [ ] **Step 4: Run tests**: PASS; render every figure from the smoke outputs of Tasks 3 to 5 into the scratchpad and look at each image.
- [ ] **Step 5: Commit** `eeg_controls/plots_phase1.py tests/test_eeg_controls_plots_phase1.py`: "feat(eeg-controls): figures of phase 1".

### Task 7: Notebook, generator and README

**Files:**
- Create: `tools/build_eeg_controls_phase1_notebook.py`, `eeg_controls_phase1.ipynb`, `tests/test_eeg_controls_phase1_notebook.py`
- Modify: `eeg_controls/README.md` (section "Phase 1"), `.gitignore` (nothing needed: `eeg_controls/results/` is already ignored)

**Interfaces:**
- Consumes: CLIs and functions of Tasks 3 to 6.
- Produces: notebook cells, each code cell starting with `# P1-`: `P1-SETUP` (participants, paths, device, `show(fig, name)` saving to `eeg_controls/results/phase1/figures/`), `P1-W-RUN`, `P1-W-PLOT`, `P1-R-RUN`, `P1-R-PLOT`, `P1-C-RUN` (training; `C_PARTICIPANTS`, `CONDITIONS = ["central", "edge", "eye_clean"]`, `HORIZONS_MS = [0, 200, 500]`), `P1-C-PLOT`; a markdown guide before each section saying what to look for in each figure.
- [ ] **Step 1: Write failing tests**: every code cell parses and starts with `# P1-`; 7 code cells; the committed notebook matches the generator; all phase-1 modules import.
- [ ] **Step 2: Run them**: FAIL.
- [ ] **Step 3: Implement** the generator (same pattern as `tools/build_eeg_controls_notebook.py`), generate the notebook, write the README section (commands, outputs, run times from the design's cost table).
- [ ] **Step 4: Run the whole suite** (command in Global Constraints): PASS. Execute the non-training cells for P1 against the scratchpad smoke checkpoints with outputs redirected to the scratchpad; all figures appear.
- [ ] **Step 5: Commit** `tools/build_eeg_controls_phase1_notebook.py eeg_controls_phase1.ipynb tests/test_eeg_controls_phase1_notebook.py eeg_controls/README.md eeg_controls/DESIGN_phase1.md eeg_controls/PLAN_phase1.md`: "feat(eeg-controls): phase-1 notebook, generator and guide".
