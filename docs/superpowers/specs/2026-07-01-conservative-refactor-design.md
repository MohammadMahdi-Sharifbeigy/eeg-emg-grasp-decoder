# Conservative Refactor Design

**Goal:** Clean up the EEG-to-EMG training codebase without changing the current research workflow or destabilizing the runnable training path.

## Scope

This refactor is intentionally conservative.

It will:
- keep the current repository layout centered on `src/`, `scripts/`, and `notebooks/`
- preserve the current executable training behavior
- preserve the current config format in `configs/default.yaml`
- preserve current dataset split semantics, preprocessing semantics, and checkpoint locations
- improve code reuse between the notebook and the terminal training script
- improve readability of `notebooks/02_train_transformer.ipynb` by adding `# @title [Function/Class Name]` to relevant cells

It will not:
- redesign the project for Method 2, Method 3, or Method 4
- force the current runnable pipeline onto the partially implemented KG-GT/GAT path
- rewrite the `rich` UI layer from scratch
- change experiment outputs unless required for correctness

## Current Structure

The project currently has two overlapping layers:

1. A reusable package-oriented layer in `src/`
2. An experiment-oriented orchestration layer in `scripts/` and `notebooks/`

The main duplication is in the training setup path:
- config-driven preprocessing closure construction
- dataset building
- extraction of unique series for CCA fitting
- EMG normalization statistics fitting
- per-batch preparation logic
- local definition of the runnable transformer regressor

The training script currently mixes:
- CLI parsing
- terminal UI rendering
- data pipeline setup
- model construction
- training orchestration
- evaluation orchestration

This makes the script harder to maintain and causes the notebook and script to drift.

## Target Design

### 1. `src/` remains the source of truth

Reusable training-pipeline logic should live in `src/`, not in the notebook or script.

The script should become a thin entrypoint that:
- parses args
- loads config
- shows UI
- calls reusable pipeline helpers
- launches training and evaluation

The notebook should remain interactive, but should rely on the same reusable setup helpers where practical.

### 2. Introduce a small shared pipeline layer

A new `src/pipelines/` package will collect conservative orchestration helpers that are too high-level for the existing `data/`, `preprocessing/`, `models/`, and `training/` modules but are still reusable.

This layer will own:
- preprocess function construction from config
- dataset construction helpers
- unique-series extraction for CCA/stat fitting
- train-time CCA fitting and projector export
- EMG normalization-stat fitting
- batch preparation helpers

This avoids pushing orchestration concerns into low-level modules while removing duplication from notebook and script entrypoints.

### 3. Move the runnable regressor into `src/models`

The currently active runnable model is a transformer encoder followed by a linear head. That definition should not live inline in the training script.

The refactor will move that model into a reusable module under `src/models/`, while keeping its behavior unchanged.

This does not resolve the larger architectural mismatch between the report and the current runnable path; it only ensures the current path is implemented cleanly.

### 4. Keep UI separate from training logic

The `rich` UI helpers in `scripts/train_transformer.py` may stay in the script if they are presentation-only, but logic that affects data preparation, model building, or training behavior should move to `src/`.

This preserves the current user experience without coupling the scientific pipeline to terminal-rendering code.

## Planned File Changes

### New files

- `src/pipelines/__init__.py`
- `src/pipelines/transformer_training.py`
- `src/models/transformer_regressor.py`

### Modified files

- `scripts/train_transformer.py`
- `src/models/__init__.py`
- `notebooks/02_train_transformer.ipynb`

### Files intentionally left unchanged unless needed for compatibility

- `src/data/loader.py`
- `src/data/dataset.py`
- `src/preprocessing/*.py`
- `src/training/train.py`
- `src/training/evaluate.py`

## Responsibilities After Refactor

### `src/pipelines/transformer_training.py`

Will provide small reusable helpers such as:
- `make_preprocess_fn(cfg)`
- `build_dataset(split, cfg, participants, root_dir)`
- `build_datasets(cfg, participants, root_dir)`
- `unique_series_arrays(ds)`
- `fit_cca_and_emg_stats(train_ds, cca_cfg, device)`
- `prepare_batch_factory(cca_projector, emg_mean, emg_std, device)`

These helpers will preserve the current training semantics.

### `src/models/transformer_regressor.py`

Will expose the currently used transformer-plus-linear-head model as a reusable class and builder function.

### `scripts/train_transformer.py`

Will keep:
- CLI parsing
- `rich` display helpers
- top-level execution flow

It will stop owning:
- inline model definition
- data setup logic that can be shared
- batch-preparation logic that can be shared

### `notebooks/02_train_transformer.ipynb`

Will remain a notebook, but readability will improve via:
- `# @title [Function/Class Name]` headers for cells defining functions/classes used in the notebook
- better alignment with shared setup helpers where this can be done without disrupting notebook usability

## Notebook Title Plan

The notebook currently defines several helper functions and one local model class. Relevant code cells should receive titles matching the primary function/class defined in each cell.

Examples:
- `# @title plot_time_series`
- `# @title make_preprocess_fn`
- `# @title build_ds`
- `# @title unique_series_arrays`
- `# @title prepare_batch`
- `# @title TransformerRegressor`
- `# @title test_untrained_inference`

If a cell contains multiple related helper functions, the title should use the dominant or first-defined function name, unless splitting the cell is trivial and clearly beneficial.

## Constraints

- No broad redesign
- No destructive git cleanup
- No change to current data paths
- No change to checkpoint filenames or resume behavior
- No change to train/val/test split mapping
- No forced migration to the unfinished GAT path

## Risks

### Behavior drift

Moving setup code out of the script could accidentally change tensor shapes, device placement, or normalization semantics.

Mitigation:
- preserve function signatures and tensor flow exactly
- keep helper functions thin
- run a compile/import validation after edits

### Notebook/script divergence

If the notebook is only partially aligned with shared helpers, drift can continue.

Mitigation:
- move the highest-value shared setup logic first
- keep notebook edits conservative and readability-focused

### Hidden dependence on script-local names

The script currently depends on locally scoped closures and constants.

Mitigation:
- extract those pieces behind explicit helper factories
- keep UI-only constants in the script

## Validation Plan

After implementation:
- run a Python compile/import check across modified `src/` and `scripts/` files
- verify `scripts/train_transformer.py` still imports and resolves its dependencies
- verify the notebook JSON remains valid after title insertion

Full training will not be used as the primary validation step because the project currently has multi-hour runtime and GPU-memory constraints.

## Success Criteria

The refactor is successful if:
- the current runnable transformer training path still works from the script entrypoint
- training setup code is no longer duplicated between script and notebook where practical
- the transformer regressor is no longer defined inline inside the training script
- notebook code cells defining key functions/classes in `02_train_transformer.ipynb` have clear `# @title ...` headers
- the codebase is easier to extend in the later optimization and GAT planning phases
