# Conservative Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the current EEG-to-EMG training path so reusable pipeline logic lives in `src/`, the script becomes a thinner entrypoint, and `notebooks/02_train_transformer.ipynb` gains clear `# @title` headers without changing training behavior.

**Architecture:** Introduce a small shared pipeline module for transformer-training setup, move the runnable transformer regressor into `src/models`, and keep `scripts/train_transformer.py` focused on CLI and presentation. Preserve the current config format, dataset semantics, CCA flow, checkpoint behavior, and notebook workflow.

**Tech Stack:** Python, PyTorch, NumPy, YAML, Jupyter Notebook JSON, PowerShell, ripgrep

## Global Constraints

- Preserve `configs/default.yaml` schema and current paths verbatim.
- Preserve dataset split behavior from `src/data/loader.py`.
- Preserve CCA fit/project semantics from `src/preprocessing/cca.py`.
- Preserve checkpoint filenames and resume behavior from `src/training/train.py`.
- Do not force migration to `kg_gat.py` / `kg_gt.py`.
- Keep notebook edits conservative and readability-focused.

---

### Task 1: Add Shared Pipeline Helpers

**Files:**
- Create: `src/pipelines/__init__.py`
- Create: `src/pipelines/transformer_training.py`
- Test: `scripts/train_transformer.py`

**Interfaces:**
- Consumes: `WAYEEGDataset`, `make_cca_from_config`, `preprocess_eeg_from_config`, `preprocess_emg_from_config`
- Produces:
  - `make_preprocess_fn(cfg: dict) -> Callable`
  - `build_dataset(split: str, cfg: dict, participants: list[int], root_dir: Path) -> WAYEEGDataset`
  - `build_datasets(cfg: dict, participants: list[int], root_dir: Path) -> tuple[WAYEEGDataset, WAYEEGDataset, WAYEEGDataset]`
  - `unique_series_arrays(ds: WAYEEGDataset) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]`
  - `fit_cca_and_emg_stats(train_ds: WAYEEGDataset, cca_cfg: dict, device: torch.device) -> tuple[object, torch.Tensor, torch.Tensor]`
  - `prepare_batch_factory(cca_projector: object, emg_mean: torch.Tensor, emg_std: torch.Tensor, device: torch.device, n_cca: int) -> Callable`

- [ ] **Step 1: Write the failing import probe**

```python
from pathlib import Path

from src.pipelines.transformer_training import (
    build_dataset,
    build_datasets,
    fit_cca_and_emg_stats,
    make_preprocess_fn,
    prepare_batch_factory,
    unique_series_arrays,
)

assert callable(make_preprocess_fn)
assert callable(build_dataset)
assert callable(build_datasets)
assert callable(unique_series_arrays)
assert callable(fit_cca_and_emg_stats)
assert callable(prepare_batch_factory)
assert Path("src/pipelines/__init__.py").exists()
```

- [ ] **Step 2: Run import probe to verify it fails**

Run: `@'from pathlib import Path\nfrom src.pipelines.transformer_training import make_preprocess_fn\n'@ | python -`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipelines'`

- [ ] **Step 3: Write minimal shared pipeline module**

```python
# src/pipelines/__init__.py
from .transformer_training import (
    build_dataset,
    build_datasets,
    fit_cca_and_emg_stats,
    make_preprocess_fn,
    prepare_batch_factory,
    unique_series_arrays,
)

__all__ = [
    "make_preprocess_fn",
    "build_dataset",
    "build_datasets",
    "unique_series_arrays",
    "fit_cca_and_emg_stats",
    "prepare_batch_factory",
]
```

```python
# src/pipelines/transformer_training.py
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.data.dataset import WAYEEGDataset
from src.preprocessing.cca import make_cca_from_config
from src.preprocessing.eeg import preprocess_eeg_from_config
from src.preprocessing.emg import preprocess_emg_from_config


def make_preprocess_fn(cfg: dict) -> Callable:
    eeg_cfg = cfg["preprocessing"]["eeg"]
    emg_cfg = cfg["preprocessing"]["emg"]

    def preprocess_fn(series: dict) -> dict:
        series = dict(series)
        series["eeg"] = preprocess_eeg_from_config(
            series["eeg"],
            float(series["fs_eeg"]),
            eeg_cfg,
            channel_names=series.get("eeg_names"),
        )
        series["emg"] = preprocess_emg_from_config(
            series["emg"],
            float(series["fs_emg"]),
            emg_cfg,
        )
        return series

    return preprocess_fn


def build_dataset(split: str, cfg: dict, participants: list[int], root_dir: Path) -> WAYEEGDataset:
    data_cfg = cfg["data"]
    return WAYEEGDataset(
        data_dir=root_dir / data_cfg["raw_dir"],
        participants=participants,
        split=split,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        preprocess_fn=make_preprocess_fn(cfg),
        cache_dir=root_dir / data_cfg["cache_dir"],
    )
```

- [ ] **Step 4: Run import probe to verify it passes**

Run: `@'from src.pipelines.transformer_training import make_preprocess_fn, build_dataset\nprint(callable(make_preprocess_fn), callable(build_dataset))\n'@ | python -`
Expected: PASS and print `True True`

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/__init__.py src/pipelines/transformer_training.py
git commit -m "refactor: add shared transformer training pipeline helpers"
```

### Task 2: Move the Runnable Transformer Regressor into `src/models`

**Files:**
- Create: `src/models/transformer_regressor.py`
- Modify: `src/models/__init__.py`
- Test: `scripts/train_transformer.py`

**Interfaces:**
- Consumes: `build_transformer_from_config(cfg: dict, input_dim: int) -> TransformerEncoder`
- Produces:
  - `class TransformerRegressor(nn.Module)`
  - `build_transformer_regressor(cfg: dict, input_dim: int, out_channels: int = 5) -> TransformerRegressor`

- [ ] **Step 1: Write the failing import probe**

```python
from src.models import TransformerRegressor, build_transformer_regressor

assert TransformerRegressor.__name__ == "TransformerRegressor"
assert callable(build_transformer_regressor)
```

- [ ] **Step 2: Run import probe to verify it fails**

Run: `@'from src.models import TransformerRegressor\n'@ | python -`
Expected: FAIL with `ImportError` because `TransformerRegressor` is not exported yet

- [ ] **Step 3: Write the shared model module**

```python
# src/models/transformer_regressor.py
from __future__ import annotations

import torch.nn as nn

from .transformer import build_transformer_from_config


class TransformerRegressor(nn.Module):
    def __init__(self, cfg: dict, input_dim: int, out_channels: int = 5):
        super().__init__()
        self.encoder = build_transformer_from_config(cfg["model"]["transformer"], input_dim)
        self.head = nn.Linear(self.encoder.d_model, out_channels)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)


def build_transformer_regressor(cfg: dict, input_dim: int, out_channels: int = 5) -> TransformerRegressor:
    return TransformerRegressor(cfg=cfg, input_dim=input_dim, out_channels=out_channels)
```

```python
# src/models/__init__.py
from .transformer_regressor import TransformerRegressor, build_transformer_regressor
```

- [ ] **Step 4: Run import probe to verify it passes**

Run: `@'from src.models import TransformerRegressor, build_transformer_regressor\nprint(TransformerRegressor.__name__)\nprint(callable(build_transformer_regressor))\n'@ | python -`
Expected: PASS, printing `TransformerRegressor` and `True`

- [ ] **Step 5: Commit**

```bash
git add src/models/transformer_regressor.py src/models/__init__.py
git commit -m "refactor: move transformer regressor into models package"
```

### Task 3: Rewrite the Training Script as a Thin Entry Point

**Files:**
- Modify: `scripts/train_transformer.py`
- Test: `scripts/train_transformer.py`

**Interfaces:**
- Consumes:
  - `build_dataset` / `build_datasets`
  - `fit_cca_and_emg_stats`
  - `prepare_batch_factory`
  - `build_transformer_regressor`
- Produces: script entrypoint that preserves CLI and UI behavior while no longer defining pipeline logic inline

- [ ] **Step 1: Write the failing regression probe**

```python
from pathlib import Path

script_text = Path("scripts/train_transformer.py").read_text(encoding="utf-8")

assert "class TransformerRegressor" not in script_text
assert "from src.pipelines import" in script_text
assert "from src.models import TransformerRegressor" in script_text or "build_transformer_regressor" in script_text
```

- [ ] **Step 2: Run regression probe to verify it fails**

Run: `@'from pathlib import Path\nscript_text = Path("scripts/train_transformer.py").read_text(encoding="utf-8")\nassert "class TransformerRegressor" not in script_text\n'@ | python -`
Expected: FAIL because the class is still defined inline

- [ ] **Step 3: Replace inline pipeline logic with imports and calls**

```python
from src.models import build_transformer_regressor
from src.pipelines import (
    build_dataset,
    fit_cca_and_emg_stats,
    prepare_batch_factory,
    unique_series_arrays,
)

train_ds = build_dataset("train", cfg, participants, ROOT)
val_ds = build_dataset("val", cfg, participants, ROOT)
test_ds = build_dataset("test", cfg, participants, ROOT)

cca_gpu, EMG_MEAN, EMG_STD = fit_cca_and_emg_stats(
    train_ds,
    cfg["preprocessing"]["cca"],
    DEVICE,
)
prepare_batch = prepare_batch_factory(cca_gpu, EMG_MEAN, EMG_STD, DEVICE, N_CCA)
model = build_transformer_regressor(cfg, input_dim=N_CCA).to(DEVICE)
```

- [ ] **Step 4: Run script help and import validation**

Run: `python scripts/train_transformer.py --help`
Expected: PASS and show the existing CLI arguments without import errors

- [ ] **Step 5: Commit**

```bash
git add scripts/train_transformer.py
git commit -m "refactor: thin training script entrypoint"
```

### Task 4: Add Notebook `# @title` Headers and Align Lightly with Shared Helpers

**Files:**
- Modify: `notebooks/02_train_transformer.ipynb`
- Test: `notebooks/02_train_transformer.ipynb`

**Interfaces:**
- Consumes: existing notebook function/class cells
- Produces: titled cells for `plot_time_series`, `make_preprocess_fn`, `build_ds`, `unique_series_arrays`, `prepare_batch`, `TransformerRegressor`, `test_untrained_inference`

- [ ] **Step 1: Write the failing notebook-title probe**

```python
import json
from pathlib import Path

nb = json.loads(Path("notebooks/02_train_transformer.ipynb").read_text(encoding="utf-8"))
sources = ["".join(cell.get("source", [])) for cell in nb["cells"] if cell.get("cell_type") == "code"]

required_titles = [
    "# @title make_preprocess_fn",
    "# @title build_ds",
    "# @title unique_series_arrays",
    "# @title prepare_batch",
    "# @title TransformerRegressor",
    "# @title test_untrained_inference",
]

for title in required_titles:
    assert any(src.startswith(title) for src in sources)
```

- [ ] **Step 2: Run notebook-title probe to verify it fails**

Run: `@'import json\nfrom pathlib import Path\nnb = json.loads(Path("notebooks/02_train_transformer.ipynb").read_text(encoding="utf-8"))\nsources = ["".join(cell.get("source", [])) for cell in nb["cells"] if cell.get("cell_type") == "code"]\nassert any(src.startswith("# @title make_preprocess_fn") for src in sources)\n'@ | python -`
Expected: FAIL because some relevant cells lack the required title

- [ ] **Step 3: Insert `# @title` headers into relevant cells**

```python
# @title make_preprocess_fn
def make_preprocess_fn(cfg):
    ...
```

```python
# @title TransformerRegressor
class TransformerRegressor(nn.Module):
    ...
```

```python
# @title test_untrained_inference
def test_untrained_inference():
    ...
```

- [ ] **Step 4: Run notebook JSON validation**

Run: `@'import json\nfrom pathlib import Path\njson.loads(Path("notebooks/02_train_transformer.ipynb").read_text(encoding="utf-8"))\nprint("ok")\n'@ | python -`
Expected: PASS and print `ok`

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_train_transformer.ipynb
git commit -m "docs: add notebook title headers for training notebook"
```

### Task 5: Final Validation Sweep

**Files:**
- Modify: none
- Test: `src/pipelines/__init__.py`, `src/pipelines/transformer_training.py`, `src/models/transformer_regressor.py`, `scripts/train_transformer.py`, `notebooks/02_train_transformer.ipynb`

**Interfaces:**
- Consumes: all outputs from Tasks 1-4
- Produces: validated conservative refactor ready for later optimization work

- [ ] **Step 1: Run Python compile check on modified source files**

Run: `python -m py_compile src/pipelines/__init__.py src/pipelines/transformer_training.py src/models/transformer_regressor.py scripts/train_transformer.py`
Expected: PASS with no output

- [ ] **Step 2: Run script CLI smoke check**

Run: `python scripts/train_transformer.py --help`
Expected: PASS and print CLI help

- [ ] **Step 3: Run notebook JSON verification**

Run: `@'import json\nfrom pathlib import Path\njson.loads(Path("notebooks/02_train_transformer.ipynb").read_text(encoding="utf-8"))\nprint("notebook valid")\n'@ | python -`
Expected: PASS and print `notebook valid`

- [ ] **Step 4: Review for placeholder regressions**

```text
Confirm:
- no inline TransformerRegressor class remains in scripts/train_transformer.py
- src/pipelines owns shared setup helpers
- notebook title headers exist for the targeted cells
- no config keys or checkpoint paths changed
```

- [ ] **Step 5: Commit**

```bash
git add src/models/__init__.py src/models/transformer_regressor.py src/pipelines/__init__.py src/pipelines/transformer_training.py scripts/train_transformer.py notebooks/02_train_transformer.ipynb
git commit -m "refactor: consolidate transformer training pipeline"
```

## Self-Review

1. **Spec coverage:** This plan covers new shared pipeline helpers, moving the runnable model to `src/models`, thinning the training script, and adding notebook title headers.
2. **Placeholder scan:** No `TODO`, `TBD`, or undefined handoff references remain.
3. **Type consistency:** The produced helper names and model names are used consistently across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-conservative-refactor.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
