# Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a current-GPU training path that avoids OOM and materially reduces epoch time through profiling, caching, loader tuning, staged loss/runtime changes, and moderate model simplification.

**Architecture:** Keep the same end-to-end EEG-to-EMG objective, but add a profiling layer, offline cache paths, controllable runtime knobs, and cheaper training defaults. Apply optimizations in measured stages so each change can be compared against baseline time, VRAM, and validation quality.

**Tech Stack:** Python, PyTorch, NumPy, YAML, PowerShell, CSV/Markdown reporting

## Global Constraints

- Optimize for the current GPU first.
- Preserve train/val/test semantics from `src/data/loader.py`.
- Keep CCA fit-on-train-only semantics.
- Permit moderate runtime/model simplifications when justified by speed or VRAM gains.
- Record measured impact after each optimization stage.

---

### Task 1: Add Repeatable Profiling and Baseline Reporting

**Files:**
- Create: `src/training/profile.py`
- Create: `scripts/profile_transformer.py`
- Create: `outputs/profiling/.gitkeep`
- Test: `scripts/profile_transformer.py`

**Interfaces:**
- Consumes: current dataset/model/training helpers
- Produces:
  - `capture_batch_profile(...) -> dict`
  - `write_profile_report(report: dict, output_path: Path) -> None`
  - CLI profiling script that runs one bounded baseline pass

- [ ] **Step 1: Write the failing import probe**

```python
from src.training.profile import capture_batch_profile, write_profile_report

assert callable(capture_batch_profile)
assert callable(write_profile_report)
```

- [ ] **Step 2: Run import probe to verify it fails**

Run: `@'from src.training.profile import capture_batch_profile\n'@ | python -`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write the profiling utilities and CLI**

```python
# src/training/profile.py
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


def capture_batch_profile(model, loader, prepare_batch, device, max_batches: int = 10) -> dict:
    timings = []
    peak_allocated = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (eeg, _kin, emg) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            t0 = time.perf_counter()
            x, y = prepare_batch(eeg, emg)
            t1 = time.perf_counter()
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_allocated = max(peak_allocated, torch.cuda.memory_allocated(device))
            t2 = time.perf_counter()
            timings.append(
                {
                    "batch_idx": batch_idx,
                    "prepare_s": t1 - t0,
                    "forward_s": t2 - t1,
                    "loss": float(loss.item()),
                }
            )
    return {"timings": timings, "peak_allocated": peak_allocated}


def write_profile_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run a bounded baseline profile**

Run: `python scripts/profile_transformer.py --participants 1 2 3 --max-batches 5 --output outputs/profiling/baseline.json`
Expected: PASS and create `outputs/profiling/baseline.json`

- [ ] **Step 5: Commit**

```bash
git add src/training/profile.py scripts/profile_transformer.py outputs/profiling/.gitkeep
git commit -m "perf: add bounded profiling baseline tools"
```

### Task 2: Add Preprocessed-Series and Optional Post-CCA Cache Support

**Files:**
- Modify: `src/pipelines/transformer_training.py`
- Create: `src/data/feature_cache.py`
- Modify: `configs/default.yaml`
- Test: `scripts/profile_transformer.py`

**Interfaces:**
- Consumes: dataset series arrays, fitted CCA projector
- Produces:
  - `save_preprocessed_series(...)`
  - `load_preprocessed_series(...)`
  - `save_projected_series(...)`
  - `load_projected_series(...)`
  - config keys for cache control

- [ ] **Step 1: Write the failing cache import probe**

```python
from src.data.feature_cache import (
    load_preprocessed_series,
    load_projected_series,
    save_preprocessed_series,
    save_projected_series,
)

assert callable(save_preprocessed_series)
assert callable(load_preprocessed_series)
assert callable(save_projected_series)
assert callable(load_projected_series)
```

- [ ] **Step 2: Run cache import probe to verify it fails**

Run: `@'from src.data.feature_cache import save_preprocessed_series\n'@ | python -`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the cache module and config entries**

```python
# src/data/feature_cache.py
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_preprocessed_series(path: Path, eeg: np.ndarray, kin: np.ndarray, emg: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, eeg=eeg, kin=kin, emg=emg)


def load_preprocessed_series(path: Path):
    if not path.exists():
        return None
    data = np.load(path)
    return data["eeg"], data["kin"], data["emg"]
```

```yaml
# configs/default.yaml
data:
  preprocessed_cache_dir: data/cache/preprocessed
  projected_cache_dir: data/cache/projected
  use_projected_cache: false
```

- [ ] **Step 4: Re-run profiling with caches enabled**

Run: `python scripts/profile_transformer.py --participants 1 2 3 --max-batches 5 --output outputs/profiling/cache-enabled.json`
Expected: PASS and create a second report suitable for comparison with the baseline

- [ ] **Step 5: Commit**

```bash
git add src/data/feature_cache.py src/pipelines/transformer_training.py configs/default.yaml
git commit -m "perf: add reusable preprocessing and projection cache hooks"
```

### Task 3: Add Configurable Loader, AMP, and Accumulation Controls

**Files:**
- Modify: `src/training/train.py`
- Modify: `scripts/train_transformer.py`
- Modify: `configs/default.yaml`
- Test: `scripts/train_transformer.py`

**Interfaces:**
- Consumes: existing `TrainConfig`
- Produces:
  - `gradient_accumulation_steps: int`
  - `log_memory_every: int`
  - loader config keys: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`

- [ ] **Step 1: Write the failing config probe**

```python
from src.training import TrainConfig

cfg = TrainConfig()
assert hasattr(cfg, "checkpoint_dir")
assert hasattr(cfg, "use_amp")
assert hasattr(cfg, "grad_clip_norm")
assert hasattr(cfg, "gradient_accumulation_steps")
```

- [ ] **Step 2: Run config probe to verify it fails**

Run: `@'from src.training import TrainConfig\ncfg = TrainConfig()\nassert hasattr(cfg, "gradient_accumulation_steps")\n'@ | python -`
Expected: FAIL because the field does not exist yet

- [ ] **Step 3: Add runtime controls**

```python
@dataclass
class TrainConfig:
    lr: float = 1e-3
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    log_memory_every: int = 50
```

```yaml
training:
  gradient_accumulation_steps: 1
  log_memory_every: 50

runtime:
  num_workers: 2
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

- [ ] **Step 4: Run a smoke training command with the new knobs**

Run: `python scripts/train_transformer.py --participants 1 --max-epochs 1 --batch-size 8`
Expected: PASS and complete one short training epoch without config or dataloader errors

- [ ] **Step 5: Commit**

```bash
git add src/training/train.py scripts/train_transformer.py configs/default.yaml
git commit -m "perf: add runtime controls for loaders amp and accumulation"
```

### Task 4: Reduce Sequence Cost and Stage Expensive Loss Behavior

**Files:**
- Modify: `configs/default.yaml`
- Modify: `src/losses/soft_dtw.py`
- Modify: `scripts/train_transformer.py`
- Test: `scripts/profile_transformer.py`

**Interfaces:**
- Consumes: existing training config
- Produces:
  - staged pure-MSE mode
  - explicit sequence-cost comparison runs using window/stride settings
  - optional fine-tune mode for Soft-DTW later

- [ ] **Step 1: Write the failing staged-loss probe**

```python
from src.losses import build_loss_from_config

loss = build_loss_from_config({"loss_lambda": 1.0, "soft_dtw_gamma": 0.1})
assert loss.loss_lambda == 1.0
```

- [ ] **Step 2: Run loss probe and baseline compare command**

Run: `python scripts/profile_transformer.py --participants 1 2 3 --max-batches 5 --output outputs/profiling/sequence-baseline.json`
Expected: PASS and preserve the current sequence-cost baseline for comparison

- [ ] **Step 3: Add explicit staged configs**

```yaml
training:
  loss_lambda: 1.0
  fine_tune_loss_lambda: 0.9

data:
  profiling_window_size_candidates: [500, 250]
  profiling_stride_candidates: [15, 50]
```

```python
def build_loss_from_config(cfg: dict) -> CombinedEMGLoss:
    return CombinedEMGLoss(
        loss_lambda=cfg.get("loss_lambda", 1.0),
        soft_dtw_gamma=cfg.get("soft_dtw_gamma", 0.1),
    )
```

- [ ] **Step 4: Run reduced-sequence comparison**

Run: `python scripts/profile_transformer.py --participants 1 2 3 --max-batches 5 --output outputs/profiling/sequence-reduced.json`
Expected: PASS and produce a report that can be compared on `prepare_s`, `forward_s`, and `peak_allocated`

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml src/losses/soft_dtw.py scripts/train_transformer.py
git commit -m "perf: stage expensive loss usage and expose sequence cost comparisons"
```

### Task 5: Add a Comparison Report and Recommended Current-GPU Preset

**Files:**
- Create: `docs/performance/current-gpu-optimization-report.md`
- Modify: `configs/default.yaml`
- Test: `docs/performance/current-gpu-optimization-report.md`

**Interfaces:**
- Consumes: profiling outputs from Tasks 1-4
- Produces:
  - one markdown comparison report
  - one recommended config preset for current-GPU training

- [ ] **Step 1: Write the failing report existence probe**

```python
from pathlib import Path

path = Path("docs/performance/current-gpu-optimization-report.md")
assert path.exists()
```

- [ ] **Step 2: Run report probe to verify it fails**

Run: `@'from pathlib import Path\nassert Path("docs/performance/current-gpu-optimization-report.md").exists()\n'@ | python -`
Expected: FAIL because the report does not exist yet

- [ ] **Step 3: Write the comparison report**

```md
# Current GPU Optimization Report

| Run | Window | Stride | Batch | AMP | Peak VRAM | Time/Batch | Notes |
|---|---:|---:|---:|---|---:|---:|---|
| Baseline | 500 | 15 | 32 | yes | ... | ... | ... |
| Cached | 500 | 15 | 32 | yes | ... | ... | ... |
| Reduced Sequence | 250 | 50 | 32 | yes | ... | ... | ... |

## Recommended Preset

- window_size: ...
- stride: ...
- batch_size: ...
- gradient_accumulation_steps: ...
- loss_lambda: ...
```

- [ ] **Step 4: Verify report structure**

Run: `Get-Content -Raw 'docs/performance/current-gpu-optimization-report.md'`
Expected: PASS and include a comparison table plus a recommended preset section

- [ ] **Step 5: Commit**

```bash
git add docs/performance/current-gpu-optimization-report.md configs/default.yaml
git commit -m "perf: document current gpu optimization results and preset"
```

## Self-Review

1. **Spec coverage:** This plan covers profiling, memory stabilization controls, caching, sequence-cost reduction, staged loss behavior, and comparison reporting.
2. **Placeholder scan:** No unresolved placeholders remain inside task steps.
3. **Type consistency:** Profiling, cache, and runtime-control names are used consistently across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-performance-optimization.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
