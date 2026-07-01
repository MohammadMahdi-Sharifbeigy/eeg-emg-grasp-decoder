# GAT Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a transformer-to-GAT EEG-to-EMG model where the 5 EMG channels are graph nodes, with both a baseline GAT and a kinematic-guided attention variant.

**Architecture:** Preserve the current preprocessing and transformer temporal encoding path, then project transformer outputs into 5 muscle-node embeddings, apply graph attention independently at each time step, and decode node embeddings back into 5 EMG predictions. Keep the plain GAT baseline and the kinematic-guided variant separate but structurally aligned.

**Tech Stack:** Python, PyTorch, YAML, NumPy

## Global Constraints

- Keep the graph node set fixed at the 5 EMG channels.
- Preserve the current transformer input path and tensor conventions.
- Keep time modeling in the transformer and graph reasoning in the GAT.
- Support both plain GAT and kinematic-guided attention variants.
- Keep the output target as continuous 5-channel EMG regression.

---

### Task 1: Implement Node Projection and Timewise Graph Tensor Shaping

**Files:**
- Create: `src/models/gat_projection.py`
- Modify: `src/models/__init__.py`
- Test: `src/models/gat_projection.py`

**Interfaces:**
- Consumes: transformer output tensor `H_temp: torch.Tensor` of shape `(B, T, D)`
- Produces:
  - `class MuscleNodeProjection(nn.Module)`
  - `forward(x: torch.Tensor) -> torch.Tensor` with output shape `(B, T, 5, F)`

- [ ] **Step 1: Write the failing shape test**

```python
import torch

from src.models.gat_projection import MuscleNodeProjection

proj = MuscleNodeProjection(input_dim=64, n_nodes=5, node_dim=32)
x = torch.randn(2, 100, 64)
y = proj(x)
assert y.shape == (2, 100, 5, 32)
```

- [ ] **Step 2: Run shape test to verify it fails**

Run: `@'import torch\nfrom src.models.gat_projection import MuscleNodeProjection\n'@ | python -`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the projection module**

```python
# src/models/gat_projection.py
from __future__ import annotations

import torch
import torch.nn as nn


class MuscleNodeProjection(nn.Module):
    def __init__(self, input_dim: int, n_nodes: int = 5, node_dim: int = 32):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.proj = nn.Linear(input_dim, n_nodes * node_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        out = self.proj(x)
        return out.view(b, t, self.n_nodes, self.node_dim)
```

- [ ] **Step 4: Run shape test to verify it passes**

Run: `@'import torch\nfrom src.models.gat_projection import MuscleNodeProjection\nproj = MuscleNodeProjection(input_dim=64, n_nodes=5, node_dim=32)\ny = proj(torch.randn(2, 100, 64))\nprint(tuple(y.shape))\n'@ | python -`
Expected: PASS and print `(2, 100, 5, 32)`

- [ ] **Step 5: Commit**

```bash
git add src/models/gat_projection.py src/models/__init__.py
git commit -m "feat: add transformer-to-muscle node projection"
```

### Task 2: Implement Baseline Multi-Head GAT over 5 Muscle Nodes

**Files:**
- Modify: `src/models/kg_gat.py`
- Test: `src/models/kg_gat.py`

**Interfaces:**
- Consumes: node tensor `(B, T, 5, F)`
- Produces:
  - `class MuscleGATLayer(nn.Module)`
  - `class MuscleGATEncoder(nn.Module)`
  - `forward(nodes: torch.Tensor) -> torch.Tensor` with output `(B, T, 5, F_out)`

- [ ] **Step 1: Write the failing baseline GAT shape test**

```python
import torch

from src.models.kg_gat import MuscleGATEncoder

gat = MuscleGATEncoder(node_dim=32, hidden_dim=32, num_heads=4, out_dim=32)
x = torch.randn(2, 100, 5, 32)
y = gat(x)
assert y.shape == (2, 100, 5, 32)
```

- [ ] **Step 2: Run shape test to verify it fails**

Run: `@'import torch\nfrom src.models.kg_gat import MuscleGATEncoder\n'@ | python -`
Expected: FAIL because the current file does not expose the planned class

- [ ] **Step 3: Implement baseline timewise GAT**

```python
class MuscleGATLayer(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.key = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.value = nn.Linear(node_dim, hidden_dim * num_heads, bias=False)
        self.out = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        b, t, n, f = nodes.shape
        flat = nodes.view(b * t, n, f)
        q = self.query(flat)
        k = self.key(flat)
        v = self.value(flat)
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = self.out(out)
        return out.view(b, t, n, -1)
```

- [ ] **Step 4: Run baseline GAT shape test**

Run: `@'import torch\nfrom src.models.kg_gat import MuscleGATEncoder\ngat = MuscleGATEncoder(node_dim=32, hidden_dim=32, num_heads=4, out_dim=32)\ny = gat(torch.randn(2, 100, 5, 32))\nprint(tuple(y.shape))\n'@ | python -`
Expected: PASS and print `(2, 100, 5, 32)`

- [ ] **Step 5: Commit**

```bash
git add src/models/kg_gat.py
git commit -m "feat: implement baseline muscle graph attention encoder"
```

### Task 3: Implement Kinematic-Guided Attention Scoring

**Files:**
- Modify: `src/models/kg_gat.py`
- Test: `src/models/kg_gat.py`

**Interfaces:**
- Consumes:
  - node tensor `(B, T, 5, F)`
  - kinematic tensor `(B, T, K)`
- Produces:
  - `class KinematicGuidedMuscleGATEncoder(nn.Module)`
  - `forward(nodes: torch.Tensor, kin: torch.Tensor) -> torch.Tensor`

- [ ] **Step 1: Write the failing kinematic-guided shape test**

```python
import torch

from src.models.kg_gat import KinematicGuidedMuscleGATEncoder

gat = KinematicGuidedMuscleGATEncoder(node_dim=32, kin_dim=13, hidden_dim=32, num_heads=4, out_dim=32)
nodes = torch.randn(2, 100, 5, 32)
kin = torch.randn(2, 100, 13)
y = gat(nodes, kin)
assert y.shape == (2, 100, 5, 32)
```

- [ ] **Step 2: Run shape test to verify it fails**

Run: `@'import torch\nfrom src.models.kg_gat import KinematicGuidedMuscleGATEncoder\n'@ | python -`
Expected: FAIL because the class does not exist yet

- [ ] **Step 3: Add kinematic-conditioned edge scoring**

```python
class KinematicGuidedMuscleGATEncoder(nn.Module):
    def __init__(self, node_dim: int, kin_dim: int, hidden_dim: int, num_heads: int, out_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.kin_proj = nn.Linear(kin_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim * 3, num_heads)
        self.out = nn.Linear(hidden_dim * num_heads, out_dim)

    def forward(self, nodes: torch.Tensor, kin: torch.Tensor) -> torch.Tensor:
        ...
```

- [ ] **Step 4: Run kinematic-guided shape test**

Run: `@'import torch\nfrom src.models.kg_gat import KinematicGuidedMuscleGATEncoder\ngat = KinematicGuidedMuscleGATEncoder(node_dim=32, kin_dim=13, hidden_dim=32, num_heads=4, out_dim=32)\ny = gat(torch.randn(2, 100, 5, 32), torch.randn(2, 100, 13))\nprint(tuple(y.shape))\n'@ | python -`
Expected: PASS and print `(2, 100, 5, 32)`

- [ ] **Step 5: Commit**

```bash
git add src/models/kg_gat.py
git commit -m "feat: add kinematic-guided muscle graph attention"
```

### Task 4: Assemble the Full Transformer-to-GAT Model

**Files:**
- Modify: `src/models/kg_gt.py`
- Modify: `src/models/__init__.py`
- Test: `src/models/kg_gt.py`

**Interfaces:**
- Consumes:
  - `build_transformer_from_config`
  - `MuscleNodeProjection`
  - `MuscleGATEncoder`
  - `KinematicGuidedMuscleGATEncoder`
- Produces:
  - `class KGGTModel(nn.Module)`
  - `build_kg_gt_from_config(cfg: dict, input_dim: int, kin_dim: int = 13) -> KGGTModel`

- [ ] **Step 1: Write the failing full-model shape test**

```python
import torch

from src.models.kg_gt import KGGTModel

model = KGGTModel(
    transformer_cfg={"d_model": 64, "n_layers": 2, "n_heads": 4, "d_k": 16, "d_v": 16, "ffn_dim": 128, "dropout": 0.1},
    input_dim=8,
    node_dim=32,
    gat_hidden_dim=32,
    gat_heads=4,
    out_channels=5,
    kin_dim=13,
    use_kinematic_guidance=False,
)
eeg = torch.randn(2, 100, 8)
kin = torch.randn(2, 100, 13)
y = model(eeg, kin)
assert y.shape == (2, 100, 5)
```

- [ ] **Step 2: Run full-model shape test to verify it fails**

Run: `@'import torch\nfrom src.models.kg_gt import KGGTModel\n'@ | python -`
Expected: FAIL because the class does not exist yet

- [ ] **Step 3: Implement the composed model**

```python
class KGGTModel(nn.Module):
    def __init__(...):
        self.transformer = build_transformer_from_config(transformer_cfg, input_dim)
        self.node_projection = MuscleNodeProjection(self.transformer.d_model, n_nodes=5, node_dim=node_dim)
        self.gat = MuscleGATEncoder(...) if not use_kinematic_guidance else KinematicGuidedMuscleGATEncoder(...)
        self.decoder = nn.Linear(gat_out_dim, 1)

    def forward(self, eeg: torch.Tensor, kin: torch.Tensor) -> torch.Tensor:
        h_temp = self.transformer(eeg)
        nodes = self.node_projection(h_temp)
        h_gat = self.gat(nodes) if not self.use_kinematic_guidance else self.gat(nodes, kin)
        return self.decoder(h_gat).squeeze(-1)
```

- [ ] **Step 4: Run full-model shape test**

Run: `@'import torch\nfrom src.models.kg_gt import KGGTModel\nmodel = KGGTModel(transformer_cfg={"d_model": 64, "n_layers": 2, "n_heads": 4, "d_k": 16, "d_v": 16, "ffn_dim": 128, "dropout": 0.1}, input_dim=8, node_dim=32, gat_hidden_dim=32, gat_heads=4, out_channels=5, kin_dim=13, use_kinematic_guidance=False)\ny = model(torch.randn(2, 100, 8), torch.randn(2, 100, 13))\nprint(tuple(y.shape))\n'@ | python -`
Expected: PASS and print `(2, 100, 5)`

- [ ] **Step 5: Commit**

```bash
git add src/models/kg_gt.py src/models/__init__.py
git commit -m "feat: assemble transformer to gat eeg emg model"
```

### Task 5: Add Config Hooks and a Minimal Train-Time Integration Path

**Files:**
- Modify: `configs/default.yaml`
- Modify: `scripts/train_transformer.py`
- Test: `scripts/train_transformer.py`

**Interfaces:**
- Consumes: `build_kg_gt_from_config`
- Produces:
  - `model.type: transformer_regressor | kg_gt | kg_gt_kinematic`
  - script path able to instantiate the selected model

- [ ] **Step 1: Write the failing model-selector probe**

```python
import yaml

cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
assert "type" in cfg["model"]
assert cfg["model"]["type"] in {"transformer_regressor", "kg_gt", "kg_gt_kinematic"}
```

- [ ] **Step 2: Run model-selector probe to verify it fails**

Run: `@'import yaml\ncfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))\nassert "type" in cfg["model"]\n'@ | python -`
Expected: FAIL because the selector key does not exist yet

- [ ] **Step 3: Add model selector and integration branch**

```yaml
model:
  type: transformer_regressor
  gat:
    node_dim: 64
    hidden_dim: 64
    heads: 4
    use_kinematic_guidance: false
```

```python
if cfg["model"]["type"] == "transformer_regressor":
    model = build_transformer_regressor(cfg, input_dim=N_CCA).to(DEVICE)
elif cfg["model"]["type"] == "kg_gt":
    model = build_kg_gt_from_config(cfg, input_dim=N_CCA, kin_dim=13).to(DEVICE)
```

- [ ] **Step 4: Run script help and config smoke validation**

Run: `python scripts/train_transformer.py --help`
Expected: PASS with no import errors after the new model-selection branch

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml scripts/train_transformer.py
git commit -m "feat: add model selector for transformer and gat variants"
```

## Self-Review

1. **Spec coverage:** This plan covers transformer-to-node projection, baseline GAT, kinematic-guided attention, full-model assembly, and configuration-level model selection.
2. **Placeholder scan:** No unresolved placeholders or undefined interfaces remain across tasks.
3. **Type consistency:** Tensor shapes are consistent across tasks: `(B, T, D)` -> `(B, T, 5, F)` -> `(B, T, 5)`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-gat-architecture.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
