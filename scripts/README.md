# scripts/

Script versions of the project notebooks — runnable from the terminal with a
rich, interactive UI (progress bars, live tables, sparklines, GPU panel).

---

## `train_transformer.py`

Script equivalent of `notebooks/02_train_transformer.ipynb`.

Implements the full KG-GT Method 1 §3.2–3.3 pipeline:

| Step | Description |
|------|-------------|
| 1 | Preprocess EEG + EMG per subject × series (cached to `.npz`) |
| 2 | Fit CCA (EEG 32 → 13 canonical components aligned with `k_t`) |
| 3 | Build Transformer encoder (L=4, H=8, d=256) + linear readout |
| 4 | Train: Adam, MSE loss, AMP, grad-clip, LR scheduler, early stop |
| 5 | Evaluate on held-out test split (RMSE / MAE / Pearson r per channel) |

### Prerequisites

```powershell
# Activate the project virtual environment
.\env\Scripts\Activate.ps1

# rich must be installed (it is almost always a transitive dep already)
pip install rich
```

### Usage

```powershell
# Show all options
python scripts\train_transformer.py --help

# Full training run (all 12 subjects, 500 epochs) — will resume last.pt if found
python scripts\train_transformer.py

# Smoke test: 1 subject, 5 epochs, small batch — runs in minutes
python scripts\train_transformer.py --participants 1 --max-epochs 5 --batch-size 32

# Force a completely fresh run (ignore any existing checkpoint)
python scripts\train_transformer.py --no-resume

# Custom participant subset with a smaller batch to avoid CUDA OOM
python scripts\train_transformer.py --participants 1 2 3 --batch-size 32

# Custom config file
python scripts\train_transformer.py --config configs/default.yaml
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `configs/default.yaml` | YAML config to load |
| `--participants N [N …]` | all in config | Participant IDs to include |
| `--batch-size N` | from config (128) | Override batch size |
| `--max-epochs N` | from config (500) | Override max epochs |
| `--resume` | ✓ | Resume from `outputs/checkpoints/last.pt` |
| `--no-resume` | — | Force fresh training run |
| `--seed N` | 42 | Random seed |

### Known CUDA Issues & Fixes

| Error | Fix |
|-------|-----|
| `MemoryError` on dataset load | Use `--participants 1 2 3` to reduce subjects |
| `OutOfMemoryError` during training | Use `--batch-size 32` (or 16) |
| VRAM fragmentation | Set env var `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

```powershell
# Recommended flags for GTX 1660 Ti (6 GB VRAM):
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python scripts\train_transformer.py --batch-size 32
```

### Outputs

```
outputs/
├── checkpoints/
│   ├── last.pt          ← saved every epoch, used for resume
│   └── best.pt          ← best val loss checkpoint
└── transformer_best.pt  ← slim inference-only checkpoint
```

### Terminal UI Components

The UI is built with **`rich`** — the Python equivalent of termcn/Ink:

| rich component | equivalent termcn component |
|---|---|
| `Panel` + `Table.grid` | Panel + KeyValue |
| `Progress` + `SpinnerColumn` | Spinner + Multi Progress |
| `Table` | Table |
| `Syntax` | Code |
| `sparkline()` helper | Sparkline |
| `Rule` | Divider |
| Colored `Text` | Badge / Status Message |

---

## TUI (termcn / Node.js)

In addition to the `rich` Python script, there is a dedicated React/Ink `termcn` frontend in the `scripts/tui/` directory. 
This frontend provides a premium interactive configuration menu and completely eliminates rendering artifacts.

### Architecture

1. **`scripts/tui/src/App.tsx`**: The main React/Ink app built with `termcn` components. It handles user input via an interactive menu.
2. **`scripts/train_transformer_json.py`**: A special backend script that runs the exact same PyTorch training loop but emits raw JSON lines to `stdout` instead of rendering `rich` UI elements.
3. The Node.js app spawns the Python backend, pipes the JSON stream, and updates the React state to drive the TUI.

### Installation

The TUI requires Node.js and uses `npm` for dependency management.

```powershell
# Navigate to the TUI directory
cd scripts/tui

# Install Node dependencies (Ink, React, termcn components, etc.)
npm install
```

### Usage

You can launch the interactive TUI directly from the `scripts/tui/` directory using the provided `npm` script. The TUI will guide you through selecting the participants, epochs, batch size, and learning rate before beginning the training.

```powershell
# Make sure you are in the scripts/tui directory
npm run tui
```
