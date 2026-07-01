#!/usr/bin/env python
"""
Script version of notebooks/02_train_transformer.ipynb.

Pipeline:
  1. Preprocess every subject x series HS file (EEG + EMG + kinematics).
  2. Fit CCA for EEG -> canonical components aligned with k_t.
  3. Build transformer encoder + linear EMG readout.
  4. Train with Adam, scheduler, early stopping, and checkpointing.
  5. Evaluate on the held-out test split.
"""

from __future__ import annotations

import argparse
import io
import signal
import sys
import time
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    from rich import box
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit(
        "rich is not installed.\n"
        "Activate the project venv and run:\n"
        "    pip install rich\n"
    )

_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.losses import build_loss_from_config
from src.models import build_transformer_regressor
from src.pipelines import (
    build_dataset as build_shared_dataset,
    fit_cca_and_emg_stats,
    prepare_batch_factory,
    unique_series_arrays,
)
from src.training import TrainConfig, evaluate, save_checkpoint, train_model
from src.utils import get_device, set_seed

EMG_NAMES = [
    "Ant. Deltoid",
    "Ext. Carpi Rad.",
    "Flex. Digitorum",
    "Ext. Dig. Comm.",
    "1st Dors. Inteross.",
]

BANNER = r"""
 ██╗  ██╗ ██████╗      ██████╗████████╗
 ██║ ██╔╝██╔════╝     ██╔════╝╚══██╔══╝
 █████╔╝ ██║  ███╗    ██║  ███╗  ██║
 ██╔═██╗ ██║   ██║    ██║   ██║  ██║
 ██║  ██╗╚██████╔╝    ╚██████╔╝  ██║
 ╚═╝  ╚═╝ ╚═════╝      ╚═════╝   ╚═╝
"""

CONSOLE = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_transformer",
        description="Train the EEG->EMG transformer encoder with a rich terminal UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), metavar="PATH")
    parser.add_argument("--participants", nargs="+", type=int, default=None, metavar="N")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N")
    parser.add_argument("--max-epochs", type=int, default=None, metavar="N")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_sparkline(values: List[float], width: int = 30) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if len(values) < 2:
        return "─" * width
    tail = values[-width:]
    lo, hi = min(tail), max(tail)
    span = hi - lo or 1e-9
    return "".join(blocks[int((v - lo) / span * (len(blocks) - 1))] for v in tail).rjust(width, "─")


def gpu_panel(device: torch.device) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="green bold", justify="right")
    table.add_column()
    table.add_row("Device", str(device))
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        cap = torch.cuda.get_device_capability(idx)
        total_mb = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 2
        alloc_mb = torch.cuda.memory_allocated(idx) / 1024 ** 2
        reserved_mb = torch.cuda.memory_reserved(idx) / 1024 ** 2
        table.add_row("GPU name", torch.cuda.get_device_name(idx))
        table.add_row("CUDA capability", f"sm_{cap[0]}{cap[1]}")
        table.add_row("VRAM total", f"{total_mb:,.0f} MB")
        table.add_row("VRAM allocated", f"{alloc_mb:,.1f} MB")
        table.add_row("VRAM reserved", f"{reserved_mb:,.1f} MB")
        table.add_row("CUDA version", torch.version.cuda or "n/a")
    else:
        table.add_row("Note", "CUDA not available - training on CPU")
    table.add_row("PyTorch", torch.__version__)
    return Panel(table, title="[bold cyan]GPU / Device Status[/bold cyan]", border_style="cyan", padding=(1, 2))


def config_panel(cfg: dict) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(style="white")
    data = cfg.get("data", {})
    model = cfg.get("model", {}).get("transformer", {})
    train = cfg.get("training", {})
    cca = cfg.get("preprocessing", {}).get("cca", {})
    rows = [
        ("participants", str(data.get("participants", "all"))),
        ("window / stride", f"{data.get('window_size')} / {data.get('stride')} samples"),
        ("CCA components", str(cca.get("n_components"))),
        ("Transformer", f"L={model.get('n_layers')}  H={model.get('n_heads')}  d={model.get('d_model')}"),
        ("batch size", str(train.get("batch_size"))),
        ("max epochs", str(train.get("max_epochs"))),
        ("lr", str(train.get("lr"))),
        ("early stop", f"{train.get('early_stop_patience')} epochs"),
        ("grad clip", str(train.get("grad_clip_norm"))),
    ]
    for key, value in rows:
        table.add_row(key, value)
    return Panel(table, title="[bold magenta]Config[/bold magenta]", border_style="magenta", padding=(1, 2))


def model_panel(model: nn.Module) -> Panel:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    syntax = Syntax(repr(model), "python", theme="dracula", line_numbers=False)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="green bold", justify="right")
    table.add_column()
    table.add_row("Trainable params", f"{n_params:,}")
    return Panel(Columns([table, syntax]), title="[bold green]Model Architecture[/bold green]", border_style="green", padding=(1, 2))


def loss_history_table(history: dict, best_val: float, early_stop_patience: int, epochs_no_improve: int) -> Table:
    train_hist = history.get("train", [])
    val_hist = history.get("val", [])
    n_rows = min(len(train_hist), len(val_hist), 12)
    table = Table(title="[bold]Training History[/bold]", box=box.SIMPLE_HEAVY, header_style="bold cyan", show_lines=False, padding=(0, 1))
    table.add_column("Epoch", justify="right", style="dim", width=6)
    table.add_column("Train ↓", justify="right", style="white", width=10)
    table.add_column("Val ↓", justify="right", style="white", width=10)
    table.add_column("Δ Val", justify="right", style="dim", width=9)
    table.add_column("Best Val", justify="right", style="green", width=10)
    start_idx = len(train_hist) - n_rows
    for i in range(n_rows):
        ep = start_idx + i + 1
        tr = train_hist[start_idx + i]
        vl = val_hist[start_idx + i]
        if i > 0:
            prev = val_hist[start_idx + i - 1]
            delta_val = vl - prev
            delta = f"[red]+{delta_val:.5f}[/red]" if delta_val > 0 else f"[green]{delta_val:.5f}[/green]"
        else:
            delta = "—"
        best_marker = "[bold green]★[/bold green]" if abs(vl - best_val) < 1e-8 else ""
        table.add_row(str(ep), f"{tr:.6f}", f"{vl:.6f}", delta, f"{best_val:.6f} {best_marker}")
    if len(train_hist) >= 2:
        table.add_row("[dim]train[/dim]", Text(make_sparkline(train_hist, 28), style="yellow"), "", "", "")
        table.add_row("[dim]val[/dim]", Text(make_sparkline(val_hist, 28), style="cyan"), "", "", "")
    remaining = early_stop_patience - epochs_no_improve
    es_color = "green" if remaining > 10 else ("yellow" if remaining > 3 else "red")
    table.add_row("[dim]early-stop[/dim]", f"[{es_color}]{remaining} left[/{es_color}]", "", "", "")
    return table


def build_dataset(split: str, cfg: dict, participants: List[int], progress: Progress, task_id):
    progress.update(task_id, description=f"[cyan]Loading {split:5s}...[/cyan]")
    t0 = time.time()
    ds = build_shared_dataset(split=split, cfg=cfg, participants=participants, root_dir=ROOT)
    elapsed = time.time() - t0
    progress.update(task_id, description=f"[green]{split:5s} ✓[/green]  {len(ds):,} windows  ({elapsed:.1f}s)", completed=1)
    return ds


def main() -> None:
    args = parse_args()

    CONSOLE.print()
    CONSOLE.print(
        Panel(
            Align.center(Text(BANNER, style="bold cyan", justify="center")),
            title="[bold white]EEG -> EMG  ·  Transformer Training[/bold white]",
            subtitle="[dim]KG-GT Method 1  ·  Phase 1[/dim]",
            border_style="bright_blue",
            padding=(0, 4),
        )
    )
    CONSOLE.print()

    set_seed(args.seed)
    device = get_device()
    CONSOLE.print(gpu_panel(device))
    CONSOLE.print()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    participants = args.participants or cfg["data"]["participants"]
    cfg["data"]["participants"] = participants
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs

    CONSOLE.print(config_panel(cfg))
    CONSOLE.print()

    n_cca = cfg["preprocessing"]["cca"]["n_components"]

    CONSOLE.print(Rule("[bold yellow]§1  Loading Datasets[/bold yellow]"))
    CONSOLE.print()

    ds_progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description:<40}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=CONSOLE,
        transient=False,
    )

    with ds_progress:
        t_train = ds_progress.add_task("[cyan]train...[/cyan]", total=1)
        t_val = ds_progress.add_task("[cyan]val...[/cyan]", total=1)
        t_test = ds_progress.add_task("[cyan]test...[/cyan]", total=1)
        train_ds = build_dataset("train", cfg, participants, ds_progress, t_train)
        val_ds = build_dataset("val", cfg, participants, ds_progress, t_val)
        test_ds = build_dataset("test", cfg, participants, ds_progress, t_test)

    summary = Table.grid(padding=(0, 3))
    summary.add_column(style="bold yellow")
    summary.add_column(style="white")
    summary.add_row("Train windows", f"{len(train_ds):,}")
    summary.add_row("Val   windows", f"{len(val_ds):,}")
    summary.add_row("Test  windows", f"{len(test_ds):,}")
    CONSOLE.print(Panel(summary, title="Dataset Summary", border_style="yellow"))
    CONSOLE.print()

    CONSOLE.print(Rule("[bold blue]§2  CCA Fit + EMG Statistics[/bold blue]"))
    CONSOLE.print()

    with CONSOLE.status("[cyan]Extracting unique train series...[/cyan]"):
        tr_eeg, _, _ = unique_series_arrays(train_ds)
    CONSOLE.print(f"  [dim]Unique train series:[/dim] {len(tr_eeg)}")

    with CONSOLE.status("[cyan]Fitting CCA...[/cyan]"):
        cca_gpu, emg_mean, emg_std = fit_cca_and_emg_stats(
            train_ds,
            cfg["preprocessing"]["cca"],
            device,
        )

    CONSOLE.print(
        f"  [green]✓[/green] CCA fitted  ->  n_components = {n_cca}"
        f"  [dim](device: {cca_gpu.device})[/dim]"
    )

    stats = Table.grid(padding=(0, 3))
    stats.add_column(style="bold blue", justify="right")
    stats.add_column()
    stats.add_row("EMG mean", " ".join(f"{v:.4f}" for v in emg_mean.cpu().tolist()))
    stats.add_row("EMG std", " ".join(f"{v:.4f}" for v in emg_std.cpu().tolist()))
    CONSOLE.print(Panel(stats, border_style="blue"))
    CONSOLE.print()

    prepare_batch = prepare_batch_factory(
        cca_projector=cca_gpu,
        emg_mean=emg_mean,
        emg_std=emg_std,
        device=device,
        n_cca=n_cca,
    )

    CONSOLE.print(Rule("[bold green]§3  Model[/bold green]"))
    CONSOLE.print()

    model = build_transformer_regressor(cfg, input_dim=n_cca).to(device)
    CONSOLE.print(model_panel(model))
    CONSOLE.print()

    batch_size = cfg["training"]["batch_size"]
    loader_kw = dict(pin_memory=True, num_workers=2, persistent_workers=True) if device.type == "cuda" else {}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)

    loss_fn = build_loss_from_config(cfg["training"]).to(device)
    checkpoint_dir = str(ROOT / "outputs" / "checkpoints")
    train_config = TrainConfig.from_config(cfg["training"], max_epochs=args.max_epochs)
    train_config.checkpoint_dir = checkpoint_dir
    train_config.checkpoint_every = 1

    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _sigint_handler)

    CONSOLE.print(Rule("[bold red]§4  Training[/bold red]"))
    CONSOLE.print()
    resume_str = "[green]resuming from last.pt[/green]" if args.resume else "[yellow]fresh run[/yellow]"
    CONSOLE.print(f"  Mode: {resume_str}  ·  batch={batch_size}  ·  epochs={train_config.max_epochs}  ·  device={device}")
    CONSOLE.print()

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        prepare_batch=prepare_batch,
        loss_fn=loss_fn,
        device=device,
        cfg=train_config,
        resume=args.resume,
    )

    CONSOLE.print()
    CONSOLE.print(loss_history_table(result.history, result.best_val, train_config.early_stop_patience, 0))

    inference_ckpt = str(ROOT / "outputs" / "transformer_best.pt")
    save_checkpoint(inference_ckpt, model, train_config, result.best_val)

    CONSOLE.print()
    CONSOLE.print(
        Panel(
            f"[green]Best val loss :[/green]  [bold]{result.best_val:.6f}[/bold]\n"
            f"[green]Resumable ckpts:[/green] {checkpoint_dir}\n"
            f"[green]Inference ckpt :[/green] {inference_ckpt}",
            title="[bold green]Training Complete[/bold green]",
            border_style="green",
        )
    )
    CONSOLE.print()

    CONSOLE.print(Rule("[bold yellow]§5  Evaluation (test split)[/bold yellow]"))
    CONSOLE.print()

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kw)
    with CONSOLE.status("[cyan]Running evaluation...[/cyan]"):
        metrics = evaluate(
            model=model,
            loader=test_loader,
            prepare_batch=prepare_batch,
            device=device,
            channel_names=EMG_NAMES,
            n_channels=5,
        )

    eval_table = Table(
        title="Test-Set Metrics (z-scored EMG)",
        box=box.SIMPLE_HEAVY,
        header_style="bold yellow",
        show_lines=False,
    )
    eval_table.add_column("Channel", style="white")
    eval_table.add_column("RMSE", style="cyan", justify="right")
    eval_table.add_column("MAE", style="blue", justify="right")
    eval_table.add_column("Pearson r", style="green", justify="right")

    for ch_name, rmse, mae, r in zip(
        metrics.channel_names,
        metrics.rmse,
        metrics.mae,
        metrics.pearson,
    ):
        r_color = "green" if r > 0.7 else ("yellow" if r > 0.4 else "red")
        eval_table.add_row(ch_name, f"{rmse:.4f}", f"{mae:.4f}", f"[{r_color}]{r:.4f}[/{r_color}]")

    CONSOLE.print(eval_table)
    CONSOLE.print()
    CONSOLE.print(
        Panel(
            f"Mean RMSE: [bold cyan]{metrics.rmse.mean():.4f}[/bold cyan]   "
            f"Mean MAE: [bold blue]{metrics.mae.mean():.4f}[/bold blue]   "
            f"Mean Pearson r: [bold green]{metrics.pearson.mean():.4f}[/bold green]",
            title="[bold]Overall Metrics[/bold]",
            border_style="yellow",
        )
    )
    CONSOLE.print()
    CONSOLE.print(
        "[bold green]✓ Done.[/bold green]  "
        "Run with [dim]--no-resume[/dim] to start fresh or "
        "[dim]--participants 1 --max-epochs 5[/dim] for a quick smoke test."
    )
    CONSOLE.print()


if __name__ == "__main__":
    main()
