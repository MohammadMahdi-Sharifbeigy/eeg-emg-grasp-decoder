#!/usr/bin/env python
"""
scripts/train_transformer_json.py
==================================
JSON-emitting backend for the termcn TUI.

Runs the full training pipeline but instead of printing human-readable text,
emits newline-delimited JSON messages to stdout that the Node.js termcn TUI
(scripts/tui/) consumes and renders.

This script is launched as a child process by the Node.js TUI, or can be
piped manually:

    python scripts\\train_transformer_json.py | node scripts\\tui\\src\\index.tsx

JSON message schema
-------------------
Every line is one JSON object with at least a ``"type"`` field:

  {"type": "phase",   "phase": "setup"}
  {"type": "gpu",     "device": "cuda", "name": "GTX 1660 Ti",
                      "vram_total_mb": 6144, "vram_alloc_mb": 25.2,
                      "cuda_version": "12.6", "torch_version": "2.8.0",
                      "amp": true}
  {"type": "config",  "participants": [1,2,3], "batch_size": 32,
                      "max_epochs": 5, "lr": 0.001, "n_cca": 13,
                      "n_layers": 4, "n_heads": 8, "d_model": 256}
  {"type": "dataset", "split": "train", "windows": 12345, "elapsed_s": 23.4}
  {"type": "cca",     "n_components": 13,
                      "emg_mean": [...], "emg_std": [...]}
  {"type": "model",   "arch": "TransformerRegressor(...)", "n_params": 3159813}
  {"type": "batch",   "epoch": 1, "batch": 100, "total_batches": 1278,
                      "loss": 0.0152}
  {"type": "epoch",   "epoch": 1, "max_epochs": 500,
                      "train_loss": 0.0123, "val_loss": 0.0145,
                      "lr": 0.001, "elapsed_s": 45.2,
                      "best_val": 0.0145, "epochs_no_improve": 0}
  {"type": "eval",    "channels": ["Ant. Deltoid", ...],
                      "rmse": [...], "mae": [...], "r": [...]}
  {"type": "done",    "best_val": 0.0098, "ckpt_dir": "...",
                      "inference_ckpt": "..."}
  {"type": "error",   "message": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List

# ── stdout must be raw / unbuffered JSON lines ────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── project imports ───────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import WAYEEGDataset
from src.losses import build_loss_from_config
from src.models.transformer import build_transformer_from_config
from src.preprocessing.cca import make_cca_from_config
from src.preprocessing.eeg import preprocess_eeg_from_config
from src.preprocessing.emg import preprocess_emg_from_config
from src.training import (
    TrainConfig,
    evaluate,
    save_checkpoint,
    train_model,
)
from src.utils import get_device, set_seed

EMG_NAMES = [
    "Ant. Deltoid",
    "Ext. Carpi Rad.",
    "Flex. Digitorum",
    "Ext. Dig. Comm.",
    "1st Dors. Inteross.",
]


# ── JSON emit ─────────────────────────────────────────────────────────────────

def emit(obj: dict[str, Any]) -> None:
    """Write one JSON line to stdout and flush immediately."""
    print(json.dumps(obj), flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_transformer_json",
        description=(
            "JSON-emitting backend for the termcn TUI.\n"
            "Designed to be piped into scripts/tui/ (the Node.js termcn app)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--participants", nargs="+", type=int, default=None)
    p.add_argument("--batch-size",   type=int,  default=None)
    p.add_argument("--max-epochs",   type=int,  default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--seed",         type=int,  default=42)
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume",    dest="resume", action="store_true",  default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


# ── Dataset helpers ───────────────────────────────────────────────────────────

def make_preprocess_fn(cfg: dict):
    eeg_cfg = cfg["preprocessing"]["eeg"]
    emg_cfg = cfg["preprocessing"]["emg"]
    def fn(s):
        s = dict(s)
        s["eeg"] = preprocess_eeg_from_config(s["eeg"], float(s["fs_eeg"]), eeg_cfg)
        s["emg"] = preprocess_emg_from_config(s["emg"], float(s["fs_emg"]), emg_cfg)
        return s
    return fn


def unique_series_arrays(ds):
    seen, eegs, kins, emgs = set(), [], [], []
    for eeg_all, kin_all, emg_all, _ in ds._windows:
        key = id(eeg_all)
        if key in seen:
            continue
        seen.add(key)
        eegs.append(eeg_all); kins.append(kin_all); emgs.append(emg_all)
    return eegs, kins, emgs


# ── Model ─────────────────────────────────────────────────────────────────────

class TransformerRegressor(nn.Module):
    def __init__(self, cfg: dict, input_dim: int, out_channels: int = 5):
        super().__init__()
        self.encoder = build_transformer_from_config(cfg["model"]["transformer"], input_dim)
        self.head = nn.Linear(self.encoder.d_model, out_channels)

    def forward(self, x):
        return self.head(self.encoder(x))


# ── Batch callback wrapper ────────────────────────────────────────────────────

class BatchEmitter:
    """Wraps prepare_batch and emits a JSON 'batch' message every N batches."""
    def __init__(self, prepare_fn, emit_every: int = 25):
        self._fn = prepare_fn
        self._every = emit_every
        self._counter = 0
        self.current_epoch = 1
        self.total_batches = 1

    def __call__(self, eeg, emg):
        return self._fn(eeg, emg)

    def on_batch(self, epoch: int, batch: int, total: int, loss: float) -> None:
        self.current_epoch = epoch
        self.total_batches = total
        self._counter += 1
        if self._counter % self._every == 0:
            emit({
                "type": "batch",
                "epoch": epoch,
                "batch": batch,
                "total_batches": total,
                "loss": round(loss, 6),
            })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        emit({"type": "phase", "phase": "setup"})

        set_seed(args.seed)
        DEVICE = get_device()

        # GPU info
        gpu_msg: dict[str, Any] = {
            "type": "gpu",
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "amp": False,
        }
        if DEVICE.type == "cuda":
            idx = DEVICE.index or 0
            cap = torch.cuda.get_device_capability(idx)
            props = torch.cuda.get_device_properties(idx)
            gpu_msg.update({
                "name": torch.cuda.get_device_name(idx),
                "cuda_version": torch.version.cuda or "",
                "capability": f"sm_{cap[0]}{cap[1]}",
                "vram_total_mb": round(props.total_memory / 1024 ** 2),
                "vram_alloc_mb": round(torch.cuda.memory_allocated(idx) / 1024 ** 2, 1),
                "amp": cap[0] >= 7,
            })
        emit(gpu_msg)

        # Config
        cfg = yaml.safe_load(open(args.config))
        participants = args.participants or cfg["data"]["participants"]
        cfg["data"]["participants"] = participants
        if args.batch_size is not None:
            cfg["training"]["batch_size"] = args.batch_size
        if args.max_epochs is not None:
            cfg["training"]["max_epochs"] = args.max_epochs
        if args.lr is not None:
            cfg["training"]["lr"] = args.lr

        N_CCA = cfg["preprocessing"]["cca"]["n_components"]
        tr = cfg["training"]
        mo = cfg["model"]["transformer"]
        emit({
            "type": "config",
            "participants": participants,
            "batch_size": tr["batch_size"],
            "max_epochs": tr["max_epochs"],
            "lr": tr["lr"],
            "n_cca": N_CCA,
            "n_layers": mo["n_layers"],
            "n_heads": mo["n_heads"],
            "d_model": mo["d_model"],
            "resume": args.resume,
        })

        # ── Datasets ──────────────────────────────────────────────────────────
        emit({"type": "phase", "phase": "datasets"})
        DATA_DIR  = ROOT / cfg["data"]["raw_dir"]
        CACHE_DIR = ROOT / cfg["data"]["cache_dir"]
        W, STRIDE = cfg["data"]["window_size"], cfg["data"]["stride"]
        preprocess_fn = make_preprocess_fn(cfg)

        datasets = {}
        for split in ("train", "val", "test"):
            t0 = time.time()
            ds = WAYEEGDataset(
                data_dir=DATA_DIR, participants=participants, split=split,
                window_size=W, stride=STRIDE,
                preprocess_fn=preprocess_fn, cache_dir=CACHE_DIR,
            )
            emit({"type": "dataset", "split": split,
                  "windows": len(ds), "elapsed_s": round(time.time() - t0, 1)})
            datasets[split] = ds

        train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

        # ── CCA + EMG stats ───────────────────────────────────────────────────
        emit({"type": "phase", "phase": "cca"})
        tr_eeg, tr_kin, tr_emg = unique_series_arrays(train_ds)
        cca = make_cca_from_config(cfg["preprocessing"]["cca"])
        cca.fit(tr_eeg, tr_kin)
        cca_gpu = cca.torch_projector(DEVICE)
        emg_concat = np.concatenate(tr_emg, axis=0)
        EMG_MEAN = torch.tensor(emg_concat.mean(0), dtype=torch.float32, device=DEVICE)
        EMG_STD  = torch.tensor(emg_concat.std(0) + 1e-8, dtype=torch.float32, device=DEVICE)
        emit({
            "type": "cca",
            "n_components": cca.n_components,
            "n_series": len(tr_eeg),
            "emg_mean": [round(v, 4) for v in EMG_MEAN.cpu().tolist()],
            "emg_std":  [round(v, 4) for v in EMG_STD.cpu().tolist()],
        })

        def prepare_batch(eeg, emg):
            B, Wn, _ = eeg.shape
            flat = eeg.reshape(-1, 32).to(DEVICE)
            proj = cca_gpu.transform(flat).reshape(B, Wn, N_CCA)
            return proj, (emg.to(DEVICE) - EMG_MEAN) / EMG_STD

        # ── Model ─────────────────────────────────────────────────────────────
        emit({"type": "phase", "phase": "model"})
        model = TransformerRegressor(cfg, input_dim=N_CCA).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emit({"type": "model", "arch": repr(model), "n_params": n_params})

        # ── DataLoaders ───────────────────────────────────────────────────────
        BATCH = cfg["training"]["batch_size"]
        loader_kw = (
            dict(pin_memory=True, num_workers=2, persistent_workers=True)
            if DEVICE.type == "cuda" else {}
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  drop_last=True, **loader_kw)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, **loader_kw)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, **loader_kw)

        loss_fn      = build_loss_from_config(cfg["training"]).to(DEVICE)
        CKPT_DIR     = str(ROOT / "outputs" / "checkpoints")
        train_config = TrainConfig.from_config(cfg["training"], max_epochs=args.max_epochs)
        train_config.checkpoint_dir   = CKPT_DIR
        train_config.checkpoint_every = 1

        # ── Training ──────────────────────────────────────────────────────────
        emit({"type": "phase", "phase": "training"})
        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            prepare_batch=prepare_batch,
            loss_fn=loss_fn,
            device=DEVICE,
            cfg=train_config,
            resume=args.resume,
        )

        # Emit full history once training is done
        hist = result.history
        for i, (tr_l, vl_l) in enumerate(
            zip(hist.get("train", []), hist.get("val", []))
        ):
            emit({
                "type": "epoch",
                "epoch": i + 1,
                "max_epochs": train_config.max_epochs,
                "train_loss": round(tr_l, 6),
                "val_loss":   round(vl_l, 6),
                "best_val":   round(result.best_val, 6),
                "epochs_no_improve": 0,  # history replay — details lost post-hoc
            })

        # Save inference checkpoint
        inference_ckpt = str(ROOT / "outputs" / "transformer_best.pt")
        save_checkpoint(inference_ckpt, model, train_config, result.best_val)

        # ── Evaluation ────────────────────────────────────────────────────────
        emit({"type": "phase", "phase": "eval"})
        metrics = evaluate(
            model=model,
            loader=test_loader,
            prepare_batch=prepare_batch,
            device=DEVICE,
            channel_names=EMG_NAMES,
            n_channels=5,
        )
        emit({
            "type": "eval",
            "channels": EMG_NAMES,
            "rmse": [round(v, 4) for v in metrics.rmse_per_channel],
            "mae":  [round(v, 4) for v in metrics.mae_per_channel],
            "r":    [round(v, 4) for v in metrics.r_per_channel],
            "mean_rmse": round(metrics.mean_rmse, 4),
            "mean_mae":  round(metrics.mean_mae,  4),
            "mean_r":    round(metrics.mean_r,    4),
        })

        emit({
            "type": "done",
            "best_val": round(result.best_val, 6),
            "ckpt_dir": CKPT_DIR,
            "inference_ckpt": inference_ckpt,
        })

    except KeyboardInterrupt:
        emit({"type": "error", "message": "Interrupted by user (Ctrl+C)"})
        sys.exit(0)
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
