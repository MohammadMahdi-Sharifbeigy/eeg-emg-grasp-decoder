#!/usr/bin/env python
"""Bounded profiling script for the current EEG->EMG training path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from torch.utils.data import DataLoader

from src.models import build_transformer_regressor
from src.pipelines import build_dataset, fit_cca_and_emg_stats, prepare_batch_factory
from src.training import capture_batch_profile, write_profile_report
from src.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded profile over the transformer training path.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--participants", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "profiling" / "baseline.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    participants = args.participants or cfg["data"]["participants"]
    cfg["data"]["participants"] = participants
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    train_ds = build_dataset("train", cfg, participants, ROOT)
    n_cca = cfg["preprocessing"]["cca"]["n_components"]
    cca_gpu, emg_mean, emg_std = fit_cca_and_emg_stats(train_ds, cfg["preprocessing"]["cca"], device)
    prepare_batch = prepare_batch_factory(cca_gpu, emg_mean, emg_std, device, n_cca)
    model = build_transformer_regressor(cfg, input_dim=n_cca).to(device)

    runtime_cfg = cfg.get("runtime", {})
    loader_kwargs = {}
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = runtime_cfg.get("pin_memory", True)
        loader_kwargs["num_workers"] = runtime_cfg.get("num_workers", 2)
        num_workers = loader_kwargs["num_workers"]
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = runtime_cfg.get("persistent_workers", True)
            loader_kwargs["prefetch_factor"] = runtime_cfg.get("prefetch_factor", 2)

    loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    report = capture_batch_profile(model, loader, prepare_batch, device, max_batches=args.max_batches)
    write_profile_report(report, Path(args.output))
    print(f"Profile written to {args.output}")


if __name__ == "__main__":
    main()
