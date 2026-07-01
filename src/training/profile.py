"""Lightweight profiling helpers for the current transformer training path."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.no_grad()
def capture_batch_profile(
    model,
    loader,
    prepare_batch,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Profile data preparation and forward cost over a bounded number of batches."""
    timings: list[dict] = []
    peak_allocated = 0
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx, (eeg, _kin, emg) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        t0 = time.perf_counter()
        x, y = prepare_batch(eeg, emg)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        t2 = time.perf_counter()
        timings.append(
            {
                "batch_idx": batch_idx,
                "prepare_s": t1 - t0,
                "forward_s": t2 - t1,
                "total_s": t2 - t0,
                "loss": float(loss.item()),
                "batch_size": int(eeg.size(0)),
            }
        )

    total_samples = sum(item["batch_size"] for item in timings)
    total_time = sum(item["total_s"] for item in timings)
    return {
        "device": str(device),
        "max_batches": max_batches,
        "peak_allocated_bytes": int(peak_allocated),
        "avg_prepare_s": (sum(item["prepare_s"] for item in timings) / len(timings)) if timings else 0.0,
        "avg_forward_s": (sum(item["forward_s"] for item in timings) / len(timings)) if timings else 0.0,
        "avg_total_s": (sum(item["total_s"] for item in timings) / len(timings)) if timings else 0.0,
        "samples_per_second": (total_samples / total_time) if total_time > 0 else 0.0,
        "timings": timings,
    }


def write_profile_report(report: dict, output_path: Path) -> None:
    """Persist a profile report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
