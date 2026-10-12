"""Memory and time check on RANDOM data. Trains nothing real and saves nothing.

The measured step is the real training step: one optimizer per stream, stepped by `step_streams`.
"""
import time

import torch

from .losses import build_loss, v2_loss
from .model import build_model, count_parameters
from .train import build_optimizers, step_streams


def _measure(cfg, device, batch, steps):
    w = cfg["tokens"]["window"]
    model = build_model(cfg).to(device).train()
    opts = build_optimizers(model, cfg["training"])
    crit = build_loss(cfg)
    use_amp = bool(cfg["training"]["use_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(steps):
        kin, hist = torch.randn(batch, w, 24, device=device), torch.rand(batch, w, 8, device=device)
        eeg, target = torch.randn(batch, w, 32, 4, device=device), torch.rand(batch, w, 5, device=device)
        hold = torch.zeros(batch, w, dtype=torch.int64, device=device)
        hold[:, w // 4: w // 2] = 1
        t0 = time.perf_counter()
        with torch.autocast(device.type, enabled=use_amp):
            loss, _, _ = v2_loss(model(kin, hist, eeg), target, hold, crit)
        scaler.scale(loss).backward()
        step_streams(model, opts, scaler, cfg["training"]["grad_clip_norm"])   # also zeroes the gradients
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
    n_params = count_parameters(model)
    del model, opts
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return peak, min(times[1:] or times), n_params        # skip the first (warm-up) step


def project_runtime(cfg, device, n_windows=1100, steps=4, mem_fraction=0.8) -> dict:
    tc = cfg["training"]
    eff = int(tc["batch_size"]) * int(tc["gradient_accumulation_steps"])
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30 if device.type == "cuda" else 0.0
    candidates = [(eff, 1)] if device.type != "cuda" else [(eff, 1), (max(eff // 2, 1), 2)]
    last = None
    for batch, accum in candidates:
        try:
            peak, sec, n_params = _measure(cfg, device, batch, steps)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            last = f"batch {batch}: out of memory"
            continue
        if device.type == "cuda" and peak > mem_fraction * total:
            last = f"batch {batch}: {peak:.2f} GB of {total:.2f} GB"
            continue
        steps_per_epoch = n_windows / batch
        return {"batch_size": batch, "gradient_accumulation_steps": accum, "peak_gb": round(peak, 3),
                "total_gb": round(total, 2), "sec_per_step": sec, "parameters": n_params,
                "projected_minutes": sec * steps_per_epoch * tc["max_epochs"] / 60}
    raise RuntimeError(f"model does not fit in {mem_fraction:.0%} of the GPU ({last}); free the GPU or reduce the window")
