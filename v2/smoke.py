"""Memory and time check on RANDOM data. Trains nothing real and saves nothing.

The measured step is the real training step: one optimizer per stream, stepped by `step_streams`.
Mixed precision is slower than float32 on some GPUs, so on the GPU both are measured and the faster
setting that fits in memory is returned.
"""
import time

import torch

from .losses import build_loss, v2_loss
from .model import build_model, count_parameters
from .train import build_optimizers, step_streams


def _measure(cfg, device, batch, steps, use_amp):
    w = cfg["tokens"]["window"]
    model = build_model(cfg).to(device).train()
    opts = build_optimizers(model, cfg["training"])
    crit = build_loss(cfg)
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


def choose_setting(candidates, mem_fraction) -> dict:
    """The measured setting with the smallest projected time among those that fit; RuntimeError if none fits."""
    fitting = [c for c in candidates if c["fits"]]
    if not fitting:
        tried = "; ".join(
            f"batch {c['batch_size']}, mixed precision {'on' if c['use_amp'] else 'off'}: "
            + ("out of memory" if c["peak_gb"] is None else f"{c['peak_gb']:.2f} GB") for c in candidates)
        raise RuntimeError(f"model does not fit in {mem_fraction:.0%} of the GPU ({tried}); "
                           f"free the GPU or reduce the window")
    return min(fitting, key=lambda c: c["projected_minutes"])


def project_runtime(cfg, device, n_windows=1100, steps=4, mem_fraction=0.8) -> dict:
    tc = cfg["training"]
    cuda = device.type == "cuda"
    eff = int(tc["batch_size"]) * int(tc["gradient_accumulation_steps"])
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30 if cuda else 0.0
    sizes = [(eff, 1), (max(eff // 2, 1), 2)] if cuda else [(eff, 1)]
    candidates, n_params = [], None
    for use_amp in ([True, False] if cuda and bool(tc["use_amp"]) else [False]):
        for batch, accum in sizes:                                 # the half batch is tried only if the full one does not fit
            cand = {"use_amp": use_amp, "batch_size": batch, "gradient_accumulation_steps": accum, "peak_gb": None,
                    "sec_per_step": None, "projected_minutes": None, "fits": False}
            candidates.append(cand)
            try:
                peak, sec, n_params = _measure(cfg, device, batch, steps, use_amp)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            cand.update(peak_gb=round(peak, 3), sec_per_step=sec,
                        projected_minutes=sec * (n_windows / batch) * tc["max_epochs"] / 60,
                        fits=not (cuda and peak > mem_fraction * total))
            if cand["fits"]:
                break
    best = choose_setting(candidates, mem_fraction)
    return {"batch_size": best["batch_size"], "gradient_accumulation_steps": best["gradient_accumulation_steps"],
            "peak_gb": best["peak_gb"], "total_gb": round(total, 2), "sec_per_step": best["sec_per_step"],
            "parameters": n_params, "projected_minutes": best["projected_minutes"], "use_amp": best["use_amp"],
            "candidates": candidates}
