"""Memory and time check of one guided step on RANDOM data. Trains nothing real and saves nothing.

The measured step is the real one: student + frozen random teacher + guide heads, one AdamW, clipped, stepped.
The guide is measured at full weight (at least scale 0.1), the most expensive case.
"""
import copy
import time

import torch

from v2.losses import build_loss
from v2.smoke import choose_setting

from .guide import build_heads, guided_loss
from .model import build_student
from .teacher import Teacher


def _measure(cfg, device, batch, steps, use_amp):
    cfg = copy.deepcopy(cfg)
    cfg["guide"].update(schedule="constant", scale=max(float(cfg["guide"]["scale"]), 0.1))
    w = cfg["tokens"]["window"]
    student, heads = build_student(cfg).to(device).train(), build_heads(cfg).to(device).train()
    teacher = Teacher(cfg["model"], None).to(device).eval()
    params = list(student.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    crit = build_loss(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(steps):
        b = {"kin": torch.randn(batch, w, 24, device=device), "hist": torch.rand(batch, w, 8, device=device),
             "eeg": torch.randn(batch, w, 32, 4, device=device), "target": torch.rand(batch, w, 5, device=device),
             "hold_id": torch.zeros(batch, w, dtype=torch.int64, device=device)}
        b["hold_id"][:, w // 4: w // 2] = 1
        t0 = time.perf_counter()
        with torch.autocast(device.type, enabled=use_amp):
            loss, _ = guided_loss(student(b["eeg"]), teacher(b["kin"], b["hist"]), b, heads, crit, cfg, epoch=0)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["training"]["grad_clip_norm"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
    n_params = sum(p.numel() for p in student.parameters())
    del student, heads, teacher, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return peak, min(times[1:] or times), n_params


def project_runtime(cfg, device, n_windows=1100, steps=4, mem_fraction=0.8) -> dict:
    tc = cfg["training"]
    cuda = device.type == "cuda"
    eff = int(tc["batch_size"]) * int(tc["gradient_accumulation_steps"])
    total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30 if cuda else 0.0
    sizes = [(eff, 1), (max(eff // 2, 1), 2)] if cuda else [(eff, 1)]
    candidates, n_params = [], None
    for use_amp in ([True, False] if cuda and bool(tc["use_amp"]) else [False]):
        for batch, accum in sizes:
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
