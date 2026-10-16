"""Guided training of the EEG-only student. One optimizer over the student and the guide heads.

The teacher is read-only. Early stopping and the best epoch use the validation loss of E alone
(`evaluate_e_loss`), never a guide term. A `fade` run cannot stop before the guide has reached zero
(`guide.off_at`), so the last stretch of every such run is pure EEG-to-EMG training.

Files in ckpt_dir:
  best.pt   - {"state": student weights of the best epoch, history, config, epoch, training_complete}; student only
  last.pt   - the same, with "state" = current weights, plus heads, optimizer, scheduler, scaler, best_state, counters
  history.json, metrics.csv
"""
import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from v2.losses import build_loss

from .guide import guide_grad_share, guide_weight, guided_loss

IDENTITY_BLOCKS = ("model", "tokens", "eeg", "history", "loss", "guide")


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_e_loss(student, loader, crit, device, use_amp) -> float:
    """Mean v2 StreamLoss of E over the loader (weighted by batch size); no guide term."""
    student.eval()
    total, n = 0.0, 0
    for batch in loader:
        b = _to(batch, device)
        with torch.autocast(device.type, enabled=use_amp):
            e = student(b["eeg"])["E"]
        total += float(crit(e, b["target"], b["hold_id"])[0]) * b["eeg"].shape[0]
        n += b["eeg"].shape[0]
    return total / max(n, 1)


def load_student(student, ckpt_dir, device) -> dict:
    ck = torch.load(Path(ckpt_dir) / "best.pt", map_location=device, weights_only=False)
    student.load_state_dict(ck["state"])
    return ck["history"]


def _config(cfg):
    return {k: copy.deepcopy(cfg[k]) for k in IDENTITY_BLOCKS}


def _save(path, epoch, state, hist, cfg, complete, extra=None):
    torch.save({"epoch": epoch, "state": state, "history": hist, "config": _config(cfg),
                "training_complete": complete, **(extra or {})}, path)


def fit_guided(student, teacher, heads, train_ds, val_ds, cfg, device, ckpt_dir, resume=True, log=print) -> dict:
    g, tc = cfg["guide"], cfg["training"]
    if teacher is None and g["scale"] > 0 and g["schedule"] != "off":
        raise ValueError("guide.scale is above zero but no teacher was given; pass scale 0.0 for the E-only control")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    student.to(device)
    heads.to(device)
    torch.manual_seed(tc["seed"])
    total, warm, patience = tc["max_epochs"], tc["warmup_epochs"], tc["patience"]
    use_amp = bool(tc["use_amp"]) and device.type == "cuda"
    min_epochs = g["off_at"] if (g["schedule"] == "fade" and g["scale"] > 0) else 0
    crit = build_loss(cfg)
    params = list(student.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])

    def lr_at(ep):
        if ep < warm:
            return (ep + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(total - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = int(tc["gradient_accumulation_steps"])
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False)

    keys = ("train_emg", "train_align", "train_distill", "train_aux", "w", "grad_share", "val_e")
    hist = {**{k: [] for k in keys}, "best_epoch": 0, "epochs_run": 0, "total_time_s": 0.0}
    best, bad, recent, start_ep, spent = math.inf, 0, [], 0, 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

    last_pt = ckpt_dir / "last.pt"
    if resume and last_pt.exists():
        ck = torch.load(last_pt, map_location=device, weights_only=False)
        differ = [b for b in IDENTITY_BLOCKS if ck.get("config", {}).get(b) != cfg.get(b)]
        if differ:
            log(f"  last.pt config mismatch ({', '.join(map(repr, differ))}); starting fresh")
        else:
            student.load_state_dict(ck["state"])
            heads.load_state_dict(ck["heads_state"])
            opt.load_state_dict(ck["optimizer_state"])
            sched.load_state_dict(ck["scheduler_state"])
            if ck.get("scaler_state") is not None:
                scaler.load_state_dict(ck["scaler_state"])
            hist, best, bad, recent = ck["history"], ck["best"], ck["bad"], ck["recent"]
            best_state, start_ep, spent = ck["best_state"], ck["epoch"], ck["history"].get("total_time_s", 0.0)
            if ck["training_complete"]:
                student.load_state_dict(best_state)
                log(f"  training already completed ({start_ep}/{total} epochs): skipped")
                return hist
            log(f"  resumed last.pt at epoch {start_ep}/{total}")

    t_start = time.time()
    for ep in range(start_ep, total):
        train_ds.set_epoch(ep)
        student.train()
        heads.train()
        w = guide_weight(ep, cfg)
        sums = {"emg": 0.0, "align": 0.0, "distill": 0.0, "aux": 0.0}
        steps, share = 0, 0.0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            b = _to(batch, device)
            with torch.autocast(device.type, enabled=use_amp):
                s_out = student(b["eeg"])
                t_out = teacher(b["kin"], b["hist"]) if w > 0 else None
                loss, parts = guided_loss(s_out, t_out, b, heads, crit, cfg, ep)
            if i == 0 and w > 0:
                l_emg = crit(s_out["E"], b["target"], b["hold_id"])[0]
                share = guide_grad_share(student.parameters(), l_emg, loss - l_emg)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                norm = float(torch.nn.utils.clip_grad_norm_(params, tc["grad_clip_norm"]))
                if math.isfinite(norm):
                    scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            for k in sums:
                sums[k] += parts[k]
            steps += 1
        sched.step()
        v = evaluate_e_loss(student, val_loader, crit, device, use_amp)
        for k in sums:
            hist["train_" + k].append(sums[k] / max(steps, 1))
        hist["w"].append(w)
        hist["grad_share"].append(share)
        hist["val_e"].append(v)
        hist["epochs_run"] = ep + 1
        recent = (recent + [v])[-tc["smooth_window"]:]
        smooth = sum(recent) / len(recent)
        improved = smooth < best - 1e-6
        if improved:
            best, bad, hist["best_epoch"] = smooth, 0, ep + 1
            best_state = {k: x.detach().cpu().clone() for k, x in student.state_dict().items()}
        else:
            bad += 1
        hist["total_time_s"] = spent + time.time() - t_start
        stop = bad >= patience and ep + 1 >= min_epochs
        done = stop or ep + 1 == total
        _save(last_pt, ep + 1, student.state_dict(), hist, cfg, done, {
            "heads_state": heads.state_dict(), "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
            "scaler_state": scaler.state_dict(), "best_state": best_state, "best": best, "bad": bad, "recent": recent})
        if improved:
            _save(ckpt_dir / "best.pt", ep + 1, best_state, hist, cfg, False)
        log(f"  epoch {ep + 1:3d}: val E {v:.4f}  guide weight {w:.3f}  guide gradient share {share:.3f}")
        if stop:
            log(f"  early stop at epoch {ep + 1} (no improvement for {patience} epochs, guide off since epoch {min_epochs})")
            break

    student.load_state_dict(best_state)
    _save(ckpt_dir / "best.pt", hist["epochs_run"], best_state, hist, cfg, True)
    (ckpt_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    pd.DataFrame({k: hist[k] for k in keys}).rename_axis("epoch").to_csv(ckpt_dir / "metrics.csv")
    return hist
