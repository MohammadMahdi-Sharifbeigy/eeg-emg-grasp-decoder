"""Single-stage training of KG-GT v2 with separate early stopping for the two streams.

The kinematic stream (parameters `kin.*`) and the EEG stream + fusion (`eeg.*`, `fusion.*`)
share no parameters and no gradients, so each keeps the weights of its own best epoch.
"""
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import build_loss, v2_loss
from .model import stream_of


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate_loss(model, loader, crit, device, use_amp):
    model.eval()
    sums, n = {"K": 0.0, "E": 0.0, "F": 0.0}, 0
    comps_sum = {}
    for batch in loader:
        b = _to(batch, device)
        with torch.autocast(device.type, enabled=use_amp):
            out = model(b["kin"], b["hist"], b["eeg"])
            _, per, comps = v2_loss(out, b["target"], b["hold_id"], crit)
        for k in sums:
            sums[k] += float(per[k])
        for k, v in comps.items():
            comps_sum[k] = comps_sum.get(k, 0.0) + v
        n += 1
    n = max(n, 1)
    return sums["K"] / n, (sums["E"] + sums["F"]) / n, {k: v / n for k, v in comps_sum.items()}


def load_best(model, ckpt_dir, device):
    ck = torch.load(Path(ckpt_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])
    return ck["history"]


def clip_per_stream(model, max_norm):
    """Clip the gradient norm of each stream on its own, so the size of one stream's step never depends on the other's.

    One clip over all parameters would scale the kinematic gradients by the EEG and fusion gradient norm.
    """
    groups = {"kin": [], "eeg": []}
    for name, p in model.named_parameters():
        if p.grad is not None:
            groups[stream_of(name)].append(p)
    return {s: float(torch.nn.utils.clip_grad_norm_(ps, max_norm)) if ps else 0.0 for s, ps in groups.items()}


def fit(model, train_ds, val_ds, cfg, device, ckpt_dir, resume=True, log=print):
    ckpt_dir = Path(ckpt_dir)
    if resume and (ckpt_dir / "best.pt").exists():
        log(f"  found {ckpt_dir / 'best.pt'}: training skipped")
        return load_best(model, ckpt_dir, device)
    tc = cfg["training"]
    torch.manual_seed(tc["seed"])
    use_amp = bool(tc["use_amp"]) and device.type == "cuda"
    crit = build_loss(cfg)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    warm, total = tc["warmup_epochs"], tc["max_epochs"]

    def lr_at(ep):                                                   # ep counts from 0
        if ep < warm:
            return (ep + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(total - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = int(tc["gradient_accumulation_steps"])
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False)

    hist = {"train": [], "val_kin": [], "val_eeg": [], "components": [], "best_epoch_kin": 0, "best_epoch_eeg": 0,
            "epochs_run": 0}
    best = {"kin": math.inf, "eeg": math.inf}
    bad = {"kin": 0, "eeg": 0}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    recent = {"kin": [], "eeg": []}

    for ep in range(total):
        train_ds.set_epoch(ep)
        model.train()
        running, steps = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            b = _to(batch, device)
            with torch.autocast(device.type, enabled=use_amp):
                out = model(b["kin"], b["hist"], b["eeg"])
                loss, _, _ = v2_loss(out, b["target"], b["hold_id"], crit)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt)
                clip_per_stream(model, tc["grad_clip_norm"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss.detach())
            steps += 1
        sched.step()
        v_kin, v_eeg, comps = evaluate_loss(model, val_loader, crit, device, use_amp)
        hist["train"].append(running / max(steps, 1))
        hist["val_kin"].append(v_kin)
        hist["val_eeg"].append(v_eeg)
        hist["components"].append(comps)
        hist["epochs_run"] = ep + 1
        state = model.state_dict()
        for stream, value in (("kin", v_kin), ("eeg", v_eeg)):
            recent[stream] = (recent[stream] + [value])[-tc["smooth_window"]:]
            smooth = sum(recent[stream]) / len(recent[stream])
            if smooth < best[stream] - 1e-6:
                best[stream], bad[stream] = smooth, 0
                hist[f"best_epoch_{stream}"] = ep + 1
                for k, v in state.items():
                    if stream_of(k) == stream:
                        best_state[k] = v.detach().cpu().clone()
            else:
                bad[stream] += 1
        log(f"  epoch {ep + 1:3d}  train {hist['train'][-1]:.4f}  val K {v_kin:.4f}  val E+F {v_eeg:.4f}"
            f"  bad K {bad['kin']}/{tc['patience']}  bad E {bad['eeg']}/{tc['patience']}")
        if bad["kin"] >= tc["patience"] and bad["eeg"] >= tc["patience"]:
            log("  both streams stopped improving: early stop")
            break

    model.load_state_dict(best_state)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "history": hist}, ckpt_dir / "best.pt")
    (ckpt_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    return hist
