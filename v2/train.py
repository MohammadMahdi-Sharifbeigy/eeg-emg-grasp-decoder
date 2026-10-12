"""Single-stage training of KG-GT v2 with separate early stopping for the two streams.

The kinematic stream (parameters `kin.*`) and the EEG stream + fusion (`eeg.*`, `fusion.*`)
share no parameters and no gradients, so each keeps the weights of its own best epoch.
Each stream also has its own optimizer, schedule, gradient clip and step, so nothing that happens
to one stream's gradients (size, NaN, a skipped step) changes the other stream's update.
"""
import copy
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


IDENTITY_BLOCKS = ("model", "tokens", "eeg", "history", "loss")    # config blocks that define the model and its data


def load_best(model, ckpt_dir, device):
    ck = torch.load(Path(ckpt_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])
    return ck["history"]


def _resume(model, ckpt_dir, cfg, device):
    """Load a finished run, after checking that it was trained with the same model, data and loss settings."""
    path = Path(ckpt_dir) / "best.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    stored = ck.get("config")
    if stored is None:
        raise RuntimeError(f"{path} does not say which config it was trained with, so it cannot be resumed safely. "
                           f"Move or delete the folder {Path(ckpt_dir)} to train again.")
    differ = [b for b in IDENTITY_BLOCKS if stored.get(b) != cfg.get(b)]
    if differ:
        raise RuntimeError(f"{path} was trained with a different config: block(s) {', '.join(map(repr, differ))} "
                           f"differ from the current config. Move or delete the folder {Path(ckpt_dir)} to train "
                           f"again, or restore the config it was trained with.")
    model.load_state_dict(ck["state"])
    return ck["history"]


def build_optimizers(model, tc):
    """One AdamW per stream: `kin.*`, and `eeg.*` + `fusion.*` with `fusion.gate` free of weight decay."""
    groups = {"kin": [], "eeg": [], "gate": []}
    for name, p in model.named_parameters():
        groups["gate" if name == "fusion.gate" else stream_of(name)].append(p)
    wd = tc["weight_decay"]
    return {"kin": torch.optim.AdamW(groups["kin"], lr=tc["lr"], weight_decay=wd),
            "eeg": torch.optim.AdamW([{"params": groups["eeg"], "weight_decay": wd},
                                      {"params": groups["gate"], "weight_decay": 0.0}], lr=tc["lr"])}


def step_streams(model, opts, scaler, max_norm):
    """Clip and step each stream on its own; a stream whose gradient norm is not finite skips this step.

    Returns {"kin": stepped, "eeg": stepped}. The scaler alone would skip only under AMP, so the norm is
    checked here too (on the CPU and without AMP a NaN gradient would otherwise be written into the weights).
    """
    for opt in opts.values():
        scaler.unscale_(opt)
    norms = clip_per_stream(model, max_norm)
    stepped = {}
    for stream, opt in opts.items():
        stepped[stream] = math.isfinite(norms[stream])
        if stepped[stream]:
            scaler.step(opt)
        opt.zero_grad(set_to_none=True)
    scaler.update()
    return stepped


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
    model.to(device)
    if resume and (ckpt_dir / "best.pt").exists():
        history = _resume(model, ckpt_dir, cfg, device)
        log(f"  found {ckpt_dir / 'best.pt'}: training skipped")
        return history
    tc = cfg["training"]
    torch.manual_seed(tc["seed"])
    use_amp = bool(tc["use_amp"]) and device.type == "cuda"
    crit = build_loss(cfg)
    opts = build_optimizers(model, tc)
    warm, total = tc["warmup_epochs"], tc["max_epochs"]

    def lr_at(ep):                                                   # ep counts from 0
        if ep < warm:
            return (ep + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(total - warm, 1)))

    scheds = [torch.optim.lr_scheduler.LambdaLR(opt, lr_at) for opt in opts.values()]
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
        for opt in opts.values():
            opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            b = _to(batch, device)
            with torch.autocast(device.type, enabled=use_amp):
                out = model(b["kin"], b["hist"], b["eeg"])
                loss, _, _ = v2_loss(out, b["target"], b["hold_id"], crit)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                step_streams(model, opts, scaler, tc["grad_clip_norm"])
            running += float(loss.detach())
            steps += 1
        for sched in scheds:
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
    # the saved model joins each stream's own best epoch, so it is validated once more as a whole
    v_kin, v_eeg, comps = evaluate_loss(model, val_loader, crit, device, use_amp)
    hist["val_combined"] = {"kin": v_kin, "eeg": v_eeg, "components": comps}
    a, b = hist["best_epoch_kin"], hist["best_epoch_eeg"]
    note = "" if a == b else (f"  (K from epoch {a}, E and fusion from epoch {b}: the fusion was not validated "
                              f"with this K during training)")
    log(f"  combined best model: val K {v_kin:.4f}  val E+F {v_eeg:.4f}{note}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "history": hist, "config": {k: copy.deepcopy(cfg[k]) for k in IDENTITY_BLOCKS}},
               ckpt_dir / "best.pt")
    (ckpt_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    return hist
