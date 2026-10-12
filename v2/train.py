"""Single-stage training of KG-GT v2 with separate early stopping for the two streams.

The kinematic stream (parameters `kin.*`) and the EEG stream + fusion (`eeg.*`, `fusion.*`)
share no parameters and no gradients, so each keeps the weights of its own best epoch.
Each stream also has its own optimizer, schedule, gradient clip and step, so nothing that happens
to one stream's gradients (size, NaN, a skipped step) changes the other stream's update.

Checkpointing (matching main_1stage patterns):
  last.pt  – full resumable snapshot written after every epoch (model + optimizer + scheduler +
              scaler state_dicts, history, best tracking).  Picked up automatically on restart.
  best.pt  – written whenever either stream improves; contains best_state + history + config.
             Also the final authoritative checkpoint written at the end of training.

Metrics files written to ckpt_dir every epoch:
  metrics.json        – full history dict
  metrics.csv         – epoch × metric table (pandas)
  metrics_report.md   – human-readable markdown table
"""
import copy
import csv
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

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


def weighted_stream_loss(components, stream, loss_cfg) -> float:
    """Loss of one output ("K", "E" or "F") from the logged components, e.g. history["components"][epoch].

    The same weighted sum as StreamLoss: w_main * main + w_ccc * ccc + w_lift * lift + w_diff * diff.
    """
    return float(sum(loss_cfg[f"w_{part}"] * components[f"{stream}/{part}"] for part in ("main", "ccc", "lift", "diff")))


IDENTITY_BLOCKS = ("model", "tokens", "eeg", "history", "loss")    # config blocks that define the model and its data


# ============================================================================
# Checkpoint helpers
# ============================================================================

def _save_last(path, epoch, model, opts, scheds, scaler, best_state, hist, best, bad, recent, cfg):
    """Full crash-safe snapshot: everything needed to resume training exactly."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {k: opt.state_dict() for k, opt in opts.items()},
        "scheduler_state_dicts": [s.state_dict() for s in scheds],
        "scaler_state_dict": scaler.state_dict(),
        "best_state": best_state,
        "history": hist,
        "best": best,
        "bad": bad,
        "recent": recent,
        "config": {k: copy.deepcopy(cfg[k]) for k in IDENTITY_BLOCKS},
    }, path)


def _save_best(path, epoch, model, opts, scheds, scaler, best_state, hist, cfg):
    """Best-model checkpoint: best weights + full optimizer/scheduler/scaler state + history + config.

    Saves everything _save_last saves, but uses best_state (not the current epoch's weights) as the
    primary model state.  This means you can resume training from the best model, not just use it for
    inference.  The 'state' key holds best_state so load_best() still works unchanged.
    """
    torch.save({
        "epoch": epoch,
        "state": best_state,                                          # inference key (load_best uses this)
        "model_state_dict": best_state,                               # resume key (mirrors last.pt schema)
        "optimizer_state_dicts": {k: opt.state_dict() for k, opt in opts.items()},
        "scheduler_state_dicts": [s.state_dict() for s in scheds],
        "scaler_state_dict": scaler.state_dict(),
        "best_state": best_state,
        "history": hist,
        "config": {k: copy.deepcopy(cfg[k]) for k in IDENTITY_BLOCKS},
    }, path)


def load_best(model, ckpt_dir, device):
    """Load inference weights from best.pt. Returns history dict."""
    ck = torch.load(Path(ckpt_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])
    return ck["history"]


def _check_config(path, ck, cfg, log):
    """Return list of mismatched identity blocks; empty means compatible."""
    stored = ck.get("config")
    if stored is None:
        return []  # old checkpoint without config — allow with warning
    return [b for b in IDENTITY_BLOCKS if stored.get(b) != cfg.get(b)]


def _resume_finished(model, ckpt_dir, cfg, device, log):
    """Load a *completed* run from best.pt (training will be skipped).

    Raises RuntimeError if the config doesn't match so callers know to delete
    the checkpoint folder rather than silently train over the wrong weights.
    """
    path = Path(ckpt_dir) / "best.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    stored = ck.get("config")
    if stored is None:
        log(f"  WARNING: {path} has no config record; loading anyway (verify manually).")
    else:
        differ = [b for b in IDENTITY_BLOCKS if stored.get(b) != cfg.get(b)]
        if differ:
            raise RuntimeError(
                f"{path} was trained with a different config: block(s) {', '.join(map(repr, differ))} "
                f"differ from the current config. Move or delete the folder {Path(ckpt_dir)} to train "
                f"again, or restore the config it was trained with."
            )
    model.load_state_dict(ck["state"])
    return ck["history"]


# ============================================================================
# Metrics reporting (mirrors main/training.py TrainResult.save_metrics)
# ============================================================================

def _save_metrics(ckpt_dir, hist, total_time_s, cfg):
    """Write metrics.json, metrics.csv, metrics_report.md to ckpt_dir."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n = hist["epochs_run"]
    best_k, best_e = hist["best_epoch_kin"], hist["best_epoch_eeg"]

    # 1. Full JSON (the raw hist dict + a summary section)
    summary = {
        "epochs_run": n,
        "best_epoch_kin": best_k,
        "best_epoch_eeg": best_e,
        "total_time_s": total_time_s,
        "history": hist,
    }
    (ckpt_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 2. CSV — one row per epoch
    rows = []
    for i in range(n):
        row = {
            "epoch": i + 1,
            "train_loss": hist["train"][i] if i < len(hist["train"]) else "",
            "val_kin": hist["val_kin"][i] if i < len(hist["val_kin"]) else "",
            "val_eeg": hist["val_eeg"][i] if i < len(hist["val_eeg"]) else "",
            "skipped_kin": hist["skipped_kin"][i] if i < len(hist["skipped_kin"]) else "",
            "skipped_eeg": hist["skipped_eeg"][i] if i < len(hist["skipped_eeg"]) else "",
            "best_K": "*" if (i + 1) == best_k else "",
            "best_E": "*" if (i + 1) == best_e else "",
        }
        rows.append(row)
    if rows:
        with open(ckpt_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 3. Markdown report
    lines = [
        "=" * 78,
        "               KG-GT v2 TRAINING METRICS REPORT",
        "=" * 78,
        f"  Completed epochs : {n}",
        f"  Best K epoch     : {best_k}",
        f"  Best E epoch     : {best_e}",
        f"  Total time       : {total_time_s / 60:.2f} min ({total_time_s:.1f} s)",
        "-" * 78,
    ]
    if rows:
        header = f"{'Epoch':^6} | {'Train':^10} | {'Val K':^10} | {'Val E+F':^10} | {'sk_K':^5} | {'sk_E':^5} | {'Best':^8}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            flag = ""
            if row["best_K"] and row["best_E"]:
                flag = "K+E"
            elif row["best_K"]:
                flag = "K"
            elif row["best_E"]:
                flag = "E"
            lines.append(
                f"{row['epoch']:^6d} | {float(row['train_loss'] or 0):^10.4f} | "
                f"{float(row['val_kin'] or 0):^10.4f} | {float(row['val_eeg'] or 0):^10.4f} | "
                f"{str(row['skipped_kin']):^5} | {str(row['skipped_eeg']):^5} | {flag:^8}"
            )
    lines.append("=" * 78)
    (ckpt_dir / "metrics_report.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# Optimizer / scheduler builders
# ============================================================================

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


# ============================================================================
# Main training loop
# ============================================================================

def fit(model, train_ds, val_ds, cfg, device, ckpt_dir, resume=True, log=print):
    ckpt_dir = Path(ckpt_dir)
    model.to(device)

    # --- skip entirely if a completed run already exists ---
    if resume and (ckpt_dir / "best.pt").exists():
        history = _resume_finished(model, ckpt_dir, cfg, device, log)
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

    hist = {"train": [], "val_kin": [], "val_eeg": [], "components": [], "skipped_kin": [], "skipped_eeg": [],
            "best_epoch_kin": 0, "best_epoch_eeg": 0, "epochs_run": 0}
    best = {"kin": math.inf, "eeg": math.inf}
    bad = {"kin": 0, "eeg": 0}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    recent = {"kin": [], "eeg": []}
    start_ep = 0
    t_start = time.time()

    # --- resume from last.pt if interrupted mid-run ---
    last_pt = ckpt_dir / "last.pt"
    if resume and last_pt.exists():
        ck = torch.load(last_pt, map_location=device, weights_only=False)
        differ = _check_config(last_pt, ck, cfg, log)
        if differ:
            log(f"  last.pt config mismatch ({', '.join(map(repr, differ))}); starting fresh")
        else:
            model.load_state_dict(ck["model_state_dict"])
            # restore optimizer state_dicts
            opt_states = ck.get("optimizer_state_dicts", {})
            for k, opt in opts.items():
                if k in opt_states:
                    opt.load_state_dict(opt_states[k])
            # restore scheduler state_dicts
            sched_states = ck.get("scheduler_state_dicts", [])
            for sched, st in zip(scheds, sched_states):
                sched.load_state_dict(st)
            # restore scaler
            scaler_st = ck.get("scaler_state_dict")
            if scaler_st is not None:
                scaler.load_state_dict(scaler_st)
            hist = ck["history"]
            best = ck["best"]
            bad = ck["bad"]
            recent = ck["recent"]
            best_state = ck["best_state"]
            start_ep = ck.get("epoch", hist["epochs_run"])
            log(f"  resumed last.pt at epoch {start_ep}/{total}  "
                f"(bad K {bad['kin']}/{tc['patience']}, bad E {bad['eeg']}/{tc['patience']})")

    for ep in range(start_ep, total):
        train_ds.set_epoch(ep)
        model.train()
        running, steps = 0.0, 0
        skipped = {"kin": 0, "eeg": 0}                               # optimizer steps left out: no finite gradient norm
        for opt in opts.values():
            opt.zero_grad(set_to_none=True)
        if _TQDM_AVAILABLE:
            bar = _tqdm(train_loader, desc=f"  Ep {ep + 1:4d}/{total}", leave=False, unit="batch",
                        dynamic_ncols=True)
        else:
            bar = train_loader
        try:
            for i, batch in enumerate(bar):
                b = _to(batch, device)
                with torch.autocast(device.type, enabled=use_amp):
                    out = model(b["kin"], b["hist"], b["eeg"])
                    loss, _, _ = v2_loss(out, b["target"], b["hold_id"], crit)
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    for stream, stepped in step_streams(model, opts, scaler, tc["grad_clip_norm"]).items():
                        skipped[stream] += not stepped
                batch_loss = float(loss.detach())
                running += batch_loss
                steps += 1
                if _TQDM_AVAILABLE:
                    bar.set_postfix({"loss": f"{batch_loss:.4f}"})
        finally:
            if _TQDM_AVAILABLE:
                bar.close()
        for sched in scheds:
            sched.step()
        v_kin, v_eeg, comps = evaluate_loss(model, val_loader, crit, device, use_amp)
        hist["train"].append(running / max(steps, 1))
        hist["val_kin"].append(v_kin)
        hist["val_eeg"].append(v_eeg)
        hist["components"].append(comps)
        hist["skipped_kin"].append(skipped["kin"])
        hist["skipped_eeg"].append(skipped["eeg"])
        hist["epochs_run"] = ep + 1
        state = model.state_dict()
        improved_any = False
        for stream, value in (("kin", v_kin), ("eeg", v_eeg)):
            recent[stream] = (recent[stream] + [value])[-tc["smooth_window"]:]
            smooth = sum(recent[stream]) / len(recent[stream])
            if smooth < best[stream] - 1e-6:
                best[stream], bad[stream] = smooth, 0
                hist[f"best_epoch_{stream}"] = ep + 1
                for k, v in state.items():
                    if stream_of(k) == stream:
                        best_state[k] = v.detach().cpu().clone()
                improved_any = True
            else:
                bad[stream] += 1
        log(f"  epoch {ep + 1:3d}  train {hist['train'][-1]:.4f}  val K {v_kin:.4f}  val E+F {v_eeg:.4f}"
            f"  bad K {bad['kin']}/{tc['patience']}  bad E {bad['eeg']}/{tc['patience']}"
            + (f"  skipped steps K {skipped['kin']}, E {skipped['eeg']}" if skipped["kin"] or skipped["eeg"] else ""))

        elapsed = time.time() - t_start

        # --- crash-safe last.pt: full optimizer + scheduler + scaler state ---
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        _save_last(ckpt_dir / "last.pt", ep + 1, model, opts, scheds, scaler,
                   best_state, hist, best, bad, recent, cfg)

        # --- best.pt whenever either stream improves ---
        if improved_any:
            _save_best(ckpt_dir / "best.pt", ep + 1, model, opts, scheds, scaler,
                       best_state, hist, cfg)

        # --- metrics.json / metrics.csv / metrics_report.md ---
        _save_metrics(ckpt_dir, hist, elapsed, cfg)

        if bad["kin"] >= tc["patience"] and bad["eeg"] >= tc["patience"]:
            log("  both streams stopped improving: early stop")
            break

    total_time_s = time.time() - t_start
    model.load_state_dict(best_state)
    # the saved model joins each stream's own best epoch, so it is validated once more as a whole
    v_kin, v_eeg, comps = evaluate_loss(model, val_loader, crit, device, use_amp)
    hist["val_combined"] = {"kin": v_kin, "eeg": v_eeg, "components": comps}
    a, b = hist["best_epoch_kin"], hist["best_epoch_eeg"]
    note = "" if a == b else (f"  (K from epoch {a}, E and fusion from epoch {b}: the fusion was not validated "
                              f"with this K during training)")
    log(f"  combined best model: val K {v_kin:.4f}  val E+F {v_eeg:.4f}{note}")

    # --- final authoritative best.pt + full history ---
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _save_best(ckpt_dir / "best.pt", hist["epochs_run"], model, opts, scheds, scaler,
               best_state, hist, cfg)
    (ckpt_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    _save_metrics(ckpt_dir, hist, total_time_s, cfg)
    log(f"  checkpoints -> {ckpt_dir}  (best.pt, last.pt, history.json, metrics.*)")
    return hist
