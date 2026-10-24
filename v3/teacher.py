"""Frozen kinematic teacher: the v2 KinStream with weights from the user's v2 checkpoint. Used in training only.

It never receives a gradient, never leaves eval mode, and is never part of a saved student checkpoint.
`forward` repeats KinStream.forward step by step only to keep the output of the second temporal layer.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.blocks import causal_mask
from v2.model import KinStream
from v2.train import load_kin

_KIN_KEYS = ("d_model", "n_heads", "ffn_dim", "kin_layers", "node_dim", "muscle_gat_layers", "causal")


class Teacher(nn.Module):
    def __init__(self, cfg_model, muscle_prior=None, n_kin=24, n_hist=8, n_muscles=5):
        super().__init__()
        self.kin = KinStream(cfg_model, n_kin, n_hist, n_muscles, muscle_prior)

    def train(self, mode: bool = True):
        return super().train(False)                           # a frozen teacher stays in eval mode

    @torch.no_grad()
    def forward(self, kin, hist):
        s = self.kin
        x = torch.cat([kin, hist], dim=-1).transpose(1, 2)
        if s.causal:
            x = F.pad(x, (s.stem.kernel_size[0] - 1, 0))
        x = s.stem(x).transpose(1, 2)
        enc = s.enc
        x = x + enc._positions(x.shape[1], x.device, x.dtype)
        mask = causal_mask(x.shape[1], x.device) if enc.causal else None
        mid = None
        for i, layer in enumerate(enc.layers):
            x = layer(x, mask)
            if i == 1:
                mid = x
        fin = enc.norm(x)
        gain = torch.pow(2.0, torch.tanh(s.gain(hist)))
        return {"K": s.head(fin) * gain, "mid": mid, "fin": fin}


def teacher_path(cfg, participant: int) -> Path:
    return Path(cfg["teacher"]["checkpoint_dir"]) / f"P{int(participant)}" / "best.pt"


def check_teacher_split(ckpt_path, participant: int, run_split: dict, data_dir=None) -> dict:
    """Raise if the teacher was trained on a series this run validates or tests on; returns the teacher's split.

    The split is read from the checkpoint ("split" in its config or history); v2 does not store it, so otherwise it
    is v2's rule, main.dataloader.stratified_split (deterministic, no seed). The teacher may share the run's
    validation series (it was only selected on them), but its training series must not reach the run's validation or
    test series, and its validation series must not reach the run's test series.
    """
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    split = (ck.get("config") or {}).get("split") or (ck.get("history") or {}).get("split")
    source = "checkpoint"
    if split is None:
        from main.dataloader import stratified_split
        split, source = stratified_split(int(participant), data_dir), "v2 split rule (main.dataloader.stratified_split)"
    t_train, t_val = {int(s) for s in split["train"]}, {int(s) for s in split["val"]}
    r_val, r_test = {int(s) for s in run_split["val"]}, {int(s) for s in run_split["test"]}
    problems = []
    if t_train & (r_val | r_test):
        problems.append(f"trained on series {sorted(t_train & (r_val | r_test))}, which this run uses for validation or test")
    if t_val & r_test:
        problems.append(f"selected on series {sorted(t_val & r_test)}, which this run tests on")
    if problems:
        raise ValueError(f"P{participant} teacher {ckpt_path} (split from the {source}): " + "; ".join(problems))
    return {"train": sorted(t_train), "val": sorted(t_val), "test": sorted(int(s) for s in split.get("test", [])),
            "source": source}


def load_teacher(cfg, muscle_prior, ckpt_path, device) -> Teacher:
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"K checkpoint not found: {path}")
    stored = torch.load(path, map_location="cpu", weights_only=False).get("config", {}).get("model")
    if stored is not None:
        for key in _KIN_KEYS:
            if stored.get(key) != cfg["model"].get(key):
                raise ValueError(f"{path}: model.{key} is {stored.get(key)!r} in the checkpoint "
                                 f"but {cfg['model'].get(key)!r} in the config; the teacher would not match")
    mp = None if muscle_prior is None else torch.as_tensor(muscle_prior, dtype=torch.float32)
    teacher = Teacher(cfg["model"], mp)
    load_kin(teacher, path)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.to(device).eval()
