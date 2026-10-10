"""
main/ablation.py — How much does a model use EEG vs kinematics?

Input-ablation evaluation on an existing loader, reusing the notebook's
``prepare_batch`` code path (normalisation + kin column drop) and then
perturbing the *normalised* model inputs:

    full         unchanged inputs
    kin_zero     kinematics set to 0 in z-space (= the per-feature training mean)
    kin_off      EEG only: the model is called with ``use_kin=False`` (both kinematic
                 pathways off: no kinematic edge bias, no kin skip).  Not the same as
                 kin_zero — zero inputs still pass through the skip MLP and the edge-bias
                 layer.  Only for models whose forward accepts ``use_kin``.
    kin_shuffle  kinematics time-shuffled within each window (fixed seeded permutation)
    eeg_shuffle  EEG time-shuffled within each window (fixed seeded permutation)

For every condition the per-muscle Pearson r over the concatenated loader
(same definition as ``compute_metrics``) is reported, plus the mean predicted
level during rest (target below a per-muscle threshold) for each condition.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .training import PrepareBatch, _forward_model

# input perturbations every model supports
ABLATION_CONDITIONS: tuple[str, ...] = ("full", "kin_zero", "kin_shuffle", "eeg_shuffle")
# ... plus kin_off for models with a ``use_kin`` switch (KGGTModel)
ALL_ABLATION_CONDITIONS: tuple[str, ...] = ("full", "kin_zero", "kin_off", "kin_shuffle", "eeg_shuffle")

ABLATION_LABELS: dict[str, str] = {
    "full": "Full model",
    "kin_zero": "Kin zeroed",
    "kin_off": "EEG only (kinematic paths off)",
    "kin_shuffle": "Kin time-shuffled",
    "eeg_shuffle": "EEG time-shuffled",
}

ABLATION_COLORS: dict[str, str] = {
    "full": "#2a78d6",
    "kin_zero": "#eb6834",
    "kin_off": "#2e9e6b",
    "kin_shuffle": "#9e9e9e",
    "eeg_shuffle": "#4a4a4a",
}


def shuffle_time(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute the time axis (dim 1) of a (B, T, F) tensor with a fixed seeded permutation.

    The same permutation is applied to every window, so the result is
    deterministic for a given seed and T; per-window value sets are preserved.
    """
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(x.shape[1], generator=g).to(x.device)
    return x.index_select(1, perm)


def apply_condition(inputs: dict, condition: str, seed: int = 0) -> dict:
    """Return a copy of the normalised model-input dict perturbed for ``condition``."""
    if condition not in ALL_ABLATION_CONDITIONS:
        raise ValueError(f"Unknown ablation condition {condition!r}; choose from {ALL_ABLATION_CONDITIONS}")
    out = dict(inputs)
    if condition == "kin_zero":
        out["kin"] = torch.zeros_like(inputs["kin"])
    elif condition == "kin_off":
        out["use_kin"] = False          # forwarded to the model: model(eeg=…, kin=…, use_kin=False)
    elif condition == "kin_shuffle":
        out["kin"] = shuffle_time(inputs["kin"], seed)
    elif condition == "eeg_shuffle":
        out["eeg"] = shuffle_time(inputs["eeg"], seed)
    return out


def supports_use_kin(model: nn.Module) -> bool:
    """True if ``model.forward`` takes a ``use_kin`` argument (EEG-only inference switch)."""
    try:
        return "use_kin" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def default_conditions(model: nn.Module) -> tuple[str, ...]:
    """All conditions the model supports: with ``kin_off`` only if it has ``use_kin``."""
    return ALL_ABLATION_CONDITIONS if supports_use_kin(model) else ABLATION_CONDITIONS


def pearson_per_channel(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-column Pearson r of (N, C) arrays (0 for a constant column), as in compute_metrics."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    p_c = pred - pred.mean(0, keepdims=True)
    t_c = target - target.mean(0, keepdims=True)
    num = (p_c * t_c).mean(0)
    denom = np.sqrt((p_c ** 2).mean(0)) * np.sqrt((t_c ** 2).mean(0))
    safe = np.where(denom > 1e-12, denom, 1.0)
    return np.where(denom > 1e-12, num / safe, 0.0)


@torch.no_grad()
def rest_threshold_from_loader(
    loader: DataLoader, prepare_batch: PrepareBatch, q: float = 10.0
) -> np.ndarray:
    """Per-muscle q-th percentile of the (normalised) targets yielded by ``prepare_batch``."""
    ys = []
    for eeg, kin, emg in loader:
        _, y = prepare_batch(eeg, kin, emg)
        ys.append(y.float().cpu().numpy().reshape(-1, y.shape[-1]))
    return np.percentile(np.concatenate(ys), q, axis=0)


@dataclass
class AblationResult:
    conditions: tuple[str, ...]
    r: dict[str, np.ndarray]                 # condition -> (C,) Pearson r
    r_mean: dict[str, float]                 # condition -> mean r over muscles
    predictions: dict[str, np.ndarray]       # condition -> (N, C) normalised predictions
    target: np.ndarray                       # (N, C) normalised targets
    rest_threshold: np.ndarray | None = None  # (C,)
    rest_pred: dict[str, np.ndarray] = field(default_factory=dict)  # condition -> (C,)
    rest_target: np.ndarray | None = None    # (C,) mean target during rest

    def as_table(self, muscle_names: Sequence[str] | None = None) -> str:
        C = self.target.shape[1]
        names = list(muscle_names) if muscle_names is not None else [f"ch{c}" for c in range(C)]
        head = f"{'condition':<13}" + "".join(f"{n:>8}" for n in names) + f"{'mean':>8}"
        lines = ["Pearson r", head, "-" * len(head)]
        for cond in self.conditions:
            lines.append(f"{cond:<13}" + "".join(f"{v:8.3f}" for v in self.r[cond])
                         + f"{self.r_mean[cond]:8.3f}")
        if self.rest_pred:
            lines += ["", "Mean level during rest (z-space, target < threshold)", head, "-" * len(head)]
            rows = [("target", self.rest_target)] + [(c, self.rest_pred[c]) for c in self.conditions
                                                     if c in self.rest_pred]
            for label, vals in rows:
                lines.append(f"{label:<13}" + "".join(f"{v:8.3f}" for v in vals)
                             + f"{float(np.nanmean(vals)):8.3f}")
        return "\n".join(lines)


@torch.no_grad()
def input_ablation(
    model: nn.Module,
    loader: DataLoader,
    prepare_batch: PrepareBatch,
    device: torch.device,
    conditions: Sequence[str] | None = None,
    seed: int = 0,
    use_amp: bool = True,
    rest_threshold: np.ndarray | None = None,
) -> AblationResult:
    """Evaluate ``model`` on ``loader`` under each input-ablation condition.

    Inputs go through ``prepare_batch`` exactly like ``collect_predictions``;
    the perturbation is applied to the normalised inputs afterwards.
    If ``rest_threshold`` (per-muscle, normalised units) is given, the mean
    prediction over samples whose target is below it is reported per condition.

    ``conditions`` = None (default): ``default_conditions(model)`` — the four input
    perturbations, plus ``kin_off`` (model called with ``use_kin=False``) when the
    model's forward has that switch.
    """
    model.eval()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp = use_amp and device.type == "cuda"
    conditions = default_conditions(model) if conditions is None else tuple(conditions)
    if "kin_off" in conditions and not supports_use_kin(model):
        raise TypeError(
            f"condition 'kin_off' needs a model whose forward accepts use_kin "
            f"({type(model).__name__}.forward does not)"
        )

    preds: dict[str, list[np.ndarray]] = {c: [] for c in conditions}
    targets: list[np.ndarray] = []
    for eeg, kin, emg in loader:
        inputs, y = prepare_batch(eeg, kin, emg)
        n_ch = y.shape[-1]
        targets.append(y.float().cpu().numpy().reshape(-1, n_ch))
        for cond in conditions:
            x = apply_condition(inputs, cond, seed=seed)
            with torch.amp.autocast(device_type=amp_device, enabled=amp):
                p = _forward_model(model, x)
            preds[cond].append(p.float().cpu().numpy().reshape(-1, n_ch))

    Y = np.concatenate(targets)
    P = {c: np.concatenate(v) for c, v in preds.items()}
    r = {c: pearson_per_channel(P[c], Y) for c in conditions}
    r_mean = {c: float(np.mean(r[c])) for c in conditions}

    rest_pred: dict[str, np.ndarray] = {}
    rest_target = None
    thr = None
    if rest_threshold is not None:
        thr = np.broadcast_to(np.asarray(rest_threshold, dtype=np.float64), (Y.shape[1],)).copy()
        mask = Y < thr                                                     # (N, C)
        cnt = mask.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rest_target = np.where(cnt > 0, (Y * mask).sum(0) / np.maximum(cnt, 1), np.nan)
            for c in conditions:
                rest_pred[c] = np.where(cnt > 0, (P[c] * mask).sum(0) / np.maximum(cnt, 1), np.nan)

    return AblationResult(
        conditions=conditions, r=r, r_mean=r_mean, predictions=P, target=Y,
        rest_threshold=thr, rest_pred=rest_pred, rest_target=rest_target,
    )
