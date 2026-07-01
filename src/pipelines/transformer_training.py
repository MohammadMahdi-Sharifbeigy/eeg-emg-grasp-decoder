"""Reusable setup helpers for the current transformer training path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.data.dataset import WAYEEGDataset
from src.preprocessing.cca import TorchCCA, make_cca_from_config
from src.preprocessing.eeg import preprocess_eeg_from_config
from src.preprocessing.emg import preprocess_emg_from_config


def make_preprocess_fn(cfg: dict) -> Callable[[dict], dict]:
    """Build the per-series preprocessing closure from the project config."""
    eeg_cfg = cfg["preprocessing"]["eeg"]
    emg_cfg = cfg["preprocessing"]["emg"]

    def preprocess_fn(series: dict) -> dict:
        series = dict(series)
        series["eeg"] = preprocess_eeg_from_config(
            series["eeg"],
            float(series["fs_eeg"]),
            eeg_cfg,
            channel_names=series.get("eeg_names"),
        )
        series["emg"] = preprocess_emg_from_config(
            series["emg"],
            float(series["fs_emg"]),
            emg_cfg,
        )
        return series

    return preprocess_fn


def build_dataset(
    split: str,
    cfg: dict,
    participants: list[int],
    root_dir: Path,
) -> WAYEEGDataset:
    """Build one dataset split using the existing config semantics."""
    data_cfg = cfg["data"]
    return WAYEEGDataset(
        data_dir=root_dir / data_cfg["raw_dir"],
        participants=participants,
        split=split,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        preprocess_fn=make_preprocess_fn(cfg),
        cache_dir=root_dir / data_cfg["cache_dir"],
    )


def build_datasets(
    cfg: dict,
    participants: list[int],
    root_dir: Path,
) -> tuple[WAYEEGDataset, WAYEEGDataset, WAYEEGDataset]:
    """Build train/val/test datasets in the current project layout."""
    return (
        build_dataset("train", cfg, participants, root_dir),
        build_dataset("val", cfg, participants, root_dir),
        build_dataset("test", cfg, participants, root_dir),
    )


def unique_series_arrays(
    ds: WAYEEGDataset,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return unique full-series arrays referenced by a sliding-window dataset."""
    seen, eegs, kins, emgs = set(), [], [], []
    for eeg_all, kin_all, emg_all, _ in ds._windows:
        key = id(eeg_all)
        if key in seen:
            continue
        seen.add(key)
        eegs.append(eeg_all)
        kins.append(kin_all)
        emgs.append(emg_all)
    return eegs, kins, emgs


def fit_cca_and_emg_stats(
    train_ds: WAYEEGDataset,
    cca_cfg: dict,
    device: torch.device,
) -> tuple[TorchCCA, torch.Tensor, torch.Tensor]:
    """Fit the train-only CCA and EMG normalization statistics."""
    tr_eeg, tr_kin, tr_emg = unique_series_arrays(train_ds)
    cca = make_cca_from_config(cca_cfg)
    cca.fit(tr_eeg, tr_kin)
    cca_gpu = cca.torch_projector(device)

    emg_concat = np.concatenate(tr_emg, axis=0)
    emg_mean = torch.tensor(emg_concat.mean(0), dtype=torch.float32, device=device)
    emg_std = torch.tensor(emg_concat.std(0) + 1e-8, dtype=torch.float32, device=device)
    return cca_gpu, emg_mean, emg_std


def prepare_batch_factory(
    cca_projector: TorchCCA,
    emg_mean: torch.Tensor,
    emg_std: torch.Tensor,
    device: torch.device,
    n_cca: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[dict[str, Any], torch.Tensor]]:
    """Create the shared train/eval batch-preparation function.

    Returns a model-input dictionary so transformer-only and transformer+GAT
    variants can share the same training loop. Callers that do not need
    kinematics can ignore the extra ``kin`` entry.
    """

    def prepare_batch(
        eeg: torch.Tensor,
        kin: torch.Tensor,
        emg: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        batch_size, window_size, n_features = eeg.shape
        flat = eeg.reshape(-1, n_features).to(device, non_blocking=True)
        proj = cca_projector.transform(flat).reshape(batch_size, window_size, n_cca)
        kin_device = kin.to(device, non_blocking=True)
        emg_norm = (emg.to(device, non_blocking=True) - emg_mean) / emg_std
        return {"eeg": proj, "kin": kin_device}, emg_norm

    return prepare_batch
