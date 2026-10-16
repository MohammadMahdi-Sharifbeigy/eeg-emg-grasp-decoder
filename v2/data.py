"""Cached 500 Hz series -> 50 Hz tokens, normalisation, training windows."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from main.dataloader import stratified_split
from main.eeg_features import eeg_bandpower

from .history import history_features, hold_ids
from .tokens import hold_to_tokens, to_tokens

N_KIN = 24
_KIN_DROP = (12, 25)          # rho_GL and its velocity in the cached [pos13 | vel13] layout
_F_L, _F_G = 10, 11           # gated load and grip force columns of the cached 26-column array


def cache_path(cache_dir, participant: int, split: str, series) -> Path:
    return Path(cache_dir) / f"P{int(participant)}_{split}_S{series}.npz"


def load_split(cache_dir, participant: int, split: str, data_dir=None) -> list[dict]:
    out = []
    for sid in stratified_split(int(participant), data_dir)[split]:
        p = cache_path(cache_dir, participant, split, sid)
        if not p.exists():
            raise FileNotFoundError(
                f"missing cache file {p.name} in {Path(cache_dir)}: run the pre-processing cell of "
                f"main_1stage.ipynb for participant {participant} once to build it")
        with np.load(p) as d:
            out.append({"series": int(sid), "eeg": d["eeg"].astype(np.float32),
                        "kin": d["kin"].astype(np.float32), "emg": d["emg"].astype(np.float32)})
    return out


def build_tokens(raw: dict, cfg: dict) -> dict:
    """One cached 500 Hz series -> token arrays.

    cfg["tokens"]["causal"] (default False) builds every input from past samples only (v2.causal);
    cfg["tokens"]["horizon"] (default 0 tokens) pairs the inputs at token t with the target at t + horizon.
    """
    fs = float(cfg["data"]["fs"])
    tc = cfg.get("tokens", {})
    causal, horizon = bool(tc.get("causal", False)), int(tc.get("horizon", 0))
    kin26, eeg, emg = raw["kin"], raw["eeg"], raw["emg"]
    if kin26.shape[1] != 26:
        raise ValueError(f"expected the 26-column cached kinematics, got {kin26.shape[1]} columns")
    T = min(len(kin26), len(eeg), len(emg))
    kin26, eeg, emg = kin26[:T], eeg[:T], emg[:T]
    h = cfg["history"]
    hist, holds = history_features(kin26[:, _F_L], kin26[:, _F_G], fs, h["force_threshold"], h["min_hold_s"])
    bands = cfg["eeg"]["bands"]
    if causal:
        from .causal import bandpower_causal, causal_kinematics, to_tokens_causal
        bp = bandpower_causal(eeg, fs, bands, smooth_hz=cfg["eeg"]["smooth_hz"])
        kin26, in_tokens = causal_kinematics(kin26, fs), to_tokens_causal
    else:
        bp = eeg_bandpower(eeg, fs, bands=bands, smooth_hz=cfg["eeg"]["smooth_hz"])    # (T, n_bands * C), band-major
        in_tokens = to_tokens
    C = eeg.shape[1]
    grid = np.asarray(bp).reshape(T, len(bands), C).transpose(0, 2, 1)                 # (T, C, n_bands)
    eeg_tok = in_tokens(grid.reshape(T, -1)).reshape(-1, C, len(bands))
    tok = {
        "series": raw.get("series"),
        "kin": in_tokens(np.delete(kin26, _KIN_DROP, axis=1)),
        "hist": hold_to_tokens(hist).astype(np.float32),
        "eeg": eeg_tok,
        "emg": np.maximum(to_tokens(emg), 0.0),
        "hold_id": hold_to_tokens(hold_ids(T, holds)),
        "emg_full": np.ascontiguousarray(emg, dtype=np.float32),
        "n_samples": int(T),
    }
    if horizon:
        from .causal import shift_horizon
        tok = shift_horizon(tok, horizon)
    return tok


def fit_norm(train_tokens: list[dict]) -> dict:
    kin = np.concatenate([t["kin"] for t in train_tokens]).astype(np.float64)
    eeg = np.concatenate([t["eeg"] for t in train_tokens]).astype(np.float64)
    emg = np.concatenate([t["emg"] for t in train_tokens]).astype(np.float64)
    return {
        "kin_mean": kin.mean(0).astype(np.float32), "kin_std": np.maximum(kin.std(0), 1e-6).astype(np.float32),
        "eeg_mean": eeg.mean(0).astype(np.float32), "eeg_std": np.maximum(eeg.std(0), 1e-6).astype(np.float32),
        "emg_scale": np.maximum(np.percentile(emg, 95, axis=0), 1e-8).astype(np.float32),
    }


def normalise(tok: dict, norm: dict, clip: float = 5.0):
    kin_z = (tok["kin"] - norm["kin_mean"]) / norm["kin_std"]
    eeg_z = np.clip((tok["eeg"] - norm["eeg_mean"]) / norm["eeg_std"], -clip, clip)
    return kin_z.astype(np.float32), eeg_z.astype(np.float32)


def to_model_domain(y, scale):
    """u = sqrt(max(y, 0) / scale); works for NumPy arrays and torch tensors."""
    if isinstance(y, torch.Tensor):
        return torch.sqrt(torch.clamp(y, min=0.0) / torch.as_tensor(scale, dtype=y.dtype, device=y.device))
    return np.sqrt(np.maximum(y, 0.0) / scale).astype(np.float32)


def from_model_domain(u, scale):
    if isinstance(u, torch.Tensor):
        return torch.as_tensor(scale, dtype=u.dtype, device=u.device) * u ** 2
    return (scale * np.asarray(u) ** 2).astype(np.float32)


class TokenWindows(Dataset):
    """Fixed-length token windows over a list of series; inputs are normalised once at construction."""

    def __init__(self, tokens, norm, window=400, stride=100, random_offset=False, seed=0, clip=5.0):
        self.window, self.stride, self.random_offset, self.seed = int(window), int(stride), bool(random_offset), int(seed)
        self.series = []
        for t in tokens:
            kin_z, eeg_z = normalise(t, norm, clip)
            self.series.append({
                "kin": torch.from_numpy(kin_z), "hist": torch.from_numpy(t["hist"]), "eeg": torch.from_numpy(eeg_z),
                "target": torch.from_numpy(to_model_domain(t["emg"], norm["emg_scale"])),
                "hold_id": torch.from_numpy(t["hold_id"].astype(np.int64)),
            })
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        off = 0
        if self.random_offset:
            off = int(np.random.default_rng(self.seed + int(epoch)).integers(0, self.stride))
        self.index = [(i, s) for i, ser in enumerate(self.series)
                      for s in range(off, ser["kin"].shape[0] - self.window + 1, self.stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, s = self.index[i]
        return {name: v[s:s + self.window] for name, v in self.series[k].items()}
