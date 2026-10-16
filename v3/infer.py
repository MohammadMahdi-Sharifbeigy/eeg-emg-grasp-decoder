"""Stitched inference of the EEG-only student: 8 s windows, 4 s hop, the centre of each window is kept."""
import numpy as np
import torch

from v2.data import from_model_domain
from v2.infer import window_plan
from v2.tokens import to_samples


def normalise_eeg(tok: dict, norm: dict, clip: float = 5.0) -> np.ndarray:
    return np.clip((tok["eeg"] - norm["eeg_mean"]) / norm["eeg_std"], -clip, clip).astype(np.float32)


@torch.no_grad()
def predict_eeg_tokens(model, tok, norm, device, window=400, hop=200, use_amp=False, eeg=None, clip=5.0):
    """Model-domain E for one series, (n_tok, 5). `eeg` replaces the normalised EEG (already normalised and clipped)."""
    eeg_z = normalise_eeg(tok, norm, clip)
    if eeg is not None:
        eeg = np.asarray(eeg, dtype=np.float32)
        if eeg.shape != eeg_z.shape:
            raise ValueError(f"eeg override has shape {eeg.shape}, expected {eeg_z.shape}")
        eeg_z = eeg
    n = eeg_z.shape[0]
    out = np.zeros((n, 5), dtype=np.float32)
    model.eval()
    for start, lo, hi in window_plan(n, window, hop, causal=getattr(model, "causal", False)):
        sl = slice(start, min(start + window, n))
        with torch.autocast(device.type, enabled=use_amp and device.type == "cuda"):
            res = model(torch.from_numpy(eeg_z[sl])[None].to(device))
        out[lo:hi] = res["E"][0, lo - start:hi - start].float().cpu().numpy()
    return out


def predict_eeg_series(model, tok, norm, device, **kw) -> np.ndarray:
    pt = predict_eeg_tokens(model, tok, norm, device, **kw)
    return np.maximum(to_samples(from_model_domain(pt, norm["emg_scale"]), tok["n_samples"]), 0.0)
