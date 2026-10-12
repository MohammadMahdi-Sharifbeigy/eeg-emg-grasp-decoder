"""Stitched inference: 8 s windows, 4 s hop, the centre of each window is kept."""
import numpy as np
import torch

from .data import from_model_domain, normalise
from .tokens import to_samples


def window_plan(n_tok: int, window: int = 400, hop: int = 200) -> list[tuple[int, int, int]]:
    if n_tok <= window:
        return [(0, 0, n_tok)]
    starts = list(range(0, n_tok - window + 1, hop))
    if starts[-1] != n_tok - window:
        starts.append(n_tok - window)
    margin = (window - hop) // 2
    plan, prev_hi = [], 0
    for i, s in enumerate(starts):
        hi = n_tok if i == len(starts) - 1 else s + window - margin
        plan.append((s, prev_hi, hi))
        prev_hi = hi
    return plan


@torch.no_grad()
def predict_tokens(model, tok, norm, device, window=400, hop=200, use_amp=False, eeg=None, hist=None, clip=5.0):
    kin_z, eeg_z = normalise(tok, norm, clip)
    eeg_z = eeg_z if eeg is None else np.asarray(eeg, dtype=np.float32)
    hist_a = tok["hist"] if hist is None else np.asarray(hist, dtype=np.float32)
    n = kin_z.shape[0]
    out = {k: np.zeros((n, 5), dtype=np.float32) for k in ("K", "E", "F")}
    model.eval()
    for start, lo, hi in window_plan(n, window, hop):
        sl = slice(start, min(start + window, n))
        with torch.autocast(device.type, enabled=use_amp and device.type == "cuda"):
            res = model(torch.from_numpy(kin_z[sl])[None].to(device), torch.from_numpy(hist_a[sl])[None].to(device),
                        torch.from_numpy(eeg_z[sl])[None].to(device))
        for k in out:
            out[k][lo:hi] = res[k][0, lo - start:hi - start].float().cpu().numpy()
    return out


def predict_series(model, tok, norm, device, **kw):
    pt = predict_tokens(model, tok, norm, device, **kw)
    return {k: np.maximum(to_samples(from_model_domain(v, norm["emg_scale"]), tok["n_samples"]), 0.0)
            for k, v in pt.items()}
