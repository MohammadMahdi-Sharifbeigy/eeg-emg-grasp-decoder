"""Residual test: can the EEG predict what the kinematic stream misses?

F = K + EEG correction can only beat K when the EEG carries information about the residual (target - K).
This asks the question directly, without the fusion module: a ridge regression from EEG features to the
residual, fitted on series K was not trained on (the validation series) and scored on the test series,
against the same regression applied to EEG shifted in time. No effect here means F = K is the honest result
for this data; an effect here that F does not show means the fusion fails to learn it.
"""
import numpy as np
import pandas as pd
import torch

from .data import from_model_domain, normalise, to_model_domain
from .evaluate import _r, _vaf, swap_p
from .infer import predict_tokens, window_plan

FEATURES = ("tokens", "bandpower")
ALPHAS = tuple(10.0 ** np.arange(-2.0, 5.0))


@torch.no_grad()
def eeg_features(model, tok, norm, device, window=400, hop=200, clip=5.0, smooth=50) -> dict:
    """EEG features of one series at the token rate.

      tokens    (Tk, d_model)  output of the EEG stream's temporal encoder (what the fusion reads), stitched
                               like v2.infer (centre of each window kept)
      bandpower (Tk, E * B)    the normalised band power the model gets, mean over `smooth` tokens (1 s)
    """
    _, eeg_z = normalise(tok, norm, clip)
    n = eeg_z.shape[0]
    tokens = None
    model.eval()
    for start, lo, hi in window_plan(n, window, hop):
        x = torch.from_numpy(np.ascontiguousarray(eeg_z[start:min(start + window, n)]))[None].to(device)
        t = model.eeg(x)[1][0].float().cpu().numpy()
        if tokens is None:
            tokens = np.zeros((n, t.shape[-1]), dtype=np.float32)
        tokens[lo:hi] = t[lo - start:hi - start]
    flat = eeg_z.reshape(n, -1).astype(np.float64)
    csum = np.concatenate([np.zeros((1, flat.shape[1])), np.cumsum(flat, axis=0)])
    a, b = np.clip(np.arange(n) - smooth // 2, 0, n), np.clip(np.arange(n) + (smooth + 1) // 2, 0, n)
    return {"tokens": tokens, "bandpower": ((csum[b] - csum[a]) / (b - a)[:, None]).astype(np.float32)}


def _ridge(x, y, alpha):
    """Weights (d + 1, C) of ridge regression on standardised x with an intercept; alpha is per sample."""
    mu, sd = x.mean(0), x.std(0) + 1e-8
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + alpha * len(z) * np.eye(z.shape[1]), z.T @ (y - y.mean(0)))
    return mu, sd, w, y.mean(0)


def _apply(fit, x):
    mu, sd, w, b = fit
    return ((x - mu) / sd) @ w + b


def fit_ridge(x, y, alphas=ALPHAS, folds=5):
    """Ridge with alpha chosen by cross-validation over `folds` contiguous blocks (lowest squared error)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    edges = np.linspace(0, len(x), folds + 1).astype(int)
    err = np.zeros(len(alphas))
    for f in range(folds):
        held = np.zeros(len(x), dtype=bool)
        held[edges[f]:edges[f + 1]] = True
        for i, a in enumerate(alphas):
            err[i] += ((_apply(_ridge(x[~held], y[~held], a), x[held]) - y[held]) ** 2).sum()
    alpha = float(alphas[int(np.argmin(err))])
    return _ridge(x, y, alpha), alpha


def residual_test(model, fit_tokens, test_tokens, norm, cfg, device, names, n_null=None, seed=None) -> pd.DataFrame:
    """One row per feature set and muscle, plus a row "MEAN" (the test: mean over muscles against the null).

      r            correlation of the predicted and the real residual on the test series (model units)
      r_null_mean  the same with the prediction shifted in time by a whole number of hops (n_null draws).
                   The encoders add the position inside the window to every token, and the stitched K can
                   carry a small error with the same period; a shift by whole hops keeps that period, so the
                   null contains it and only information tied to the actual time counts. The band-power
                   features hold no position at all.
      p            one-sided empirical p of r against that null, never below 1 / n_shifts
      d_vaf        VAF of K + predicted residual minus VAF of K (original units, token rate, percentage points)
      d_vaf_null_mean, p_d_vaf   the same for the gain
      r_model      correlation of the model's own correction F_eeg with the residual, for reference
    """
    n_null = int(cfg["evaluation"]["n_null"] if n_null is None else n_null)
    rng = np.random.default_rng(cfg["evaluation"]["seed"] if seed is None else seed)
    tc = cfg["tokens"]
    kw = dict(window=tc["window"], hop=tc["hop"], clip=cfg["eeg"]["clip"])
    scale = norm["emg_scale"]

    def collect(toks):
        feats, k_all, f_all, y_all = {k: [] for k in FEATURES}, [], [], []
        for tok in toks:
            pt = predict_tokens(model, tok, norm, device, **kw)
            fe = eeg_features(model, tok, norm, device, **kw)
            for k in FEATURES:
                feats[k].append(fe[k])
            k_all.append(pt["K"]); f_all.append(pt["F"]); y_all.append(np.asarray(tok["emg"], np.float32))
        cat = np.concatenate
        return {k: cat(v) for k, v in feats.items()}, cat(k_all), cat(f_all), cat(y_all)

    x_fit, k_fit, _, y_fit = collect(fit_tokens)
    x_te, k_te, f_te, y_te = collect(test_tokens)
    res_fit = to_model_domain(y_fit, scale).astype(np.float64) - k_fit
    res_te = to_model_domain(y_te, scale).astype(np.float64) - k_te
    vaf_k = _vaf(from_model_domain(k_te, scale), y_te)
    gain = lambda pred: _vaf(from_model_domain(np.maximum(k_te + pred, 0.0), scale), y_te) - vaf_k
    r_model = _r(f_te - k_te, res_te)
    n, hop = len(res_te), int(tc["hop"])
    shifts = np.arange(max(tc["window"] // hop, 1), n // hop - max(tc["window"] // hop, 1) + 1) * hop
    if len(shifts) < 2:
        raise ValueError(f"the test series have {n} tokens: too short for a time-shift null with hop {hop}")
    p_of = lambda obs, null: swap_p(float(obs), null, len(shifts))
    rows = []
    for feat in FEATURES:
        fit, alpha = fit_ridge(x_fit[feat], res_fit)
        pred = _apply(fit, x_te[feat].astype(np.float64))
        r, d = _r(pred, res_te), gain(pred)
        r_null, d_null = np.zeros((n_null, len(names))), np.zeros((n_null, len(names)))
        for i in range(n_null):
            alt = np.roll(pred, int(rng.choice(shifts)), axis=0)                 # the same prediction, wrong time
            r_null[i], d_null[i] = _r(alt, res_te), gain(alt)
        common = {"features": feat, "alpha": alpha, "n_fit": len(res_fit), "n_test": n, "n_null": n_null, "n_shifts": len(shifts)}
        for c, name in enumerate(names):
            rows.append({**common, "muscle": name, "r": r[c], "r_null_mean": r_null[:, c].mean(),
                         "p": p_of(r[c], r_null[:, c]), "d_vaf": d[c], "d_vaf_null_mean": d_null[:, c].mean(),
                         "p_d_vaf": p_of(d[c], d_null[:, c]), "r_model": r_model[c]})
        rows.append({**common, "muscle": "MEAN", "r": r.mean(), "r_null_mean": r_null.mean(),
                     "p": p_of(r.mean(), r_null.mean(1)), "d_vaf": d.mean(), "d_vaf_null_mean": d_null.mean(),
                     "p_d_vaf": p_of(d.mean(), d_null.mean(1)), "r_model": r_model.mean()})
    return pd.DataFrame(rows)
