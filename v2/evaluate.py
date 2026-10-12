"""Evaluation of KG-GT v2 on the test series, scored at 500 Hz against the original target."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from main.dataloader import series_conditions
from main.magnitude_eval import lift_hold_segments, magnitude_by_load, magnitude_report_by_series

from .data import normalise
from .infer import predict_series

OUTPUTS = ("K", "E", "F")


def _vaf(pred, target):
    return 100.0 * (1.0 - np.var(target - pred, axis=0) / np.maximum(np.var(target, axis=0), 1e-20))


def _r(pred, target):
    return np.array([np.corrcoef(pred[:, c], target[:, c])[0, 1] if pred[:, c].std() > 0 else 0.0
                     for c in range(target.shape[1])])


def standard_metrics(pred, target, names) -> pd.DataFrame:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    pm, tm = pred.mean(0), target.mean(0)
    cov = ((pred - pm) * (target - tm)).mean(0)
    ccc = 2 * cov / (pred.var(0) + target.var(0) + (pm - tm) ** 2 + 1e-20)
    return pd.DataFrame({"muscle": list(names), "r": _r(pred, target), "ccc": ccc,
                         "rmse": np.sqrt(((pred - target) ** 2).mean(0)), "vaf": _vaf(pred, target)})


def smooth_ceiling(target, names, fs: float = 500.0, cutoff_hz: float = 2.0) -> pd.DataFrame:
    target = np.asarray(target, np.float64)
    smooth = sosfiltfilt(butter(4, cutoff_hz, "low", fs=fs, output="sos"), target, axis=0)
    return pd.DataFrame({"muscle": list(names), "ceiling_vaf": _vaf(smooth, target)})


def modulation_depth(target, names) -> pd.DataFrame:
    lo, hi = np.percentile(target, 20, axis=0), np.percentile(target, 80, axis=0)
    return pd.DataFrame({"muscle": list(names), "modulation": hi / np.maximum(lo, 1e-12)})


def block_shuffle(x, block: int, rng) -> np.ndarray:
    out = np.array(x, copy=True)
    for a in range(0, len(out), block):
        out[a:a + block] = out[a:a + block][rng.permutation(len(out[a:a + block]))]
    return out


def block_swap(x, block: int, rng) -> np.ndarray:
    out = np.array(x, copy=True)
    n = len(out) // block
    if n < 2:
        return out
    perm = np.roll(np.arange(n), int(rng.integers(1, n)))                     # a rotation is a derangement
    for dst, src in enumerate(perm):
        out[dst * block:(dst + 1) * block] = np.asarray(x)[src * block:(src + 1) * block]
    return out


def empirical_p(observed: float, null) -> float:
    null = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def baseline_comparison(standard_v2: pd.DataFrame, subject: str, path="results/magnitude_report_1stage.csv"):
    path = Path(path)
    if not path.exists():
        return None
    base = pd.read_csv(path)
    base = base[(base.subject == subject) & (base.group == "all test") & (base.muscle != "MEAN")]
    if base.empty:
        return None
    k = standard_v2[standard_v2.output == "K"].set_index("muscle")
    b = base.set_index("muscle").loc[k.index]
    return pd.DataFrame({"muscle": k.index, "baseline_r": b.r.values, "v2_K_r": k.r.values,
                         "baseline_vaf": b.vaf.values, "v2_K_vaf": k.vaf.values,
                         "d_vaf": k.vaf.values - b.vaf.values}).reset_index(drop=True)


def evaluate_subject(model, test_tokens, norm, cfg, device, participant, names, data_dir=None) -> dict:
    fs = float(cfg["data"]["fs"])
    win, hop = cfg["tokens"]["window"], cfg["tokens"]["hop"]
    use_amp = bool(cfg["training"]["use_amp"])
    kw = dict(window=win, hop=hop, use_amp=use_amp, clip=cfg["eeg"]["clip"])
    preds = {}
    for tok in test_tokens:
        p = predict_series(model, tok, norm, device, **kw)
        preds[tok["series"]] = {**p, "target": tok["emg_full"]}
    cat = lambda key: np.concatenate([preds[s][key] for s in preds])
    target = cat("target")
    cond = series_conditions(int(participant), data_dir)

    standard = pd.concat([standard_metrics(cat(o), target, names).assign(output=o) for o in OUTPUTS], ignore_index=True)
    magnitude = pd.concat([
        magnitude_report_by_series({s: (preds[s][o], preds[s]["target"]) for s in preds}, cond, fs=fs,
                                   channel_names=names).assign(output=o) for o in OUTPUTS], ignore_index=True)
    by_load = []
    for s in preds:
        seg = lift_hold_segments(int(participant), int(s), fs=fs, data_dir=data_dir, n_samples=len(preds[s]["target"]))
        for o in OUTPUTS:
            for group in ("CurW", "CurS"):
                by_load.append(magnitude_by_load(preds[s][o], preds[s]["target"], seg, group=group,
                                                 channel_names=names).assign(output=o, series=s))
    ceiling = smooth_ceiling(target, names, fs).merge(modulation_depth(target, names), on="muscle")

    # nulls: EEG shuffled inside each window-sized block, and EEG blocks swapped across the test tokens
    rng = np.random.default_rng(cfg["evaluation"]["seed"])
    n_null = int(cfg["evaluation"]["n_null"])
    eeg_z = [normalise(t, norm, cfg["eeg"]["clip"])[1] for t in test_tokens]
    lens = np.cumsum([0] + [len(e) for e in eeg_z])
    all_eeg = np.concatenate(eeg_z)
    r_e, vaf_k, vaf_f = _r(cat("E"), target), _vaf(cat("K"), target), _vaf(cat("F"), target)
    null = {"E_shuffle": [], "E_swap": [], "dF_swap": []}
    for _ in range(n_null):
        for kind, fn in (("shuffle", block_shuffle), ("swap", block_swap)):
            alt = fn(all_eeg, win, rng)
            e_parts, f_parts = [], []
            for i, tok in enumerate(test_tokens):
                p = predict_series(model, tok, norm, device, eeg=alt[lens[i]:lens[i + 1]], **kw)
                e_parts.append(p["E"])
                f_parts.append(p["F"])
            null[f"E_{kind}"].append(_r(np.concatenate(e_parts), target))
            if kind == "swap":
                null["dF_swap"].append(_vaf(np.concatenate(f_parts), target) - vaf_k)
    null = {k: np.asarray(v) for k, v in null.items()}
    gate = model.fusion.gate.detach().float().cpu().numpy()
    d_f = vaf_f - vaf_k
    nulls = pd.DataFrame({
        "muscle": list(names), "E_r": r_e,
        "E_r_shuffle_mean": null["E_shuffle"].mean(0), "E_r_shuffle_sd": null["E_shuffle"].std(0),
        "E_r_swap_mean": null["E_swap"].mean(0), "E_r_swap_sd": null["E_swap"].std(0),
        "E_p_swap": [empirical_p(r_e[c], null["E_swap"][:, c]) for c in range(len(names))],
        "dF_vaf": d_f, "dF_vaf_swap_mean": null["dF_swap"].mean(0), "dF_vaf_swap_sd": null["dF_swap"].std(0),
        "dF_p_swap": [empirical_p(d_f[c], null["dF_swap"][:, c]) for c in range(len(names))],
        "gate": gate,
    })

    # history check: K with the history of another test series (tiled or cut to length)
    hist_rows = pd.DataFrame({"muscle": list(names), "K_vaf": vaf_k})
    if len(test_tokens) >= 2:
        parts = []
        for i, tok in enumerate(test_tokens):
            other = test_tokens[(i + 1) % len(test_tokens)]["hist"]
            reps = int(np.ceil(len(tok["hist"]) / len(other)))
            parts.append(predict_series(model, tok, norm, device, hist=np.tile(other, (reps, 1))[:len(tok["hist"])], **kw)["K"])
        hist_rows["K_vaf_other_history"] = _vaf(np.concatenate(parts), target)
    return {"standard": standard, "magnitude": magnitude, "by_load": pd.concat(by_load, ignore_index=True),
            "ceiling": ceiling, "nulls": nulls, "history": hist_rows, "predictions": preds}
