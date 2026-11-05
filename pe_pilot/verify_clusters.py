"""Verification of the pilot's cluster tests. Reads results/pe_pilot/cache, writes one JSON file (argv[1]); no training.

1. false-positive rate: labels shuffled within each participant and current weight -> clusters should appear in
   about 5% of runs
2. power: a known effect planted into the unexpected lifts of label-shuffled data -> detection rate per amplitude
3. lift-off-aligned re-analysis of the real labels
4. TFCE (threshold-free) on the real labels, touch-aligned
"""
import json
import sys
import time

import numpy as np
from mne.stats import permutation_cluster_1samp_test

from pe_pilot.config import CONFIG
from pe_pilot.data import load_epochs
from pe_pilot.stats import channel_adjacency, matched_contrast, time_adjacency_only

OUT = sys.argv[1]
RNG = np.random.default_rng(0)
NPERM = 200
eps = {p: load_epochs(p, results_dir="results/pe_pilot") for p in range(1, 13)}
any_ep = eps[1]
times = any_ep["times"]
names = list(any_ep["ch_names"])
ADJ = channel_adjacency(names)
ROI = [names.index(c) for c in ["C3", "CP1", "CP5", "FC1", "FC5", "Cz"]]
res = {}


def shuffled_table(t, kind):
    """Shuffle the labels expected/kind among usable lifts of the same current weight."""
    t = t.copy()
    ch = t["change"].to_numpy(object).copy()
    ok = ~t["rejected"].to_numpy(bool)
    w = t["CurW"].to_numpy(float)
    for cw in np.unique(w):
        idx = np.flatnonzero(ok & (w == cw) & np.isin(ch, ["expected", kind]))
        ch[idx] = RNG.permutation(ch[idx])
    t["change"] = ch
    return t


def stack(signal, kind, tables=None, plant=None):
    X = []
    for p, ep in eps.items():
        t = ep["table"] if tables is None else tables[p]
        x = ep[signal]
        if plant is not None:
            x = x.copy()
            x[(t["change"] == kind).to_numpy()] += plant
        d, n = matched_contrast(x, t, kind)
        if d is not None:
            X.append(d)
    return np.stack(X)


def clusters(X, adj, threshold=None):
    t_obs, cl, pv, _ = permutation_cluster_1samp_test(X, adjacency=adj, n_permutations=NPERM, tail=0,
                                                     threshold=threshold, out_type="mask", verbose=False,
                                                     seed=int(RNG.integers(1e9)))
    return t_obs, cl, pv


t0 = time.time()
# ---------------------------------------------------------------- 1. false-positive rate
N_NULL = 40
fp = {}
for kind in ("heavier", "lighter"):
    for signal in ("slow", "beta"):
        hits = 0
        for r in range(N_NULL):
            tabs = {p: shuffled_table(ep["table"], kind) for p, ep in eps.items()}
            _, cl, pv = clusters(stack(signal, kind, tabs), ADJ)
            hits += int(len(pv) and pv.min() < 0.05)
        fp[f"{kind}_{signal}"] = hits / N_NULL
        print("FP", kind, signal, hits / N_NULL, f"{time.time() - t0:.0f}s", flush=True)
res["false_positive_rate"] = fp
res["n_null_runs"] = N_NULL

# ---------------------------------------------------------------- 2. power with a planted effect
bump_t = np.exp(-0.5 * ((times - 0.30) / 0.05) ** 2)          # 300 ms, sd 50 ms
space = np.zeros(len(names)); space[ROI] = 1.0
# noise level: SD over participants of the real contrast at C3, 250-350 ms
real = {k: stack("slow", k) for k in ("heavier", "lighter")}
win = (times >= 0.25) & (times <= 0.35)
res["between_subject_sd_uV"] = {k: float(real[k][:, win][:, :, names.index("C3")].mean(1).std(ddof=1)) for k in real}
real_b = {k: stack("beta", k) for k in ("heavier", "lighter")}
res["between_subject_sd_dB"] = {k: float(real_b[k][:, win][:, :, names.index("C3")].mean(1).std(ddof=1)) for k in real_b}
N_POW = 8
power = {}
for signal, amps in (("slow", [3.0, 6.0, 10.0]), ("beta", [0.5, 1.0, 2.0])):
    for kind in ("heavier", "lighter"):
        for a in amps:
            plant = (a * bump_t[:, None] * space[None, :]).astype(np.float32)
            det = 0
            for r in range(N_POW):
                tabs = {p: shuffled_table(ep["table"], kind) for p, ep in eps.items()}
                _, cl, pv = clusters(stack(signal, kind, tabs, plant), ADJ)
                ok = False
                for m, pp in zip(cl, pv):
                    if pp < 0.05 and m[win][:, ROI].any():
                        ok = True
                det += int(ok)
            power[f"{signal}_{kind}_{a}"] = det / N_POW
            print("POWER", signal, kind, a, det / N_POW, f"{time.time() - t0:.0f}s", flush=True)
res["power"] = power
res["n_power_runs"] = N_POW

# ---------------------------------------------------------------- 3. lift-off-aligned
FS_EP = 100.0
rel = np.arange(-0.4, 1.0, 0.01)


def liftoff_epochs(ep, signal):
    t = ep["table"]
    lo = (t["tLiftOff"] - t["tBothDigitTouch"]).to_numpy(float)
    x = ep[signal]
    out = np.full((len(t), len(rel), x.shape[2]), np.nan, np.float32)
    for i, d in enumerate(lo):
        if not np.isfinite(d):
            continue
        idx = np.round((rel + d - times[0]) * FS_EP).astype(int)
        if idx.min() >= 0 and idx.max() < x.shape[1]:
            out[i] = x[i, idx]
    return out


lo_res = {}
for kind in ("heavier", "lighter"):
    for signal in ("slow", "alpha", "beta", "emg", "force"):
        X = []
        for p, ep in eps.items():
            x = liftoff_epochs(ep, signal)
            t = ep["table"].copy()
            t["rejected"] = t["rejected"].to_numpy(bool) | np.isnan(x).any(axis=(1, 2))
            d, n = matched_contrast(np.nan_to_num(x), t, kind)
            if d is not None:
                X.append(d)
        X = np.stack(X)
        adj = ADJ if signal in ("slow", "alpha", "beta") else time_adjacency_only(X.shape[2])
        _, cl, pv = clusters(X, adj)
        sig = [(float(rel[np.nonzero(m)[0].min()]), float(rel[np.nonzero(m)[0].max()]), float(pp),
                sorted({names[c] for c in np.nonzero(m)[1]}) if signal in ("slow", "alpha", "beta") else
                sorted({int(c) for c in np.nonzero(m)[1]}))
               for m, pp in zip(cl, pv) if pp < 0.1]
        lo_res[f"{kind}_{signal}"] = sorted(sig, key=lambda z: z[2])
        print("LIFTOFF", kind, signal, lo_res[f"{kind}_{signal}"][:3], flush=True)
res["liftoff_aligned"] = lo_res

# ---------------------------------------------------------------- 4. TFCE, touch-aligned
tfce = {}
for kind in ("heavier", "lighter"):
    for signal in ("slow", "alpha", "beta"):
        X = stack(signal, kind)
        t_obs, _, pv = clusters(X, ADJ, threshold=dict(start=0, step=0.25))
        pv = pv.reshape(t_obs.shape)
        k = np.unravel_index(np.argmin(pv), pv.shape)
        sigmask = pv < 0.05
        tfce[f"{kind}_{signal}"] = {"min_p": float(pv.min()), "at_s": float(times[k[0]]), "at_ch": names[k[1]],
                                    "n_sig_points": int(sigmask.sum()),
                                    "first_sig_s": float(times[np.nonzero(sigmask.any(1))[0].min()]) if sigmask.any() else None}
        print("TFCE", kind, signal, tfce[f"{kind}_{signal}"], flush=True)
res["tfce"] = tfce
json.dump(res, open(OUT, "w"), indent=1)
print("done", f"{time.time() - t0:.0f}s")
