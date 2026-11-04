"""The four analyses of the pilot. Each takes {participant: epochs} (pe_pilot.data.load_epochs) and returns tables.

1. behaviour   : do the forces show the classic weight-expectation effects? (sanity check of the labels)
2. error       : EEG of unexpectedly heavier / lighter lifts minus expected lifts of the same weight, touch-aligned,
                 next to the same contrast on EMG and force, so the EEG effect can be dated against the movement
3. expectation : does the EEG before contact tell the previous weight (the expected weight)? with EMG, hand-speed,
                 pre-cue and peripheral-channel decoders as controls
4. learning    : does the size of the EEG response on an unexpected lift predict how much the grip force rate
                 changes on the next lift? (exploratory)
"""
import numpy as np
import pandas as pd

from pe_pilot.config import CONFIG
from pe_pilot.stats import (channel_adjacency, cluster_test, decode, group_wilcoxon, matched_contrast,
                            partial_spearman, time_adjacency_only)

SIGNALS_EEG = ("slow", "alpha", "beta")
SIGNALS_BODY = ("emg", "force")


def behaviour(epochs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per participant: weight-matched difference (unexpected minus expected) of dGF_Max, GF_Max, Dur_LoadPhase.

    Expected direction (Johansson & Westling 1988): heavier -> longer load phase and lower grip force rate than the
    weight needs; lighter -> higher grip force rate.
    """
    rows = []
    for p, ep in epochs.items():
        for kind in ("heavier", "lighter"):
            for col in ("dGF_Max", "GF_Max", "Dur_LoadPhase"):
                t = ep["table"].assign(rejected=ep["table"][col].isna())   # behaviour does not depend on the EEG
                d, n = matched_contrast(t[col].to_numpy(float)[:, None], t, kind)
                rows.append({"participant": p, "kind": kind, "measure": col,
                             "diff": np.nan if d is None else float(d[0]), "n_unexpected": n})
    per = pd.DataFrame(rows)
    group = pd.DataFrame([{"kind": k, "measure": m, **group_wilcoxon(g["diff"])}
                          for (k, m), g in per.groupby(["kind", "measure"])])
    return per, group


def contrasts(epochs: dict, kind: str, signal: str) -> tuple[np.ndarray, list]:
    """Stack the matched contrast of one signal over participants: (n_participants, n_times, n_channels)."""
    out, used = [], []
    for p, ep in epochs.items():
        d, n = matched_contrast(ep[signal], ep["table"], kind)
        if d is not None:
            out.append(d)
            used.append(p)
    return np.stack(out), used


def first_time(times: np.ndarray, clusters: pd.DataFrame, alpha: float) -> float:
    sig = clusters[clusters["p"] < alpha]
    return float(times[sig["t_first"].min()]) if len(sig) else np.nan


def error(epochs: dict) -> tuple[pd.DataFrame, dict]:
    """Cluster tests of the matched contrast per kind and signal.

    Returns a summary (one row per kind and signal: the earliest significant cluster time and the number of
    significant clusters) and the group-mean contrasts with their cluster tables for the figures.
    """
    any_ep = next(iter(epochs.values()))
    times, names = any_ep["times"], list(any_ep["ch_names"])
    adj_eeg = channel_adjacency(names)
    rows, detail = [], {}
    for kind in ("heavier", "lighter"):
        for signal in SIGNALS_EEG + SIGNALS_BODY:
            X, used = contrasts(epochs, kind, signal)
            adj = adj_eeg if signal in SIGNALS_EEG else time_adjacency_only(X.shape[2])
            cl = cluster_test(X, adj, CONFIG["n_permutations"])
            sig = cl[cl["p"] < CONFIG["cluster_p"]]
            rows.append({"kind": kind, "signal": signal, "n_participants": len(used),
                         "n_clusters": len(sig), "first_s": first_time(times, cl, CONFIG["cluster_p"]),
                         "best_p": float(cl["p"].min()) if len(cl) else np.nan})
            detail[(kind, signal)] = {"mean": X.mean(0), "clusters": cl, "participants": used}
    return pd.DataFrame(rows), {"times": times, "ch_names": names, "emg_names": list(any_ep["emg_names"]),
                                "contrasts": detail}


DECODERS = {                                         # name -> (feature key, channel filter)
    "eeg_pre_contact": ("eeg_pre", "central"),
    "eeg_pre_contact_peripheral": ("eeg_pre", "peripheral"),
    "eeg_pre_cue": ("eeg_cue", "central"),
    "emg_pre_contact": ("emg_pre", None),
    "hand_speed_pre_contact": ("kin_pre", None),
}


def _select(ep, key, which):
    X = ep[key]
    if which is None:
        return X
    names = list(ep["ch_names"])
    per = [n in CONFIG["peripheral"] for n in names]
    keep = np.flatnonzero(np.asarray(per) if which == "peripheral" else ~np.asarray(per))
    n_ch = len(names)                                # eeg features are [slow | alpha | beta] per channel
    return X[:, np.concatenate([keep + k * n_ch for k in range(X.shape[1] // n_ch)])]


def expectation(epochs: dict, light: float = 1, heavy: float = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode the previous weight (light vs heavy) from features before contact, leave one series out."""
    rows = []
    for p, ep in epochs.items():
        t = ep["table"]
        keep = (~t["rejected"].to_numpy(bool)) & t["PrevW"].isin([light, heavy]).to_numpy()
        y = (t["PrevW"].to_numpy(float)[keep] == heavy).astype(int)
        groups = t["Run"].to_numpy(int)[keep]
        for name, (key, which) in DECODERS.items():
            res = decode(_select(ep, key, which)[keep], y, groups, CONFIG["n_label_shuffles"], seed=p)
            rows.append({"participant": p, "decoder": name, **res})
    per = pd.DataFrame(rows)
    group = pd.DataFrame([{"decoder": d, **group_wilcoxon(g["auc"], 0.5, "greater")}
                          for d, g in per.groupby("decoder", sort=False)])
    return per, group


def error_size(ep: dict) -> np.ndarray:
    """Single-lift error response: beta dB change in the a-priori window and region (see CONFIG)."""
    names = list(ep["ch_names"])
    roi = [names.index(c) for c in CONFIG["error_roi"]]
    tw = (ep["times"] >= CONFIG["error_window"][0]) & (ep["times"] <= CONFIG["error_window"][1])
    return ep["beta"][:, tw][:, :, roi].mean(axis=(1, 2))


def learning(epochs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per participant, over unexpected lifts followed by a lift of the same weight: partial Spearman between the
    EEG error size and the change of dGF_Max on the next lift, controlling the current lift's peak grip force, load
    phase duration and EMG peak (so a bigger movement does not explain both)."""
    rows = []
    for p, ep in epochs.items():
        t = ep["table"]
        keep = ((~t["rejected"]) & t["change"].isin(["heavier", "lighter"]) & (t["next_W"] == t["CurW"])
                & t["next_dGF"].notna() & t["dGF_Max"].notna()).to_numpy(bool)
        if keep.sum() < 8:
            rows.append({"participant": p, "rho": np.nan, "n": int(keep.sum())})
            continue
        sub = t[keep]
        update = (sub["next_dGF"] - sub["dGF_Max"]).to_numpy(float)
        # the sign of the update flips between heavier and lighter: compare sizes
        size = np.abs(error_size(ep)[keep])
        cov = np.column_stack([sub["GF_Max"], sub["Dur_LoadPhase"], ep["emg"][keep].max(axis=(1, 2))])
        rows.append({"participant": p, "rho": partial_spearman(size, np.abs(update), cov), "n": int(keep.sum())})
    per = pd.DataFrame(rows)
    return per, pd.DataFrame([group_wilcoxon(np.arctanh(np.clip(per["rho"], -0.999, 0.999)))])
