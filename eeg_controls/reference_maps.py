"""Experiment 4: reference maps from classical analyses, compared with the rollout maps of E.

The rollout maps say which electrodes E reads. Two classical analyses say where scalp EEG is tied to
the movement, without any model:

  ERD   event-related (de)synchronisation: percent change of alpha (8-13 Hz) and beta (13-30 Hz) power
        in a window before the event, against a baseline before the hand starts to move:
            ERD% = 100 * (P_window / P_baseline - 1)          (negative = desynchronisation)
        baseline  [hand_start - 2 s, hand_start - 1 s]
        windows   [event - 1 s, event] for event = first_touch and lift_off, and the hold
                  [lift_off + 0.5 s, lift_off + 1.5 s] (beta rebound / steady grip)
        Power: zero-phase band-pass, squared Hilbert envelope, 5 Hz smoothing, on the preprocessed EEG of
        the cache. One value per lift; the map is the median over lifts.
  CMC   corticomuscular coherence between each electrode and each muscle in the hold
        [lift_off, replace]: raw EMG (4000 Hz) band-passed 30-300 Hz, rectified, reduced to 500 Hz; 1 s
        epochs (Hann window), coherence = |mean S_xy|^2 / (mean S_xx * mean S_yy), mean over 15-30 Hz.
        The 95% significance level of one coherence value is 1 - 0.05^(1 / (L - 1)) for L epochs.

Comparison, per participant: Spearman correlation over the 32 electrodes between the rollout share
of E (outputs/interpretability_v2/P*/electrodes.csv, read only) and each reference map:
  - rollout (and gradient-weighted rollout, mean over muscles) vs -ERD% (more desynchronisation = more
    expected importance), for alpha and beta, at the same event;
  - gradient-weighted rollout of muscle m vs CMC with muscle m (and the mean over muscles).
Group: mean rho over participants with a sign-flip permutation p (does the correlation hold across
participants?), and the correlation of the group-mean maps with an electrode-shuffle p.

Output (default eeg_controls/results/reference/): reference_maps.csv (every map, long format),
rollout_vs_reference.csv (per participant), group.csv, and two figures (group_erd.png, group_cmc.png).

Run from the repository root:
    env/Scripts/python.exe -m eeg_controls.reference_maps --participants all
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, hilbert, sosfiltfilt
from scipy.stats import spearmanr

from main.dataloader import emg_channel_names, load_hs
from v2.data import load_split
from v2.electrodes import EEG_NAMES

from .common import ROOT, cache_dir, data_dir, lifts, parse_subjects

FS = 500.0
BANDS = {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
WINDOWS = {"first_touch": ("first_touch", -1.0, 0.0), "lift_off": ("lift_off", -1.0, 0.0),
           "hold": ("lift_off", 0.5, 1.5)}
BASELINE = ("hand_start", -2.0, -1.0)
CMC_BAND = (15.0, 30.0)
EPOCH = 500                                                        # 1 s at 500 Hz
IV2 = ROOT / "outputs" / "interpretability_v2"
RNG = np.random.default_rng(0)
N_PERM = 5000


# ----------------------------------------------------------------------------------- ERD
def band_power(eeg: np.ndarray, band, smooth_hz: float = 5.0) -> np.ndarray:
    """Linear band power (T, C): zero-phase band-pass, squared Hilbert envelope, zero-phase 5 Hz smoothing."""
    x = sosfiltfilt(butter(4, band, btype="bandpass", fs=FS, output="sos"), np.asarray(eeg, np.float64), axis=0)
    p = np.abs(hilbert(x, axis=0)) ** 2
    return sosfiltfilt(butter(1, smooth_hz, fs=FS, output="sos"), p, axis=0)


def _mean_in(p, lift, ev, d0, d1):
    if lift.get(ev) is None:
        return None
    a, b = int(lift[ev] + d0 * FS), int(lift[ev] + d1 * FS)
    if a < 0 or b > len(p) or b <= a:
        return None
    return p[a:b].mean(0)


def erd_rows(participant: int, series_list) -> list[dict]:
    """Per-lift ERD% for every band, window and electrode of the given series."""
    rows = []
    for raw in series_list:
        lift_list = lifts(participant, raw["series"], len(raw["eeg"]))
        for band, edges in BANDS.items():
            p = band_power(raw["eeg"], edges)
            for lift in lift_list:
                base = _mean_in(p, lift, *BASELINE)
                if base is None:
                    continue
                for win, (ev, d0, d1) in WINDOWS.items():
                    w = _mean_in(p, lift, ev, d0, d1)
                    if w is None:
                        continue
                    erd = 100.0 * (w / np.maximum(base, 1e-20) - 1.0)
                    rows += [{"series": raw["series"], "lift": lift["lift"], "band": band, "window": win,
                              "electrode": e, "erd": float(v)} for e, v in zip(EEG_NAMES, erd)]
    return rows


# ----------------------------------------------------------------------------------- CMC
def emg_for_cmc(participant: int, series: int, n: int) -> np.ndarray:
    """Raw EMG -> band-pass 30-300 Hz (zero phase) -> rectified -> 500 Hz, cut to n samples."""
    hs = load_hs(data_dir() / f"P{participant}" / f"HS_P{participant}_S{series}.mat")
    fs = float(hs["fs_emg"])
    x = sosfiltfilt(butter(4, (30.0, 300.0), btype="bandpass", fs=fs, output="sos"),
                    np.asarray(hs["emg"], np.float64), axis=0)
    x = np.abs(x)
    q = int(round(fs / FS))
    x = decimate(x, q, axis=0, zero_phase=True)
    if len(x) < n:
        raise ValueError(f"P{participant} S{series}: EMG at 500 Hz has {len(x)} samples, the cache {n}")
    return x[:n]


def hold_epochs(lift_list, n: int) -> list[int]:
    """Start samples of non-overlapping 1 s epochs inside [lift_off, replace] of every lift."""
    starts = []
    for lift in lift_list:
        a, b = lift.get("lift_off"), lift.get("replace")
        if a is None or b is None:
            continue
        starts += list(range(int(a), min(int(b), n) - EPOCH + 1, EPOCH))
    return starts


def cmc(eeg_epochs: np.ndarray, emg_epochs: np.ndarray) -> np.ndarray:
    """(C_eeg, C_emg) mean coherence over CMC_BAND. Epoch arrays are (L, EPOCH, C)."""
    win = np.hanning(EPOCH)[None, :, None]
    fx = np.fft.rfft((eeg_epochs - eeg_epochs.mean(1, keepdims=True)) * win, axis=1)
    fy = np.fft.rfft((emg_epochs - emg_epochs.mean(1, keepdims=True)) * win, axis=1)
    f = np.fft.rfftfreq(EPOCH, 1.0 / FS)
    sel = (f >= CMC_BAND[0]) & (f <= CMC_BAND[1])
    fx, fy = fx[:, sel], fy[:, sel]                                  # (L, F, C)
    sxy = np.einsum("lfe,lfm->fem", fx, np.conj(fy)) / len(fx)
    sxx = (np.abs(fx) ** 2).mean(0)                                  # (F, C_eeg)
    syy = (np.abs(fy) ** 2).mean(0)                                  # (F, C_emg)
    coh = np.abs(sxy) ** 2 / (sxx[:, :, None] * syy[:, None, :] + 1e-30)
    return coh.mean(0)


def cmc_map(participant: int, series_list) -> tuple[np.ndarray, int]:
    eeg_ep, emg_ep = [], []
    for raw in series_list:
        n = len(raw["eeg"])
        emg = emg_for_cmc(participant, raw["series"], n)
        for s in hold_epochs(lifts(participant, raw["series"], n), n):
            eeg_ep.append(raw["eeg"][s:s + EPOCH])
            emg_ep.append(emg[s:s + EPOCH])
    if len(eeg_ep) < 2:
        raise ValueError(f"P{participant}: fewer than two hold epochs")
    return cmc(np.asarray(eeg_ep, np.float64), np.asarray(emg_ep, np.float64)), len(eeg_ep)


# ----------------------------------------------------------------------------------- comparison
def rollout_maps(participant: int) -> pd.DataFrame:
    path = IV2 / f"P{participant}" / "electrodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the interpretability cell of main_v2.ipynb first")
    d = pd.read_csv(path)
    return d[d["output"] == "E"]


def _vec(df, col, key="electrode"):
    return df.set_index(key).loc[EEG_NAMES, col].to_numpy(dtype=float)


def compare(participant: int, ref: pd.DataFrame, names) -> list[dict]:
    """Spearman over electrodes between rollout maps and reference maps of one participant."""
    ro = rollout_maps(participant)
    rows = []

    def add(view, muscle, event, kind, a, b):
        rho = spearmanr(a, b)[0]
        rows.append({"subject": f"P{participant}", "view": view, "muscle": muscle, "event": event,
                     "reference": kind, "rho": float(rho)})

    for event in ("first_touch", "lift_off"):
        r_roll = _vec(ro[(ro.event == event) & (ro.view == "rollout")].groupby("electrode", as_index=False)["mean"].mean(), "mean")
        rel = ro[(ro.event == event) & (ro.view == "relevance")]
        r_rel = _vec(rel.groupby("electrode", as_index=False)["mean"].mean(), "mean")
        for band in BANDS:
            for win in (event, "hold"):
                e = ref[(ref.kind == "erd") & (ref.band == band) & (ref.window == win)]
                if e.empty:
                    continue
                neg = -_vec(e, "value")
                add("rollout", "mean", event, f"-ERD {band} {win}", r_roll, neg)
                add("relevance", "mean", event, f"-ERD {band} {win}", r_rel, neg)
        c = ref[ref.kind == "cmc"]
        if not c.empty:
            add("rollout", "mean", event, "CMC beta hold", r_roll, _vec(c.groupby("electrode", as_index=False)["value"].mean(), "value"))
            add("relevance", "mean", event, "CMC beta hold", r_rel, _vec(c.groupby("electrode", as_index=False)["value"].mean(), "value"))
            for m in names:
                rm = rel[rel.muscle == m]
                cm = c[c.muscle == m]
                if not rm.empty and not cm.empty:
                    add("relevance", m, event, "CMC beta hold", _vec(rm, "mean"), _vec(cm, "value"))
    return rows


def sign_flip_p(x: np.ndarray, n: int = N_PERM) -> float:
    """Two-sided p of the mean of x against 0, by random sign flips."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    obs = abs(x.mean())
    null = np.abs((RNG.choice([-1.0, 1.0], size=(n, len(x))) * x).mean(1))
    return float((1 + (null >= obs).sum()) / (n + 1))


def shuffle_p(a: np.ndarray, b: np.ndarray, n: int = N_PERM) -> tuple[float, float]:
    """Spearman rho of two maps and its two-sided p against electrode-label shuffles."""
    rho = spearmanr(a, b)[0]
    null = np.array([spearmanr(a, RNG.permutation(b))[0] for _ in range(n)])
    return float(rho), float((1 + (np.abs(null) >= abs(rho)).sum()) / (n + 1))


def run_participant(participant: int, splits=("train", "val", "test")) -> pd.DataFrame:
    series = []
    for s in splits:
        series += load_split(cache_dir(), participant, s, data_dir())
    names = emg_channel_names(participant, data_dir())
    erd = pd.DataFrame(erd_rows(participant, series))
    maps = (erd.groupby(["band", "window", "electrode"])["erd"].agg(value="median", n="count").reset_index()
            .assign(kind="erd", muscle=""))
    coh, n_ep = cmc_map(participant, series)
    cm = pd.DataFrame([{"kind": "cmc", "band": "beta", "window": "hold", "electrode": e, "muscle": m,
                        "value": float(coh[i, j]), "n": n_ep} for i, e in enumerate(EEG_NAMES) for j, m in enumerate(names)])
    out = pd.concat([maps, cm], ignore_index=True).assign(subject=f"P{participant}",
                                                           cmc_sig_level=1.0 - 0.05 ** (1.0 / max(n_ep - 1, 1)))
    return out


def group_tables(ref: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (view, muscle, event, kind), d in cmp.groupby(["view", "muscle", "event", "reference"]):
        rows.append({"level": "participants", "view": view, "muscle": muscle, "event": event, "reference": kind,
                     "mean_rho": d["rho"].mean(), "n_positive": int((d["rho"] > 0).sum()), "n": len(d),
                     "p": sign_flip_p(d["rho"].to_numpy())})
    return pd.DataFrame(rows)


def group_map_rows(ref: pd.DataFrame, subjects) -> pd.DataFrame:
    """Group-mean rollout maps against group-mean reference maps (electrode-shuffle p)."""
    ro = pd.concat([rollout_maps(p).assign(subject=f"P{p}") for p in subjects])
    rows = []
    for event in ("first_touch", "lift_off"):
        for view in ("rollout", "relevance"):
            g = ro[(ro.event == event) & (ro.view == view)].groupby("electrode")["mean"].mean().loc[EEG_NAMES].to_numpy()
            for (kind, band, win), d in ref.groupby(["kind", "band", "window"]):
                if kind == "erd" and win not in (event, "hold"):
                    continue
                v = d.groupby("electrode")["value"].mean().loc[EEG_NAMES].to_numpy()
                v = -v if kind == "erd" else v
                rho, p = shuffle_p(g, v)
                rows.append({"level": "group map", "view": view, "event": event,
                             "reference": f"{'-ERD' if kind == 'erd' else 'CMC'} {band} {win}", "rho": rho, "p": p})
    return pd.DataFrame(rows)


def rollout_group(subjects, event: str = "lift_off", view: str = "rollout") -> np.ndarray:
    """Group-mean rollout share of E per electrode (mean over muscles), in EEG_NAMES order."""
    ro = pd.concat([rollout_maps(p) for p in subjects])
    return ro[(ro.event == event) & (ro.view == view)].groupby("electrode")["mean"].mean().loc[EEG_NAMES].to_numpy()


def figures(ref: pd.DataFrame, subjects, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    from v2.interp import plots as P

    from . import plots as PP
    with P.style():
        PP.erd_maps(ref).savefig(out / "group_erd.png", dpi=150)
        PP.cmc_maps(ref).savefig(out / "group_cmc.png", dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participants", default="all")
    ap.add_argument("--splits", default="train,val,test", help="series used for the reference maps")
    ap.add_argument("--out", default=str(ROOT / "eeg_controls" / "results" / "reference"))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(a.participants)
    refs, cmps = [], []
    for p in subjects:
        ref = run_participant(p, tuple(s.strip() for s in a.splits.split(",")))
        refs.append(ref)
        cmp = pd.DataFrame(compare(p, ref, emg_channel_names(p, data_dir())))
        cmps.append(cmp)
        b = ref[(ref.kind == "erd") & (ref.band == "beta") & (ref.window == "lift_off")].set_index("electrode")["value"]
        top_erd = ", ".join(b.sort_values().index[:3])
        c = ref[ref.kind == "cmc"].groupby("electrode")["value"].mean().sort_values(ascending=False)
        main_rho = cmp[(cmp.view == "relevance") & (cmp.muscle == "mean") & (cmp.event == "lift_off")]
        print(f"P{p}: strongest beta ERD at {top_erd}; strongest CMC at {', '.join(c.index[:3])} "
              f"({int(ref[ref.kind == 'cmc']['n'].iloc[0])} hold epochs); "
              + "; ".join(f"{r.reference}: rho {r.rho:+.2f}" for r in main_rho.itertuples()), flush=True)
    ref_all, cmp_all = pd.concat(refs, ignore_index=True), pd.concat(cmps, ignore_index=True)
    ref_all.to_csv(out / "reference_maps.csv", index=False)
    cmp_all.to_csv(out / "rollout_vs_reference.csv", index=False)
    grp = pd.concat([group_tables(ref_all, cmp_all), group_map_rows(ref_all, subjects)], ignore_index=True)
    grp.to_csv(out / "group.csv", index=False)
    show = grp[(grp.muscle.isin(["mean", np.nan]) | grp.muscle.isna()) & (grp.event == "lift_off")]
    print("\n" + show[["level", "view", "reference", "mean_rho", "n_positive", "n", "rho", "p"]].round(3).to_string(index=False))
    if not a.no_figures:
        figures(ref_all, subjects, out)
    print("saved", out)


if __name__ == "__main__":
    main()
