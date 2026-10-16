"""Scalp-distance prior for the 32-electrode montage of WAY-EEG-GAL (order of hs.eeg.names)."""
import numpy as np

EEG_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
             "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"]


def electrode_prior(names=EEG_NAMES) -> np.ndarray:
    """(N, N) attention bias: 0 on the diagonal, -distance / median off-diagonal distance elsewhere."""
    import mne

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    missing = [n for n in names if n not in pos]
    if missing:
        raise ValueError(f"unknown electrode name(s): {missing}")
    xyz = np.stack([pos[n] for n in names])
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    off = dist[~np.eye(len(names), dtype=bool)]
    return (-dist / np.median(off)).astype(np.float32)


def select_channels(names, available=EEG_NAMES) -> list[int]:
    """Indices of `names` in `available` (the current channel list); None -> every channel."""
    available = list(available)
    if names is None:
        return list(range(len(available)))
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"unknown or already removed electrode name(s): {missing}")
    return [available.index(n) for n in names]


def eye_regress(eeg: np.ndarray, refs: list[str], names=EEG_NAMES):
    """Least-squares removal (with intercept) of the reference channels from every other channel of one series.

    Returns the residual channels (T, C - len(refs)) and their names; the reference channels are dropped.
    """
    names = list(names)
    ref_idx = select_channels(refs, names)
    keep = [i for i in range(len(names)) if i not in ref_idx]
    x = np.asarray(eeg, dtype=np.float64)
    design = np.column_stack([x[:, ref_idx], np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(design, x[:, keep], rcond=None)
    resid = x[:, keep] - design @ beta
    return resid.astype(np.asarray(eeg).dtype), [names[i] for i in keep]


def channel_names(cfg) -> list[str]:
    """Electrode names the model sees: after the optional eye regression, then the optional channel subset."""
    e = cfg["eeg"]
    names = list(EEG_NAMES)
    refs = e.get("eye_regress")
    if refs:
        select_channels(refs, names)
        names = [n for n in names if n not in refs]
    chans = e.get("channels")
    if chans is not None:
        names = [names[i] for i in select_channels(chans, names)]
    return names
