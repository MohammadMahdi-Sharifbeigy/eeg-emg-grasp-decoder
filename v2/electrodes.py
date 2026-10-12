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
