"""
synergy_decoding/nmf_extractor.py
==================================
Multi-subject Non-Negative Matrix Factorization (NMF) pipeline for muscle synergy extraction.

Provides:
- extract_nmf_synergies(): Per-subject NMF on EMG envelopes
- compute_vaf(): Variance Accounted For metric
- vaf_curve(): VAF vs. k sweep for component count selection
- align_synergies(): Hungarian-algorithm-based cross-subject synergy alignment
- CrossSubjectSimilarity: Cosine similarity matrix across all subjects' W matrices
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import NMF


# ============================================================================
# VAF Computation
# ============================================================================

def compute_vaf(M: np.ndarray, W: np.ndarray, C: np.ndarray) -> float:
    """Compute Variance Accounted For (VAF) of NMF reconstruction.

    VAF = 1 - SS_res / SS_tot, where SS_tot is computed relative to zero mean
    (not the grand mean), consistent with the motor synergy literature
    (d'Avella & Tresch, 2002).

    Args:
        M: Ground-truth EMG matrix of shape (T, n_muscles).
        W: Synergy weight matrix of shape (k, n_muscles).
        C: Activation coefficient matrix of shape (T, k).

    Returns:
        VAF in [0, 1]; 1.0 = perfect reconstruction.
    """
    M_hat = C @ W  # (T, n_muscles)
    ss_res = np.sum((M - M_hat) ** 2)
    ss_tot = np.sum(M ** 2)  # Relative to zero, per motor synergy convention
    if ss_tot < 1e-12:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def vaf_curve(
    M: np.ndarray,
    k_range: Tuple[int, int] = (1, 6),
    n_init: int = 10,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[int, float]:
    """Compute VAF for each synergy count k in k_range.

    Runs NMF n_init times per k (with different random seeds) and takes the
    best (lowest reconstruction error) solution, consistent with best practice.

    Args:
        M: EMG envelope matrix of shape (T, n_muscles). Must be non-negative.
        k_range: (k_min, k_max_exclusive) tuple.
        n_init: Number of NMF random restarts per k.
        max_iter: Maximum NMF iterations per restart.
        random_state: Base random seed.

    Returns:
        Dict mapping k -> VAF (float).
    """
    M = np.maximum(M, 0.0)
    vaf_dict: Dict[int, float] = {}
    for k in range(k_range[0], k_range[1]):
        best_err = np.inf
        best_W, best_C = None, None
        for seed in range(n_init):
            model = NMF(
                n_components=k,
                init="nndsvda",
                max_iter=max_iter,
                random_state=random_state + seed,
                tol=1e-5,
            )
            try:
                C_fit = model.fit_transform(M)  # (T, k)
                W_fit = model.components_        # (k, n_muscles)
                err = model.reconstruction_err_
                if err < best_err:
                    best_err = err
                    best_W = W_fit
                    best_C = C_fit
            except Exception:
                continue
        if best_W is not None:
            vaf_dict[k] = compute_vaf(M, best_W, best_C)
        else:
            vaf_dict[k] = 0.0
    return vaf_dict


# ============================================================================
# Single-Subject NMF Extraction
# ============================================================================

def extract_nmf_synergies(
    M: np.ndarray,
    k: int = 3,
    n_init: int = 20,
    max_iter: int = 2000,
    random_state: int = 42,
    normalize_W: bool = True,
    smooth_C: bool = True,
    smooth_cutoff_hz: float = 4.0,
    fs: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract k muscle synergies from a single-subject EMG matrix via NMF.

    Returns the best solution across n_init random restarts (lowest reconstruction
    error). Optionally normalizes W and smooths C.

    Args:
        M: EMG envelope of shape (T, n_muscles). Must be >= 0.
        k: Number of synergy components.
        n_init: Number of random restarts.
        max_iter: Maximum NMF iterations per restart.
        random_state: Base random seed.
        normalize_W: If True, normalize each row of W to unit L1 norm and
            scale C accordingly, so columns of C represent absolute activation.
        smooth_C: If True, low-pass filter each column of C at smooth_cutoff_hz.
        smooth_cutoff_hz: Cutoff for the causal low-pass smoothing filter (Hz).
        fs: Sampling frequency of M (Hz).

    Returns:
        W: Synergy weight matrix of shape (k, n_muscles). Each row is a synergy.
        C: Synergy activation coefficients of shape (T, k). C >= 0.
        vaf: VAF of this decomposition.
    """
    M = np.maximum(M, 0.0).astype(np.float64)

    best_err = np.inf
    best_W, best_C = None, None

    for seed in range(n_init):
        model = NMF(
            n_components=k,
            init="nndsvda",
            max_iter=max_iter,
            random_state=random_state + seed,
            tol=1e-6,
            l1_ratio=0.0,
        )
        try:
            C_fit = model.fit_transform(M)
            W_fit = model.components_
            if model.reconstruction_err_ < best_err:
                best_err = model.reconstruction_err_
                best_W = W_fit.copy()
                best_C = C_fit.copy()
        except Exception:
            continue

    if best_W is None:
        raise RuntimeError("NMF failed to converge in all restarts.")

    W, C = best_W, best_C

    # ── Normalize W rows to unit L1 → absorb scale into C ──────────────────
    if normalize_W:
        row_norms = np.maximum(W.sum(axis=1, keepdims=True), 1e-10)
        C = C * row_norms.T  # (T, k) * (1, k)
        W = W / row_norms    # (k, n_muscles)

    # ── Low-pass smooth C to isolate envelope dynamics ──────────────────────
    if smooth_C:
        C = _causal_lowpass(C, cutoff_hz=smooth_cutoff_hz, fs=fs)

    # ── Enforce non-negativity after smoothing ───────────────────────────────
    C = np.maximum(C, 0.0)

    vaf = compute_vaf(M, W, C)
    return W.astype(np.float32), C.astype(np.float32), vaf


def _causal_lowpass(C: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Apply causal (forward-only) Butterworth LP filter to each column of C."""
    from scipy.signal import butter, sosfilt
    sos = butter(2, cutoff_hz / (fs / 2.0), btype="low", output="sos")
    out = np.zeros_like(C)
    for col in range(C.shape[1]):
        out[:, col] = sosfilt(sos, C[:, col])
    return out


# ============================================================================
# Multi-Subject Extraction
# ============================================================================

def extract_all_subjects(
    emg_list: List[np.ndarray],
    subject_ids: Optional[List[int]] = None,
    k: int = 3,
    n_init: int = 20,
    max_iter: int = 2000,
    random_state: int = 42,
    smooth_C: bool = True,
    smooth_cutoff_hz: float = 4.0,
    fs: float = 500.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, float]]:
    """Run subject-specific NMF extraction for all subjects.

    Args:
        emg_list: List of EMG arrays, each of shape (T_s, n_muscles).
        subject_ids: Optional integer IDs for each subject. Defaults to 0..N-1.
        k: Fixed synergy count.
        (remaining args passed to extract_nmf_synergies)

    Returns:
        W_dict: {subject_id: W_s} where W_s is (k, n_muscles).
        C_dict: {subject_id: C_s} where C_s is (T_s, k).
        vaf_dict: {subject_id: float VAF}.
    """
    if subject_ids is None:
        subject_ids = list(range(len(emg_list)))

    W_dict: Dict[int, np.ndarray] = {}
    C_dict: Dict[int, np.ndarray] = {}
    vaf_dict: Dict[int, float] = {}

    for sid, M in zip(subject_ids, emg_list):
        W, C, vaf = extract_nmf_synergies(
            M,
            k=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            smooth_C=smooth_C,
            smooth_cutoff_hz=smooth_cutoff_hz,
            fs=fs,
        )
        W_dict[sid] = W
        C_dict[sid] = C
        vaf_dict[sid] = vaf
        print(f"  Subject {sid:>2}: VAF = {vaf:.4f}")

    return W_dict, C_dict, vaf_dict


# ============================================================================
# Cross-Subject Synergy Alignment & Similarity
# ============================================================================

def align_synergies(
    W_ref: np.ndarray,
    W_tgt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align synergy rows of W_tgt to best match W_ref using Hungarian algorithm.

    Uses cosine similarity as the matching criterion, consistent with the
    cross-subject synergy literature (Cheung et al., 2005).

    Args:
        W_ref: Reference W of shape (k, n_muscles).
        W_tgt: Target W of shape (k, n_muscles) to be reordered.

    Returns:
        W_aligned: Reordered W_tgt rows to best match W_ref.
        perm: Permutation indices applied to W_tgt rows.
    """
    # Cosine similarity matrix: (k, k)
    sim = 1.0 - cdist(W_ref, W_tgt, metric="cosine")
    # Hungarian: maximize similarity = minimize negative similarity
    row_ind, col_ind = linear_sum_assignment(-sim)
    perm = col_ind[np.argsort(row_ind)]
    return W_tgt[perm], perm


class CrossSubjectSimilarity:
    """Compute and store cosine similarity matrix across subjects' synergy matrices.

    After fitting, provides:
        self.sim_matrix: (n_subjects, n_subjects, k) array of mean per-synergy
            cosine similarity. The (i, j) entry is the mean synergy-pair cosine
            similarity after Hungarian alignment of subject j to subject i.
        self.global_sim: (n_subjects, n_subjects) overall mean across k synergies.
    """

    def __init__(self) -> None:
        self.subject_ids: List[int] = []
        self.W_list: List[np.ndarray] = []
        self.sim_matrix: Optional[np.ndarray] = None
        self.global_sim: Optional[np.ndarray] = None

    def fit(self, W_dict: Dict[int, np.ndarray]) -> "CrossSubjectSimilarity":
        """Compute all pairwise synergy similarity values.

        Args:
            W_dict: {subject_id: W_s} with W_s of shape (k, n_muscles).
        """
        self.subject_ids = sorted(W_dict.keys())
        self.W_list = [W_dict[sid] for sid in self.subject_ids]
        n = len(self.subject_ids)
        k = self.W_list[0].shape[0]

        sim_matrix = np.zeros((n, n, k), dtype=np.float32)
        global_sim = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(n):
                W_i = self.W_list[i]  # reference
                W_j_aligned, _ = align_synergies(W_i, self.W_list[j])
                # Per-synergy cosine similarity
                for s in range(k):
                    sim_val = 1.0 - float(
                        cdist(W_i[s:s+1], W_j_aligned[s:s+1], metric="cosine")[0, 0]
                    )
                    sim_matrix[i, j, s] = sim_val
                global_sim[i, j] = sim_matrix[i, j].mean()

        self.sim_matrix = sim_matrix
        self.global_sim = global_sim
        return self

    def report(self) -> str:
        """Return a formatted similarity report."""
        if self.global_sim is None:
            return "Not yet fitted."
        n = len(self.subject_ids)
        lines = ["Cross-Subject Synergy Cosine Similarity (mean across k synergies):"]
        header = "    " + "".join(f"  S{sid:02d}" for sid in self.subject_ids)
        lines.append(header)
        for i, sid_i in enumerate(self.subject_ids):
            row = f"S{sid_i:02d} " + "".join(
                f" {self.global_sim[i, j]:.3f}" for j in range(n)
            )
            lines.append(row)
        off_diag = self.global_sim[~np.eye(n, dtype=bool)]
        lines.append(f"\nMean off-diagonal similarity: {off_diag.mean():.4f} +/- {off_diag.std():.4f}")
        return "\n".join(lines)
