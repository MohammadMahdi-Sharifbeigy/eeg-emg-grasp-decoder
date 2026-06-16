"""
CCA-based EEG alignment for KG-GT pipeline (method1.tex §3.2).

Goal: reduce EEG 32 → 16 canonical components maximally correlated
with kinematic state k_t (13-dim), so the Transformer sees neural
activity most relevant to motor execution.

Algorithm:
  1. Fit CCA on training-set (eeg, kin) pairs using sklearn CCA.
     - n_components = 16  (hardcoded in default.yaml)
     - eeg shape: (T_train, 32)   delta-band preprocessed
     - kin shape: (T_train, 13)   k_t from preprocess_kinematics
  2. Transform any split: project eeg (T, 32) → (T, 16) canonical scores.

Note: CCA requires both eeg and kin to fit, but only eeg to transform.
This is different from PCA/ICA — the projection is supervised by kinematics.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA


class EEGKinCCA:
    """CCA alignment: EEG 32 → n_components canonical variates.

    Fit once on all training series. Then call transform(eeg) on any split
    to project EEG into the kinematically-aligned subspace.

    Usage:
        cca = EEGKinCCA(n_components=16)

        # Fit on train: provide lists of aligned (eeg, kin) arrays
        cca.fit(train_eeg_list, train_kin_list)

        # Transform any split
        eeg_aligned = cca.transform(val_eeg)     # (T, 16)
    """

    def __init__(self, n_components: int = 16) -> None:
        self.n_components = n_components
        self._cca: CCA | None = None

    def fit(
        self,
        eeg_arrays: list[np.ndarray],
        kin_arrays: list[np.ndarray],
    ) -> "EEGKinCCA":
        """Fit CCA on concatenated training (eeg, kin) pairs.

        Args:
            eeg_arrays: list of (T_i, 32) preprocessed EEG arrays (train split)
            kin_arrays: list of (T_i, 13) k_t kinematic arrays (same split)

        All arrays must have the same number of samples T_i per pair.
        Arrays from different series are concatenated before fitting.
        """
        eeg_concat = np.concatenate(eeg_arrays, axis=0).astype(np.float64)
        kin_concat = np.concatenate(kin_arrays, axis=0).astype(np.float64)

        if eeg_concat.shape[0] != kin_concat.shape[0]:
            raise ValueError(
                f"EEG and kin sample counts differ: "
                f"{eeg_concat.shape[0]} vs {kin_concat.shape[0]}"
            )

        # sklearn CCA (canonical mode) requires n_components <= min(n, p, q)
        max_components = min(eeg_concat.shape[0], eeg_concat.shape[1], kin_concat.shape[1])
        actual = min(self.n_components, max_components)
        if actual < self.n_components:
            import warnings
            warnings.warn(
                f"EEGKinCCA: n_components={self.n_components} exceeds upper bound "
                f"{max_components} (min of n_samples={eeg_concat.shape[0]}, "
                f"n_eeg={eeg_concat.shape[1]}, n_kin={kin_concat.shape[1]}). "
                f"Clamped to {actual}.",
                UserWarning, stacklevel=2,
            )
            self.n_components = actual

        self._cca = CCA(n_components=self.n_components, max_iter=1000)
        self._cca.fit(eeg_concat, kin_concat)
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        """Project EEG into canonical space.

        Args:
            eeg: ndarray (T, 32) preprocessed EEG (any split)

        Returns:
            ndarray (T, n_components) float32 canonical EEG scores
        """
        if self._cca is None:
            raise RuntimeError("EEGKinCCA must be fit before transform.")
        # sklearn CCA.transform returns (X_scores, Y_scores); we only need X
        eeg_scores, _ = self._cca.transform(
            eeg.astype(np.float64),
            np.zeros((eeg.shape[0], 13), dtype=np.float64),
        )
        return eeg_scores.astype(np.float32)

    def fit_transform(
        self,
        eeg_arrays: list[np.ndarray],
        kin_arrays: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Fit and transform all training arrays in one call.

        Returns:
            list of (T_i, n_components) arrays in same order as input
        """
        self.fit(eeg_arrays, kin_arrays)
        return [self.transform(e) for e in eeg_arrays]

    def state_dict(self) -> dict:
        """Serialise CCA weights for saving to .npz or checkpoint."""
        if self._cca is None:
            raise RuntimeError("EEGKinCCA has not been fit yet.")
        return {
            "n_components":  self.n_components,
            "x_weights_":    self._cca.x_weights_,     # (32, n_components)
            "y_weights_":    self._cca.y_weights_,     # (13, n_components)
            "x_mean_":       self._cca.x_mean_,        # (32,)
            "y_mean_":       self._cca.y_mean_,        # (13,)
            "x_std_":        self._cca.x_std_,         # (32,) or scalar
            "y_std_":        self._cca.y_std_,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore CCA weights without re-fitting."""
        self.n_components = int(d["n_components"])
        self._cca = CCA(n_components=self.n_components)
        # Manually restore sklearn internal attributes
        self._cca.x_weights_ = np.asarray(d["x_weights_"], dtype=np.float64)
        self._cca.y_weights_ = np.asarray(d["y_weights_"], dtype=np.float64)
        self._cca.x_mean_    = np.asarray(d["x_mean_"],    dtype=np.float64)
        self._cca.y_mean_    = np.asarray(d["y_mean_"],    dtype=np.float64)
        self._cca.x_std_     = np.asarray(d["x_std_"],     dtype=np.float64)
        self._cca.y_std_     = np.asarray(d["y_std_"],     dtype=np.float64)
        # Derive rotation matrices sklearn needs for transform()
        self._cca.x_rotations_ = self._cca.x_weights_
        self._cca.y_rotations_ = self._cca.y_weights_


# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------

def make_cca_from_config(cfg: dict) -> EEGKinCCA:
    """Build EEGKinCCA from preprocessing.cca config section."""
    return EEGKinCCA(n_components=cfg.get("n_components", 16))
