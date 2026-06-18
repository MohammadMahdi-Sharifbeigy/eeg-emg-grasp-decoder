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
import torch
from sklearn.cross_decomposition import CCA
from torch import Tensor


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

    def __init__(self, n_components: int = 16, max_fit_samples: int = 500_000,
                 random_state: int = 42) -> None:
        self.n_components = n_components
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
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
        Arrays from different series are concatenated before fitting. If the
        total exceeds ``max_fit_samples``, a random subset is drawn — CCA
        weights converge on a fraction of the data and the full set (tens of
        millions of rows) would exhaust memory (sklearn copies X and y to
        float64 internally).
        """
        lengths = [e.shape[0] for e in eeg_arrays]
        for i, (n_e, k) in enumerate(zip(lengths, kin_arrays)):
            if n_e != k.shape[0]:
                raise ValueError(
                    f"EEG and kin sample counts differ in series {i}: "
                    f"{n_e} vs {k.shape[0]}"
                )

        n_total = int(np.sum(lengths))
        if self.max_fit_samples and n_total > self.max_fit_samples:
            # Sample per-array (proportional) and only materialise the subset,
            # so we never build the full (n_total, 32) concatenation in memory.
            rng = np.random.default_rng(self.random_state)
            eeg_parts, kin_parts = [], []
            for eeg_i, kin_i, n_i in zip(eeg_arrays, kin_arrays, lengths):
                take = max(1, round(self.max_fit_samples * n_i / n_total))
                take = min(take, n_i)
                sel = rng.choice(n_i, size=take, replace=False)
                sel.sort()
                eeg_parts.append(eeg_i[sel])
                kin_parts.append(kin_i[sel])
            eeg_concat = np.concatenate(eeg_parts, axis=0).astype(np.float64)
            kin_concat = np.concatenate(kin_parts, axis=0).astype(np.float64)
        else:
            eeg_concat = np.concatenate(eeg_arrays, axis=0).astype(np.float64)
            kin_concat = np.concatenate(kin_arrays, axis=0).astype(np.float64)

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


    def torch_projector(self, device: torch.device | str = "cpu") -> "TorchCCA":
        """Export the fitted projection as an on-device TorchCCA.

        sklearn's CCA.transform reduces to an affine map followed by a matmul:
            X_scores = ((X - x_mean_) / x_std_) @ x_rotations_
        Replicating it in torch keeps the whole batch on the GPU and removes
        the per-batch CPU round-trip (.cpu().numpy() -> sklearn -> .to(device)).

        Args:
            device: Device to place the projection tensors on.

        Returns:
            TorchCCA with mean/std/rotation tensors on ``device``.
        """
        if self._cca is None:
            raise RuntimeError("EEGKinCCA must be fit before export.")

        # sklearn renamed these to private (_x_mean) around 1.3; support both.
        def _attr(*names):
            for n in names:
                if hasattr(self._cca, n):
                    return getattr(self._cca, n)
            raise AttributeError(f"CCA missing all of {names}")

        x_mean = np.asarray(_attr("_x_mean", "x_mean_"), dtype=np.float32).reshape(-1)
        x_std = np.asarray(_attr("_x_std", "x_std_"), dtype=np.float32).reshape(-1)
        x_rot = np.asarray(self._cca.x_rotations_, dtype=np.float32)  # (n_eeg, k)
        return TorchCCA(x_mean, x_std, x_rot, device=device)


class TorchCCA:
    """On-device EEG -> canonical projection (matmul only, no autograd needed).

    Mirrors EEGKinCCA.transform but runs entirely in torch so it can sit inside
    the GPU training loop. Not an nn.Module: the projection is a fixed,
    pre-fitted transform, so its tensors are plain buffers.
    """

    def __init__(
        self,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        x_rotations: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32, device=self.device)
        self.x_std = torch.as_tensor(x_std, dtype=torch.float32, device=self.device)
        self.x_rotations = torch.as_tensor(
            x_rotations, dtype=torch.float32, device=self.device
        )
        self.n_components = self.x_rotations.shape[1]

    def to(self, device: torch.device | str) -> "TorchCCA":
        """Move projection tensors to ``device`` in place."""
        self.device = torch.device(device)
        self.x_mean = self.x_mean.to(self.device)
        self.x_std = self.x_std.to(self.device)
        self.x_rotations = self.x_rotations.to(self.device)
        return self

    @torch.no_grad()
    def transform(self, eeg: Tensor) -> Tensor:
        """Project EEG into canonical space on-device.

        Args:
            eeg: Tensor of shape (..., n_eeg) on any device.

        Returns:
            Tensor of shape (..., n_components) on this projector's device.
        """
        eeg = eeg.to(self.device, dtype=torch.float32)
        return ((eeg - self.x_mean) / self.x_std) @ self.x_rotations


# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------

def make_cca_from_config(cfg: dict) -> EEGKinCCA:
    """Build EEGKinCCA from preprocessing.cca config section."""
    return EEGKinCCA(
        n_components=cfg.get("n_components", 16),
        max_fit_samples=cfg.get("max_fit_samples", 500_000),
        random_state=cfg.get("random_state", 42),
    )
