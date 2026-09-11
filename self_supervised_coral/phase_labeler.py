"""PhaseLabeler — Converts continuous kinematics into discrete movement phases.

WAY-EEG-GAL contains continuous grasping trials. We discretize each windowed
segment into one of three movement phases based on kinematic velocity magnitude:

    REST   (0): ‖v‖ < low_threshold — quiescent baseline
    ACTIVE (1): ‖v‖ ≥ high_threshold — clear movement epochs
    TRANSIT(2): low_threshold ≤ ‖v‖ < high_threshold — transition zones

These labels are used in PhaseAwareInfoNCELoss to mask out same-phase pairs
from the negative denominator, preventing the contrastive objective from
pushing semantically identical neural states apart.

Usage:
    labeler = PhaseLabeler(fs=500, low_pct=25, high_pct=75)
    phases = labeler(kin_windows)   # (N,) int64 label tensor
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class MovementPhase(IntEnum):
    """Discrete movement phase labels."""
    REST    = 0
    TRANSIT = 2
    ACTIVE  = 1


class PhaseLabeler(torch.nn.Module):
    """Discretizes kinematic windows into movement phases.

    Computes the mean velocity magnitude per window and classifies it
    relative to dataset-level percentile thresholds, which are computed
    lazily on the first batch or can be pre-fit on training data.

    Args:
        fs: Sampling rate in Hz (default: 500).
        low_pct: Percentile for REST/TRANSIT boundary (default: 25).
        high_pct: Percentile for TRANSIT/ACTIVE boundary (default: 75).
        velocity_channels: Slice of kinematic channels to use as velocity.
                           WAY-EEG-GAL kin features: positions 0-17, velocity 18-35.
                           Set to None to use all channels as velocity proxy.
        window_stride: Stride when computing velocity via finite difference (default: 1).
    """

    def __init__(
        self,
        fs: float = 500.0,
        low_pct: float = 25.0,
        high_pct: float = 75.0,
        velocity_channels: Optional[slice] = None,
        window_stride: int = 1,
    ) -> None:
        super().__init__()
        self.fs = fs
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.velocity_channels = velocity_channels  # e.g., slice(18, 36) for WAY-EEG-GAL
        self.window_stride = window_stride

        # Thresholds are set after fit() or on first batch via lazy fitting
        self._low_thresh: Optional[float] = None
        self._high_thresh: Optional[float] = None

    def fit(self, kin_array: np.ndarray) -> "PhaseLabeler":
        """Pre-compute velocity percentile thresholds from a population of windows.

        Args:
            kin_array: (N, T, K) numpy array of kinematic windows from training set.

        Returns:
            self (for chaining)
        """
        vel_mags = self._compute_velocity_magnitudes_np(kin_array)
        self._low_thresh = float(np.percentile(vel_mags, self.low_pct))
        self._high_thresh = float(np.percentile(vel_mags, self.high_pct))
        return self

    def _compute_velocity_magnitudes_np(self, kin: np.ndarray) -> np.ndarray:
        """Compute per-window mean velocity magnitude from (N, T, K) array."""
        if self.velocity_channels is not None:
            kin = kin[:, :, self.velocity_channels]
        # Finite-difference velocity (causal: v[t] = x[t] - x[t-1])
        vel = np.diff(kin, axis=1, prepend=kin[:, :1, :]) * self.fs  # (N, T, K)
        mag = np.linalg.norm(vel, axis=-1)  # (N, T)
        return mag.mean(axis=1)  # (N,)

    def _compute_velocity_magnitudes(self, kin: Tensor) -> Tensor:
        """Compute per-window mean velocity magnitude from (B, T, K) tensor."""
        if self.velocity_channels is not None:
            kin = kin[:, :, self.velocity_channels]
        # Causal finite difference: prepend first frame to avoid lookahead
        kin_prev = torch.cat([kin[:, :1, :], kin[:, :-1, :]], dim=1)
        vel = (kin - kin_prev) * self.fs  # (B, T, K)
        mag = torch.norm(vel, dim=-1)      # (B, T)
        return mag.mean(dim=1)             # (B,)

    def _lazy_fit(self, vel_mags: Tensor) -> None:
        """Lazily set thresholds from first batch if not already fit."""
        if self._low_thresh is None:
            mags_np = vel_mags.detach().cpu().numpy()
            self._low_thresh = float(np.percentile(mags_np, self.low_pct))
            self._high_thresh = float(np.percentile(mags_np, self.high_pct))

    def forward(self, kin: Tensor) -> Tensor:
        """
        Classify kinematic windows into discrete movement phases.

        Args:
            kin: (B, T, K) kinematic feature tensor (positions, velocities, etc.)

        Returns:
            phases: (B,) int64 tensor with values in {REST=0, ACTIVE=1, TRANSIT=2}
        """
        vel_mags = self._compute_velocity_magnitudes(kin)  # (B,)
        self._lazy_fit(vel_mags)

        low = self._low_thresh
        high = self._high_thresh

        phases = torch.full(
            (kin.shape[0],),
            fill_value=int(MovementPhase.TRANSIT),
            dtype=torch.long,
            device=kin.device,
        )
        phases[vel_mags < low] = int(MovementPhase.REST)
        phases[vel_mags >= high] = int(MovementPhase.ACTIVE)
        return phases

    def label_from_emg_power(self, emg: Tensor) -> Tensor:
        """Alternative phase labeling using EMG RMS power (for EMG-side labeling).

        Args:
            emg: (B, T, n_muscles) EMG envelope tensor.

        Returns:
            phases: (B,) int64 tensor
        """
        # RMS power per window per muscle, then mean across muscles
        rms = emg.pow(2).mean(dim=1).sqrt()  # (B, n_muscles)
        power = rms.mean(dim=-1)             # (B,)
        self._lazy_fit(power)

        low = self._low_thresh
        high = self._high_thresh

        phases = torch.full(
            (emg.shape[0],),
            fill_value=int(MovementPhase.TRANSIT),
            dtype=torch.long,
            device=emg.device,
        )
        phases[power < low] = int(MovementPhase.REST)
        phases[power >= high] = int(MovementPhase.ACTIVE)
        return phases

    @property
    def thresholds(self) -> Tuple[Optional[float], Optional[float]]:
        """Returns (low_thresh, high_thresh) in velocity magnitude units."""
        return self._low_thresh, self._high_thresh

    def __repr__(self) -> str:
        return (
            f"PhaseLabeler(low_pct={self.low_pct}, high_pct={self.high_pct}, "
            f"thresholds=({self._low_thresh:.4f}, {self._high_thresh:.4f}))"
            if self._low_thresh is not None
            else f"PhaseLabeler(low_pct={self.low_pct}, high_pct={self.high_pct}, thresholds=not_fit)"
        )
