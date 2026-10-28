"""The six conditions of Critical experiment II, under the names used by every table, notebook step and README.

  causal_cue  past LED and event times only                       how much of the envelope is the task clock
  K           past hand kinematics and force                      the biomechanical explanation
  B           K + past task events                                the primary no-EEG baseline
  F           K + past task events + EEG history                  the primary neural candidate
  E0          EEG only, trained without the kinematic guide       the EEG-only reference
  Eguided     EEG only, trained with the frozen K teacher         the secondary training intervention (v3)
The primary contrast is F against B (eeg_skill = 1 - MSE(F) / MSE(B)); E0 against Eguided is secondary.
"""
import pandas as pd

CONDITIONS = {
    "causal_cue": {"inputs": "past LED and event times only", "role": "measures the task timing", "uses_eeg": False,
                   "trained": False, "source": "v3.timing (ridge, no training)"},
    "K": {"inputs": "past hand kinematics and force", "role": "biomechanical explanation", "uses_eeg": False,
          "trained": False, "source": "kinematic stream of outputs/checkpoints_v2_horizon/h{h}ms/P{p}"},
    "B": {"inputs": "K + past task events", "role": "primary no-EEG baseline", "uses_eeg": False, "trained": False,
          "source": "v3.contrast (ridge on the residual of K)"},
    "F": {"inputs": "K + past task events + EEG history", "role": "primary neural candidate", "uses_eeg": True,
          "trained": False, "source": "v3.contrast (ridge on the residual of K)"},
    "E0": {"inputs": "EEG only, no guide", "role": "EEG-only reference", "uses_eeg": True, "trained": True,
           "source": "v3.train, guide.scale 0"},
    "Eguided": {"inputs": "EEG only, trained with the K teacher", "role": "secondary training intervention",
                "uses_eeg": True, "trained": True, "source": "v3.train, guide.scale > 0"},
}


def _get(name: str) -> dict:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; the conditions are {', '.join(CONDITIONS)}")
    return CONDITIONS[name]


def describe(name: str) -> str:
    """One line, e.g. "B: K + past task events (primary no-EEG baseline)"."""
    c = _get(name)
    return f"{name}: {c['inputs']} ({c['role']})"


def condition_table() -> pd.DataFrame:
    return pd.DataFrame([{"condition": n, **c} for n, c in CONDITIONS.items()])
