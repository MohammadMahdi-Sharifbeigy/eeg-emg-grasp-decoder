"""Configuration of the EEG-only guided model (spec 2026-10-16-eeg-base-kin-guide-design.md)."""
import copy

from v2.config import CONFIG_V2

CONFIG_V3 = copy.deepcopy(CONFIG_V2)
CONFIG_V3["guide"] = {
    "scale": 0.1,                                   # one number for the strength of the help; 0 = the E-only control
    "scales": [0.0, 0.05, 0.1, 0.3],
    "terms": ["align", "distill"],                  # also available: "kin_aux"
    "w_align": 1.0, "w_distill": 0.5, "w_aux": 0.5,
    "align_at": "final",                            # "final" tokens or "mid" (after the first temporal layer)
    "aux_columns": list(range(12, 24)),             # standardised kinematic features the optional aux head predicts
    "schedule": "fade",                             # "fade", "constant" or "off"
    "hold_until": 40, "off_at": 60,                 # epochs: full strength until hold_until, zero from off_at
}
CONFIG_V3["teacher"] = {"checkpoint_dir": "outputs/checkpoints_v2"}
CONFIG_V3["outputs"] = {"checkpoint_dir": "outputs/checkpoints_v3", "results_dir": "results/v3"}
