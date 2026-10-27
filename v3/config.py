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
CONFIG_V3["training"]["seeds"] = [0, 1, 2]           # every scale is trained once per seed; tests average the seeds
CONFIG_V3["evaluation"].update({
    "n_swap": 10,                                   # cue-aligned trial swaps per run (v3.nulls)
    "swap_match_condition": False,                  # True: donors only from trials with the same weight and surface
    "residual_folds": 5,                            # trial groups of the cross-validated ridge in residual_over_timing
})
CONFIG_V3["timing"] = {
    "variants": ["phase", "cue", "cue_causal", "events_causal"],
    "horizons_ms": [0, 200, 500, 1000],             # the causal variants only; phase and cue read the future
    "primary": "cue_causal",                        # the timing model of the beyond_timing criterion
    "cache_dir": "results/v3/timing_preds",
}
CONFIG_V3["contrast"] = {                           # rows C and F of the paper's table (v3.contrast); no training
    "checkpoint_dir": "outputs/checkpoints_v2_horizon",   # the causal v2 horizon models, h{h}ms/P{p}, read only
    "horizons_ms": [0, 200, 500, 1000],
    "sources": ["eeg", "swap", "bandpower", "peripheral"],
    "results_dir": "results/v3/contrast",
}
CONFIG_V3["muscle"] = {"shared": ["pc1", "mean"]}    # v3.evaluate.muscle_specificity: the shared activation removed

# Artifact controls (configs only): which electrodes and bands carry the student's information?
CENTRAL = ["C3", "C4", "Cz", "CP1", "CP2", "CP5", "CP6", "FC1", "FC2", "FC5", "FC6"]
PERIPHERAL = ["Fp1", "Fp2", "T7", "T8", "TP9", "TP10", "O1", "Oz", "O2", "PO9", "PO10"]   # frontal pole, temporal, occipital
EYE_REFS = ["Fp1", "Fp2", "F7", "F8"]                                                   # as eeg_controls.run_conditions
PRESETS = {
    "central": {"channels": CENTRAL},
    "peripheral": {"channels": PERIPHERAL},
    "eye_regressed": {"eye_regress": EYE_REFS},
    "no_gamma": {"bands": [b for b in CONFIG_V2["eeg"]["bands"] if b != [30, 40]]},
}


def preset_config(name: str, base: dict = None) -> dict:
    """A copy of `base` (default CONFIG_V3) with the eeg block of one artifact-control preset."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    cfg = copy.deepcopy(CONFIG_V3 if base is None else base)
    cfg["eeg"].update(copy.deepcopy(PRESETS[name]))
    return cfg
