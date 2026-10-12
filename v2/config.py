"""Configuration of the KG-GT v2 dual-stream token model (spec 2026-10-12-kg-gt-v2-design.md)."""

CONFIG_V2 = {
    "data": {
        "data_root": "data/way-eeg/raw",
        "cache_dir": "data/cache_main_stride500_square_pose_force_sg51",
        "fs": 500.0,
    },
    "tokens": {"window": 400, "stride": 100, "hop": 200},
    "eeg": {"bands": [[4, 8], [8, 13], [13, 30], [30, 40]], "smooth_hz": 5.0, "clip": 5.0},
    "history": {"force_threshold": 1.0, "min_hold_s": 0.5},
    "model": {
        "d_model": 64, "n_heads": 4, "ffn_dim": 256, "dropout": 0.2,
        "kin_layers": 3, "eeg_layers": 2,
        "electrode_dim": 32, "electrode_gat_layers": 2, "electrode_dropout": 0.2,
        "node_dim": 32, "muscle_gat_layers": 2,
    },
    "loss": {"w_main": 1.0, "w_ccc": 0.3, "w_lift": 0.5, "w_diff": 0.1, "huber_delta": 0.5},
    "training": {
        "lr": 1e-3, "weight_decay": 0.01, "warmup_epochs": 3, "max_epochs": 150, "patience": 15,
        "batch_size": 32, "gradient_accumulation_steps": 1, "grad_clip_norm": 1.0, "use_amp": True,
        "smooth_window": 3, "seed": 42,
    },
    "evaluation": {"n_null": 20, "seed": 42},
    "interp": {
        "residual": 0.5, "flow_bin": 4, "events": ["first_touch", "lift_off"], "max_events": 6,
        "deletion_fractions": [0.05, 0.1, 0.2, 0.4], "n_random": 10, "n_perm": 1000, "seed": 42,
    },
    "outputs": {
        "checkpoint_dir": "outputs/checkpoints_v2",
        "results_dir": "results/v2",
        "interpretability_dir": "outputs/interpretability_v2",
    },
}
