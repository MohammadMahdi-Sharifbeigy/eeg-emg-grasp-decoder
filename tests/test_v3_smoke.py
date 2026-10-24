import copy

import pytest
import torch

from v3.config import CONFIG_V3
from v3.smoke import project_runtime


def test_cpu_projection_runs():
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["model"].update(d_model=16, n_heads=2, ffn_dim=32, kin_layers=2, eeg_layers=2, electrode_dim=16, node_dim=8,
                        electrode_gat_layers=1, muscle_gat_layers=1)
    cfg["tokens"].update(window=40, stride=20)
    cfg["training"].update(batch_size=4, use_amp=False, max_epochs=2)
    out = project_runtime(cfg, torch.device("cpu"), n_windows=8, steps=2)
    assert out["parameters"] > 0 and out["projected_minutes"] > 0 and out["use_amp"] is False
    assert out["batch_size"] == 4


def test_run_budget_counts_runs_and_hours():
    from v3.smoke import run_budget
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["model"].update(d_model=16, n_heads=2, ffn_dim=32, kin_layers=2, eeg_layers=2, electrode_dim=16, node_dim=8,
                        electrode_gat_layers=1, muscle_gat_layers=1)
    cfg["tokens"].update(window=40, stride=20)
    cfg["training"].update(batch_size=4, use_amp=False, max_epochs=2)
    b = run_budget(cfg, torch.device("cpu"), [0.0, 0.1], [0, 1, 2], 12, n_windows=8, steps=2)
    assert b["n_runs"] == 72 and set(b["settings"]) == {"control", "guided"}
    expected = 3 * 12 * (b["per_run_minutes"][0.0] + b["per_run_minutes"][0.1]) / 60
    assert b["total_gpu_hours"] == pytest.approx(expected) and b["min_epochs_guided"] == cfg["guide"]["off_at"]
