import copy

import torch

from v2.config import CONFIG_V2
from v2.smoke import project_runtime


def test_project_runtime_on_cpu_reports_time_and_keeps_effective_batch():
    c = copy.deepcopy(CONFIG_V2)
    c["tokens"]["window"] = 100                       # keep the CPU test quick
    c["training"]["batch_size"] = 4
    out = project_runtime(c, torch.device("cpu"), n_windows=40, steps=2)
    assert out["batch_size"] * out["gradient_accumulation_steps"] == 4
    assert out["sec_per_step"] > 0 and out["projected_minutes"] > 0 and out["peak_gb"] == 0.0
    assert out["parameters"] <= 1_000_000
    expected = out["sec_per_step"] * (40 / out["batch_size"]) * c["training"]["max_epochs"] / 60
    assert abs(out["projected_minutes"] - expected) < 1e-6
