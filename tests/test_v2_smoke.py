import copy

import pytest
import torch

from v2.config import CONFIG_V2
from v2.smoke import choose_setting, project_runtime


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
    assert out["use_amp"] is False and len(out["candidates"]) == 1          # mixed precision is not tried on the CPU
    only = out["candidates"][0]
    assert only["fits"] is True and only["use_amp"] is False
    assert all(only[k] == out[k] for k in ("batch_size", "gradient_accumulation_steps", "peak_gb", "sec_per_step",
                                           "projected_minutes"))


def _cand(use_amp, batch, minutes, fits, peak=2.0):
    return {"use_amp": use_amp, "batch_size": batch, "gradient_accumulation_steps": 32 // batch, "peak_gb": peak,
            "sec_per_step": minutes / 100, "projected_minutes": minutes, "fits": fits}


def test_the_fastest_setting_that_fits_is_chosen():
    slow_amp, fast_plain = _cand(True, 32, 80.0, True), _cand(False, 32, 25.0, True)
    assert choose_setting([slow_amp, fast_plain], 0.8) is fast_plain
    assert choose_setting([fast_plain, slow_amp], 0.8) is fast_plain                 # the order does not matter
    assert choose_setting([_cand(True, 32, 20.0, True), fast_plain], 0.8)["use_amp"] is True


def test_a_faster_setting_that_does_not_fit_is_not_chosen():
    too_big = _cand(False, 32, 25.0, False, peak=5.5)
    half = _cand(False, 16, 40.0, True)
    amp = _cand(True, 32, 80.0, True)
    assert choose_setting([amp, too_big, half], 0.8) is half
    out_of_memory = {**_cand(False, 32, 0.0, False), "peak_gb": None, "sec_per_step": None, "projected_minutes": None}
    assert choose_setting([out_of_memory, amp], 0.8) is amp


def test_nothing_fits_raises():
    with pytest.raises(RuntimeError, match="does not fit in 80% of the GPU"):
        choose_setting([_cand(True, 32, 80.0, False, peak=5.0), _cand(True, 16, 90.0, False, peak=4.9)], 0.8)
