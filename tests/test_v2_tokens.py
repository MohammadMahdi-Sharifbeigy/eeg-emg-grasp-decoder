import numpy as np
import pytest

from v2 import tokens
from v2.config import CONFIG_V2


def test_token_constants_and_lengths():
    assert tokens.FS_IN == 500.0 and tokens.Q == 10 and tokens.FS_TOK == 50.0
    assert tokens.n_tokens(4000) == 400 and tokens.n_tokens(4001) == 401 and tokens.n_tokens(4009) == 401
    x = np.random.default_rng(0).standard_normal((4003, 3))
    assert tokens.to_tokens(x).shape == (401, 3) and tokens.to_tokens(x).dtype == np.float32
    assert tokens.hold_to_tokens(np.arange(4003)).shape == (401,)


@pytest.mark.parametrize("freq,tol", [(1.0, 0.005), (4.0, 0.01), (8.0, 0.03)])
def test_round_trip_keeps_content_up_to_8_hz(freq, tol):
    t = np.arange(10000) / tokens.FS_IN
    x = np.stack([np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)], 1)
    back = tokens.to_samples(tokens.to_tokens(x), len(x))
    mid = slice(1000, 9000)                       # ignore filter edges
    err = np.sqrt(np.mean((back[mid] - x[mid]) ** 2)) / np.sqrt(np.mean(x[mid] ** 2))
    assert back.shape == x.shape and err < tol


def test_anti_alias_removes_content_above_nyquist():
    t = np.arange(10000) / tokens.FS_IN
    x = np.sin(2 * np.pi * 60.0 * t)[:, None]     # above the 25 Hz token Nyquist
    assert np.abs(tokens.to_tokens(x)[100:900]).max() < 0.02


def test_short_input_raises():
    with pytest.raises(ValueError, match="too short"):
        tokens.to_tokens(np.zeros((300, 2)))


def test_config_values_from_the_spec():
    c = CONFIG_V2
    assert c["tokens"] == {"window": 400, "stride": 100, "hop": 200}
    assert c["model"]["d_model"] == 64 and c["model"]["kin_layers"] == 3 and c["model"]["eeg_layers"] == 2
    assert c["loss"] == {"w_main": 1.0, "w_ccc": 0.3, "w_lift": 0.5, "w_diff": 0.1, "huber_delta": 0.5}
    assert c["training"]["max_epochs"] == 150 and c["training"]["patience"] == 15 and "max_train_minutes" not in c["training"]
    assert c["data"]["cache_dir"] == "data/cache_main_stride500_square_pose_force_sg51"
    assert c["outputs"]["checkpoint_dir"] == "outputs/checkpoints_v2" and c["outputs"]["results_dir"] == "results/v2"
    assert c["outputs"]["interpretability_dir"] == "outputs/interpretability_v2"
    assert c["interp"] == {"residual": 0.5, "flow_bin": 4, "events": ["first_touch", "lift_off"], "max_events": 6,
                           "deletion_fractions": [0.05, 0.1, 0.2, 0.4], "n_random": 10, "n_perm": 1000, "seed": 42}
