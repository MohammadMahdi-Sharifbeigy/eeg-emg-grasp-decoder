"""Causal mode and forecast horizon of KG-GT v2 (EEG control experiment C)."""
import copy

import numpy as np
import pytest
import torch

from v2.causal import bandpower_causal, causal_kinematics, causal_velocity, shift_horizon, to_tokens_causal
from v2.config import CONFIG_V2
from v2.infer import window_plan
from v2.model import build_model

B, T = 2, 60


def _cfg(causal=True):
    cfg = copy.deepcopy(CONFIG_V2)
    cfg["model"]["causal"] = causal
    return cfg


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 24, generator=g), torch.rand(B, T, 8, generator=g), torch.randn(B, T, 32, 4, generator=g)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_causal_model_ignores_the_future(which):
    torch.manual_seed(0)
    m = build_model(_cfg(True)).eval()
    x = list(_inputs())
    t = 37
    with torch.no_grad():
        a = m(*x)
        y = [v.clone() for v in x]
        y[which][:, t + 1:] = torch.randn_like(y[which][:, t + 1:]) * 5
        b = m(*y)
    for k in ("K", "E", "F", "F_eeg", "K_base"):
        assert torch.allclose(a[k][:, :t + 1], b[k][:, :t + 1], atol=1e-5), k
    changed = "E" if which == 2 else "K"                            # the stream that reads the changed input
    assert not torch.allclose(a[changed][:, t + 1:], b[changed][:, t + 1:])


def test_default_model_is_unchanged_and_sees_the_future():
    torch.manual_seed(0)
    m = build_model(_cfg(False)).eval()
    assert m.causal is False and m.kin.stem.padding == (2,)
    x = list(_inputs())
    with torch.no_grad():
        a = m(*x)
        x[2][:, 50:] += 3.0
        b = m(*x)
    assert not torch.allclose(a["E"][:, :40], b["E"][:, :40])        # bidirectional attention: earlier tokens move


def test_causal_and_default_models_share_the_parameter_shapes():
    a, b = build_model(_cfg(True)).state_dict(), build_model(_cfg(False)).state_dict()
    assert a.keys() == b.keys() and all(a[k].shape == b[k].shape for k in a)


def test_causal_stem_matrix_is_lower_triangular():
    m = build_model(_cfg(True))
    s = m.kin.stem_matrix(12)
    assert torch.allclose(s.sum(-1), torch.ones(12)) and torch.allclose(s, s.tril())
    d = build_model(_cfg(False)).kin.stem_matrix(12)
    assert d[5, 7] > 0 and d[5, 8] == 0                               # centred: two taps after the token


@pytest.mark.parametrize("n,window,hop", [(1000, 400, 200), (1234, 400, 200), (401, 400, 200), (300, 400, 200)])
def test_causal_window_plan_covers_every_token_once(n, window, hop):
    plan = window_plan(n, window, hop, causal=True)
    kept = np.concatenate([np.arange(lo, hi) for _, lo, hi in plan])
    assert np.array_equal(kept, np.arange(n))
    for i, (start, lo, hi) in enumerate(plan):
        assert start <= lo and hi <= start + window and hi <= n
        if i > 0:
            assert lo - start >= window - hop                       # every later token sees >= window - hop before it


def test_to_tokens_causal_uses_no_future_sample():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 3))
    a = to_tokens_causal(x)
    x2 = x.copy()
    x2[1001:] += 10
    b = to_tokens_causal(x2)
    assert a.shape == (200, 3)
    assert np.allclose(a[:101], b[:101]) and not np.allclose(a[101:], b[101:])
    c = to_tokens_causal(np.full((1000, 1), 4.0))
    assert np.allclose(c, 4.0, atol=1e-4)                             # no start-up transient for a constant


def test_bandpower_causal_uses_no_future_sample_and_finds_the_band():
    rng = np.random.default_rng(1)
    fs, n = 500.0, 5000
    t = np.arange(n) / fs
    eeg = rng.standard_normal((n, 2)) * 0.1
    eeg[:, 0] += np.sin(2 * np.pi * 20 * t)                           # beta in channel 0 only
    bands = [[4, 8], [13, 30]]
    a = bandpower_causal(eeg, fs, bands)
    assert a.shape == (n, 4) and np.isfinite(a).all()
    assert a[1000:, 2].mean() > a[1000:, 3].mean() + 3               # beta power of channel 0 >> channel 1
    e2 = eeg.copy()
    e2[3001:] *= 10
    b = bandpower_causal(e2, fs, bands)
    # the floor uses the series median, which the change moves; compare above the floor
    above = (a[:3001] > a[:3001].min(0)) & (b[:3001] > b[:3001].min(0))
    assert np.allclose(a[:3001][above], b[:3001][above], atol=1e-5)


def test_causal_velocity_is_exact_for_a_cubic_and_one_sided():
    fs = 500.0
    t = np.arange(400) / fs
    pos = (1 + 2 * t + 3 * t ** 2 - t ** 3)[:, None]
    v = causal_velocity(pos, fs)
    true = (2 + 6 * t - 3 * t ** 2)[:, None]
    assert np.allclose(v[60:], true[60:], atol=1e-3)
    p2 = pos.copy()
    p2[201:] += 5
    assert np.allclose(causal_velocity(p2, fs)[:201], v[:201])
    kin = np.random.default_rng(2).standard_normal((300, 26)).astype(np.float32)
    out = causal_kinematics(kin, fs)
    assert out.shape == (300, 26) and np.array_equal(out[:, :13], kin[:, :13])


def test_shift_horizon_pairs_input_t_with_target_t_plus_h():
    n = 1003
    n_tok = -(-n // 10)
    tok = {"series": 4, "kin": np.arange(n_tok)[:, None].astype(np.float32), "hist": np.zeros((n_tok, 8)),
           "eeg": np.zeros((n_tok, 32, 4)), "emg": np.arange(n_tok)[:, None] * np.ones((1, 5)),
           "hold_id": np.arange(n_tok), "emg_full": np.arange(n)[:, None] * np.ones((1, 5)), "n_samples": n}
    s = shift_horizon(tok, 7)
    assert len(s["kin"]) == len(s["emg"]) == len(s["hold_id"]) == n_tok - 7
    assert s["emg"][0, 0] == 7 and s["kin"][0, 0] == 0 and s["hold_id"][0] == 7
    assert s["emg_full"][0, 0] == 70 and s["n_samples"] == n - 70 == len(s["emg_full"])
    assert shift_horizon(tok, 0)["n_samples"] == n
    with pytest.raises(ValueError):
        shift_horizon(tok, -1)
