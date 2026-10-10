"""Package 2 — Phase B: make the EEG path contribute.

1. log band-power EEG features (main.eeg_features) + WAYEEGDataset(eeg_transform=…)
2. per-feature z-score (train statistics) + clip in the prepare_batch closure
3. kinematic-pathway dropout, EEG-only forward (use_kin=False), EEG-only auxiliary loss
4. electrode dropout (whole electrode, all bands, whole window)
5. ablation condition kin_off, electrode / band groups for the EEG occlusion
6. notebooks: CONFIG values, every output path depends on outputs.run_tag, cells compile

Synthetic data, CPU only; no notebook cell is executed except the CONFIG cell (a dict
literal plus three path helpers).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.ablation import (  # noqa: E402
    ABLATION_CONDITIONS,
    ABLATION_LABELS,
    ALL_ABLATION_CONDITIONS,
    apply_condition,
    default_conditions,
    input_ablation,
)
from main.dataset import WAYEEGDataset  # noqa: E402
from main.eeg_features import (  # noqa: E402
    DEFAULT_BANDS,
    build_eeg_transform_from_config,
    clip_fraction,
    eeg_band_groups,
    eeg_bandpower,
    eeg_electrode_groups,
    eeg_feature_names,
    fit_feature_stats,
)
from main.interp import decompose_prediction, eeg_occlusion_attribution  # noqa: E402
from main.losses import CompositeEMGLoss, build_loss_from_config  # noqa: E402
from main.model import (  # noqa: E402
    KGGTModel,
    TransformerOnlyModel,
    build_kg_gt_from_config,
    build_transformer_only_from_config,
    electrode_dropout,
)
from main.training import normalized_prepare_batch_factory  # noqa: E402

FS = 500.0
NOTEBOOKS = ["main.ipynb", "main_1stage.ipynb"]
RUN_TAG = "p2b"


# ════════════════════════════════════════════════════════════════════════════
# 1. Log band power
# ════════════════════════════════════════════════════════════════════════════

def _noise(T=10000, C=3, seed=0, sd=1.0):
    return np.random.default_rng(seed).standard_normal((T, C)) * sd


def test_bandpower_shape_layout_dtype_finite():
    x = _noise()
    f = eeg_bandpower(x, FS)
    assert f.shape == (10000, len(DEFAULT_BANDS) * 3)
    assert f.dtype == np.float32 and np.isfinite(f).all()
    # band-major layout: column b * C + c is band b of channel c
    one = eeg_bandpower(x[:, [2]], FS)
    np.testing.assert_allclose(f[:, 2::3], one, rtol=1e-5, atol=1e-5)
    two_bands = eeg_bandpower(x, FS, bands=[[8, 13], [13, 30]])
    assert two_bands.shape == (10000, 6)
    np.testing.assert_allclose(two_bands, f[:, 3:9], rtol=1e-5, atol=1e-5)
    # float32 input (the cache dtype) gives the same features
    np.testing.assert_allclose(eeg_bandpower(x.astype(np.float32), FS), f, rtol=1e-3, atol=1e-3)


def test_bandpower_rejects_bad_input():
    with pytest.raises(ValueError):
        eeg_bandpower(np.zeros(100), FS)                       # not (T, C)
    with pytest.raises(ValueError):
        eeg_bandpower(_noise(), FS, bands=[[30, 300]])         # above Nyquist
    with pytest.raises(ValueError):
        eeg_bandpower(_noise(), FS, bands=[[13, 8]])


def test_feature_names_and_groups_are_band_major():
    names = ["Fp1", "C3", "Cz"]
    feat = eeg_feature_names(names, DEFAULT_BANDS)
    assert len(feat) == 12
    assert feat[0] == "Fp1 4-8Hz" and feat[1] == "C3 4-8Hz" and feat[3] == "Fp1 8-13Hz"
    assert feat[-1] == "Cz 30-40Hz"
    groups = eeg_electrode_groups(names, 4)
    assert list(groups) == names
    assert groups["C3"] == [1, 4, 7, 10]
    assert eeg_electrode_groups(3, 4) == {0: [0, 3, 6, 9], 1: [1, 4, 7, 10], 2: [2, 5, 8, 11]}
    bands = eeg_band_groups(3, DEFAULT_BANDS)
    assert list(bands) == ["4-8Hz", "8-13Hz", "13-30Hz", "30-40Hz"]
    assert bands["8-13Hz"] == [3, 4, 5]
    # every column belongs to exactly one electrode and one band
    assert sorted(i for v in groups.values() for i in v) == list(range(12))
    assert sorted(i for v in bands.values() for i in v) == list(range(12))


def test_alpha_burst_raises_only_that_channels_alpha_feature():
    T, C = 10000, 3
    x = _noise(T, C, seed=1)
    t = np.arange(T) / FS
    t0 = 12.0                                                  # burst centre (s)
    env = np.exp(-0.5 * ((t - t0) / 0.6) ** 2)
    x[:, 1] += 6.0 * env * np.sin(2 * np.pi * 10.0 * t)
    f = eeg_bandpower(x, FS)
    burst = np.abs(t - t0) < 0.3
    rest = np.abs(t - t0) > 3.0
    rise = f[burst].mean(axis=0) - f[rest].mean(axis=0)        # log units, per feature
    alpha_ch1 = 1 * C + 1
    assert rise[alpha_ch1] > 4.0                               # > e^4 = 55x the resting alpha power
    others = np.delete(rise, alpha_ch1)
    assert np.abs(others).max() < 1.0, rise.round(2)
    assert rise[alpha_ch1] > 5 * np.abs(others).max()


def test_bandpower_is_zero_phase():
    """The feature peaks where the burst peaks: the filters add no delay."""
    T = 10000
    t = np.arange(T) / FS
    for t0 in (7.3, 12.0):
        x = _noise(T, 2, seed=5, sd=0.01)
        x[:, 0] += np.exp(-0.5 * ((t - t0) / 0.6) ** 2) * np.sin(2 * np.pi * 10.0 * t)
        f = eeg_bandpower(x, FS)
        peak = int(np.argmax(f[:, 1 * 2 + 0]))                 # alpha of channel 0
        assert abs(peak - int(round(t0 * FS))) <= 5, (t0, peak)   # 10 ms; a causal filter chain lags > 100 ms
        # symmetric burst -> symmetric feature around the peak
        left, right = f[peak - 250:peak, 2][::-1], f[peak + 1:peak + 251, 2]
        assert np.abs(left - right).max() < 0.15


def test_relative_floor_limits_a_silent_stretch():
    T = 20000
    x = _noise(T, 2, seed=2)
    x[6000:9000, 0] = 0.0                                      # amplifier dropout: exact zeros
    f = eeg_bandpower(x, FS)
    for b in range(len(DEFAULT_BANDS)):
        col = f[:, b * 2 + 0].astype(np.float64)
        bound = np.median(col) + np.log(1e-3)                  # log(1e-3 x median band power)
        assert col.min() >= bound - 1e-3
        assert col[7000:8000].max() <= bound + 1e-3            # the silent stretch sits ON the floor
        other = f[:, b * 2 + 1].astype(np.float64)
        # normal data: only the deepest fades of the narrow-band envelope reach the floor
        other_bound = np.median(other) + np.log(1e-3)
        assert other.min() >= other_bound - 1e-3
        assert (other <= other_bound + 1e-3).mean() < 0.001
    # the default smoothing kernel is positive; a 4th-order Butterworth rings below zero
    # and would pin far more samples of ordinary data to the floor
    def on_floor(feat):
        col = feat[:, 3 * 2 + 1].astype(np.float64)            # 30-40 Hz of the normal channel
        return float((col <= np.median(col) + np.log(1e-3) + 1e-3).mean())
    assert on_floor(eeg_bandpower(x, FS, smooth_order=4)) > 0.01 > 10 * on_floor(f)
    # an all-zero channel stays finite (absolute floor)
    z = eeg_bandpower(np.zeros((4000, 1)), FS)
    assert np.isfinite(z).all() and np.ptp(z) == 0.0


def test_transform_from_config():
    assert build_eeg_transform_from_config({}, FS) is None
    assert build_eeg_transform_from_config({"eeg_features": None}, FS) is None
    cfg = {"eeg_features": {"type": "bandpower", "bands": [[4, 8], [8, 13]], "smooth_hz": 3.0}}
    fn = build_eeg_transform_from_config(cfg, FS)
    x = _noise(4000, 2, seed=3)
    np.testing.assert_array_equal(fn(x), eeg_bandpower(x, FS, bands=[[4, 8], [8, 13]], smooth_hz=3.0))
    with pytest.raises(ValueError):
        build_eeg_transform_from_config({"eeg_features": {"type": "wavelet"}}, FS)


def test_feature_stats_are_fitted_on_the_given_series_only():
    rng = np.random.default_rng(4)
    train = [rng.standard_normal((500, 4)) * 2 + 1, rng.standard_normal((300, 4)) * 2 + 1]
    test = [rng.standard_normal((400, 4)) * 9 - 7]
    mean, std = fit_feature_stats(train)
    cat = np.concatenate(train)
    np.testing.assert_allclose(mean, cat.mean(0), rtol=1e-5)
    np.testing.assert_allclose(std, cat.std(0), rtol=1e-5)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    m2, s2 = fit_feature_stats(train + test)
    assert not np.allclose(m2, mean, atol=0.5)                  # the test series would change them
    const_mean, const_std = fit_feature_stats([np.full((50, 2), 3.0)])
    np.testing.assert_allclose(const_mean, 3.0)
    assert (const_std >= 1e-6).all()                            # clamped, never 0
    frac = clip_fraction(train, mean, std, clip=2.0)
    assert 0.03 < frac < 0.06                                   # ~4.6 % of a normal beyond 2 SD
    assert clip_fraction(train, mean, std, clip=50.0) == 0.0


# ════════════════════════════════════════════════════════════════════════════
# 2. Dataset: eeg_transform on every split, never in the cache
# ════════════════════════════════════════════════════════════════════════════

_SPLITS = {"train": [1, 2], "val": [3], "test": [4]}
_T, _C_EEG, _C_KIN, _C_EMG, _WIN = 1200, 3, 4, 5, 100


def _fake_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    for split, sids in _SPLITS.items():
        for sid in sids:
            np.savez_compressed(
                cache / f"P1_{split}_S{sid}.npz",
                eeg=rng.standard_normal((_T, _C_EEG)).astype(np.float32),
                kin=rng.standard_normal((_T, _C_KIN)).astype(np.float32),
                emg=rng.random((_T, _C_EMG)).astype(np.float32),
            )
    return cache


def _hashes(folder: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def _double(eeg):
    """Stand-in transform: (T, C) -> (T, 2C), depends on the WHOLE series."""
    eeg = np.asarray(eeg, dtype=np.float64)
    return np.concatenate([eeg - eeg.mean(axis=0, keepdims=True), 2.0 * eeg], axis=1).astype(np.float32)


@pytest.fixture
def patched_split(monkeypatch):
    import main.dataset as ds_mod
    monkeypatch.setattr(ds_mod, "get_split_series", lambda split, p, data_dir=None: _SPLITS[split])


def test_dataset_applies_eeg_transform_to_every_split_and_leaves_the_cache_alone(tmp_path, patched_split):
    cache = _fake_cache(tmp_path)
    before = _hashes(cache)
    for split, sids in _SPLITS.items():
        plain = WAYEEGDataset(None, participants=[1], split=split, window_size=_WIN, stride=_WIN,
                              cache_dir=cache)
        ds = WAYEEGDataset(None, participants=[1], split=split, window_size=_WIN, stride=_WIN,
                           cache_dir=cache, eeg_transform=_double)
        assert len(ds) == len(plain) == len(sids) * (_T // _WIN)
        e0, k0, m0 = plain[5]
        e1, k1, m1 = ds[5]
        assert e0.shape == (_WIN, _C_EEG) and e1.shape == (_WIN, 2 * _C_EEG)
        assert e1.dtype == torch.float32
        torch.testing.assert_close(k0, k1)
        torch.testing.assert_close(m0, m1)
        # the transform saw the whole series, not the window
        full_raw = np.load(cache / f"P1_{split}_S{sids[0]}.npz")["eeg"]
        expected = _double(full_raw)[5 * _WIN:6 * _WIN]
        np.testing.assert_allclose(e1.numpy(), expected, rtol=1e-6, atol=1e-6)
        assert not np.allclose(e1.numpy()[:, :_C_EEG], _double(full_raw[5 * _WIN:6 * _WIN])[:, :_C_EEG], atol=1e-3)
    assert _hashes(cache) == before                             # no file rewritten, none added
    assert np.load(cache / "P1_train_S1.npz")["eeg"].shape == (_T, _C_EEG)


def test_dataset_default_has_no_transform(tmp_path, patched_split):
    cache = _fake_cache(tmp_path)
    ds = WAYEEGDataset(None, participants=[1], split="train", window_size=_WIN, stride=_WIN, cache_dir=cache)
    raw = np.load(cache / "P1_train_S1.npz")["eeg"]
    np.testing.assert_array_equal(ds[0][0].numpy(), raw[:_WIN])


def test_dataset_caches_the_untransformed_eeg_when_it_preprocesses(tmp_path, patched_split):
    cache = _fake_cache(tmp_path)
    ds = WAYEEGDataset(None, participants=[1], split="val", window_size=_WIN, stride=_WIN, cache_dir=cache)
    n0 = len(ds)
    rng = np.random.default_rng(1)
    series = {"series": 9, "eeg": rng.standard_normal((_T, _C_EEG)).astype(np.float32),
              "kin": rng.standard_normal((_T, _C_KIN)).astype(np.float32),
              "emg": rng.random((_T, _C_EMG)).astype(np.float32)}
    ds.eeg_transform = _double
    ds._index_series(series, participant=1)                     # the "not cached yet" path
    saved = np.load(cache / "P1_val_S9.npz")
    np.testing.assert_array_equal(saved["eeg"], series["eeg"])  # raw EEG in the cache
    assert len(ds) == n0 + _T // _WIN
    assert ds._windows[-1][0].shape == (_T, 2 * _C_EEG)         # transformed EEG in memory


def test_dataset_rejects_a_transform_that_changes_the_length(tmp_path, patched_split):
    cache = _fake_cache(tmp_path)
    with pytest.raises(ValueError):
        WAYEEGDataset(None, participants=[1], split="val", window_size=_WIN, stride=_WIN,
                      cache_dir=cache, eeg_transform=lambda e: e[:-1])


# ════════════════════════════════════════════════════════════════════════════
# 3. prepare_batch: frozen train statistics, EEG z-score + clip
# ════════════════════════════════════════════════════════════════════════════

def _reference_prepare_batch(emg_mean, emg_std, kin_mean, kin_std, device, drop_kin_indices=None):
    """The notebook closure as it was before Phase B (kept here as the reference)."""
    def prepare_batch(eeg, kin, emg):
        eeg = eeg.to(device, non_blocking=True)
        kin = kin.to(device, non_blocking=True)
        emg = emg.to(device, non_blocking=True)
        kin_norm = (kin - kin_mean) / kin_std
        if drop_kin_indices is not None:
            keep_idx = [i for i in range(kin_norm.shape[-1]) if i not in drop_kin_indices]
            kin_norm = kin_norm[..., keep_idx]
        emg_norm = (emg - emg_mean) / emg_std
        return {"eeg": eeg, "kin": kin_norm}, emg_norm
    return prepare_batch


def _stats(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g), torch.rand(n, generator=g) + 0.5


def test_prepare_batch_without_eeg_stats_is_the_old_closure():
    dev = torch.device("cpu")
    em, es = _stats(5, 0)
    km, ks = _stats(6, 1)
    g = torch.Generator().manual_seed(2)
    eeg, kin, emg = torch.randn(3, 20, 8, generator=g), torch.randn(3, 20, 6, generator=g), torch.rand(3, 20, 5, generator=g)
    for drop in (None, [1, 4]):
        ref_in, ref_y = _reference_prepare_batch(em, es, km, ks, dev, drop)(eeg, kin, emg)
        new_in, new_y = normalized_prepare_batch_factory(em, es, km, ks, dev, drop_kin_indices=drop)(eeg, kin, emg)
        assert set(new_in) == {"eeg", "kin"}
        assert torch.equal(new_in["eeg"], ref_in["eeg"]) and torch.equal(new_in["kin"], ref_in["kin"])
        assert torch.equal(new_y, ref_y)


def test_prepare_batch_zscores_and_clips_the_eeg_features():
    dev = torch.device("cpu")
    em, es = _stats(5, 0)
    km, ks = _stats(6, 1)
    rng = np.random.default_rng(3)
    train = [rng.standard_normal((400, 8)) * 3 + 10, rng.standard_normal((300, 8)) * 3 + 10]
    mean, std = fit_feature_stats(train)                       # numpy, train series only
    pb = normalized_prepare_batch_factory(em, es, km, ks, dev, eeg_mean=mean, eeg_std=std)
    eeg = torch.as_tensor(np.stack([train[0][:50], train[1][:50]]), dtype=torch.float32)
    eeg[0, 3, 2] = 1e4                                          # artefact
    eeg[1, 7, 5] = -1e4
    kin, emg = torch.zeros(2, 50, 6), torch.zeros(2, 50, 5)
    inputs, _ = pb(eeg, kin, emg)
    z = inputs["eeg"]
    assert z.dtype == torch.float32 and z.shape == eeg.shape
    assert z.max().item() == 5.0 and z.min().item() == -5.0
    assert z[0, 3, 2].item() == 5.0 and z[1, 7, 5].item() == -5.0
    expected = ((eeg - torch.as_tensor(mean)) / torch.as_tensor(std)).clamp(-5.0, 5.0)
    torch.testing.assert_close(z, expected)
    assert abs(z.mean().item()) < 0.3 and 0.7 < z.std().item() < 1.3
    # other clip value / tensors as statistics
    pb3 = normalized_prepare_batch_factory(em, es, km, ks, dev, eeg_mean=torch.as_tensor(mean),
                                           eeg_std=torch.as_tensor(std), eeg_clip=3.0)
    assert pb3(eeg, kin, emg)[0]["eeg"].abs().max().item() == 3.0
    with pytest.raises(ValueError):
        normalized_prepare_batch_factory(em, es, km, ks, dev, eeg_mean=mean)      # mean without std


# ════════════════════════════════════════════════════════════════════════════
# 4. Model: kinematic-pathway dropout, use_kin, electrode dropout
# ════════════════════════════════════════════════════════════════════════════

WIN, EEG_DIM, KIN_DIM, N_CH = 40, 12, 8, 5
_T_CFG = {"n_layers": 1, "n_heads": 2, "d_model": 16, "d_k": 8, "d_v": 8,
          "ffn_dim": 32, "dropout": 0.0, "chunk_size": WIN}


def _randomise(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give the zero-initialised layers (GAT out, kin skip, slope) real weights."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return model


def _kggt(seed=0, **kw) -> KGGTModel:
    kw.setdefault("gat_dropout", 0.0)
    torch.manual_seed(seed)
    m = KGGTModel(_T_CFG, input_dim=EEG_DIM, node_dim=8, gat_hidden_dim=8, gat_heads=2,
                  out_channels=N_CH, kin_dim=KIN_DIM, use_kinematic_guidance=True, **kw)
    return _randomise(m, seed)


def _batch(B=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, WIN, EEG_DIM, generator=g), torch.randn(B, WIN, KIN_DIM, generator=g),
            torch.randn(B, WIN, N_CH, generator=g))


def _seeded(model, *args, seed=0, **kw):
    torch.manual_seed(seed)
    return model(*args, **kw)


def test_kin_dropout_zero_is_identical():
    eeg, kin, _ = _batch()
    base, same = _kggt(), _kggt(kin_dropout=0.0)
    assert base.kin_dropout == 0.0
    for mode in (True, False):
        base.train(mode)
        same.train(mode)
        assert torch.equal(_seeded(base, eeg, kin), _seeded(same, eeg, kin))


def test_kin_dropout_is_off_in_eval():
    eeg, kin, _ = _batch()
    ref = _kggt().eval()
    for p in (0.3, 1.0):
        m = _kggt(kin_dropout=p).eval()
        m.load_state_dict(ref.state_dict())
        assert torch.equal(m(eeg, kin), ref(eeg, kin))
    with pytest.raises(ValueError):
        _kggt(kin_dropout=1.5)


def test_kin_dropout_one_equals_use_kin_false_and_ignores_kin():
    eeg, kin, _ = _batch()
    m = _kggt(kin_dropout=1.0).train()
    out = _seeded(m, eeg, kin)
    off = _seeded(m, eeg, kin, use_kin=False)
    torch.testing.assert_close(out, off, rtol=0, atol=1e-6)
    other_kin = kin * -3.0 + 7.0
    torch.testing.assert_close(_seeded(m, eeg, other_kin), out, rtol=0, atol=1e-6)
    # ... while the full model does depend on the kinematics
    full = _kggt().train()
    full.load_state_dict(m.state_dict())
    assert not torch.allclose(_seeded(full, eeg, kin), _seeded(full, eeg, other_kin), atol=1e-3)


def test_kin_dropout_switches_whole_samples():
    eeg, kin, _ = _batch(B=64)
    m = _kggt(kin_dropout=0.5).train()
    out = _seeded(m, eeg, kin, seed=3)
    off = _seeded(m, eeg, kin, seed=3, use_kin=False)
    dropped = (out - off).abs().amax(dim=(1, 2)) < 1e-6        # per sample: equal to the EEG-only output
    assert 12 < int(dropped.sum()) < 52                        # about half of 64
    kept = ~dropped
    assert ((out - off).abs().amax(dim=(1, 2))[kept] > 1e-3).all()


def test_use_kin_false_is_not_the_same_as_zero_kinematics():
    eeg, kin, _ = _batch()
    m = _kggt().eval()
    off = m(eeg, kin, use_kin=False)
    zeros = m(eeg, torch.zeros_like(kin))
    assert not torch.allclose(off, zeros, atol=1e-3)           # zeros still pass the skip MLP / edge bias
    assert torch.equal(m(eeg, None, use_kin=False), off)       # kin is not needed
    assert torch.equal(m.eeg_only_forward(eeg), off)
    assert torch.equal(m(eeg, kin * 5.0, use_kin=False), off)
    with pytest.raises(ValueError):
        m(eeg, None)                                           # the full model still needs kin


def test_use_kin_false_is_the_graph_path_with_the_kinematic_edge_bias_zeroed():
    eeg, kin, _ = _batch()
    m = _kggt().eval()
    parts = decompose_prediction(m, eeg, kin, as_numpy=False)
    torch.testing.assert_close(m(eeg, kin, use_kin=False), parts["graph_no_kin_bias"], rtol=0, atol=1e-5)
    # the decomposition is still exact
    torch.testing.assert_close(parts["graph"] + parts["kin"], parts["total"], rtol=0, atol=1e-4)
    torch.testing.assert_close(parts["total"], m(eeg, kin), rtol=0, atol=1e-5)


def test_non_guided_and_transformer_only_models_accept_use_kin():
    eeg, kin, _ = _batch()
    torch.manual_seed(0)
    plain = KGGTModel(_T_CFG, input_dim=EEG_DIM, node_dim=8, gat_hidden_dim=8, gat_heads=2,
                      out_channels=N_CH, kin_dim=KIN_DIM, use_kinematic_guidance=False, gat_dropout=0.0).eval()
    assert torch.equal(plain(eeg, kin, use_kin=False), plain(eeg, kin))
    t_only = TransformerOnlyModel(_T_CFG, input_dim=EEG_DIM, node_dim=8, out_channels=N_CH).eval()
    assert torch.equal(t_only(eeg, kin, use_kin=False), t_only(eeg))


def test_electrode_dropout_function_zeroes_whole_electrodes():
    B, T, E, G = 200, 7, 4, 3
    x = torch.ones(B, T, E * G)
    assert electrode_dropout(x, 0.0, G, training=True) is x
    assert electrode_dropout(x, 0.5, G, training=False) is x
    torch.manual_seed(0)
    y = electrode_dropout(x, 0.25, G, training=True)
    per_electrode = y.view(B, T, G, E)                         # band-major: column = band * E + electrode
    for b in range(B):
        for e in range(E):
            vals = per_electrode[b, :, :, e]
            assert (vals == 0).all() or torch.allclose(vals, torch.full_like(vals, 1 / 0.75))
    dropped = (per_electrode[:, 0, 0, :] == 0).float().mean().item()
    assert 0.18 < dropped < 0.32
    assert abs(y.mean().item() - 1.0) < 0.08                   # inverted scaling keeps the expectation
    with pytest.raises(ValueError):
        electrode_dropout(torch.ones(2, 3, 10), 0.2, 4, training=True)   # 10 features, 4 bands


@pytest.mark.parametrize("kind", ["kggt", "transformer_only"])
def test_models_apply_electrode_dropout_only_in_training(kind):
    groups = 4
    if kind == "kggt":
        m = _kggt(eeg_channel_dropout=0.5, eeg_groups=groups)
    else:
        torch.manual_seed(0)
        m = TransformerOnlyModel(_T_CFG, input_dim=EEG_DIM, node_dim=8, out_channels=N_CH,
                                 eeg_channel_dropout=0.5, eeg_groups=groups)
    seen = []
    m.encoder.embed.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    eeg, kin, _ = _batch(B=16)
    eeg = eeg.abs() + 1.0                                      # no zeros in the input
    m.train()
    _seeded(m, eeg, kin)
    x = seen[-1]
    E = EEG_DIM // groups
    zero = (x == 0).view(16, WIN, groups, E)
    assert zero.any() and not zero.all()
    assert (zero == zero[:, :1, :1, :]).all()                  # same electrodes for all bands and time steps
    m.eval()
    m(eeg, kin)
    assert torch.equal(seen[-1], eeg)
    with pytest.raises(ValueError):
        KGGTModel(_T_CFG, input_dim=EEG_DIM, node_dim=8, gat_hidden_dim=8, gat_heads=2,
                  out_channels=N_CH, kin_dim=KIN_DIM, use_kinematic_guidance=True,
                  eeg_channel_dropout=0.2, eeg_groups=5)


def _full_cfg(**model_extra):
    cfg = {
        "model": {"type": "transformer_regressor", "transformer": dict(_T_CFG),
                  "gat": {"n_layers": 1, "node_dim": 8, "hidden_dim": 8, "heads": 2, "dropout": 0.0,
                          "use_kinematic_guidance": True},
                  "decoder": {"out_channels": N_CH}},
        "training": {"loss": {}},
    }
    cfg["model"].update(model_extra)
    return cfg


def test_builders_read_the_phase_b_keys():
    old = build_kg_gt_from_config(_full_cfg(), input_dim=EEG_DIM, kin_dim=KIN_DIM)
    assert (old.kin_dropout, old.eeg_channel_dropout, old.eeg_groups, old.eeg_only_aux) == (0.0, 0.0, 1, False)
    cfg = _full_cfg(kin_dropout=0.3, eeg_channel_dropout=0.2, eeg_groups=4)
    cfg["training"]["loss"]["w_eeg_only"] = 0.5
    new = build_kg_gt_from_config(cfg, input_dim=EEG_DIM, kin_dim=KIN_DIM)
    assert (new.kin_dropout, new.eeg_channel_dropout, new.eeg_groups, new.eeg_only_aux) == (0.3, 0.2, 4, True)
    assert set(new.state_dict()) == set(old.state_dict())      # no new parameters: old checkpoints still load
    t_old = build_transformer_only_from_config(_full_cfg(), input_dim=EEG_DIM)
    t_new = build_transformer_only_from_config(cfg, input_dim=EEG_DIM)
    assert (t_old.eeg_channel_dropout, t_old.eeg_groups) == (0.0, 1)
    assert (t_new.eeg_channel_dropout, t_new.eeg_groups) == (0.2, 4)
    no_training = {"model": cfg["model"]}                      # configs without a training section
    assert build_kg_gt_from_config(no_training, input_dim=EEG_DIM, kin_dim=KIN_DIM).eeg_only_aux is False
    loss = build_loss_from_config(cfg["training"])
    assert loss.w_eeg_only == 0.5
    assert build_loss_from_config({"loss": {}}).w_eeg_only == 0.0


# ════════════════════════════════════════════════════════════════════════════
# 5. EEG-only auxiliary loss
# ════════════════════════════════════════════════════════════════════════════

def test_eeg_only_weight_zero_is_the_old_loss():
    eeg, kin, y = _batch()
    m = _kggt(eeg_only_aux=True).train()
    pred = _seeded(m, eeg, kin)
    old, new = CompositeEMGLoss(), CompositeEMGLoss(w_eeg_only=0.0)
    a = old(pred, y, model=m)
    b = new(pred, y, model=m)
    assert a.item() == b.item()
    assert new.last_components["eeg_only"] == 0.0
    assert {k: v for k, v in new.last_components.items() if k != "eeg_only"} == \
        {k: v for k, v in old.last_components.items() if k != "eeg_only"}


def test_eeg_only_term_adds_exactly_w_times_the_loss_of_the_eeg_only_prediction():
    eeg, kin, y = _batch()
    w = 0.5
    kwargs = dict(w_rest_over=1.0, w_height=0.5, huber_delta=2.5, asymmetry=1.5,
                  channel_weights=[0.5, 1.0, 1.5, 1.0, 1.0])
    m = _kggt(eeg_only_aux=True).train()
    pred = _seeded(m, eeg, kin)
    eeg_only = m.last_eeg_only_output
    assert eeg_only is not None and eeg_only.shape == pred.shape and eeg_only.requires_grad
    torch.testing.assert_close(eeg_only, _seeded(m, eeg, kin, use_kin=False), rtol=0, atol=1e-6)
    m.last_eeg_only_output = eeg_only                           # (the forward above reset it)
    base = CompositeEMGLoss(**kwargs)
    main_with_kl = base(pred, y, model=m).item()
    aux = base(eeg_only, y).item()                              # no model -> no KL term
    loss = CompositeEMGLoss(w_eeg_only=w, **kwargs)
    total = loss(pred, y, model=m)
    assert total.item() == pytest.approx(main_with_kl + w * aux, rel=1e-5)
    assert loss.last_components["eeg_only"] == pytest.approx(aux, rel=1e-5)
    assert loss.last_components["total"] == pytest.approx(total.item(), rel=1e-6)
    assert loss.last_components["kl_reg"] > 0
    assert m.last_eeg_only_output is None                       # consumed: nothing stale left on the model


def test_eeg_only_term_is_off_in_eval_and_without_a_model():
    eeg, kin, y = _batch()
    m = _kggt(eeg_only_aux=True)
    loss, base = CompositeEMGLoss(w_eeg_only=0.5), CompositeEMGLoss()
    m.eval()
    pred = m(eeg, kin)
    assert m.last_eeg_only_output is None                       # no extra pass in eval
    assert loss(pred, y, model=m).item() == base(pred, y, model=m).item()
    assert loss(pred, y).item() == base(pred, y).item()         # validation call: no model
    assert loss.last_components["eeg_only"] == 0.0


def test_eeg_only_term_needs_the_models_cached_output():
    eeg, kin, y = _batch()
    loss = CompositeEMGLoss(w_eeg_only=0.5)
    m = _kggt().train()                                         # guided, but built without eeg_only_aux
    with pytest.raises(RuntimeError, match="eeg_only_aux"):
        loss(_seeded(m, eeg, kin), y, model=m)
    # a model without kinematic pathways already is EEG-only: nothing to add
    t_only = TransformerOnlyModel(_T_CFG, input_dim=EEG_DIM, node_dim=8, out_channels=N_CH).train()
    pred = _seeded(t_only, eeg, kin)
    assert loss(pred, y, model=t_only).item() == CompositeEMGLoss()(pred, y, model=t_only).item()


def test_eeg_only_term_sends_gradients_to_the_encoder_even_if_the_main_output_ignores_the_eeg():
    eeg, kin, y = _batch()
    m = _kggt(eeg_only_aux=True).train()

    def encoder_grad(w):
        m.zero_grad(set_to_none=True)
        pred = _seeded(m, eeg, kin).detach()                    # main output cut off from the EEG path
        CompositeEMGLoss(w_eeg_only=w)(pred, y, model=m).backward()
        g = m.encoder.embed.weight.grad
        return 0.0 if g is None else float(g.abs().sum())

    assert encoder_grad(0.0) == 0.0
    assert encoder_grad(0.5) > 1e-6
    # the kinematic pathways get nothing from the EEG-only term
    skip_grad = m.kin_skip_proj[0].weight.grad
    assert skip_grad is None or float(skip_grad.abs().sum()) == 0.0


def test_eeg_only_term_under_autocast_and_gradient_accumulation():
    m = _kggt(eeg_only_aux=True, kin_dropout=0.3, eeg_channel_dropout=0.2, eeg_groups=4).train()
    loss = CompositeEMGLoss(w_eeg_only=0.5, w_height=0.5, w_rest_over=1.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    for step in range(2):                                       # two accumulated batches, one optimizer step
        eeg, kin, y = _batch(seed=step)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            pred = m(eeg, kin)
            value = loss(pred, y, model=m)
        assert torch.isfinite(value)
        assert loss.last_components["eeg_only"] > 0
        (value / 2).backward()
        assert m.last_eeg_only_output is None
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert float(m.encoder.embed.weight.grad.abs().sum()) > 0
    opt.step()


def test_model_with_a_cached_eeg_only_output_can_still_be_deep_copied():
    eeg, kin, _ = _batch()
    m = _kggt(eeg_only_aux=True).train()
    _seeded(m, eeg, kin)
    assert m.last_eeg_only_output is not None
    clone = copy.deepcopy(m)                                    # notebooks deep-copy the trained model
    assert clone.last_eeg_only_output is None
    assert m.pop_eeg_only_output() is not None and m.pop_eeg_only_output() is None


# ════════════════════════════════════════════════════════════════════════════
# 6. Ablation: kin_off
# ════════════════════════════════════════════════════════════════════════════

class _SwitchModel(nn.Module):
    """EEG part + kinematic part with a constant offset; ``use_kin=False`` drops the latter."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.eeg = nn.Linear(4, 2)
        self.kin = nn.Linear(3, 2)
        self.calls = []

    def forward(self, eeg, kin=None, use_kin=True):
        self.calls.append(bool(use_kin))
        out = self.eeg(eeg)
        return out + self.kin(kin) + 3.0 if use_kin else out


class _OldModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.eeg = nn.Linear(4, 2)

    def forward(self, eeg, kin=None):
        return self.eeg(eeg)


def _abl_loader(model=None):
    g = torch.Generator().manual_seed(5)
    eeg, kin = torch.randn(6, 40, 4, generator=g), torch.randn(6, 40, 3, generator=g)
    emg = torch.randn(6, 40, 2, generator=g)
    if model is not None:
        with torch.no_grad():
            emg = model.eeg(eeg) + 0.1 * emg                    # the target follows the EEG part
    return DataLoader(TensorDataset(eeg, kin, emg), batch_size=4)


def _plain_pb(eeg, kin, emg):
    return {"eeg": eeg, "kin": kin}, emg


def test_kin_off_condition_is_listed_and_labelled():
    assert ABLATION_CONDITIONS == ("full", "kin_zero", "kin_shuffle", "eeg_shuffle")   # unchanged
    assert ALL_ABLATION_CONDITIONS == ("full", "kin_zero", "kin_off", "kin_shuffle", "eeg_shuffle")
    assert ABLATION_LABELS["kin_off"] == "EEG only (kinematic paths off)"
    import main
    assert main.ALL_ABLATION_CONDITIONS is ALL_ABLATION_CONDITIONS
    assert main.ABLATION_COLORS["kin_off"] not in {main.ABLATION_COLORS[c] for c in ABLATION_CONDITIONS}
    x = {"eeg": torch.ones(1, 3, 4), "kin": torch.ones(1, 3, 3)}
    off = apply_condition(x, "kin_off")
    assert off["use_kin"] is False and off["eeg"] is x["eeg"] and "use_kin" not in x
    assert default_conditions(_SwitchModel()) == ALL_ABLATION_CONDITIONS
    assert default_conditions(_OldModel()) == ABLATION_CONDITIONS


def test_input_ablation_reports_kin_off_for_a_model_with_use_kin():
    model = _SwitchModel()
    loader = _abl_loader(model)
    res = input_ablation(model, loader, _plain_pb, torch.device("cpu"),
                         rest_threshold=np.zeros(2))
    assert res.conditions == ALL_ABLATION_CONDITIONS
    assert False in model.calls and True in model.calls
    with torch.no_grad():
        eeg_part = torch.cat([model.eeg(e) for e, _, _ in loader]).reshape(-1, 2).numpy()
    np.testing.assert_allclose(res.predictions["kin_off"], eeg_part, rtol=1e-6, atol=1e-6)
    assert res.r_mean["kin_off"] > 0.95                         # the target is the EEG part
    # zero kinematics are something else: the kin bias and the offset stay
    assert not np.allclose(res.predictions["kin_off"], res.predictions["kin_zero"], atol=1e-3)
    table = res.as_table(["a", "b"])
    assert "kin_off" in table and "kin_zero" in table
    assert "kin_off" in res.rest_pred


def test_input_ablation_default_is_unchanged_for_models_without_use_kin():
    model = _OldModel()
    res = input_ablation(model, _abl_loader(), _plain_pb, torch.device("cpu"))
    assert res.conditions == ABLATION_CONDITIONS
    with pytest.raises(TypeError, match="use_kin"):
        input_ablation(model, _abl_loader(), _plain_pb, torch.device("cpu"), conditions=("full", "kin_off"))


def test_input_ablation_kin_off_on_the_real_model():
    eeg, kin, emg = _batch(B=4)
    m = _kggt().eval()
    loader = DataLoader(TensorDataset(eeg, kin, emg), batch_size=2)
    res = input_ablation(m, loader, _plain_pb, torch.device("cpu"))
    assert "kin_off" in res.conditions
    with torch.no_grad():
        expected = m(eeg, kin, use_kin=False).reshape(-1, N_CH).numpy()
    np.testing.assert_allclose(res.predictions["kin_off"], expected, rtol=1e-5, atol=1e-5)


# ════════════════════════════════════════════════════════════════════════════
# 7. EEG occlusion with band-power features: electrodes = groups of n_bands columns
# ════════════════════════════════════════════════════════════════════════════

class _OneElectrodeModel(nn.Module):
    """Output = sum of the 4 band columns of electrode 1 (of 3), band 2 counted 10x."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, eeg, kin=None):
        cols = eeg[..., [1, 4, 7, 10]] * torch.tensor([1.0, 1.0, 10.0, 1.0])
        return cols.sum(-1, keepdim=True).repeat(1, 1, 2)


def test_occlusion_attribution_groups_the_band_columns_per_electrode():
    names = ["Fz", "C3", "Pz"]
    g = torch.Generator().manual_seed(0)
    eeg = torch.randn(4, 50, 12, generator=g)
    occ = eeg_occlusion_attribution(
        _OneElectrodeModel(), eeg, None, time_bin=10,
        electrode_groups=eeg_electrode_groups(names, 4), band_groups=eeg_band_groups(3, DEFAULT_BANDS),
    )
    assert occ["electrode_labels"] == names
    assert occ["electrode_abs"].shape == (4, 3, 2)
    assert (occ["electrode_abs"][:, 1] > 0.1).all()
    np.testing.assert_allclose(occ["electrode_abs"][:, [0, 2]], 0.0, atol=1e-7)
    assert occ["band_labels"] == ["4-8Hz", "8-13Hz", "13-30Hz", "30-40Hz"]
    assert occ["band_abs"].shape == (4, 4, 2) and occ["band_signed"].shape == (4, 4, 2)
    assert (occ["band_abs"][:, 2] > 3 * occ["band_abs"][:, [0, 1, 3]].max(axis=1)).all()
    # default call: every column alone, no band keys (as before)
    plain = eeg_occlusion_attribution(_OneElectrodeModel(), eeg, None, time_bin=10)
    assert plain["electrode_abs"].shape == (4, 12, 2) and "band_abs" not in plain


def test_event_locked_windows_with_electrode_and_band_groups(tmp_path):
    """The notebook path: band-power layout -> one bar per electrode, plus per-band numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.interp import concat_event_locked, event_locked_summary, event_locked_windows
    from main.interp_plots import plot_event_locked

    names = ["Fz", "C3", "Pz"]                                  # 3 electrodes x 4 bands = EEG_DIM
    rng = np.random.default_rng(0)
    T = 400
    eeg = rng.standard_normal((T, EEG_DIM)).astype(np.float32)
    kin = rng.standard_normal((T, KIN_DIM)).astype(np.float32)
    emg = rng.random((T, N_CH)).astype(np.float32)
    m = _kggt().eval()
    ev = event_locked_windows(
        m, eeg, kin, emg, [100, 200, 300, None], _plain_pb, pre=WIN // 2, post=WIN // 2,
        time_bin=10, batch_size=2,
        electrode_groups=eeg_electrode_groups(names, 4), band_groups=eeg_band_groups(3, DEFAULT_BANDS),
    )
    assert ev["electrode_labels"] == names and ev["band_labels"] == ["4-8Hz", "8-13Hz", "13-30Hz", "30-40Hz"]
    assert ev["occ_electrode_abs"].shape == (3, 3, N_CH) and ev["occ_band_abs"].shape == (3, 4, N_CH)
    summ = event_locked_summary(concat_event_locked([ev, ev]))
    assert summ["occ_band_abs_mean"].shape == (4, N_CH) and summ["band_labels"] == ev["band_labels"]
    fig = plot_event_locked(summ, ["AD", "BR", "FD", "ED", "FDI"], fs=20.0, eeg_effect=0.5)
    ticks = [lab.get_text() for ax in fig.axes for lab in ax.get_xticklabels()]
    assert all(n in ticks for n in names)                       # electrode names, not 12 feature columns
    fig.savefig(tmp_path / "event.png", dpi=40)
    plt.close(fig)
    # without groups nothing changes: every column alone, no band keys
    plain = event_locked_windows(m, eeg, kin, emg, [100], _plain_pb, pre=WIN // 2, post=WIN // 2,
                                 time_bin=10, deletion=False, rollout=False)
    assert plain["occ_electrode_abs"].shape == (1, EEG_DIM, N_CH) and "occ_band_abs" not in plain
    assert plain["band_labels"] is None


# ════════════════════════════════════════════════════════════════════════════
# 8. Notebooks
# ════════════════════════════════════════════════════════════════════════════

def _nb(name: str) -> dict:
    with open(ROOT / name, encoding="utf-8") as f:
        return json.load(f)


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell_src(nb, needle: str) -> str:
    hits = [s for s in _code_cells(nb) if needle in s]
    assert len(hits) == 1, f"{needle!r}: {len(hits)} cells"
    return hits[0]


def _config_ns(name: str) -> dict:
    ns = {"copy": copy}
    exec(_cell_src(_nb(name), "CONFIG = {"), ns)               # dict literal + path helpers
    return ns


def _strip_comments(code: str) -> str:
    """Code without comment-only lines and without trailing ``#`` comments."""
    out = []
    for line in code.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(re.sub(r"\s+#.*$", "", line))
    return "\n".join(out)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_code_cells_compile(name):
    for i, c in enumerate(_nb(name)["cells"]):
        if c["cell_type"] == "code":
            code = "\n".join("pass" if re.match(r"^\s*[%!]", ln) else ln for ln in _src(c).splitlines())
            compile(code, f"{name}[cell {i}]", "exec")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_config_turns_phase_b_on(name):
    cfg = _config_ns(name)["CONFIG"]
    assert cfg["outputs"]["run_tag"] == RUN_TAG
    feats = cfg["preprocessing"]["eeg_features"]
    assert feats == {"type": "bandpower", "bands": [[4, 8], [8, 13], [13, 30], [30, 40]], "smooth_hz": 5.0}
    assert cfg["data"]["cache_dir"] == "data/cache_main_stride500_square_pose_force_sg51"   # same cache
    model = cfg["model"]
    t = model["transformer"]
    assert (t["n_layers"], t["n_heads"], t["d_model"], t["d_k"], t["d_v"], t["ffn_dim"], t["dropout"]) == \
        (2, 4, 64, 16, 16, 256, 0.3)
    assert t["chunk_size"] == cfg["dataset"]["window_size"] == 500
    assert model["eeg_channel_dropout"] == 0.2 and model["eeg_groups"] == len(feats["bands"]) == 4
    assert model["kin_dropout"] == 0.3
    tr = cfg["training"]
    assert tr["weight_decay"] == 0.01
    assert tr["loss"]["w_eeg_only"] == 0.5
    # everything of Step 1 that Phase B does not touch
    assert tr["loss"]["w_height"] == 0.5 and tr["loss"]["channel_weights"] == "auto"
    assert model["gat"]["kin_gat_dim"] == 12 and model["gat"]["kin_skip_indices"] == list(range(10, 24))


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_model_builds_with_the_phase_b_config(name):
    cfg = _config_ns(name)["CONFIG"]
    n_feat = cfg["data"]["n_eeg_channels"] * len(cfg["preprocessing"]["eeg_features"]["bands"])
    assert n_feat == 128
    torch.manual_seed(0)
    m = build_kg_gt_from_config(cfg, input_dim=n_feat, kin_dim=cfg["data"]["n_kin_features"])
    assert (m.kin_dropout, m.eeg_channel_dropout, m.eeg_groups, m.eeg_only_aux) == (0.3, 0.2, 4, True)
    mhsa = m.encoder.layers[0].mhsa
    assert len(m.encoder.layers) == 2 and m.encoder.d_model == 64
    assert mhsa.W_Q.weight.shape == (4 * 16, 64) and mhsa.W_O.weight.shape == (64, 4 * 16)
    assert m.encoder.embed.in_features == 128 and m.node_projection.proj.in_features == 64
    eeg, kin = torch.randn(2, 500, n_feat), torch.randn(2, 500, 24)
    for mode in (True, False):
        m.train(mode)
        out = m(eeg, kin)
        assert out.shape == (2, 500, 5) and torch.isfinite(out).all()
    assert m(eeg, kin, use_kin=False).shape == (2, 500, 5)
    s1 = build_transformer_only_from_config(cfg, input_dim=n_feat)
    assert s1.train()(eeg).shape == (2, 500, 5) and s1.eeg_channel_dropout == 0.2
    loss = build_loss_from_config(cfg["training"], channel_weights=[1.0] * 5)
    assert loss.w_eeg_only == 0.5


@pytest.mark.parametrize("name,ckpt", [("main.ipynb", "outputs/checkpoints_main_p2b"),
                                       ("main_1stage.ipynb", "outputs/checkpoints_1stage_p2b")])
def test_notebook_paths_come_from_the_run_tag(name, ckpt):
    ns = _config_ns(name)
    cfg = ns["CONFIG"]
    assert ns["RUN_TAG"] == RUN_TAG
    assert ns["RESULTS_DIR"] == f"results/{RUN_TAG}"
    assert ns["tagged_output"]("interpretability") == f"outputs/interpretability_{RUN_TAG}"
    assert cfg["training"]["checkpoint_dir"] == ckpt
    cfg_src = _cell_src(_nb(name), "CONFIG = {")
    assert 'endswith("_" + RUN_TAG)' in cfg_src                 # a new tag without a new folder fails loudly
    assert "p2b" in cfg_src and "overwrit" in cfg_src.lower()   # the comment explaining the tag


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_untagged_output_path(name):
    """Outside the CONFIG cell no code names results/ or outputs/ directly."""
    nb = _nb(name)
    # a path fragment, or "outputs" / "results" as an os.path.join argument (not the CONFIG key)
    pattern = re.compile(r"""results/|outputs/|["'](?:results|outputs)["'](?!\s*\])""")
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = _src(c)
        code = _strip_comments(s)
        if "CONFIG = {" in s:                                   # the only cell that may spell a path out
            for line in code.splitlines():
                if re.search(r"results/|outputs/", line):
                    assert "RUN_TAG" in line or f'_{RUN_TAG}"' in line, line
            continue
        hit = [ln for ln in code.splitlines() if pattern.search(ln)]
        assert not hit, f"{name} cell {c.get('id')}: untagged path in {hit}"
    # ... and every writer goes through the tagged names
    code = "\n".join(_strip_comments(s) for s in _code_cells(nb))
    for needle in ("RESULTS_DIR", "tagged_output(", 'os.makedirs(RESULTS_DIR, exist_ok=True)'):
        assert needle in code, needle
    for var in ("REPORT_PATH", "MAGNITUDE_CSV", "MAGNITUDE_PNG", "MAGNITUDE_LOAD_CSV", "MAGNITUDE_LOAD_PNG",
                "ABLATION_PNG", "ABLATION_CSV"):
        line = next(ln for ln in code.splitlines() if re.match(rf"\s*{var}\s*=", ln))
        assert "RESULTS_DIR" in line, line
    for folder in ("interpretability", "kin_gate"):
        assert re.search(rf'tagged_output\(\s*"{folder}', code), folder
    # checkpoints: only through training.checkpoint_dir
    for s in _code_cells(nb):
        assert "checkpoints_main_stage" not in _strip_comments(s)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_helpers_build_the_transform_and_normalise_the_eeg(name):
    nb = _nb(name)
    helper = _cell_src(nb, "def build_dataset_split(")
    assert "build_eeg_transform_from_config(" in helper and "eeg_transform=" in helper
    assert "normalized_prepare_batch_factory(" in helper and "eeg_mean=eeg_mean" in helper
    assert "eeg_clip=EEG_CLIP" in helper and "EEG_CLIP = 5.0" in helper
    assert "def fit_eeg_normalisation(" in helper
    assert "fit_feature_stats(train_eegs)" in helper and "clip_fraction(" in helper
    assert "eeg_electrode_groups(" in helper and "eeg_band_groups(" in helper
    pre = _cell_src(nb, "PER-SUBJECT PRE-PROCESSING LOOP")
    assert re.search(r"fit_eeg_normalisation\(\s*_cfg, _train_eegs, device,", pre)   # TRAIN series only
    assert "_train_eegs, _train_kins, _train_emgs = unique_series_arrays(_train_ds)" in pre
    assert "eeg_mean=_eeg_mean, eeg_std=_eeg_std" in pre
    assert '_true_eeg_dim = _batch_in["eeg"].shape[-1]' in pre
    for split in ("train", "val", "test"):                      # one builder for all splits
        assert f'build_dataset_split(_cfg, participants=[p_id], split="{split}")' in pre
    train = _cell_src(nb, "PER-SUBJECT TRAINING LOOP")
    assert 'eeg_mean=_sd["eeg_mean"]' in train and 'eeg_groups=_sd["eeg_groups"]' in train
    loso = _cell_src(nb, "[OPTIONAL] Automatic LOSOCV")
    assert "local_prepare_batch_factory(" in loso and "fit_eeg_normalisation(" in loso
    assert "_tr_eegs, _tr_kins, _tr_emgs = unique_series_arrays(_train_ds)" in loso
    assert "def _make_pb(" not in loso                          # one prepare_batch for every cell
    event = _cell_src(nb, "Event-locked interpretability (all test lifts)")
    assert "electrode_groups=" in event and "band_groups=" in event


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_ablation_cell_reports_kin_off_and_the_acceptance_criteria(name):
    nb = _nb(name)
    abl = _cell_src(nb, "input_ablation(")
    assert "kin_off" in abl and "ALL_ABLATION_CONDITIONS" in abl
    assert "EEG-only r" in abl and "0.35" in abl and "0.88" in abl
    assert '_r_off = _abl.r_mean.get("kin_off"' in abl and "_abl.conditions" in abl
    assert "Δr (full − eeg_shuffle)" in abl and "full r" in abl and "Phase B acceptance" in abl
    title = _src(nb["cells"][0])
    assert nb["cells"][0]["cell_type"] == "markdown"
    for needle in ("Phase B", "band power", "kin_dropout", "w_eeg_only", "0.35", "0.88", RUN_TAG,
                   "electrode"):
        assert needle in title, needle


def test_both_notebooks_share_data_model_and_loss():
    a, b = _config_ns("main.ipynb")["CONFIG"], _config_ns("main_1stage.ipynb")["CONFIG"]
    for k in ("data", "preprocessing", "dataset", "model", "outputs"):
        assert a[k] == b[k], k
    assert a["training"]["loss"] == b["training"]["loss"]
    assert a["training"]["weight_decay"] == b["training"]["weight_decay"]
    assert a["training"]["checkpoint_dir"] != b["training"]["checkpoint_dir"]


def test_library_defaults_are_unchanged():
    """Old configs (no Phase-B keys) give the old behaviour."""
    assert CompositeEMGLoss().w_eeg_only == 0.0
    import inspect
    sig = inspect.signature(WAYEEGDataset.__init__).parameters
    assert sig["eeg_transform"].default is None
    sig = inspect.signature(KGGTModel.__init__).parameters
    assert sig["kin_dropout"].default == 0.0 and sig["eeg_channel_dropout"].default == 0.0
    assert sig["eeg_groups"].default == 1 and sig["eeg_only_aux"].default is False
    assert inspect.signature(KGGTModel.forward).parameters["use_kin"].default is True
    import main
    for name in ("eeg_bandpower", "eeg_feature_names", "eeg_electrode_groups", "eeg_band_groups",
                 "build_eeg_transform_from_config", "fit_feature_stats", "clip_fraction",
                 "normalized_prepare_batch_factory", "electrode_dropout", "default_conditions"):
        assert name in main.__all__ and callable(getattr(main, name)), name
