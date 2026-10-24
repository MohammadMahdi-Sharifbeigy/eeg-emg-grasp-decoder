import copy

import numpy as np
import pytest
import torch

import v3.guide
import v3.train
from v2.data import TokenWindows, fit_norm
from v3.config import CONFIG_V3
from v3.guide import build_heads
from v3.model import build_student
from v3.teacher import Teacher
from v3.train import fit_guided, load_student

DEV = torch.device("cpu")


def _cfg(**guide):
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["model"].update(d_model=16, n_heads=2, ffn_dim=32, kin_layers=2, eeg_layers=2, electrode_dim=16, node_dim=8,
                        electrode_gat_layers=1, muscle_gat_layers=1)
    cfg["tokens"].update(window=40, stride=20)
    cfg["training"].update(max_epochs=6, warmup_epochs=1, patience=1, smooth_window=1, batch_size=8, use_amp=False)
    cfg["guide"].update(hold_until=2, off_at=4, scale=0.1)
    cfg["guide"].update(guide)
    return cfg


def _series(seed, n=160):
    r = np.random.default_rng(seed)
    hold = np.zeros(n, dtype=np.int64)
    hold[30:70] = 1
    return {"kin": r.normal(size=(n, 24)).astype(np.float32), "hist": r.random((n, 8)).astype(np.float32),
            "eeg": r.normal(size=(n, 32, 4)).astype(np.float32), "emg": np.abs(r.normal(size=(n, 5))).astype(np.float32),
            "hold_id": hold}


def _data(cfg):
    train = [_series(0), _series(1)]
    norm = fit_norm(train)
    tc = cfg["tokens"]
    return (TokenWindows(train, norm, tc["window"], tc["stride"], random_offset=True, seed=0),
            TokenWindows([_series(2)], norm, tc["window"], tc["stride"]))


def _parts(cfg):
    torch.manual_seed(0)
    return build_student(cfg), Teacher(cfg["model"], None).eval(), build_heads(cfg)


def _quiet(*_a, **_k):
    return None


def test_best_pt_holds_the_student_only(tmp_path):
    cfg = _cfg()
    train, val = _data(cfg)
    s, t, h = _parts(cfg)
    fit_guided(s, t, h, train, val, cfg, DEV, tmp_path, log=_quiet)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["state"] and all(k.startswith("eeg.") for k in best["state"])
    assert "heads_state" not in best and not any(k.startswith(("kin.", "align", "aux")) for k in best["state"])
    assert "heads_state" in torch.load(tmp_path / "last.pt", weights_only=False)


def test_no_early_stop_before_off_at(tmp_path):
    cfg = _cfg()
    train, val = _data(cfg)
    s, t, h = _parts(cfg)
    hist = fit_guided(s, t, h, train, val, cfg, DEV, tmp_path, log=_quiet)
    assert hist["epochs_run"] >= cfg["guide"]["off_at"]


def test_selection_uses_the_e_loss_only(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg["training"]["patience"] = 50
    cfg["training"]["max_epochs"] = 8
    train, val = _data(cfg)
    s, t, h = _parts(cfg)
    seq = iter([5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    monkeypatch.setattr(v3.train, "evaluate_e_loss", lambda *a, **k: next(seq))
    monkeypatch.setattr(v3.guide, "align_loss", lambda *a, **k: torch.tensor(1e6, requires_grad=True) * 0 + 1e6)
    hist = fit_guided(s, t, h, train, val, cfg, DEV, tmp_path, log=_quiet)
    assert hist["val_e"] == [5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert hist["best_epoch"] == 4


def test_scale_zero_never_calls_teacher(tmp_path):
    cfg = _cfg(scale=0.0)
    train, val = _data(cfg)
    s, t, h = _parts(cfg)

    class Boom(torch.nn.Module):
        def forward(self, *a):
            raise AssertionError("teacher was called")

    fit_guided(s, Boom(), h, train, val, cfg, DEV, tmp_path / "a", log=_quiet)
    fit_guided(*_parts(cfg)[:1], None, h, train, val, cfg, DEV, tmp_path / "b", log=_quiet)
    cfg2 = _cfg(scale=0.1)
    with pytest.raises(ValueError):
        fit_guided(s, None, h, train, val, cfg2, DEV, tmp_path / "c", log=_quiet)


def test_teacher_unchanged(tmp_path):
    cfg = _cfg()
    train, val = _data(cfg)
    s, t, h = _parts(cfg)
    before = {k: v.clone() for k, v in t.state_dict().items()}
    fit_guided(s, t, h, train, val, cfg, DEV, tmp_path, log=_quiet)
    assert all(torch.equal(v, before[k]) for k, v in t.state_dict().items())


def test_resume_continues_weight(tmp_path):
    cfg = _cfg()
    cfg["training"]["patience"] = 50
    train, val = _data(cfg)
    full = fit_guided(*_parts(cfg), train, val, cfg, DEV, tmp_path / "full", log=_quiet)

    def stop_after_three(msg):
        if msg.strip().startswith("epoch   3"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fit_guided(*_parts(cfg), train, val, cfg, DEV, tmp_path / "cut", log=stop_after_three)
    resumed = fit_guided(*_parts(cfg), train, val, cfg, DEV, tmp_path / "cut", log=_quiet)
    assert resumed["w"] == full["w"] and resumed["epochs_run"] == 6


def test_completed_run_is_skipped(tmp_path, monkeypatch):
    cfg = _cfg()
    train, val = _data(cfg)
    first = fit_guided(*_parts(cfg), train, val, cfg, DEV, tmp_path, log=_quiet)

    def boom(*a, **k):
        raise AssertionError("trained again")

    monkeypatch.setattr(v3.train, "guided_loss", boom)
    s, t, h = _parts(cfg)
    again = fit_guided(s, t, h, train, val, cfg, DEV, tmp_path, log=_quiet)
    assert again["epochs_run"] == first["epochs_run"]
    load_student(s, tmp_path, DEV)


def test_run_dir_seeds_and_legacy_paths(tmp_path):
    from v3.train import run_dir
    assert run_dir(tmp_path, 0.1, 2, "P3") == tmp_path / "g0.1" / "s2" / "P3"
    assert run_dir(tmp_path, 0.1, 0, "P3") == tmp_path / "g0.1" / "s0" / "P3"
    legacy = tmp_path / "g0.1" / "P3"
    legacy.mkdir(parents=True)
    (legacy / "best.pt").write_bytes(b"x")
    assert run_dir(tmp_path, 0.1, 0, "P3") == legacy                       # an old run is read as seed 0
    assert run_dir(tmp_path, 0.1, 1, "P3") == tmp_path / "g0.1" / "s1" / "P3"
    (tmp_path / "g0.1" / "s0" / "P3").mkdir(parents=True)
    assert run_dir(tmp_path, 0.1, 0, "P3") == tmp_path / "g0.1" / "s0" / "P3"   # a new seed-0 run wins
