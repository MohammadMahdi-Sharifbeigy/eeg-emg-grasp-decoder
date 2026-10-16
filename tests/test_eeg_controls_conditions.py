"""Parts S and E of the EEG controls, phase 1: the condition runner (electrode subsets, eye regression)."""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import eeg_controls.run_conditions as RC
from eeg_controls.run_conditions import CONDITIONS, condition_config, run_one, summarise
from v2.electrodes import channel_names

CENTRAL = ["C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6"]
EDGE = ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "TP9", "TP10", "P7", "P8", "PO9", "PO10", "O1", "Oz", "O2"]
EYE = ["Fp1", "Fp2", "F7", "F8"]


def test_condition_configs():
    assert set(CONDITIONS) == {"central", "edge", "eye_clean"}
    for h in (0, 200, 500):
        for name in CONDITIONS:
            cfg = condition_config(name, h)
            assert cfg["model"]["causal"] and cfg["tokens"]["causal"]
            assert cfg["tokens"]["horizon"] == h // 20
            assert cfg["training"]["freeze_kin"] is True
    assert condition_config("central", 0)["eeg"]["channels"] == CENTRAL
    assert condition_config("edge", 0)["eeg"]["channels"] == EDGE
    eye = condition_config("eye_clean", 0)["eeg"]
    assert eye["eye_regress"] == EYE and eye.get("channels") is None
    assert len(channel_names(condition_config("eye_clean", 0))) == 28
    assert "eye_regress" not in condition_config("central", 0)["eeg"]
    smoke = condition_config("edge", 200, smoke=True)
    assert smoke["training"]["max_epochs"] == 1 and smoke["evaluation"]["n_null"] == 5
    # the module-level CONDITIONS are not changed by building a config
    condition_config("central", 0)["eeg"]["channels"].append("Pz")
    assert CONDITIONS["central"]["channels"] == CENTRAL
    with pytest.raises(ValueError, match="central.*edge.*eye_clean"):
        condition_config("frontal", 0)


def _args(tmp_path, **kw):
    a = dict(kin_checkpoints=str(tmp_path / "kin"), checkpoints=str(tmp_path / "ckpt"),
             results=str(tmp_path / "res"), smoke=True, amp=False, overwrite=False)
    a.update(kw)
    return Namespace(**a)


def test_missing_kin_checkpoint_message(tmp_path):
    (tmp_path / "kin").mkdir()
    expected = tmp_path / "kin" / "h200ms" / "P3" / "best.pt"
    with pytest.raises(FileNotFoundError) as e:
        run_one(3, "central", 200, _args(tmp_path), torch.device("cpu"))
    assert str(expected) in str(e.value)
    assert not (tmp_path / "res").exists()


class _Model:
    def __init__(self):
        self.fusion = Namespace(gate=torch.zeros(2))

    def to(self, device):
        return self


def _fake_eval_frames():
    std = pd.DataFrame([{"muscle": m, "output": o, "r": 0.5, "ccc": 0.5, "rmse": 0.1, "vaf": 50.0}
                        for m in ("AD", "BR") for o in ("K", "E", "F")])
    res = pd.DataFrame([{"features": f, "muscle": "MEAN", "r": 0.1, "p": 0.5} for f in ("tokens", "bandpower")])
    seg = pd.DataFrame([{"segment": s, "skill_F": 0.0, "skill_ridge": 0.01, "r_residual_ridge": 0.1,
                         "p_skill_ridge": 0.5} for s in ("pre_move", "reach", "all")])
    pr = pd.DataFrame([{"features": "eeg", "segment": "all", "fit": "pooled", "skill": 0.0}])
    return std, res, seg, pr


def test_run_one_loads_k_before_fit_and_records_it(tmp_path, monkeypatch):
    kin = tmp_path / "kin" / "h0ms" / "P2" / "best.pt"
    kin.parent.mkdir(parents=True)
    kin.write_bytes(b"")
    calls = []
    std, res, seg, pr = _fake_eval_frames()

    def fake_fit(model, tr, va, cfg, device, ckpt_dir, log=print):
        calls.append(("fit", Path(ckpt_dir), cfg["training"]["freeze_kin"]))
        return {"epochs_run": 1, "best_epoch_kin": 0, "best_epoch_eeg": 1, "frozen_epoch": {"kin": -1, "eeg": 0}}

    monkeypatch.setattr(RC, "subject_data", lambda p, cfg: {
        "prior": None, "train_ds": None, "val_ds": None, "tokens": {"val": [], "test": []}, "norm": None, "names": []})
    monkeypatch.setattr(RC, "build_model", lambda cfg, prior: _Model())
    monkeypatch.setattr(RC, "load_kin", lambda model, path: calls.append(("load_kin", Path(path))) or 7)
    monkeypatch.setattr(RC, "fit", fake_fit)
    monkeypatch.setattr(RC, "standard", lambda *a, **k: std.copy())
    monkeypatch.setattr(RC, "residual_test", lambda *a, **k: res.copy())
    monkeypatch.setattr(RC, "segments", lambda *a, **k: seg.copy())
    monkeypatch.setattr(RC, "score_model", lambda *a, **k: pr.copy())

    args = _args(tmp_path)
    out = run_one(2, "central", 0, args, torch.device("cpu"))
    assert [c[0] for c in calls] == ["load_kin", "fit"]
    assert calls[0][1] == kin
    assert calls[1][1] == tmp_path / "ckpt" / "central" / "h0ms" / "P2" and calls[1][2] is True
    assert out == tmp_path / "res" / "central" / "h0ms" / "P2"
    for f in ("standard.csv", "residual.csv", "segments.csv", "phase_ridge.csv", "history.json"):
        assert (out / f).exists(), f
    hist = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert hist["kin_checkpoint"] == str(kin) and hist["condition"] == "central"
    assert hist["eeg"]["channels"] == CENTRAL
    assert pd.read_csv(out / "standard.csv").condition.eq("central").all()

    # finished: skipped, nothing called again
    calls.clear()
    assert run_one(2, "central", 0, args, torch.device("cpu")) == out
    assert calls == []
    # --overwrite: load_kin again before fit
    run_one(2, "central", 0, _args(tmp_path, overwrite=True), torch.device("cpu"))
    assert [c[0] for c in calls] == ["load_kin", "fit"]


def _write_run(d: Path, cond, h, subj, e_r, f_k):
    d.mkdir(parents=True)
    std = pd.DataFrame([{"muscle": m, "output": o, "r": e_r if o == "E" else 0.8, "ccc": 0.5, "rmse": 0.1,
                         "vaf": 60.0 + (f_k if o == "F" else 0.0)} for m in ("AD", "BR") for o in ("K", "E", "F")])
    std.assign(subject=subj, horizon_ms=float(h)).to_csv(d / "standard.csv", index=False)
    pd.DataFrame([{"features": f, "muscle": "MEAN", "r": e_r / 2, "p": 0.5} for f in ("tokens", "bandpower")]
                 ).assign(subject=subj, horizon_ms=float(h)).to_csv(d / "residual.csv", index=False)
    pd.DataFrame([{"segment": s, "skill_F": 0.0, "skill_ridge": e_r / 10, "r_residual_ridge": e_r / 3,
                   "p_skill_ridge": 0.5} for s in ("pre_move", "reach", "all")]
                 ).assign(subject=subj, horizon_ms=float(h)).to_csv(d / "segments.csv", index=False)


def test_summarise_reads_all_and_conditions(tmp_path):
    base, res = tmp_path / "base", tmp_path / "res"
    for i, subj in enumerate(("P1", "P2", "P3")):
        for h in (0, 500):
            _write_run(base / f"h{h}ms" / subj, "all", h, subj, 0.3 + 0.01 * i, 1.0)
            _write_run(res / "central" / f"h{h}ms" / subj, "central", h, subj, 0.1 + 0.01 * i, 0.5)
        _write_run(res / "eye_clean" / "h0ms" / subj, "eye_clean", 0, subj, 0.2 + 0.01 * i, 0.2)
    (res / "edge" / "h0ms" / "P1").mkdir(parents=True)                # unfinished run: ignored
    df = summarise(res, base)
    assert len(df) == 3 * 2 + 3 * 2 + 3
    assert not df.duplicated(["condition", "horizon_ms", "subject"]).any()
    assert set(df.condition) == {"all", "central", "eye_clean"}
    for c in ("E_r", "F_minus_K_vaf", "res_r_tokens", "pre_move_r", "reach_r", "pre_move_skill", "reach_skill"):
        assert c in df.columns
    row = df[(df.condition == "central") & (df.horizon_ms == 0) & (df.subject == "P1")].iloc[0]
    assert row.E_r == pytest.approx(0.1) and row.F_minus_K_vaf == pytest.approx(0.5)
    assert row.pre_move_r == pytest.approx(0.1 / 3) and row.reach_skill == pytest.approx(0.01)
    assert (res / "summary.csv").exists()
    vs = pd.read_csv(res / "vs_all.csv")
    assert "p" in vs.columns and set(vs.condition) == {"central", "eye_clean"}
    e = vs[(vs.condition == "central") & (vs.horizon_ms == 0) & (vs.metric == "E_r")].iloc[0]
    assert e.n == 3 and e.mean_diff == pytest.approx(-0.2) and np.isfinite(e.p)
