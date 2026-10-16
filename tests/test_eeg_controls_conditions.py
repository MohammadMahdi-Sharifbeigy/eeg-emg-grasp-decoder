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
    assert list(CONDITIONS) == ["all_frozen", "central", "edge", "eye_clean"]
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
    full = condition_config("all_frozen", 0)["eeg"]
    assert "channels" not in full and "eye_regress" not in full
    assert len(channel_names(condition_config("all_frozen", 0))) == 32
    smoke = condition_config("edge", 200, smoke=True)
    assert smoke["training"]["max_epochs"] == 1 and smoke["evaluation"]["n_null"] == 5
    # the module-level CONDITIONS are not changed by building a config
    condition_config("central", 0)["eeg"]["channels"].append("Pz")
    assert CONDITIONS["central"]["channels"] == CENTRAL
    with pytest.raises(ValueError, match="all_frozen.*central.*edge.*eye_clean"):
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


class _Model(torch.nn.Module):
    """A model with one K tensor (kin.w) and one E tensor (eeg.w), as the guard sees a v2 model."""
    def __init__(self):
        super().__init__()
        self.kin = torch.nn.Linear(2, 2)
        self.eeg = torch.nn.Linear(2, 2)
        self.fusion = Namespace(gate=torch.zeros(2))


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


def _patch_run(monkeypatch, fit_fn):
    std, res, seg, pr = _fake_eval_frames()
    monkeypatch.setattr(RC, "subject_data", lambda p, cfg: {
        "prior": None, "train_ds": None, "val_ds": None, "tokens": {"val": [], "test": []}, "norm": None, "names": []})
    monkeypatch.setattr(RC, "build_model", lambda cfg, prior: _Model())
    monkeypatch.setattr(RC, "load_kin", lambda model, path: 2)
    monkeypatch.setattr(RC, "fit", fit_fn)
    monkeypatch.setattr(RC, "standard", lambda *a, **k: std.copy())
    monkeypatch.setattr(RC, "residual_test", lambda *a, **k: res.copy())
    monkeypatch.setattr(RC, "segments", lambda *a, **k: seg.copy())
    monkeypatch.setattr(RC, "score_model", lambda *a, **k: pr.copy())


def _hist():
    return {"epochs_run": 1, "best_epoch_kin": 0, "best_epoch_eeg": 1, "frozen_epoch": {"kin": -1, "eeg": 0}}


def _kin_file(tmp_path, p, h, config=None):
    f = tmp_path / "kin" / f"h{h}ms" / f"P{p}" / "best.pt"
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({} if config is None else {"config": config}, f)
    return f


def test_k_guard_raises_when_fit_changes_k(tmp_path, monkeypatch):
    _kin_file(tmp_path, 1, 0)

    def fit_changes_k(model, *a, **k):
        with torch.no_grad():
            model.eeg.weight.add_(1.0)              # E may change
            model.kin.bias.add_(1e-3)               # K must not
        return _hist()

    _patch_run(monkeypatch, fit_changes_k)
    with pytest.raises(RuntimeError, match=r"kin\.bias"):
        run_one(1, "central", 0, _args(tmp_path), torch.device("cpu"))
    assert not (tmp_path / "res" / "central" / "h0ms" / "P1" / "history.json").exists()


def test_k_guard_passes_when_only_e_changes(tmp_path, monkeypatch):
    _kin_file(tmp_path, 1, 0)

    def fit_changes_e(model, *a, **k):
        with torch.no_grad():
            model.eeg.weight.add_(1.0)
        return _hist()

    _patch_run(monkeypatch, fit_changes_e)
    out = run_one(1, "all_frozen", 0, _args(tmp_path), torch.device("cpu"))
    assert (out / "history.json").exists()


def test_check_lists_every_missing_k_checkpoint_before_training(tmp_path, monkeypatch):
    _kin_file(tmp_path, 1, 0)
    called = []
    monkeypatch.setattr(RC, "run_one", lambda *a, **k: called.append(a))
    with pytest.raises(FileNotFoundError) as e:
        RC.run_all(["central", "edge"], [0, 500], [1, 2], _args(tmp_path), torch.device("cpu"))
    msg = str(e.value)
    for p, h in ((2, 0), (1, 500), (2, 500)):
        assert str(tmp_path / "kin" / f"h{h}ms" / f"P{p}" / "best.pt") in msg
    assert str(tmp_path / "kin" / "h0ms" / "P1" / "best.pt") not in msg
    assert called == []


def test_check_rejects_a_k_checkpoint_with_another_k_config(tmp_path, monkeypatch):
    good = {k: condition_config("central", 0, smoke=True)[k] for k in ("model", "tokens", "eeg", "history", "loss")}
    _kin_file(tmp_path, 1, 0, good)                                  # eeg block differs by design: accepted
    bad = {k: dict(v) for k, v in good.items()}
    bad["tokens"]["horizon"] = 25                                    # a 500 ms K for a 0 ms run
    _kin_file(tmp_path, 2, 0, bad)
    monkeypatch.setattr(RC, "run_one", lambda *a, **k: pytest.fail("trained despite a wrong K"))
    with pytest.raises(RuntimeError) as e:
        RC.run_all(["central"], [0], [1, 2], _args(tmp_path), torch.device("cpu"))
    assert "P2" in str(e.value) and "tokens" in str(e.value) and "P1" not in str(e.value)


def test_check_skips_finished_runs(tmp_path):
    out = tmp_path / "res" / "edge" / "h0ms" / "P4"
    out.mkdir(parents=True)
    for f in RC.FILES:
        (out / f).write_text("x")
    RC.check_kin_checkpoints(["edge"], [0], [4], _args(tmp_path))  # finished: its K is not needed
    with pytest.raises(FileNotFoundError):
        RC.check_kin_checkpoints(["edge"], [0], [4], _args(tmp_path, overwrite=True))


def test_run_all_logs_a_failure_and_goes_on(tmp_path, monkeypatch, capsys):
    for p in (1, 2, 3):
        _kin_file(tmp_path, p, 0)
    done = []

    def fake_run_one(p, name, h, args, device):
        if p == 2:
            raise ValueError("boom in P2")
        done.append(p)

    monkeypatch.setattr(RC, "run_one", fake_run_one)
    failed = RC.run_all(["central"], [0], [1, 2, 3], _args(tmp_path), torch.device("cpu"))
    assert failed == 1 and done == [1, 3]
    log = (tmp_path / "res" / "failures.log").read_text(encoding="utf-8")
    assert "central h=0 ms P2" in log and "Traceback" in log and "boom in P2" in log
    printed = capsys.readouterr().out
    assert "FAILED central P2" in printed and "1 of 3 runs failed" in printed


def test_smoke_does_not_write_to_default_folders(tmp_path):
    a = Namespace(results=str(RC.ROOT / "results" / "v2_controls"), checkpoints=str(tmp_path / "ck"))
    RC.smoke_paths(a, {"results": str(RC.ROOT / "results" / "v2_controls"),
                       "checkpoints": str(RC.ROOT / "outputs" / "checkpoints_v2_controls")})
    assert a.results == str(RC.ROOT / "results" / "v2_controls_smoke")
    assert a.checkpoints == str(tmp_path / "ck")                     # an explicit path is kept


def test_main_smoke_redirects_default_paths(monkeypatch):
    seen = {}
    monkeypatch.setattr(RC, "run_all", lambda names, hs, ps, a, device: seen.update(a=a) or 0)
    monkeypatch.setattr(RC, "summarise", lambda *a: pd.DataFrame())
    RC.main(["--smoke", "--participants", "1", "--conditions", "central", "--horizons-ms", "0"])
    assert seen["a"].results.endswith("v2_controls_smoke")
    assert seen["a"].checkpoints.endswith("checkpoints_v2_controls_smoke")


def test_summarise_compares_with_all_frozen_and_all(tmp_path):
    base, res = tmp_path / "base", tmp_path / "res"
    for i, subj in enumerate(("P1", "P2", "P3")):
        _write_run(base / "h0ms" / subj, "all", 0, subj, 0.30 + 0.01 * i, 1.0)
        _write_run(res / "all_frozen" / "h0ms" / subj, "all_frozen", 0, subj, 0.25 + 0.01 * i, 0.8)
        _write_run(res / "central" / "h0ms" / subj, "central", 0, subj, 0.10 + 0.01 * i, 0.5)
    df = summarise(res, base)
    assert list(dict.fromkeys(df.condition)) == ["all", "all_frozen", "central"]
    vs = pd.read_csv(res / "vs_all.csv")
    pairs = set(zip(vs.condition, vs.reference))
    assert pairs == {("central", "all_frozen"), ("central", "all"), ("all_frozen", "all")}
    e = vs[(vs.metric == "E_r")].set_index(["condition", "reference"])
    assert e.loc[("central", "all_frozen"), "mean_diff"] == pytest.approx(-0.15)
    assert e.loc[("central", "all"), "mean_diff"] == pytest.approx(-0.20)
    assert e.loc[("all_frozen", "all"), "mean_diff"] == pytest.approx(-0.05)
    assert e.loc[("central", "all_frozen"), "mean_reference"] == pytest.approx(0.26)
