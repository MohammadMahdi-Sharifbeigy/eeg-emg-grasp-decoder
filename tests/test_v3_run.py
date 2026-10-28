import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import v3.run as R
from v3.contrast import COLUMNS
from v3.run import STEP_NAMES, Context, run_steps, student_dir

SUBJECTS = [f"P{i}" for i in range(1, 13)]
MUSCLES = ["AD", "BR", "ED", "FD", "FDI"]


def test_steps_and_folders(tmp_path):
    assert STEP_NAMES == ["setup", "data", "timing_baselines", "b_vs_f", "budget", "train_E0", "train_Eguided",
                          "evaluate_students", "verdict"]
    ctx = Context(root=tmp_path)
    assert ctx.folder("b_vs_f") == tmp_path / "results" / "v3" / "step3_b_vs_f"
    assert Context(root=tmp_path, preset="central").folder("verdict") == tmp_path / "results" / "v3" / "central" / "step8_verdict"
    assert ctx.guided_scales == [0.05, 0.1, 0.3] and ctx.seeds == [0, 1, 2]
    with pytest.raises(ValueError, match="unknown step"):
        run_steps(ctx, ["nope"])


def test_dry_run_prints_every_step_and_writes_nothing(tmp_path):
    lines = []
    run_steps(Context(root=tmp_path, participants="1", dry_run=True, log=lines.append), "all")
    headers = [l for l in lines if l.startswith("Step ")]
    assert [h.split(":")[0] for h in headers] == [f"Step {i}/8 {n}" for i, n in enumerate(STEP_NAMES)]
    assert any("would write" in l and "b_vs_f_summary.csv" in l for l in lines)
    assert "(trains)" in headers[5] and "(no training)" in headers[3]
    assert list(tmp_path.iterdir()) == []


def test_main_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    R.main(["--step", "b_vs_f,verdict", "--participants", "1", "--dry-run"])
    out = capsys.readouterr().out
    assert "Step 3/8 b_vs_f" in out and "Step 8/8 verdict" in out and list(tmp_path.iterdir()) == []


def test_student_dir_new_names_and_old_layouts(tmp_path):
    assert student_dir(tmp_path, 0.0, 1, "P2") == tmp_path / "E0" / "s1" / "P2"
    assert student_dir(tmp_path, 0.1, 2, "P2") == tmp_path / "Eguided_g0.1" / "s2" / "P2"
    old = tmp_path / "g0.1" / "s2" / "P2"
    old.mkdir(parents=True)
    (old / "last.pt").write_bytes(b"x")
    assert student_dir(tmp_path, 0.1, 2, "P2") == old
    legacy = tmp_path / "g0.0" / "P3"
    legacy.mkdir(parents=True)
    (legacy / "best.pt").write_bytes(b"x")
    assert student_dir(tmp_path, 0.0, 0, "P3") == legacy


def _fake_scores(p, h, ckpt_root, n_null, sources):
    rows = []
    for src in sources:
        for seg in ("pre_move", "reach", "load", "hold", "release", "other", "all"):
            row = dict.fromkeys(COLUMNS, 0.1)
            row.update(segment=seg, eeg_skill=0.02 if src == "eeg" else 0.0, p=0.01)
            rows.append(dict(row, subject=f"P{p}", horizon_ms=h, source=src))
    onset = pd.DataFrame({"subject": f"P{p}", "horizon_ms": h, "condition": ["B", "F"], "lift": 0, "muscle": "AD",
                          "onset_true_s": 0.4, "onset_pred_s": [0.45, 0.42], "error_ms": [50.0, 20.0]})
    return pd.DataFrame(rows), onset


def test_b_vs_f_step_skips_missing_checkpoints_and_writes_its_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_horizon_scores", _fake_scores)
    ctx = Context(root=tmp_path, participants="1,2", log=(lines := []).append)
    ctx.cfg["b_vs_f"]["horizons_ms"] = [0, 200]
    for p in (1, 2):
        d = tmp_path / ctx.cfg["b_vs_f"]["checkpoint_dir"] / "h0ms" / f"P{p}"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"x")
    out = R.step_b_vs_f(ctx)
    assert sum("no horizon checkpoint" in l for l in lines) == 2
    folder = ctx.folder("b_vs_f")
    for f in ("b_vs_f.csv", "b_vs_f_summary.csv", "onset.csv", "onset_summary.csv", "b_vs_f.png"):
        assert (folder / f).exists(), f
    assert set(out["summary"].horizon_ms) == {0} and (folder / "runs" / "h0ms_P2.csv").exists()
    monkeypatch.setattr(R, "_horizon_scores", lambda *a: (_ for _ in ()).throw(AssertionError("recomputed")))
    R.step_b_vs_f(ctx)                                               # cached runs are reused


def test_verdict_step_reads_the_saved_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_horizon_scores", _fake_scores)
    ctx = Context(root=tmp_path, participants="1-12", log=(lines := []).append,
                  v2_standard="v2.csv", phase_metrics="phase.csv")
    ctx.cfg["b_vs_f"]["horizons_ms"] = [0]
    for p in range(1, 13):
        d = tmp_path / ctx.cfg["b_vs_f"]["checkpoint_dir"] / "h0ms" / f"P{p}"
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"x")
    R.step_b_vs_f(ctx)
    j = np.linspace(-0.01, 0.01, 12)
    std = pd.DataFrame([dict(muscle=m, r=0.6 + j[i] + (0.05 if sc else 0) + 0.001 * i * (sc > 0), vaf=30, subject=s,
                             scale=sc, seed=0, rate=500, condition="E0" if sc == 0 else "Eguided")
                        for i, s in enumerate(SUBJECTS) for sc in (0.0, 0.1) for m in MUSCLES])
    ctx.save(std, "evaluate_students", "standard.csv")
    pd.DataFrame([dict(muscle=m, r=0.6, ccc=0, rmse=0, vaf=30, output=o, subject=s) for s in SUBJECTS for m in MUSCLES
                  for o in ("E", "K")]).to_csv(tmp_path / "v2.csv", index=False)
    pd.DataFrame([dict(muscle=m, r=0.8, ccc=0, rmse=0, vaf=60, subject=s, variant=v, alpha=1, n_features=1,
                       n_test_samples=1) for s in SUBJECTS for m in MUSCLES for v in ("phase", "cue")]
                 ).to_csv(tmp_path / "phase.csv", index=False)
    out = R.step_verdict(ctx)["verdict"]
    assert (ctx.folder("verdict") / "verdict.csv").exists() and (ctx.folder("verdict") / "eguided_vs_e0.csv").exists()
    assert bool(out.Eguided_beats_E0.iloc[0]) and out.onset_abs_ms_B.iloc[0] == 50.0
    assert any("->" in l for l in lines)
