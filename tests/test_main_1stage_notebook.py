"""Static checks of main_1stage.ipynb (single-stage KG-GT notebook).

Cells are never executed except the CONFIG cell, which is a plain dict literal.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "main_1stage.ipynb"


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _strip_magics(code: str) -> str:
    return "\n".join(
        "pass" if re.match(r"^\s*[%!]", line) else line for line in code.splitlines()
    )


def _cell(nb, needle):
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(hits) == 1, f"{needle!r}: {len(hits)} cells"
    return hits[0]


def _exec_config(nb) -> dict:
    ns = {"copy": copy}
    exec(_src(_cell(nb, "CONFIG = {")), ns)  # plain dict literal + deepcopy
    return ns["CONFIG"]


@pytest.fixture(scope="module")
def nb():
    with open(NB_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def head_nb():
    try:
        raw = subprocess.run(["git", "show", "HEAD:main.ipynb"], cwd=ROOT,
                             check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"git HEAD:main.ipynb unavailable: {e}")
    return json.loads(raw.decode("utf-8"))


def test_no_outputs(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == []
            assert c["execution_count"] is None


def test_all_code_cells_compile(nb):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile(_strip_magics(_src(c)), f"main_1stage.ipynb[cell {i}]", "exec")


def test_config_single_stage_and_unchanged_preprocessing(nb, head_nb):
    cfg, head = _exec_config(nb), _exec_config(head_nb)
    tr = cfg["training"]
    assert tr["checkpoint_dir"] == "outputs/checkpoints_1stage"
    assert tr["single_stage"]["lr"] == 1e-3
    assert tr["single_stage"]["epochs"] == 200
    assert tr["scheduler"] == "cosine" and tr["optimizer"] == "adamw" and tr["warmup_epochs"] == 5
    assert cfg["data"]["cache_dir"] == head["data"]["cache_dir"]
    assert cfg["data"] == head["data"]
    assert cfg["preprocessing"] == head["preprocessing"]
    assert cfg["dataset"] == head["dataset"]
    assert cfg["model"] == head["model"]
    assert tr["loss"] == head["training"]["loss"]
    # only additions in training: checkpoint_dir, single_stage, lr (plot folder key)
    for k, v in head["training"].items():
        assert tr[k] == v, k
    assert set(tr) - set(head["training"]) == {"checkpoint_dir", "single_stage", "lr"}


def test_training_cell_is_single_stage(nb):
    train = _cell(nb, "PER-SUBJECT TRAINING LOOP")
    s = _src(train)
    assert "build_transformer_only_from_config" not in s
    assert "checkpoints_main_stage2" not in s
    assert "checkpoints_main" not in s
    assert "train_single_stage(" in s
    assert "stage1_model=None" in s
    helper = _src(_cell(nb, "def train_single_stage("))
    assert "build_transformer_only_from_config" not in helper
    assert "requires_grad = True" in helper and "resume=True" in helper
    assert "build_kg_gt_from_config(" in helper


def test_no_two_stage_paths_anywhere(nb):
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert "checkpoints_main_stage" not in _src(c)
    loso = _src(_cell(nb, "[OPTIONAL] Automatic LOSOCV"))
    assert "build_transformer_only_from_config" not in loso
    assert "train_single_stage(" in loso


def test_magnitude_then_ablation_cells_follow_evaluation(nb):
    cells = nb["cells"]
    i = next(k for k, c in enumerate(cells) if "# EVALUATION — Full metric suite" in _src(c))
    mag = _src(cells[i + 1])
    assert "magnitude_report_by_series(" in mag
    assert 'MAGNITUDE_CSV = "results/magnitude_report_1stage.csv"' in mag
    assert 'MAGNITUDE_PNG = "results/magnitude_report_1stage.png"' in mag
    assert "input_ablation(" in _src(cells[i + 2])


def test_package1_config_matches_main(nb, head_nb):
    cfg, head = _exec_config(nb), _exec_config(head_nb)
    assert cfg["preprocessing"]["kinematics"]["gated_force"] == {"near_cm": 12.0, "far_cm": 20.0, "smooth_s": 0.1}
    assert cfg["model"]["gat"]["kin_gat_dim"] == 12
    loss = cfg["training"]["loss"]
    assert (loss["asymmetry"], loss["w_rest_over"], loss["huber_delta"], loss["channel_weights"]) == \
        (1.2, 1.0, 1.0, "auto")
    assert loss == head["training"]["loss"]
