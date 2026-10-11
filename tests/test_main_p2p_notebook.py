"""Static checks of main_p2p.ipynb (single-stage KG-GT notebook, Package 2 / Phase B).

The notebook was called main_1stage.ipynb until the original (pre Phase B) notebook was
restored under that name (tests/test_main_1stage_original.py).

Cells are never executed except the CONFIG cell, which is a plain dict literal.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "main_p2p.ipynb"


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


# The two lines the user edits in the notebook itself (which participants to run, when to
# stop): their VALUES are not compared with the generator, everything else is.
_USER_KNOBS = (
    (re.compile(r"^subject_list = .*$", re.M), "subject_list = <user>"),
    (re.compile(r'^            "early_stop_patience": \d+,', re.M),
     '            "early_stop_patience": <user>,'),
)


def _comparable(nb) -> list:
    """(cell type, source) of every cell: outputs, execution counts and metadata ignored."""
    out = []
    for c in nb["cells"]:
        s = _src(c)
        for pat, repl in _USER_KNOBS:
            s = pat.sub(repl, s)
        out.append((c["cell_type"], s))
    return out


def test_notebook_is_the_generator_output_up_to_outputs(nb):
    """main_p2p.ipynb = tools/build_main_p2p_notebook.py applied to HEAD:main.ipynb.

    The notebook is committed WITH the outputs of the user's runs (the generator clears
    them), so cell outputs / execution counts are ignored here, and so are the values of
    the two user knobs above."""
    spec = importlib.util.spec_from_file_location(
        "build_main_p2p_notebook", ROOT / "tools" / "build_main_p2p_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        built, log = mod.build("HEAD")
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"git HEAD:main.ipynb unavailable: {e}")
    assert log
    assert len(built["cells"]) == len(nb["cells"])
    for i, (a, b) in enumerate(zip(_comparable(built), _comparable(nb))):
        assert a == b, f"cell {i} differs from the generator output"
    for c in built["cells"]:                       # what the generator writes has no outputs
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    # the knobs are applied to exactly one line each
    for pat, _ in _USER_KNOBS:
        assert sum(len(pat.findall(_src(c))) for c in nb["cells"]) == 1, pat.pattern


def test_generator_writes_main_p2p_and_never_main_1stage():
    gen = (ROOT / "tools" / "build_main_p2p_notebook.py").read_text(encoding="utf-8")
    assert 'default=str(ROOT / "main_p2p.ipynb")' in gen
    assert 'ROOT / "main_1stage.ipynb"' not in gen
    assert not (ROOT / "tools" / "build_main_1stage_notebook.py").exists()
    res = (ROOT / "tools" / "build_main_residual_notebook.py").read_text(encoding="utf-8")
    assert 'SOURCE = "main_p2p.ipynb"' in res and ":main_p2p.ipynb" in res
    assert "main_1stage.ipynb" not in res          # the restored original is not its source


def test_all_code_cells_compile(nb):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile(_strip_magics(_src(c)), f"main_p2p.ipynb[cell {i}]", "exec")


def test_config_single_stage_and_unchanged_preprocessing(nb, head_nb):
    cfg, head = _exec_config(nb), _exec_config(head_nb)
    tr = cfg["training"]
    # Phase B: every output path carries the run tag (tests/test_phase_b.py)
    assert tr["checkpoint_dir"] == "outputs/checkpoints_1stage_p2b"
    assert head["training"]["checkpoint_dir"] == "outputs/checkpoints_main_p2b"
    assert tr["single_stage"]["lr"] == 1e-3
    assert tr["single_stage"]["epochs"] == 200
    assert tr["scheduler"] == "cosine" and tr["optimizer"] == "adamw" and tr["warmup_epochs"] == 5
    assert cfg["data"]["cache_dir"] == head["data"]["cache_dir"]
    assert cfg["data"] == head["data"]
    assert cfg["preprocessing"] == head["preprocessing"]
    assert cfg["dataset"] == head["dataset"]
    assert cfg["model"] == head["model"]
    assert tr["loss"] == head["training"]["loss"]
    # only differences in training: checkpoint_dir (own folder), single_stage, lr (plot folder key)
    for k, v in head["training"].items():
        if k != "checkpoint_dir":
            assert tr[k] == v, k
    assert set(tr) - set(head["training"]) == {"single_stage", "lr"}


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
    # Phase B: results go to RESULTS_DIR = results/<run tag>/
    assert 'MAGNITUDE_CSV = f"{RESULTS_DIR}/magnitude_report_1stage.csv"' in mag
    assert 'MAGNITUDE_PNG = f"{RESULTS_DIR}/magnitude_report_1stage.png"' in mag
    assert "input_ablation(" in _src(cells[i + 2])


def test_package1_config_matches_main(nb, head_nb):
    cfg, head = _exec_config(nb), _exec_config(head_nb)
    assert cfg["preprocessing"]["kinematics"]["gated_force"] == {"near_cm": 12.0, "far_cm": 20.0, "smooth_s": 0.1}
    assert cfg["model"]["gat"]["kin_gat_dim"] == 12
    loss = cfg["training"]["loss"]
    # Step 1 values (Package 1 had asymmetry 1.2, huber_delta 1.0, no w_height)
    assert (loss["asymmetry"], loss["w_rest_over"], loss["huber_delta"], loss["channel_weights"]) == \
        (1.5, 1.0, 2.5, "auto")
    assert loss["w_height"] == 0.5
    assert cfg["model"]["gat"]["kin_skip_indices"] == list(range(10, 24))
    assert loss == head["training"]["loss"]
