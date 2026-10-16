import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ["# V3-SETUP", "# V3-CONFIG", "# V3-BUDGET", "# V3-DATA", "# V3-TRAIN", "# V3-EVAL", "# V3-COMPARE"]


def _builder():
    spec = importlib.util.spec_from_file_location("build_v3", ROOT / "tools" / "build_main_v3_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_notebook_has_the_generators_cells():
    nb = json.loads((ROOT / "main_v3.ipynb").read_text(encoding="utf-8"))
    cells = lambda book: [(c["cell_type"], "".join(c["source"])) for c in book["cells"]]
    assert cells(nb) == cells(_builder().build())


def test_cells_compile_and_are_in_order():
    code = _code(_builder().build())
    assert [c.split("\n")[0].strip() for c in code] == MARKERS
    for i, c in enumerate(code):
        compile(c, f"cell{i}", "exec")


def test_writes_only_to_v3_folders():
    text = "\n".join(_code(_builder().build()))
    assert "checkpoints_v2" not in text and "results/v2" not in text.replace('"results/v2/standard.csv"', "")
    for m in re.finditer(r"(to_csv|savefig)\(([^,)]*)", text):
        assert m.group(2).strip().startswith("RESULTS"), m.group(0)
    assert "CKPT / " in text and 'CONFIG["outputs"]["checkpoint_dir"]' in text


def test_train_cell_skips_missing_teacher_and_passes_no_teacher_for_the_control():
    train = _code(_builder().build())[MARKERS.index("# V3-TRAIN")]
    assert "no v2 K checkpoint at" in train and "skipped" in train
    assert "None if scale == 0" in train


def test_markdown_states_criteria_numbers():
    md = "".join(_builder().build()["cells"][0]["source"])
    for needle in ("0.645", "0.81", "0.05", "not available at test time", "ceiling"):
        assert needle in md, needle


def test_compare_cell_prints_the_three_flags_and_saves_the_tables():
    comp = _code(_builder().build())[MARKERS.index("# V3-COMPARE")]
    for needle in ("subject_means", "group_tests", "beats_E", "close_to_timing", "beats_timing", "guide_strength.png",
                   "compare.csv", "group_tests.csv"):
        assert needle in comp, needle
