import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ["# V3-SETUP", "# V3-CONFIG", "# V3-BUDGET", "# V3-DATA", "# V3-TIMING", "# V3-TRAIN", "# V3-EVAL", "# V3-SWAP",
           "# V3-RESIDUAL", "# V3-MUSCLE", "# V3-VERDICT", "# V3-CONTRAST"]


def _builder():
    spec = importlib.util.spec_from_file_location("build_v3", ROOT / "tools" / "build_main_v3_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(marker):
    return _code(_builder().build())[MARKERS.index(marker)]


def test_committed_notebook_has_the_generators_cells_and_no_outputs():
    nb = json.loads((ROOT / "main_v3.ipynb").read_text(encoding="utf-8"))
    cells = lambda book: [(c["cell_type"], "".join(c["source"])) for c in book["cells"]]
    assert cells(nb) == cells(_builder().build())
    assert all(c.get("outputs", []) == [] and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")


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
    assert "run_dir(CKPT," in text and 'CONFIG["outputs"]["checkpoint_dir"]' in text
    assert 'TIMING_CACHE = RESULTS / "timing_preds"' in text


def test_train_cell_checks_the_teacher_and_passes_no_teacher_for_the_control():
    train = _cell("# V3-TRAIN")
    assert "no v2 K checkpoint at" in train and "skipped" in train
    assert "None if scale == 0" in train and "check_teacher_split(ck" in train
    assert "for seed in run_seeds" in train and "for scale in run_scales" in train


def test_stage_runs_the_cheap_control_first():
    cfg = _cell("# V3-CONFIG")
    assert 'STAGE = "control"' in cfg and "run_scales = [0.0] if STAGE" in cfg and "run_seeds = SEEDS[:1]" in cfg
    assert 'SEEDS = CONFIG["training"]["seeds"]' in cfg


def test_markdown_states_the_question_and_the_numbers():
    md = "".join(_builder().build()["cells"][0]["source"])
    for needle in ("0.645", "0.81", "0.58", "0.05", "not available at test time", "ceiling", "beyond the task clock",
                   "null result", "beyond_timing", "survives_swap", "timing_level", "STAGE"):
        assert needle in md, needle


def test_new_cells_write_their_tables():
    for marker, needles in {
        "# V3-TIMING": ("cached_timing", "score_timing", "timing_causal.csv"),
        "# V3-EVAL": ("evaluate_student", "standard.csv", '"rate"'),
        "# V3-SWAP": ("evaluate_swap", "swap_group_test", "swap.csv", "r_swap is close to r_true"),
        "# V3-RESIDUAL": ("residual_over_timing", "residual_group_test", "residual_over_timing.csv"),
        "# V3-VERDICT": ("subject_means", "criteria(", "beats_control", "beyond_timing", "survives_swap", "timing_level",
                         "verdict.csv", "swap_and_residual.png", "seed SD"),
        "# V3-BUDGET": ("run_budget", "GPU hours"),
        "# V3-MUSCLE": ("muscle_specificity", "muscle_group_test", "muscle_specificity.csv", 'CONFIG["muscle"]["shared"]'),
        "# V3-CONTRAST": ("contrast_participant", "contrast_group_test", "contrast.csv", "contrast_tests.csv",
                          'CONFIG["contrast"]', "swap control", "contrast.png"),
    }.items():
        cell = _cell(marker)
        for n in needles:
            assert n in cell, (marker, n)
