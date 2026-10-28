import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v3.run import STEP_NAMES, STEPS  # noqa: E402

MARKERS = [f"# STEP {i} {n}" for i, n in enumerate(STEP_NAMES)]


def _builder():
    spec = importlib.util.spec_from_file_location("build_v3", ROOT / "tools" / "build_main_v3_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_notebook_has_the_generators_cells_and_no_outputs():
    nb = json.loads((ROOT / "main_v3.ipynb").read_text(encoding="utf-8"))
    cells = lambda book: [(c["cell_type"], "".join(c["source"])) for c in book["cells"]]
    assert cells(nb) == cells(_builder().build())
    assert all(c.get("outputs", []) == [] and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")


def test_one_cell_per_step_in_order_and_compiling():
    code = _code(_builder().build())
    assert [c.split("\n")[0].strip() for c in code] == MARKERS
    for i, c in enumerate(code):
        compile(c, f"cell{i}", "exec")


def test_every_step_cell_calls_the_runner_and_has_a_heading():
    nb = _builder().build()
    code, md = _code(nb), ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
    for i, (name, _, _) in enumerate(STEPS[1:], start=1):
        assert f"run.step_{name}(ctx)" in code[i], name
        assert any(m.startswith(f"## Step {i}: {name}") for m in md), name
    assert not re.search(r"to_csv|savefig", "\n".join(code))           # the runner writes; the cells only call it


def test_title_states_the_question_conditions_and_numbers():
    md = "".join(_builder().build()["cells"][0]["source"])
    for needle in ("beyond what past movement and past task events", "`causal_cue`", "`K`", "`B`", "`F`", "`E0`",
                   "`Eguided`", "1 - MSE(F) / MSE(B)", "0.81", "0.58", "0.645", "null result", "phase-matched",
                   "Never an input".lower(), "v3.run --step b_vs_f"):
        assert needle.lower() in md.lower(), needle
