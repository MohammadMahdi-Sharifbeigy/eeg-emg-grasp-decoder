import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("build_eeg_controls_notebook", ROOT / "tools" / "build_eeg_controls_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_code_cell_parses_and_has_a_marker():
    nb = _builder().build()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 7
    for src in code:
        ast.parse(src)
        assert src.startswith("# CTRL-")


def test_committed_notebook_matches_the_generator():
    committed = json.loads((ROOT / "eeg_controls.ipynb").read_text(encoding="utf-8"))
    built = _builder().build()
    src = lambda nb: [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
    assert src(committed) == src(built), "run tools/build_eeg_controls_notebook.py"


def test_eeg_controls_modules_import():
    for m in ("eeg_controls.common", "eeg_controls.phase_baseline", "eeg_controls.reference_maps", "eeg_controls.run_horizon", "eeg_controls.plots"):
        __import__(m)
