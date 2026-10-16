import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ["# P1-SETUP", "# P1-W-RUN", "# P1-W-PLOT", "# P1-R-RUN", "# P1-R-PLOT", "# P1-C-RUN", "# P1-C-PLOT"]


def _builder():
    spec = importlib.util.spec_from_file_location("build_eeg_controls_phase1_notebook",
                                                  ROOT / "tools" / "build_eeg_controls_phase1_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_code_cell_parses_and_has_a_marker():
    nb = _builder().build()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 7
    for src in code:
        ast.parse(src)
        assert src.startswith("# P1-")
    assert [s.split()[0] + " " + s.split()[1] for s in code] == MARKERS


def test_training_cell_has_the_agreed_settings():
    code = {"".join(c["source"]).split("\n")[0].split()[1]: "".join(c["source"])
            for c in _builder().build()["cells"] if c["cell_type"] == "code"}
    assert 'CONDITIONS = ["central", "edge", "eye_clean"]' in code["P1-C-RUN"]
    assert "HORIZONS_MS = [0, 200, 500]" in code["P1-C-RUN"]
    assert "C_PARTICIPANTS = PARTICIPANTS" in code["P1-C-RUN"]


def test_every_section_has_a_guide_and_no_emojis():
    nb = _builder().build()
    kinds = [c["cell_type"] for c in nb["cells"]]
    for i, k in enumerate(kinds):
        if k == "code" and "-RUN" in "".join(nb["cells"][i]["source"]).split("\n")[0]:
            assert kinds[i - 1] == "markdown"
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert not [ch for ch in text if ord(ch) >= 0x2190], "no emojis or symbols outside plain text"


def test_committed_notebook_matches_the_generator():
    committed = json.loads((ROOT / "eeg_controls_phase1.ipynb").read_text(encoding="utf-8"))
    built = _builder().build()
    src = lambda nb: [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
    assert src(committed) == src(built), "run tools/build_eeg_controls_phase1_notebook.py"


def test_phase1_modules_import():
    for m in ("eeg_controls.common", "eeg_controls.weight_check", "eeg_controls.phase_ridge",
              "eeg_controls.run_conditions", "eeg_controls.plots_phase1"):
        __import__(m)
