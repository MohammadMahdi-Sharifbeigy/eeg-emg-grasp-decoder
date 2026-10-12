import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ["# V2-SETUP", "# V2-CONFIG", "# V2-BUDGET", "# V2-DATA", "# V2-TRAIN", "# V2-EVAL", "# V2-COMPARE", "# V2-INTERP"]


def _builder():
    spec = importlib.util.spec_from_file_location("build_v2", ROOT / "tools" / "build_main_v2_notebook.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_notebook_equals_the_generator_output_and_has_no_outputs():
    nb = json.loads((ROOT / "main_v2.ipynb").read_text(encoding="utf-8"))
    assert nb == _builder().build()
    assert all(c.get("outputs", []) == [] and c.get("execution_count") is None for c in nb["cells"] if c["cell_type"] == "code")


def test_cells_compile_and_are_in_order():
    code = _code(_builder().build())
    assert [c.split("\n")[0].strip() for c in code] == MARKERS
    for i, c in enumerate(code):
        compile(c, f"cell{i}", "exec")


def test_writes_only_to_v2_folders_and_never_touches_the_baseline():
    text = "\n".join(_code(_builder().build()))
    assert not re.search(r"checkpoints_1stage|checkpoints_main|results/p2|interpretability_1stage", text)
    for literal in re.findall(r"[\"'](outputs/[^\"']*|results/[^\"']*)[\"']", text):
        assert literal == "results/magnitude_report_1stage.csv", literal        # the only literal path: the baseline table it reads
    assert "CONFIG[\"outputs\"]" in text and "stratified" not in text.replace("stratified_split", "")
    assert "interpret_subject(" in text and "CONFIG[\"outputs\"][\"interpretability_dir\"]" in text


def test_markdown_states_the_three_outputs_the_criteria_and_the_interpretability_views():
    nb = _builder().build()
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    for phrase in ("K (kinematics-led)", "E (EEG only)", "F (fused)", "75.5", "0.869", "0.47", "EEG adds nothing beyond kinematics",
                   "attention rollout", "attention flow", "gradient-weighted rollout", "deletion", "weight-shuffle"):
        assert phrase in md, phrase
    for phrase in ("Mixed precision", "one test over all muscles", "time-locked", "the same for all muscles"):
        assert phrase in md, phrase


def _cell(marker):
    return next(c for c in _code(_builder().build()) if c.startswith(marker))


def test_budget_cell_copies_the_measured_mixed_precision_setting_into_the_config():
    cell = _cell("# V2-BUDGET")
    assert 'CONFIG["training"]["use_amp"] = budget["use_amp"]' in cell
    assert 'budget["candidates"]' in cell


def test_f_verdict_is_the_single_test_over_all_muscles_and_is_saved():
    cell = _cell("# V2-EVAL")
    assert 'res["verdict"]' in cell and "any(" not in cell
    for key in ("dF_vaf_mean", "dF_p_swap", "E_r_mean", "E_p_swap"):
        assert f'v["{key}"]' in cell or f"v['{key}']" in cell, key
    assert 'v["dF_vaf_mean"] > 0 and np.isfinite(v["dF_p_swap"]) and v["dF_p_swap"] <= 0.05' in cell
    assert 'RESULTS / "verdict.csv"' in cell
    assert "EEG adds information beyond kinematics" in cell and "EEG adds nothing beyond kinematics" in cell


def test_interp_cell_uses_the_columns_of_the_current_report():
    cell = _cell("# V2-INTERP")
    for column in ('"spearman"', '"n_nan"', "'n_pairs'", "np.isfinite("):
        assert column in cell, column
    assert "rep['files']" in cell or 'rep["files"]' in cell or "rep.get('files'" in cell
