import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ["# V2-SETUP", "# V2-CONFIG", "# V2-BUDGET", "# V2-DATA", "# V2-TRAIN", "# V2-EVAL", "# V2-COMPARE", "# V2-INTERP", "# V2-TRIPTYCH", "# V2-RESIDUAL"]


def _builder():
    spec = importlib.util.spec_from_file_location("build_v2", ROOT / "tools" / "build_main_v2_notebook.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _code(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_notebook_has_the_generators_cells():
    """The notebook is committed with the user's outputs, so only the cells' kind and text are compared."""
    nb = json.loads((ROOT / "main_v2.ipynb").read_text(encoding="utf-8"))
    cells = lambda book: [(c["cell_type"], "".join(c["source"])) for c in book["cells"]]
    assert cells(nb) == cells(_builder().build())


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


def test_train_cell_says_whether_e_and_f_can_be_interpreted():
    cell = _cell("# V2-TRAIN")
    assert cell.count("weighted_stream_loss(") == 3 and "w_main" not in cell          # the arithmetic lives in v2.train
    assert 'history["components"]' in cell and 'vc["components"]' in cell and "np.argmin(" in cell
    assert "fusion_checked = same_k or l_f <= l_k" in cell and "ep_e > k_frozen" in cell and "fusion_checked=fusion_checked" in cell
    assert "weighted_stream_loss" in _cell("# V2-SETUP")


def test_eval_cell_uses_the_fusion_flag_counts_passes_and_prints_criterion_2():
    cell = _cell("# V2-EVAL")
    assert "not interpretable (fusion and K come from different epochs and F is worse than K on validation)" in cell
    assert 'results[subj]["fusion_checked"]' in cell and '"fusion_checked": ' in cell
    assert 'v["E_p_swap"] <= 0.05' in cell and "about 5 % pass by chance" in cell and "n_pass" in cell
    for name in ('res["by_load"]', '"grouping"', '"CurW"', '"height_ratio"', '"r_lift"', '"n_lifts"'):
        assert name in cell, name


def test_compare_and_interp_wording():
    cell = _cell("# V2-COMPARE")
    assert "allc.subject.nunique()" in cell and "means over all 12 participants" in cell
    assert "see the Interpretability section of the first cell" in _cell("# V2-INTERP")


def test_null_draws():
    from v2.config import CONFIG_V2
    assert CONFIG_V2["evaluation"] == {"n_null": 99, "seed": 42}
