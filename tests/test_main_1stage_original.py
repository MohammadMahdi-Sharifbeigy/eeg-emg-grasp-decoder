"""Static checks of main_1stage.ipynb = the ORIGINAL single-stage notebook.

main_1stage.ipynb is the notebook of commit 4d34ce2 (before Package 2 / Phase B): raw EEG
input, 24 kinematic features, 4-layer encoder, untagged output paths.  It is maintained by
hand (no generator) and is the notebook that is run for more participants, so its results
must land next to the existing P1 results.  The Phase-B experiment is main_p2p.ipynb
(tests/test_main_p2p_notebook.py), the EEG-residual experiment main_1stage_residual.ipynb.

Nothing is executed except the CONFIG cell (a dict literal) and, on CPU with random
weights, the model / loss builders.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_NAME = "main_1stage.ipynb"


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


@pytest.fixture(scope="module")
def nb():
    with open(ROOT / NB_NAME, encoding="utf-8") as f:
        return json.load(f)


def _code(nb, needle: str) -> str:
    hits = [_src(c) for c in nb["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(hits) == 1, f"{needle!r}: {len(hits)} cells"
    return hits[0]


def _all_code(nb) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.fixture(scope="module")
def cfg(nb) -> dict:
    ns = {"copy": copy}
    exec(_code(nb, "CONFIG = {"), ns)             # plain dict literal + deepcopy
    assert ns["notebook_cfg"] == ns["CONFIG"]
    return ns["CONFIG"]


def test_all_code_cells_compile(nb):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            code = "\n".join("pass" if re.match(r"^\s*[%!]", ln) else ln for ln in _src(c).splitlines())
            compile(code, f"{NB_NAME}[cell {i}]", "exec")


def test_title_says_which_notebook_this_is(nb):
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown"
    md = _src(first)
    assert "Original single-stage notebook (pre Package 2 Phase B)" in md
    assert "main_p2p.ipynb" in md and "main_1stage_residual.ipynb" in md


def test_notebook_is_not_generated():
    for tool in (ROOT / "tools").glob("*.py"):
        text = tool.read_text(encoding="utf-8")
        assert 'ROOT / "main_1stage.ipynb"' not in text, tool.name      # nobody writes it
        assert ":main_1stage.ipynb" not in text, tool.name                # nobody derives from it


def test_config_is_the_pre_phase_b_one(cfg):
    # raw EEG input: no band-power features
    assert "eeg_features" not in cfg["preprocessing"]
    assert cfg["data"]["n_eeg_channels"] == 32
    # 24 kinematic features: pose 10 + gated F_L, F_G + pose velocity 10 + gated-force velocity 2
    assert cfg["data"]["n_kin_features"] == 24
    assert cfg["preprocessing"]["kinematics"]["drop_indices"] == [12, 25]
    assert cfg["model"]["gat"]["kin_gat_dim"] == 12
    assert cfg["model"]["gat"]["kin_skip_indices"] == list(range(10, 24))
    assert cfg["data"]["cache_dir"] == "data/cache_main_stride500_square_pose_force_sg51"
    # original encoder
    tf = cfg["model"]["transformer"]
    assert (tf["n_layers"], tf["d_model"], tf["n_heads"], tf["dropout"]) == (4, 128, 8, 0.2)
    # none of the Phase-B switches
    for k in ("kin_dropout", "eeg_channel_dropout", "eeg_groups"):
        assert not cfg["model"].get(k), k
    assert not cfg["training"]["loss"].get("w_eeg_only")
    assert "residual" not in cfg
    # training: one stage, into the untagged folder
    tr = cfg["training"]
    assert tr["checkpoint_dir"] == "outputs/checkpoints_1stage"
    ss = tr["single_stage"]
    assert (ss["lr"], ss["epochs"]) == (1e-3, 200)
    assert type(ss["early_stop_patience"]) is int and 0 < ss["early_stop_patience"] <= 50
    assert (tr["optimizer"], tr["scheduler"], tr["warmup_epochs"], tr["weight_decay"]) == ("adamw", "cosine", 5, 1e-2)


def test_no_phase_b_run_tag_anywhere(nb, cfg):
    code = _all_code(nb)
    for bad in ("p2b", "p2c", "RUN_TAG", "RESULTS_DIR", "tagged_output(", "eeg_transform",
                "normalized_prepare_batch_factory", "use_kin=False", "checkpoints_1stage_"):
        assert bad not in code, bad
    assert "p2b" not in json.dumps(cfg)
    # "step1" only names the plot folder results/P{id}/main/<config>_step1 (main.plots)
    assert cfg["outputs"]["run_tag"] == "step1"


def test_output_paths_are_the_untagged_ones(nb):
    ev = _code(nb, "# EVALUATION — Full metric suite")
    assert 'REPORT_PATH = "results/evaluation_report_1stage.md"' in ev
    assert 'os.path.join(_cfg["training"]["checkpoint_dir"], _subj, "best.pt")' in ev
    mag = _code(nb, "# MAGNITUDE EVALUATION")
    assert 'MAGNITUDE_CSV = "results/magnitude_report_1stage.csv"' in mag
    assert 'MAGNITUDE_PNG = "results/magnitude_report_1stage.png"' in mag
    assert 'MAGNITUDE_LOAD_CSV = "results/magnitude_by_load_1stage.csv"' in mag
    assert 'MAGNITUDE_LOAD_PNG = "results/magnitude_by_load_1stage.png"' in mag
    abl = _code(nb, "HOW MUCH DOES THE MODEL USE EEG vs KINEMATICS?")
    assert 'ABLATION_PNG    = "results/input_ablation_1stage.png"' in abl
    assert 'ABLATION_CSV    = "results/input_ablation_1stage.csv"' in abl
    interp = _code(nb, "INTERPRETABILITY — Full-Signal Triptych")
    assert 'os.path.join("outputs", "interpretability_1stage", _subj)' in interp
    event = _code(nb, "Event-locked interpretability (all test lifts)")
    assert 'os.path.join("outputs", "interpretability_1stage_event", _subj)' in event
    # per-subject checkpoint folder: outputs/checkpoints_1stage/{subj}, resumable
    helper = _code(nb, "def train_single_stage(")
    assert 'ckpt_dir = str(Path(tr["checkpoint_dir"]) / subj)' in helper
    assert "resume=True" in helper
    assert 'tc.early_stop_patience = ss.get("early_stop_patience", tc.early_stop_patience)' in helper


def test_cells_are_in_the_expected_order(nb):
    cells = nb["cells"]

    def idx(needle):
        hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and needle in _src(c)]
        assert len(hits) == 1, (needle, hits)
        return hits[0]

    order = [idx(n) for n in ("CONFIG = {", "def train_single_stage(", "PER-SUBJECT PRE-PROCESSING LOOP",
                              "PER-SUBJECT TRAINING LOOP", "# EVALUATION — Full metric suite",
                              "# MAGNITUDE EVALUATION", "HOW MUCH DOES THE MODEL USE EEG vs KINEMATICS?",
                              "INTERPRETABILITY — Full-Signal Triptych",
                              "Event-locked interpretability (all test lifts)")]
    assert order == sorted(order)
    pre = _src(cells[order[2]])
    assert len(re.findall(r"^subject_list = \[[\d, ]+\]", pre, re.M)) == 1      # the user's list of participants
    assert "local_prepare_batch_factory(" in pre                                # raw EEG, no EEG statistics
    assert "MAX_WINDOWS_PER_SERIES = None" in _src(cells[order[7]])             # all windows plotted


def test_everything_the_notebook_imports_from_main_still_exists(nb):
    """The library moved on (Phase B / C); every name the original notebook imports must still be there."""
    import importlib

    code = "\n".join(ln for ln in _all_code(nb).splitlines() if not re.match(r"^\s*[%!]", ln))
    pat = re.compile(r"^\s*from (main(?:\.\w+)*) import (\([^)]*\)|[^\n]+)", re.M)
    seen = 0
    for mod_name, names in pat.findall(code):
        mod = importlib.import_module(mod_name)
        names = re.sub(r"#[^\n]*", "", names).strip("()")
        for name in (n.strip().split(" as ")[0] for n in names.replace("\n", ",").split(",")):
            if name:
                assert hasattr(mod, name), f"{mod_name}.{name}"
                seen += 1
    assert seen > 40


def test_model_and_loss_built_from_the_config_are_the_original_ones(cfg):
    """Current library + original CONFIG = the original model: no Phase-B pathway is on."""
    import torch
    from main import build_kg_gt_from_config, build_loss_from_config

    torch.manual_seed(0)
    model = build_kg_gt_from_config(cfg, input_dim=32, kin_dim=24, edge_prior=torch.rand(5, 5))
    # parameter count of the checkpoints in outputs/checkpoints_1stage (model of commit 4d34ce2)
    assert sum(p.numel() for p in model.parameters()) == 1_211_107
    assert len(model.encoder.layers) == 4
    assert model.gat_kin_dim == 12 and model.kin_skip_proj[0].in_features == 14
    assert not getattr(model, "kin_dropout", 0.0) and not getattr(model, "eeg_channel_dropout", 0.0)
    eeg, kin = torch.randn(2, 500, 32), torch.randn(2, 500, 24)
    model.train()                                  # no kinematic-pathway dropout: kin always used
    torch.manual_seed(1)
    a = model(eeg, kin)
    torch.manual_seed(1)
    b = model(eeg, kin, use_kin=True, use_eeg=True)
    assert a.shape == (2, 500, 5) and torch.equal(a, b)
    loss = build_loss_from_config(cfg["training"], channel_weights=[1.0] * 5)
    assert not getattr(loss, "w_eeg_only", 0.0)
    assert torch.isfinite(loss(a, torch.randn(2, 500, 5)))
