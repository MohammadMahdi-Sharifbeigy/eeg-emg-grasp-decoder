"""Build ``main_1stage_residual.ipynb`` from the git-HEAD version of ``main_p2p.ipynb``.

Package 2 — Phase C: does EEG explain what kinematics cannot?

main_1stage_residual.ipynb keeps the imports, device, CONFIG, helper and pre-processing
cells of main_p2p.ipynb (same cache, same condition-stratified split, same band-power EEG
features and normalisation, same model and loss settings) and replaces everything after the
pre-processing loop by:

    Stage K   kinematics-only KG-GT model (encoder output replaced by zeros), trained with
              the normal composite loss, then frozen
    Stage E   EEG-only KG-GT model (both kinematic pathways off) trained on the residual
              emg_z − kin_pred_z with main.losses.ResidualLoss
    evaluation  main.residual_eval: ΔVAF / residual r, two null distributions, per-lift
              analysis, a verdict per muscle

CONFIG differs only in outputs.run_tag ("p2c"), training.checkpoint_dir
("outputs/checkpoints_residual_p2c") and the new "residual" block.

The source is ``git show <rev>:main_p2p.ipynb`` (default HEAD), so the user's working
copy of main_p2p.ipynb (which may carry cell outputs) is neither read nor written.
Every cell and every edit is looked up by a substring that must match exactly once,
otherwise the script fails loudly.

Usage:
    env/Scripts/python.exe tools/build_main_residual_notebook.py [--rev HEAD] [--out main_1stage_residual.ipynb]
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "main_p2p.ipynb"
OUT_NAME = "main_1stage_residual.ipynb"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_head_notebook(rev: str = "HEAD") -> dict:
    raw = subprocess.run(
        ["git", "show", f"{rev}:main_p2p.ipynb"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    return json.loads(raw.decode("utf-8"))


def src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_src(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def find_cell(nb: dict, needle: str, cell_type: str | None = None) -> int:
    hits = [
        i for i, c in enumerate(nb["cells"])
        if needle in src(c) and (cell_type is None or c["cell_type"] == cell_type)
    ]
    if len(hits) != 1:
        raise AssertionError(f"expected exactly 1 cell containing {needle!r}, found {len(hits)}: {hits}")
    return hits[0]


def replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise AssertionError(f"replacement target must occur exactly once (found {n}): {old[:80]!r}")
    return text.replace(old, new)


def new_cell(cell_type: str, cell_id: str, text: str) -> dict:
    cell = {"cell_type": cell_type, "id": cell_id, "metadata": {}}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    set_src(cell, text)
    return cell


def clear_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


def dumps(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


# ── new cell contents ────────────────────────────────────────────────────────

TITLE_MD = """\
# Does EEG explain what kinematics cannot? — residual KG-GT notebook (Package 2, Phase C, run tag `p2c`)

**Question.** The single-stage models reach Pearson r ≈ 0.89 on the P1 test series, but the output comes from the kinematics and the gated forces: time-shuffling the EEG changes r by −0.003. A per-lift linear analysis suggested that EEG band power explains none of the lift-to-lift magnitude that force does not explain, and that one muscle (Flexor Digitorum, FD) has a series-level amplitude drift that no input explains. This notebook measures the **unique** contribution of the EEG: fit the best kinematics-only model, freeze it, and let an EEG-only model predict what is left.

## The two stages

**Stage K — kinematics-only model.** The same KG-GT architecture, called with `use_eeg=False` (wrapper `KinOnly`): the EEG encoder is not run and its output is replaced by zeros. The node projection is a linear layer, so the muscle-node embeddings are then its **bias** only — one learned vector per muscle, the same at every time step. The kinematics-guided GAT (static + kinematic edge bias), the kinematic skip highway, the per-muscle decoder and the lag alignment work as usual. No gradient reaches the encoder. It is trained with the composite loss of `CONFIG` with the EEG-specific parts off (`w_eeg_only` = 0, `kin_dropout` = 0, no EEG-only auxiliary output), early stopping on the smoothed validation loss, then the best checkpoint is loaded and the model is frozen.

**Stage E — EEG residual model.** For every batch the target is the residual in normalised units, `res = emg_z − kin_pred_z`, with `kin_pred_z` from the frozen Stage-K model (eval mode, no gradient). The model is the same architecture with both kinematic pathways off (`use_kin=False`, wrapper `EEGOnly`): band-power EEG → encoder → node projection → GAT with the static edge bias only → decoder. The regularised encoder and the electrode dropout of `CONFIG` stay on. The residual is signed and has mean ≈ 0, so the envelope-specific loss terms do not apply; the loss is `ResidualLoss` = Huber(δ = 1) + 0.2 · (1 − Pearson r along time, per window and muscle; windows whose residual is constant are skipped). The cell prints the loss of predicting zero residual: a trained model has to be below it on the validation series.

Both stages use `residual.lr`, `residual.early_stop_patience` and their own number of epochs; optimiser, cosine schedule, warm-up, parameter groups (encoder + node projection at lr × 0.1, kin skip at lr / 3), gradient clipping, AMP, batch size and gradient accumulation are the shared `training` keys of `main_p2p.ipynb`.

## What the numbers mean

All numbers are for the **test** series, predicted in 1 s windows through `main.inference` (the same path as the other notebooks). VAF and RMSE are in original EMG units, residual statistics in z units.

| Number | Meaning |
|---|---|
| kin r / VAF / RMSE | the kinematics-only model alone |
| kin + EEG r / VAF / RMSE | kinematics-only prediction + predicted residual |
| ΔVAF, Δr | (kin + EEG) − (kin only); ΔVAF in percentage points. Negative = the EEG model makes the prediction worse |
| res r | Pearson r between predicted and true residual |
| res FVE | fraction of the residual variance explained, 1 − Σ(res − pred)² / Σ(res − mean)²; negative = worse than predicting the mean residual |
| null "shuffle" | the same EEG model, EEG time-shuffled within each window. The window keeps its values, so its mean level survives: this null removes only the timing inside the window |
| null "swap" | the same EEG model, each window fed the EEG of a **different** test window (a random derangement of all test windows: no window keeps its own EEG). EEG statistics stay realistic, but the EEG is unrelated to this window's EMG |
| p | (1 + number of null draws ≥ observed) / (1 + `n_null`), per muscle and for the mean over muscles. With `n_null` = 20 the smallest possible p is 1/21 ≈ 0.048 |
| lift r | per test lift: mean true residual vs mean predicted residual over the hold phase (lift-off + 0.3 s … replace − 0.3 s), then Pearson r across lifts — per series and pooled, with the same for the "swap" null. This is the question whether EEG explains lift-to-lift magnitude left over by the kinematics (FD especially) |
| lift r within | pooled lift r after subtracting each series' mean from both sides. The pooled lift r of two test series can be large because of one between-series offset (two points); this one cannot |

## What would count as evidence

A muscle gets the verdict **"EEG adds unique information"** only if all three hold:

1. ΔVAF > 0 over all test series, **and**
2. p ≤ 0.05 for ΔVAF against the "swap" null, **and**
3. the residual r is > 0 in **both** test series — per lift in both, or per sample in both.

Otherwise the verdict is **"no evidence"**. "No evidence" is not evidence of absence: it means this model, on these test series, did not find a contribution. Five muscles are tested, so one p ≤ 0.05 by chance is not unlikely; a single muscle passing should be read with that in mind.

## Outputs (nothing of an earlier run is overwritten)

| Output | Path |
|---|---|
| Stage K checkpoints | `outputs/checkpoints_residual_p2c/kin_only/{subject}/` (`last.pt` + `best.pt`, resumable) |
| Stage E checkpoints | `outputs/checkpoints_residual_p2c/eeg_residual/{subject}/` |
| Report table | `results/p2c/residual_report.csv` (per muscle + MEAN; per test series and all test series; nulls, p-values, verdict) |
| Per-lift table | `results/p2c/residual_per_lift.csv` (one row per test lift and muscle) |
| Figures | `results/p2c/residual_delta_vaf.png`, `results/p2c/residual_per_lift.png` |
| Training curves saved by `main.plots` | `results/P{id}/main/<config>_p2c/plots/` |

Stage E resumes from its `last.pt`. Its targets depend on the Stage-K model, so if Stage K is trained again, delete the `eeg_residual` folder (or change the run tag) before running Stage E.

## What is the same as `main_p2p.ipynb`

`CONFIG` is that of `main_p2p.ipynb` except `outputs.run_tag` (`p2c`), `training.checkpoint_dir` and the new `residual` block. Same participants and condition-stratified split, same preprocessing and cache (`data/cache_main_stride500_square_pose_force_sg51`, read only), same 128 log band-power EEG inputs (z-scored with the training series, clipped to ±5), same 24 kinematic inputs, same model sizes and loss weights. The helper and pre-processing cells are copied unchanged. This notebook has no interpretability cells.
"""

STAGE_K_MD = """\
### Stage K — kinematics-only model

`KinOnly(model)` calls the KG-GT model with `use_eeg=False`: zeros instead of the encoder output, so the node embeddings are the bias of the node projection. Loss = the composite loss of `CONFIG` with `w_eeg_only` = 0; `kin_dropout` = 0 and no EEG-only auxiliary output (they exist to make the EEG path work and have no meaning without it).
"""

STAGE_E_MD = """\
### Stage E — EEG model on the residual

The Stage-K model is frozen. Target of every batch: `emg_z − kin_pred_z`. `EEGOnly(model)` calls a new KG-GT model with `use_kin=False` (no kinematic edge bias, no kin skip). The printed "zero residual" loss is the value of predicting 0 everywhere, i.e. of adding nothing to the kinematics-only model.
"""

EVAL_MD = """\
### Evaluation — what does the EEG residual model add on the test series?

Tables, null distributions, per-lift analysis and the verdict per muscle (rules in the first cell). A ΔVAF inside the grey band of the first figure is what the same model gives with EEG that belongs to another window.
"""

TOOLS_CELL = '''\
# ============================================================================
# RESIDUAL TOOLS  (Phase C — used by Stage K and Stage E below)
# ============================================================================
from main import (
    KinOnly, EEGOnly, ResidualLoss,
    residual_prepare_batch_factory, zero_prediction_loss,
)
from main.training import compute_mean_baseline_loss
from main.plots import plot_training_history


def train_stage(cfg, stage, subj, model, train_loader, val_loader, prepare_batch, loss_fn,
                max_epochs, device):
    """Train one stage with main.training.train_model (unchanged) and load its best checkpoint.

    stage: "kin_only" (Stage K) or "eeg_residual" (Stage E) -> checkpoints in
    stage_checkpoint_dir(cfg, stage, subj) = <training.checkpoint_dir>/<stage>/<subj>/
    (last.pt + best.pt; an interrupted run resumes from last.pt).
    lr and early-stopping patience come from cfg["residual"]; optimiser, cosine schedule,
    warm-up, transformer_lr_scale, grad clipping, AMP and grad accumulation from the shared
    cfg["training"] keys. Early stopping and best.pt use the SMOOTHED validation loss
    (mean of the last 5 epochs, main/training.py).

    Returns (best_model in eval mode, TrainResult, checkpoint folder).
    """
    tr = cfg["training"]
    res = cfg["residual"]
    ckpt_dir = stage_checkpoint_dir(cfg, stage, subj)

    tc = TrainConfig.from_config(tr, max_epochs=max_epochs, lr=res["lr"])
    tc.checkpoint_dir = ckpt_dir
    tc.early_stop_patience = res["early_stop_patience"]
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  Parameters           : {n_all:,} (a wrapped model only updates the paths it uses)")
    print(f"  Optimizer / schedule : {tc.optimizer} | {tc.scheduler} | warmup {tc.warmup_epochs} ep "
          f"| lr {tc.lr:.1e} (encoder x{tc.transformer_lr_scale}) | max {tc.max_epochs} ep "
          f"| early-stop patience {tc.early_stop_patience}")
    print(f"  Checkpoints          : {ckpt_dir}")

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        prepare_batch=prepare_batch,
        loss_fn=loss_fn,
        device=device,
        cfg=tc,
        resume=True,
    )

    # ── Load BEST checkpoint from disk into a clean model copy ────────────
    best = copy.deepcopy(model)
    best_path = Path(ckpt_dir) / "best.pt"
    if best_path.exists():
        load_checkpoint(str(best_path), best, device)
        print(f"  Loaded best checkpoint: {best_path}")
    else:
        print(f"  [WARNING] best.pt not found at {best_path}; using end-of-training weights.")
    best.eval()
    return best, result, ckpt_dir
'''

STAGE_K_CELL = '''\
# ============================================================================
# STAGE K — KINEMATICS-ONLY MODEL  (per subject)
# ============================================================================
# Same KG-GT architecture, called with use_eeg=False (KinOnly wrapper): the encoder is not
# run, its output is replaced by zeros, the node embeddings are the node-projection bias.
# Loss: the composite loss of CONFIG with the EEG-specific parts off
#   training.loss.w_eeg_only = 0, model.kin_dropout = 0, model.eeg_only_aux = False
# (they only exist to make the EEG path contribute). Everything else is CONFIG.
# Checkpoints: <training.checkpoint_dir>/kin_only/{subject}/ ; the best one is loaded
# and kept in stage_k[subject]["model"] for Stage E and the evaluation.
# ─────────────────────────────────────────────────────────────────────────────
import time

_res_cfg = notebook_cfg["residual"]
stage_k = {}   # keyed by "P{id}"

for _subj, _sd in subject_data.items():
    _t_fold = time.time()
    print(f"\\n{'='*70}")
    print(f"  STAGE K (kinematics only)  {_subj}")
    print(f"{'='*70}")

    _cfg_k = copy.deepcopy(_sd["cfg"])
    _cfg_k["model"]["kin_dropout"] = 0.0
    _cfg_k["model"]["eeg_only_aux"] = False
    _cfg_k["training"]["loss"]["w_eeg_only"] = 0.0
    _loss_k = build_loss_from_config(_cfg_k["training"], channel_weights=_sd["channel_weights"])

    torch.manual_seed(_res_cfg["seed"])
    _kin_model = KinOnly(build_kg_gt_from_config(
        _cfg_k, input_dim=_sd["true_eeg_dim"], kin_dim=_sd["true_kin_dim"], edge_prior=_sd["edge_prior"],
    )).to(device)

    _baseline = compute_mean_baseline_loss(
        _sd["train_loader"], loss_fn=_loss_k, device=device, prepare_batch=_sd["prepare_batch"]
    )
    print(f"  Mean-predictor baseline loss (train): {_baseline:.4f}")

    _kin_best, _r, _ckpt_dir = train_stage(
        _cfg_k, "kin_only", _subj, _kin_model,
        _sd["train_loader"], _sd["val_loader"], _sd["prepare_batch"], _loss_k,
        max_epochs=_res_cfg["kin_epochs"], device=device,
    )
    if _r.history["train"]:
        print(f"  Stage K final train loss : {_r.history['train'][-1]:.4f}")
        print(f"  Stage K final val   loss : {_r.history['val'][-1]:.4f}")
    print(f"  Stage K best val loss    : {_r.best_val:.4f} (epoch {_r.best_epoch})")
    _fig_hist = plot_training_history(_r.history, stage=f"Stage K kinematics only {_subj}", cfg=_cfg_k)
    plt.show()

    stage_k[_subj] = dict(model=_kin_best, result=_r, cfg=_cfg_k, checkpoint_dir=_ckpt_dir)
    del _kin_model
    print(f"  [{_subj}] Stage K complete in {(time.time() - _t_fold) / 60.0:.1f} min")

print(f"\\n✓  Stage K done for {list(stage_k.keys())}")
'''

STAGE_E_CELL = '''\
# ============================================================================
# STAGE E — EEG MODEL ON THE RESIDUAL  (per subject; needs Stage K)
# ============================================================================
# The Stage-K model is frozen (eval mode, requires_grad False). For every batch
#   target = emg_z − kin_pred_z        (normalised units; kin_pred_z under torch.no_grad())
# computed inside the wrapped prepare_batch (residual_prepare_batch_factory), so train_model
# is the same function as everywhere else. Model: a NEW KG-GT model called with
# use_kin=False (EEGOnly wrapper: no kinematic edge bias, no kin skip); the encoder size,
# its dropout and the electrode dropout are those of CONFIG.
# Loss: ResidualLoss = Huber(delta 1) + 0.2 · (1 − Pearson r along time per window and
# muscle; windows with a constant residual are skipped). The residual is signed and
# zero-mean, so the envelope terms of the composite loss do not apply.
# "Zero residual" loss = predicting 0 everywhere = adding nothing to Stage K.
# Checkpoints: <training.checkpoint_dir>/eeg_residual/{subject}/ (resumable). If Stage K was
# trained again, delete that folder first: its targets belong to the old Stage-K model.
# ─────────────────────────────────────────────────────────────────────────────
import time

_res_cfg = notebook_cfg["residual"]
stage_e = {}   # keyed by "P{id}"

for _subj, _sd in subject_data.items():
    _t_fold = time.time()
    print(f"\\n{'='*70}")
    print(f"  STAGE E (EEG on the residual)  {_subj}")
    print(f"{'='*70}")

    _kin_best = stage_k[_subj]["model"]
    _res_pb = residual_prepare_batch_factory(_sd["prepare_batch"], _kin_best)   # freezes the Stage-K model

    _cfg_e = copy.deepcopy(_sd["cfg"])
    _cfg_e["model"]["kin_dropout"] = 0.0       # no kinematic pathway is used at all
    _cfg_e["model"]["eeg_only_aux"] = False
    _loss_e = ResidualLoss(huber_delta=1.0, w_pearson=0.2)

    torch.manual_seed(_res_cfg["seed"])
    _eeg_model = EEGOnly(build_kg_gt_from_config(
        _cfg_e, input_dim=_sd["true_eeg_dim"], kin_dim=_sd["true_kin_dim"], edge_prior=_sd["edge_prior"],
    )).to(device)
    print(f"  EEG input dim {_sd['true_eeg_dim']} | electrode dropout {_eeg_model.model.eeg_channel_dropout} "
          f"(x{_eeg_model.model.eeg_groups} columns) | encoder dropout {_cfg_e['model']['transformer']['dropout']}")

    _zero_train = zero_prediction_loss(_sd["train_loader"], _loss_e, _res_pb)
    _zero_val   = zero_prediction_loss(_sd["val_loader"],   _loss_e, _res_pb)
    print(f"  Zero-residual baseline loss: train {_zero_train:.4f} | val {_zero_val:.4f}   "
          f"(a model that adds nothing to Stage K)")

    _eeg_best, _r, _ckpt_dir = train_stage(
        _cfg_e, "eeg_residual", _subj, _eeg_model,
        _sd["train_loader"], _sd["val_loader"], _res_pb, _loss_e,
        max_epochs=_res_cfg["eeg_epochs"], device=device,
    )
    if _r.history["train"]:
        print(f"  Stage E final train loss : {_r.history['train'][-1]:.4f}")
        print(f"  Stage E final val   loss : {_r.history['val'][-1]:.4f}")
    print(f"  Stage E best val loss    : {_r.best_val:.4f} (epoch {_r.best_epoch}; smoothed over 5 epochs)   "
          f"zero-residual val loss {_zero_val:.4f}   difference {_r.best_val - _zero_val:+.4f} "
          f"(negative = below the baseline)")
    _fig_hist = plot_training_history(_r.history, stage=f"Stage E EEG residual {_subj}", cfg=_cfg_e)
    plt.show()

    stage_e[_subj] = dict(model=_eeg_best, result=_r, cfg=_cfg_e, checkpoint_dir=_ckpt_dir,
                          zero_train=_zero_train, zero_val=_zero_val)
    del _eeg_model
    print(f"  [{_subj}] Stage E complete in {(time.time() - _t_fold) / 60.0:.1f} min")

print(f"\\n✓  Stage E done for {list(stage_e.keys())}")
'''

EVAL_CELL = '''\
# ============================================================================
# RESIDUAL EVALUATION — what does the EEG residual model add on the test series?
# ============================================================================
# Test series only, 1 s windows through main.inference (same path as the other notebooks;
# the tail shorter than one window is dropped, like the test loader). Per muscle and MEAN,
# per test series and over all test series (main.residual_eval):
#   kin r / VAF / RMSE          kinematics-only model            (VAF, RMSE: original EMG units)
#   kin+EEG r / VAF / RMSE      kinematics-only prediction + predicted residual
#   ΔVAF, Δr                    (kin + EEG) − (kin only)
#   res r, res FVE              predicted vs true residual (z units); FVE can be negative
#   nulls (residual.n_null draws, seeded): the SAME EEG model with the EEG
#     shuffle = time-shuffled within each window (keeps the window's mean level)
#     swap    = taken from a DIFFERENT test window (random derangement of the test windows)
#     → mean ± SD of ΔVAF and res r under each null, p = (1 + #null >= observed) / (1 + n_null)
#   per lift: mean true vs mean predicted residual over the hold phase of every test lift
#     → Pearson r across lifts (per series, pooled, pooled after removing the series means),
#       and the same for the swap null
# Verdict per muscle (rules in the first cell): "EEG adds unique information" only if
#   ΔVAF > 0  AND  p <= 0.05 against the swap null  AND  residual r > 0 in both test series
#   (per lift in both, or per sample in both); otherwise "no evidence".
# Saves {RESULTS_DIR}/residual_report.csv, residual_per_lift.csv, residual_delta_vaf.png,
# residual_per_lift.png.
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import (
    get_split_series, series_conditions, lift_hold_segments,
    predict_residuals, residual_report, residual_verdict, format_residual_report,
    plot_residual_delta_vaf, plot_residual_per_lift,
)
from main.dataloader import emg_channel_names

RESIDUAL_CSV      = f"{RESULTS_DIR}/residual_report.csv"
RESIDUAL_LIFT_CSV = f"{RESULTS_DIR}/residual_per_lift.csv"
RESIDUAL_VAF_PNG  = f"{RESULTS_DIR}/residual_delta_vaf.png"
RESIDUAL_LIFT_PNG = f"{RESULTS_DIR}/residual_per_lift.png"


def evaluate_residual(subj, sd, kin_model, eeg_model, n_null, seed, max_windows=None):
    """Residual report of one subject's test series.

    max_windows: None = every test window (the result); an int keeps only the first
    max_windows windows of each test series (quick checks only).
    Returns (report, per_lift, verdict) — see main.residual_eval.
    """
    cfg = sd["cfg"]
    p_id = int(subj.lstrip("P"))
    test_ds = sd["test_ds"]
    window = cfg["dataset"]["window_size"]
    fs = cfg["data"].get("fs_eeg", 500.0)
    names = emg_channel_names(subj, test_ds.data_dir)      # true channel order, from the HS files

    # unique full series, in the dataset's (sorted) test-split order
    unique, seen = [], set()
    for _e, _k, _m, _start in test_ds._windows:
        if id(_e) not in seen:
            seen.add(id(_e))
            unique.append((_e, _k, _m))
    ids = get_split_series("test", p_id, test_ds.data_dir)
    assert len(ids) == len(unique), (ids, len(unique))
    if max_windows is not None:
        unique = [tuple(a[:max_windows * window] for a in arrays) for arrays in unique]
    series = dict(zip(ids, unique))

    preds = predict_residuals(
        kin_model, eeg_model, series, sd["prepare_batch"],
        window=window, device=device, n_null=n_null, seed=seed,
        batch_size=cfg["training"]["batch_size"],
    )
    segments = {
        sid: lift_hold_segments(p_id, sid, fs=fs, data_dir=test_ds.data_dir,
                                n_samples=len(preds.target[sid]))
        for sid in ids
    }
    report, per_lift = residual_report(preds, sd["emg_mean"], sd["emg_std"],
                                       channel_names=names, segments=segments)
    return report, per_lift, residual_verdict(report)


_res_cfg = notebook_cfg["residual"]
os.makedirs(RESULTS_DIR, exist_ok=True)
residual_reports, residual_lifts, residual_verdicts = {}, {}, {}

for _subj, _sd in subject_data.items():
    _report, _per_lift, _verdict = evaluate_residual(
        _subj, _sd, stage_k[_subj]["model"], stage_e[_subj]["model"],
        n_null=_res_cfg["n_null"], seed=_res_cfg["seed"],
    )
    residual_reports[_subj], residual_lifts[_subj], residual_verdicts[_subj] = _report, _per_lift, _verdict
    _conds = series_conditions(int(_subj.lstrip("P")), _sd["test_ds"].data_dir)
    print(f"\\n{'─' * 100}\\n  {_subj} — test series "
          + ", ".join(f"{_s} ({_conds.get(_s, '?')})" for _s in get_split_series(
              "test", int(_subj.lstrip("P")), _sd["test_ds"].data_dir))
          + f"\\n{'─' * 100}")
    print(format_residual_report(_report))
    print(f"\\n  Verdict per muscle [{_subj}]  (ΔVAF over all test series; p against the window-swap null; "
          f"residual r per test series)")
    for _, _v in _verdict.iterrows():
        print(f"    {_v['muscle']:<5} ΔVAF {_v['d_vaf']:+7.2f}   p {_v['p_swap']:.3f}   "
              f"res r [{_v['res_r_by_series']}]   lift r [{_v['lift_r_by_series']}]   "
              f"→ {_v['verdict']}   ({_v['why']})")

# ── tables ──────────────────────────────────────────────────────────────────
_rep_frames = []
for _subj, _report in residual_reports.items():
    _f = _report.copy()
    _f["verdict"] = ""
    _is_all = _f["group"] == "all test"
    _f.loc[_is_all, "verdict"] = _f.loc[_is_all, "muscle"].map(
        residual_verdicts[_subj].set_index("muscle")["verdict"])
    _f.insert(0, "subject", _subj)
    _rep_frames.append(_f)
residual_df = pd.concat(_rep_frames, ignore_index=True)
residual_df.to_csv(RESIDUAL_CSV, index=False)

_lift_frames = []
for _subj, _per_lift in residual_lifts.items():
    _f = _per_lift.copy()
    _f.insert(0, "subject", _subj)
    _lift_frames.append(_f)
residual_lift_df = pd.concat(_lift_frames, ignore_index=True)
residual_lift_df.to_csv(RESIDUAL_LIFT_CSV, index=False)
print(f"\\n✓  Saved {RESIDUAL_CSV} and {RESIDUAL_LIFT_CSV}")

# ── figures ─────────────────────────────────────────────────────────────────
fig_res = plot_residual_delta_vaf(
    residual_reports, title="Kinematics only vs kinematics + EEG residual (all test series)")
fig_res.savefig(RESIDUAL_VAF_PNG, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓  Saved {RESIDUAL_VAF_PNG}")

if len(residual_lift_df):
    fig_lift = plot_residual_per_lift(
        residual_lift_df, title="Hold-phase residual per test lift: true vs predicted by the EEG model")
    fig_lift.savefig(RESIDUAL_LIFT_PNG, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✓  Saved {RESIDUAL_LIFT_PNG}")
else:
    print("  No lift hold phase lies inside the evaluated samples: per-lift figure skipped.")

# ── verdict ─────────────────────────────────────────────────────────────────
print("\\n" + "=" * 100)
print("  VERDICT — does the EEG explain what the kinematics-only model leaves over?")
print("  Rule: 'EEG adds unique information' only if ΔVAF > 0 AND p <= 0.05 against the window-swap null")
print("        AND residual r > 0 in both test series (per lift in both, or per sample in both).")
print("=" * 100)
for _subj, _verdict in residual_verdicts.items():
    for _, _v in _verdict.iterrows():
        print(f"  {_subj} {_v['muscle']:<5} {_v['verdict']:<28} {_v['why']}")
'''


# ── build ────────────────────────────────────────────────────────────────────

KEEP = [  # (needle, cell type) of the main_1stage cells that are kept, in order
    ("# Single-stage KG-GT notebook", "markdown"),          # replaced by TITLE_MD
    ('ROOT = Path(".")', "code"),                           # imports
    ("device = get_device()", "code"),
    ("CONFIG = {", "code"),                                 # edited
    ("### EMG channel names (true order)", "markdown"),
    ("### Train / val / test split", "markdown"),
    ("def local_prepare_batch_factory(", "code"),           # helpers, unchanged
    ("PER-SUBJECT PRE-PROCESSING LOOP", "code"),            # unchanged
]

CONFIG_EDITS = [
    (
        'CONFIG = {\n'
        '    # ── Package 2, Phase B: make the EEG path contribute (run tag "p2b") ────────\n',
        'CONFIG = {\n'
        '    # ── Package 2, Phase C: does EEG explain what kinematics cannot? (run tag "p2c") ──\n'
        '    # Identical to main_p2p.ipynb\'s CONFIG except: outputs.run_tag, training.checkpoint_dir\n'
        '    # and the "residual" block (Stage K = kinematics-only model, Stage E = EEG model on the\n'
        '    # residual). kin_dropout, w_eeg_only and the EEG-only auxiliary output are switched off\n'
        '    # per stage in the stage cells, not here.\n'
        '    # ── Settings inherited from Phase B ──────────────────────────────────────────\n',
    ),
    (
        '    # * outputs.run_tag "p2b": every file this notebook writes goes to a tagged path\n',
        '    # * outputs.run_tag "p2c": every file this notebook writes goes to a tagged path\n',
    ),
    (
        '        # folder), so the Step-1 results and checkpoints are not overwritten by a p2b run.\n',
        '        # folder), so the Step-1 and Phase B results and checkpoints are not overwritten by this run.\n',
    ),
    ('        "run_tag": "p2b",\n', '        "run_tag": "p2c",\n'),
    (
        '        # ── single-stage (this notebook) ──────────────────────────────────\n'
        '        # Checkpoints of THIS run -> <checkpoint_dir>/{subject}/ — never the two-stage folders;\n'
        '        # must end in the run tag (checked below the dict).\n'
        '        "checkpoint_dir": "outputs/checkpoints_1stage_p2b",\n'
        '        # The full KG-GT model is trained ONCE, jointly, nothing frozen.\n',
        '        # ── checkpoints of THIS run (must end in the run tag, checked below the dict) ──\n'
        '        # Stage K -> <checkpoint_dir>/kin_only/{subject}/, Stage E -> <checkpoint_dir>/eeg_residual/{subject}/\n'
        '        "checkpoint_dir": "outputs/checkpoints_residual_p2c",\n'
        '        # ── single_stage: main_p2p.ipynb only (kept so the two CONFIGs stay comparable;\n'
        '        #    this notebook takes lr / epochs / patience from the "residual" block below) ──\n'
        '        # The full KG-GT model is trained ONCE, jointly, nothing frozen.\n',
    ),
    (
        '    "dataset": {\n'
        '        # Keep in sync with data.window_size / data.stride above\n',
        '    # ── Phase C: the two stages and the evaluation ─────────────────────────────\n'
        '    "residual": {\n'
        '        "kin_epochs": 200,           # Stage K (kinematics-only model): max epochs\n'
        '        "eeg_epochs": 200,           # Stage E (EEG model on the residual): max epochs\n'
        '        "early_stop_patience": 20,   # both stages: epochs without a new best SMOOTHED val loss\n'
        '        "lr": 1e-3,                  # both stages (encoder + node projection x transformer_lr_scale)\n'
        '        "n_null": 20,                # draws per null distribution in the evaluation (smallest p = 1/21)\n'
        '        "seed": 42,                  # model initialisation of both stages and the null draws\n'
        '    },\n'
        '    "dataset": {\n'
        '        # Keep in sync with data.window_size / data.stride above\n',
    ),
    (
        'assert CONFIG["training"]["checkpoint_dir"].endswith("_" + RUN_TAG), \\\n',
        'RESIDUAL_STAGES = ("kin_only", "eeg_residual")\n'
        '\n'
        '\n'
        'def stage_checkpoint_dir(cfg, stage, subj):\n'
        '    """Checkpoint folder of one stage and subject: <training.checkpoint_dir>/<stage>/<subj>."""\n'
        '    assert stage in RESIDUAL_STAGES, stage\n'
        '    base = cfg["training"]["checkpoint_dir"]\n'
        '    assert base.endswith("_" + str(cfg["outputs"]["run_tag"])), base\n'
        '    return f"{base}/{stage}/{subj}"\n'
        '\n'
        '\n'
        'assert CONFIG["training"]["checkpoint_dir"].endswith("_" + RUN_TAG), \\\n',
    ),
    (
        '# Plot folders (main.plots.get_dynamic_fig_dir) are keyed on training["lr"], which\n'
        '# otherwise falls back to stage2_lr (= the two-stage folder) — point them at the\n'
        '# single-stage LR so saved figures do not overwrite main.ipynb\'s.\n'
        'CONFIG["training"]["lr"] = CONFIG["training"]["single_stage"]["lr"]\n',
        '# Plot folders (main.plots.get_dynamic_fig_dir) are keyed on training["lr"] and the run tag:\n'
        '# use the LR of the two stages (the folder also ends in the run tag, so nothing is shared\n'
        '# with main.ipynb / main_p2p.ipynb).\n'
        'CONFIG["training"]["lr"] = CONFIG["residual"]["lr"]\n',
    ),
]


def build_from(source_nb: dict) -> tuple[dict, dict]:
    """Residual notebook from a main_1stage notebook dict (not modified)."""
    nb = copy.deepcopy(source_nb)
    clear_outputs(nb)
    log: dict[str, str] = {}

    kept = []
    for needle, cell_type in KEEP:
        i = find_cell(nb, needle, cell_type=cell_type)
        kept.append(nb["cells"][i])
        log[f"keep {needle[:34]!r}"] = f"source cell {i}"
    title, _imports, _device, config, _md_names, _md_split, _helpers, _pre = kept

    set_src(title, TITLE_MD)
    text = src(config)
    for old, new in CONFIG_EDITS:
        text = replace_once(text, old, new)
    set_src(config, text)
    log["config"] = f"{len(CONFIG_EDITS)} edits"

    nb["cells"] = kept + [
        new_cell("code", "p2ctools", TOOLS_CELL),
        new_cell("markdown", "p2cstagekmd", STAGE_K_MD),
        new_cell("code", "p2cstagek", STAGE_K_CELL),
        new_cell("markdown", "p2cstageemd", STAGE_E_MD),
        new_cell("code", "p2cstagee", STAGE_E_CELL),
        new_cell("markdown", "p2cevalmd", EVAL_MD),
        new_cell("code", "p2ceval", EVAL_CELL),
    ]
    log["new cells"] = "tools, Stage K, Stage E, evaluation (+ 3 markdown)"

    # sanity: nothing of the Phase B run or of the single-stage training is left in the code
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = src(c)
        for bad in ("checkpoints_1stage", "checkpoints_main_", '"run_tag": "p2b"', "_1stage.md", "_1stage.csv"):
            if bad in s:
                raise AssertionError(f"leftover {bad!r} in cell {c.get('id')}")
    ids = [c.get("id") for c in nb["cells"]]
    if len(set(ids)) != len(ids) or not all(ids):
        raise AssertionError(f"cell ids must be unique and non-empty: {ids}")
    return nb, log


def build(rev: str = "HEAD") -> tuple[dict, dict]:
    return build_from(load_head_notebook(rev))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", default="HEAD", help="git revision of main_p2p.ipynb to copy (default HEAD)")
    ap.add_argument("--out", default=str(ROOT / "main_1stage_residual.ipynb"))
    args = ap.parse_args()
    nb, log = build(args.rev)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:   # LF on Windows too
        f.write(dumps(nb))
    for k, v in log.items():
        print(f"  {k:<40} {v}")
    print(f"Wrote {args.out}  ({len(nb['cells'])} cells, outputs cleared)")


if __name__ == "__main__":
    main()
