"""Build ``main_1stage.ipynb`` from the git-HEAD version of ``main.ipynb``.

main_1stage.ipynb = main.ipynb with the two-stage scheme (Transformer-only
pre-training -> frozen-encoder KG-GT fine-tuning) replaced by ONE stage that
trains the full KG-GT model jointly from scratch.  Preprocessing, cache, data
split and loss are untouched; checkpoints/results go to separate *_1stage
locations so the two notebooks never overwrite each other.

The source is ``git show <rev>:main.ipynb`` (default HEAD) so uncommitted edits
in the working copy of main.ipynb are not inherited.  Every edit is a substring
lookup that must match exactly once, otherwise the script fails loudly.

Usage:
    env/Scripts/python.exe tools/build_main_1stage_notebook.py [--rev HEAD] [--out main_1stage.ipynb]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_head_notebook(rev: str = "HEAD") -> dict:
    raw = subprocess.run(
        ["git", "show", f"{rev}:main.ipynb"], cwd=ROOT, check=True, capture_output=True
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


def replace_between(text: str, start: str, end: str, new: str) -> str:
    """Replace text[start-marker : end-marker) with ``new`` (end marker is kept)."""
    for m in (start, end):
        if text.count(m) != 1:
            raise AssertionError(f"marker must occur exactly once (found {text.count(m)}): {m[:80]!r}")
    i, j = text.index(start), text.index(end)
    if j <= i:
        raise AssertionError(f"end marker precedes start marker: {start[:40]!r} / {end[:40]!r}")
    return text[:i] + new + text[j:]


def edit_cell(nb: dict, needle: str, edits: list[tuple[str, str]]) -> int:
    i = find_cell(nb, needle)
    text = src(nb["cells"][i])
    for old, new in edits:
        text = replace_once(text, old, new)
    set_src(nb["cells"][i], text)
    return i


def clear_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


# ── new cell contents ────────────────────────────────────────────────────────

TITLE_MD = """\
# Single-stage KG-GT notebook (Using `main` module)

This notebook is a copy of `main.ipynb` that trains **one stage**: the full KG-GT model
(Transformer encoder → muscle-node projection → kinematics-guided GAT → per-muscle decoder,
plus the kinematic skip highway and lag alignment) is trained **jointly from scratch**, with
every parameter trainable from the first epoch.

**What is different from `main.ipynb`**

| | `main.ipynb` (two-stage) | `main_1stage.ipynb` (this notebook) |
|---|---|---|
| Stage 1 | Transformer-only EEG model, `stage1_lr`=1e-3, 75 epochs | — (none) |
| Stage 2 / training | KG-GT, encoder/node-proj loaded from stage 1, everything except GAT, decoder, kin-skip, node-proj and the last 1–2 encoder layers **frozen**, `stage2_lr`=1e-5, `reduce` scheduler, no warmup | KG-GT, **nothing frozen**, `single_stage.lr`=1e-3, `cosine` + 5 warmup epochs (AdamW), `single_stage.epochs`=200 |
| GAT / decoder init | re-initialised between stages | trained together with the encoder from epoch 1 |
| Checkpoints | `outputs/checkpoints_main_stage1/…`, `outputs/checkpoints_main_stage2/…` | `outputs/checkpoints_1stage/{subject}/` (`last.pt` + `best.pt`, resumable) |
| Results | `results/evaluation_report.md`, `outputs/interpretability/`, `outputs/losocv_results.csv` | `results/evaluation_report_1stage.md`, `outputs/interpretability_1stage/`, `outputs/losocv_results_1stage.csv`, `results/input_ablation_1stage.*` |

Everything else is identical to `main.ipynb`: same participants and condition-stratified
train/val/test split, same preprocessing and **same cache** (`data/cache_main_stride500_square_pose_sg51`;
preprocessing is identical, only training differs), same 20 kinematic inputs (10 hand-pose → GAT
edge bias, 10 SG-51 pose velocities → kin skip highway), same loss weights, batch size,
gradient accumulation, grad clipping, AMP and seed. The parameter groups used by `train_model`
are the same as in stage 1 of `main.ipynb` (`transformer_lr_scale`=0.1 → encoder + node projection at
1e-4, GAT/decoder at 1e-3, kin-skip at lr/3 with stronger weight decay).

There is also a new cell after the evaluation: **"How much does the model use EEG vs kinematics?"**
— Pearson r on the test set with the kinematics zeroed / time-shuffled and with the EEG time-shuffled.

**How to compare with `main.ipynb`:** run both notebooks on the same `subject_list`; compare
`results/evaluation_report.md` (two-stage) with `results/evaluation_report_1stage.md` (this one) —
same test series, same metrics — and the best epoch / final val loss printed by the training loops.
The input-ablation cell can also be pasted into `main.ipynb` to see whether the two training schemes
rely on EEG and kinematics differently.
"""

SINGLE_STAGE_HELPER = '''

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-STAGE TRAINING ROUTINE (used by the per-subject loop and LOSOCV below)
# ─────────────────────────────────────────────────────────────────────────────
def train_single_stage(cfg, subj, eeg_dim, kin_dim, edge_prior,
                       train_loader, val_loader, prepare_batch, loss_fn, device):
    """Train the full KG-GT model jointly in ONE stage (no Transformer-only
    pre-training, nothing frozen).

    Hyper-parameters: lr / epochs from cfg["training"]["single_stage"]; optimizer,
    scheduler (cosine), warmup_epochs, transformer_lr_scale, early_stop_patience,
    grad clipping, AMP and grad accumulation from the shared cfg["training"] keys.
    Checkpoints: {cfg["training"]["checkpoint_dir"]}/{subj}/ (last.pt + best.pt);
    an interrupted run resumes from last.pt.

    Returns (end_of_training_model, best_model, TrainResult, ckpt_dir).
    """
    tr = cfg["training"]
    ss = tr["single_stage"]
    ckpt_dir = str(Path(tr["checkpoint_dir"]) / subj)
    # never write into the two-stage notebook's folders
    assert "checkpoints_main" not in Path(ckpt_dir).as_posix(), ckpt_dir

    model = build_kg_gt_from_config(
        cfg, input_dim=eeg_dim, kin_dim=kin_dim, edge_prior=edge_prior,
    ).to(device)
    for p in model.parameters():          # everything trainable from the start
        p.requires_grad = True
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == n_all

    tc = TrainConfig.from_config(tr, max_epochs=ss["epochs"], lr=ss["lr"])
    tc.checkpoint_dir = ckpt_dir
    print(f"  Trainable parameters : {n_trainable:,} / {n_all:,}")
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
    return model, best, result, ckpt_dir
'''

TRAINING_CELL = '''\
# ============================================================================
# PER-SUBJECT TRAINING LOOP  (SINGLE STAGE — full KG-GT trained jointly)
# ============================================================================
# Iterates over subject_data (populated by the pre-processing loop above).
# For each subject the full KG-GT model (encoder + node projection + GAT +
# decoder + kin skip highway) is trained ONCE, from scratch, with every
# parameter trainable (see train_single_stage in the helper cell).
#
# Checkpoints are saved to:
#   outputs/checkpoints_1stage/{subject}/   (last.pt + best.pt, resumable)
#
# After the loop, the LAST subject's variables are exported to global scope
# so all downstream single-subject cells (evaluation, interpretability) work
# without modification.
#
# Results are collected in `per_subject_results` dict keyed by "P{id}".
# For compatibility with cells written for the two-stage notebook,
# stage1_model is None and result_stage1 / result_stage2 both hold the single
# training result (also available as result_single).
# ─────────────────────────────────────────────────────────────────────────────
from main.training import compute_mean_baseline_loss
from main.plots import plot_training_history
import gc, time
from pathlib import Path

per_subject_results = {}   # keyed by "P{id}"

for _fold_idx, (_subj, _sd) in enumerate(subject_data.items()):
    _t_fold = time.time()
    print(f"\\n{'='*70}")
    print(f"  TRAINING {_subj}  (fold {_fold_idx + 1}/{len(subject_data)})")
    print(f"{'='*70}")

    _cfg          = _sd["cfg"]
    _train_loader = _sd["train_loader"]
    _val_loader   = _sd["val_loader"]
    _test_loader  = _sd["test_loader"]
    _train_ds     = _sd["train_ds"]
    _val_ds       = _sd["val_ds"]
    _test_ds      = _sd["test_ds"]
    _prepare_batch= _sd["prepare_batch"]
    _emg_mean     = _sd["emg_mean"]
    _emg_std      = _sd["emg_std"]
    _kin_mean     = _sd["kin_mean"]
    _kin_std      = _sd["kin_std"]
    _edge_prior   = _sd["edge_prior"]
    _loss_fn      = _sd["loss_fn"]
    _eeg_dim      = _sd["true_eeg_dim"]
    _kin_dim      = _sd["true_kin_dim"]

    # ── SINGLE STAGE: full KG-GT joint training ───────────────────────────
    print(f"\\n--- SINGLE-STAGE: Full KG-GT joint training [{_subj}] ---")
    _baseline_mse = compute_mean_baseline_loss(
        _train_loader, loss_fn=_loss_fn, device=device, prepare_batch=_prepare_batch
    )
    print(f"  Mean-predictor baseline MSE: {_baseline_mse:.4f}")

    _model, _bm, _r, _ckpt_dir = train_single_stage(
        _cfg, _subj, _eeg_dim, _kin_dim, _edge_prior,
        _train_loader, _val_loader, _prepare_batch, _loss_fn, device,
    )
    print(f"  Single-stage final train loss : {_r.history['train'][-1]:.4f}")
    print(f"  Single-stage final val   loss : {_r.history['val'][-1]:.4f}")
    print(f"  Best val loss                 : {_r.best_val:.4f}")

    _fig_hist = plot_training_history(_r.history, stage=f"Single-stage {_subj}", cfg=_cfg)
    plt.show()

    # ── Store results ──────────────────────────────────────────────────────
    per_subject_results[_subj] = dict(
        model=_bm,
        stage1_model=None,                 # no Stage 1 in this notebook
        train_ds=_train_ds, val_ds=_val_ds, test_ds=_test_ds,
        train_loader=_train_loader, val_loader=_val_loader, test_loader=_test_loader,
        prepare_batch=_prepare_batch,
        emg_mean=_emg_mean, emg_std=_emg_std,
        kin_mean=_kin_mean, kin_std=_kin_std,
        result_stage1=_r, result_stage2=_r, result_single=_r,
        cfg=_cfg,
        edge_prior=_edge_prior,
        true_eeg_dim=_eeg_dim,
        true_kin_dim=_kin_dim,
        checkpoint_dir=_ckpt_dir,
    )

    elapsed = (time.time() - _t_fold) / 60.0
    print(f"\\n  [{_subj}] Complete in {elapsed:.1f} min")

    # ── After last subject: export to global scope for downstream cells ────
    if _fold_idx == len(subject_data) - 1:
        model         = _model
        best_model    = _bm
        stage1_model  = None
        train_ds      = _train_ds
        val_ds        = _val_ds
        test_ds       = _test_ds
        train_loader  = _train_loader
        val_loader    = _val_loader
        test_loader   = _test_loader
        prepare_batch = _prepare_batch
        emg_mean      = _emg_mean
        emg_std       = _emg_std
        kin_mean      = _kin_mean
        kin_std       = _kin_std
        result_single = _r
        result_stage1 = _r
        result_stage2 = _r
        subject_str   = _subj
        edge_prior    = _edge_prior
        true_eeg_dim  = _eeg_dim
        true_kin_dim  = _kin_dim
        notebook_cfg  = _cfg
        print(f"\\n[!] Global scope updated with last subject ({_subj}) variables.")
    else:
        # Free GPU memory between subjects
        del _model, _bm
        torch.cuda.empty_cache()
        gc.collect()

print(f"\\n{'='*70}")
print(f"  ALL SUBJECTS TRAINED (single stage): {list(per_subject_results.keys())}")
print(f"{'='*70}")
'''

ABLATION_CELL = '''\
# ============================================================================
# HOW MUCH DOES THE MODEL USE EEG vs KINEMATICS?  (test set, input ablation)
# ============================================================================
# For each subject the best model is run on the TEST loader through the same
# prepare_batch (normalisation + kin column drop) as the evaluation above; the
# normalised inputs are then perturbed:
#   full         unchanged
#   kin_zero     kinematics = 0 in z-space (= per-feature training mean)
#   kin_shuffle  kinematics time-shuffled within each window (fixed, seeded permutation)
#   eeg_shuffle  EEG time-shuffled within each window (fixed, seeded permutation)
# Reported: Pearson r per muscle + mean, and the mean predicted level during
# rest (test target below the per-muscle 10th percentile of the TRAINING target;
# the notebook has no absolute rest threshold — the loss uses a per-window
# relative one), full vs kin-zeroed.  All values in normalised (z) units.
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from main import (
    input_ablation, rest_threshold_from_loader,
    ABLATION_CONDITIONS, ABLATION_LABELS, ABLATION_COLORS,
)

EMG_NAMES       = ["FDI", "APB", "ADM", "ECR", "FCR"]
REST_PERCENTILE = 10.0
ABLATION_SEED   = CONFIG["training"]["seed"]

ablation_results = {}
_abl_rows = []
for _subj, _res in per_subject_results.items():
    _cfg = _res["cfg"]
    _pb  = _res["prepare_batch"]
    _bm  = _res["model"]          # best.pt already loaded by the training loop
    _bm.eval()

    # rest threshold from the training targets (unshuffled, no dropped windows)
    _train_eval_loader = DataLoader(_res["train_ds"], batch_size=_cfg["training"]["batch_size"],
                                    shuffle=False, num_workers=0)
    _rest_thr = rest_threshold_from_loader(_train_eval_loader, _pb, q=REST_PERCENTILE)

    _abl = input_ablation(
        _bm, _res["test_loader"], _pb, device,
        seed=ABLATION_SEED,
        use_amp=_cfg["training"].get("use_amp", False),
        rest_threshold=_rest_thr,
    )
    _abl.predictions.clear()      # keep only the summaries in memory
    ablation_results[_subj] = _abl

    print(f"\\n{'─'*60}\\n  {_subj}\\n{'─'*60}")
    print(_abl.as_table(EMG_NAMES))
    _rf = _abl.r_mean["full"]
    print(f"  Δr (full − kin_zero)    = {_rf - _abl.r_mean['kin_zero']:+.3f}")
    print(f"  Δr (full − kin_shuffle) = {_rf - _abl.r_mean['kin_shuffle']:+.3f}")
    print(f"  Δr (full − eeg_shuffle) = {_rf - _abl.r_mean['eeg_shuffle']:+.3f}")
    _abl_rows.append({
        "Subject": _subj,
        **{f"r_{c}": _abl.r_mean[c] for c in ABLATION_CONDITIONS},
        "rest_target": float(np.nanmean(_abl.rest_target)),
        "rest_pred_full": float(np.nanmean(_abl.rest_pred["full"])),
        "rest_pred_kin_zero": float(np.nanmean(_abl.rest_pred["kin_zero"])),
    })

ablation_df = pd.DataFrame(_abl_rows)
if len(ablation_df) > 1:
    _gm = ablation_df.drop(columns="Subject").mean().to_dict()
    _gm["Subject"] = "GRAND MEAN"
    ablation_df = pd.concat([ablation_df, pd.DataFrame([_gm])], ignore_index=True)
print("\\n" + "=" * 65)
print("  INPUT ABLATION SUMMARY (mean Pearson r over muscles; rest level in z units)")
print("=" * 65)
print(ablation_df.round(3).to_string(index=False))

# ── Bar plot: r per muscle, one bar per condition, one panel per subject ────
_n = len(ablation_results)
_ncols = min(3, _n)
_nrows = int(np.ceil(_n / _ncols))
fig_abl, _axes = plt.subplots(_nrows, _ncols, figsize=(4.8 * _ncols, 3.2 * _nrows),
                              squeeze=False, sharey=True, facecolor="white")
_x = np.arange(len(EMG_NAMES) + 1)
_w = 0.8 / len(ABLATION_CONDITIONS)
for _ax, (_subj, _abl) in zip(_axes.flat, ablation_results.items()):
    for _k, _c in enumerate(ABLATION_CONDITIONS):
        _vals = list(_abl.r[_c]) + [_abl.r_mean[_c]]
        _ax.bar(_x + (_k - (len(ABLATION_CONDITIONS) - 1) / 2) * _w, _vals, _w,
                color=ABLATION_COLORS[_c], label=ABLATION_LABELS[_c])
    _ax.axhline(0, color="black", linewidth=0.6)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(EMG_NAMES + ["mean"])
    _ax.set_title(_subj, fontsize=10)
    _ax.grid(axis="y", alpha=0.3)
for _ax in list(_axes.flat)[_n:]:
    _ax.axis("off")
_axes[0, 0].set_ylabel("Pearson r (test)")
_axes[0, 0].legend(fontsize=7, frameon=False)
fig_abl.suptitle("EEG vs kinematics: input ablation (single-stage KG-GT)", fontsize=11)
fig_abl.tight_layout()
os.makedirs("results", exist_ok=True)
fig_abl.savefig("results/input_ablation_1stage.png", dpi=200, bbox_inches="tight")
ablation_df.to_csv("results/input_ablation_1stage.csv", index=False)
plt.show()
print("✓  Saved results/input_ablation_1stage.png and results/input_ablation_1stage.csv")
'''

LOSO_TRAIN = '''\
        # ── SINGLE STAGE: full KG-GT trained jointly (same routine as above) ──
        _ss = notebook_cfg["training"]["single_stage"]
        print(f"  Single-stage: full KG-GT joint training ({_ss['epochs']} epochs, lr={_ss['lr']:.0e}) ...")
        _model, _bm, _r, _ckpt_dir = train_single_stage(
            notebook_cfg, _subj, _eeg_dim, _kin_dim, _edge_prior,
            _tr_loader, _val_loader, _pb, _loss_fn, device,
        )

        # ── Evaluation on held-out TEST subject (best.pt already loaded) ─────
'''

LOSO_HIST = '''\
        # 1. Training History (single stage)
        fig_hist, ax1 = plt.subplots(1, 1, figsize=(8, 4), facecolor="white")
        ss_epochs = np.arange(1, len(_r.history["train"]) + 1)
        ax1.plot(ss_epochs, _r.history["train"], label="Single-stage Train", color="#4C9BE8", linewidth=1.5)
        ax1.plot(ss_epochs, _r.history["val"],   label="Single-stage Val",   color="#E87B4C", linewidth=1.5)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title(f"Single-stage — {_subj}")
        ax1.legend(); ax1.grid(alpha=0.3)

        plt.suptitle(f"Single-stage Training History — {_subj}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"outputs/figures/history_1stage_{_subj}.png", dpi=300)
        plt.close(fig_hist)

'''


# ── build ────────────────────────────────────────────────────────────────────

def build(rev: str = "HEAD") -> tuple[dict, dict]:
    nb = load_head_notebook(rev)
    clear_outputs(nb)
    log: dict[str, str] = {}

    # 1. title markdown
    i = find_cell(nb, "# Full Method 1 Notebook", cell_type="markdown")
    set_src(nb["cells"][i], TITLE_MD)
    log["title"] = f"cell {i} replaced"

    # 2. CONFIG
    i = edit_cell(nb, "CONFIG = {", [
        (
            '        "stage2_lr": 1e-5,\n',
            '        "stage2_lr": 1e-5,          # two-stage only (unused here)\n',
        ),
        (
            '        "stage1_epochs":       75,\n',
            '        "stage1_epochs":       75,  # two-stage only (unused here)\n',
        ),
        (
            '        "checkpoint_every":     1,\n',
            '        "checkpoint_every":     1,\n'
            '        # ── single-stage (this notebook) ──────────────────────────────────\n'
            '        # Checkpoints -> outputs/checkpoints_1stage/{subject}/ — never the two-stage folders.\n'
            '        "checkpoint_dir": "outputs/checkpoints_1stage",\n'
            '        # The full KG-GT model is trained ONCE, jointly, nothing frozen.\n'
            '        # Only lr / epochs live here; everything else is the shared key above:\n'
            '        #   optimizer="adamw", scheduler="cosine" (eta_min 1e-6), warmup_epochs=5,\n'
            '        #   transformer_lr_scale=0.1 (encoder+node-proj at lr*0.1, as in stage 1),\n'
            '        #   early_stop_patience=50, grad_clip_norm, use_amp, batch_size,\n'
            '        #   gradient_accumulation_steps, seed.\n'
            '        "single_stage": {\n'
            '            "lr":     1e-3,     # = stage1_lr (GAT/decoder group; encoder x transformer_lr_scale)\n'
            '            "epochs": 200,      # = max_epochs\n'
            '        },\n',
        ),
        (
            'notebook_cfg = copy.deepcopy(CONFIG)',
            '# Plot folders (main.plots.get_dynamic_fig_dir) are keyed on training["lr"], which\n'
            '# otherwise falls back to stage2_lr (= the two-stage folder) — point them at the\n'
            '# single-stage LR so saved figures do not overwrite main.ipynb\'s.\n'
            'CONFIG["training"]["lr"] = CONFIG["training"]["single_stage"]["lr"]\n'
            'notebook_cfg = copy.deepcopy(CONFIG)',
        ),
    ])
    log["config"] = f"cell {i} edited"

    # 3. helper cell: add the single-stage routine
    i = find_cell(nb, "def local_prepare_batch_factory(")
    set_src(nb["cells"][i], src(nb["cells"][i]).rstrip("\n") + "\n" + SINGLE_STAGE_HELPER)
    log["helpers"] = f"cell {i} edited (train_single_stage added)"

    # 4. training loop
    i = find_cell(nb, "PER-SUBJECT TRAINING LOOP  (Stage 1 + Stage 2)")
    set_src(nb["cells"][i], TRAINING_CELL)
    log["training"] = f"cell {i} replaced"

    # 5. evaluation: single-stage checkpoints + separate report
    i_eval = edit_cell(nb, "# EVALUATION — Full metric suite", [
        ('REPORT_PATH = "results/evaluation_report.md"',
         'REPORT_PATH = "results/evaluation_report_1stage.md"   # main.ipynb writes evaluation_report.md'),
        ('    _ckpt = f"outputs/checkpoints_main_stage2/{_subj}/best.pt"',
         '    _ckpt = os.path.join(_cfg["training"]["checkpoint_dir"], _subj, "best.pt")   # single-stage'),
        ('_md  = f"# EMG Decoding Evaluation Report\\n\\n"',
         '_md  = f"# EMG Decoding Evaluation Report (single-stage KG-GT)\\n\\n"'),
    ])
    log["evaluation"] = f"cell {i_eval} edited"

    # 6. new ablation cell right after evaluation
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "abl1stage",
        "metadata": {},
        "outputs": [],
        "source": [],
    }
    set_src(new_cell, ABLATION_CELL)
    nb["cells"].insert(i_eval + 1, new_cell)
    log["ablation"] = f"cell {i_eval + 1} added"

    # 7. interpretability: separate output folder
    i = edit_cell(nb, "INTERPRETABILITY — Full-Signal Triptych", [
        ('    _save_dir = os.path.join("outputs", "interpretability", _subj)',
         '    _save_dir = os.path.join("outputs", "interpretability_1stage", _subj)'),
        ('print(f"   Plots saved under: outputs/interpretability/")',
         'print(f"   Plots saved under: outputs/interpretability_1stage/")'),
        ('#   • Saves to: outputs/interpretability/{subject}/',
         '#   • Saves to: outputs/interpretability_1stage/{subject}/'),
    ])
    log["interpretability"] = f"cell {i} edited"

    # 8. LOSOCV: single-stage per fold
    i = find_cell(nb, "[OPTIONAL] Automatic LOSOCV")
    text = src(nb["cells"][i])
    text = replace_once(text,
        "# Checkpoints saved to: outputs/checkpoints_main_stage1|2/LOSO_P{N}/",
        "# Training: single stage (train_single_stage) — full KG-GT trained jointly per fold.\n"
        "# Checkpoints saved to: outputs/checkpoints_1stage/LOSO_P{N}/")
    text = replace_once(text,
        "# Summary CSV saved to: outputs/losocv_results.csv",
        "# Summary CSV saved to: outputs/losocv_results_1stage.csv")
    text = replace_once(text,
        "    build_transformer_only_from_config, build_kg_gt_from_config,\n",
        "    build_kg_gt_from_config,\n")
    text = replace_between(text,
        "        # ── STAGE 1: Transformer pre-training",
        "        _P, _Y = collect_predictions(",
        LOSO_TRAIN)
    text = replace_between(text,
        "        # 1. Training History (Stage 1 & Stage 2)",
        "        # 2. Interpretability Triptych",
        LOSO_HIST)
    text = replace_once(text,
        'plt.savefig(f"outputs/figures/interpretability_{_subj}.png", dpi=300)',
        'plt.savefig(f"outputs/figures/interpretability_1stage_{_subj}.png", dpi=300)')
    text = replace_once(text,
        'plt.savefig(f"outputs/figures/kin_skip_{_subj}.png", dpi=300)',
        'plt.savefig(f"outputs/figures/kin_skip_1stage_{_subj}.png", dpi=300)')
    text = replace_once(text,
        "            stage1_model = _s1\n            model = _s2\n",
        "            stage1_model = None\n            model = _model\n")
    text = replace_once(text,
        "            result_stage1 = _r1\n            result_stage2 = _r2\n",
        "            result_single = _r\n            result_stage1 = _r\n            result_stage2 = _r\n")
    text = replace_once(text,
        "            del _s1, _s2, _bm, _train_ds, _val_ds, _test_ds\n",
        "            del _model, _bm, _train_ds, _val_ds, _test_ds\n")
    text = replace_once(text,
        '    cv_df.to_csv("outputs/losocv_results.csv", index=False)\n'
        '    print("Full results exported to: outputs/losocv_results.csv")',
        '    cv_df.to_csv("outputs/losocv_results_1stage.csv", index=False)\n'
        '    print("Full results exported to: outputs/losocv_results_1stage.csv")')
    set_src(nb["cells"][i], text)
    log["losocv"] = f"cell {i} edited"

    # sanity: no two-stage leftovers in code cells
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = src(c)
        for bad in ("checkpoints_main_stage", "_stage1/", "_stage2/", "stage2_lr\"]", "stage1_lr\"]"):
            if bad in s:
                raise AssertionError(f"two-stage leftover {bad!r} in cell {c.get('id')}")
    return nb, log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", default="HEAD", help="git revision of main.ipynb to copy (default HEAD)")
    ap.add_argument("--out", default=str(ROOT / "main_1stage.ipynb"))
    args = ap.parse_args()
    nb, log = build(args.rev)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    for k, v in log.items():
        print(f"  {k:<17} {v}")
    print(f"Wrote {args.out}  ({len(nb['cells'])} cells, outputs cleared)")


if __name__ == "__main__":
    main()
