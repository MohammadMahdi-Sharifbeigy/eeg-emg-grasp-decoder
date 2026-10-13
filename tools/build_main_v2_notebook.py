"""Builds main_v2.ipynb (KG-GT v2). Run from the repo root:  env/Scripts/python.exe tools/build_main_v2_notebook.py"""
import json
from pathlib import Path

TITLE = """# KG-GT v2: dual-stream token model

Three reported outputs:

- **K (kinematics-led)**: kinematic stream only. Judged on accuracy.
- **E (EEG only)**: EEG stream only. Judged against shuffled-EEG and swapped-EEG nulls.
- **F (fused)**: K held fixed plus a gated EEG correction. The gate starts at zero.

Everything runs on 50 Hz tokens with 8 s windows; predictions are interpolated back to 500 Hz and scored
against the same target and test series as `main_1stage.ipynb`. Outputs go to `outputs/checkpoints_v2/`,
`results/v2/` and `outputs/interpretability_v2/`; the baseline's files are only read.

**Acceptance criteria**

1. K: mean VAF above 75.5 % and mean r above 0.869 (the kinematics-only reference).
2. Magnitude: light- and middle-weight height ratios within 0.95-1.05, per-lift correlation at a fixed weight
   above the baseline's, rest level no worse.
3. E: EEG-only r above both nulls; 0.47 is the number to beat.
4. F: if F does not beat K against the null, the result is "EEG adds nothing beyond kinematics". That is an accepted outcome.
5. Budget: the V2-BUDGET cell passes and prints the projected run time before training.

**Interpretability (cell V2-INTERP)**

The model mixes information between time steps, electrodes, muscles and streams only through recorded
attention layers (plus a 5-tap input convolution with a known footprint), so every output value can be
traced back through all layers. For a window centred on each lift event of the test series the cell gives:

- **attention rollout** through every layer (the same for all muscles),
- **attention flow** (maximum flow through the same layers),
- **gradient-weighted rollout**, specific to one muscle and one time point,
- the attention of every single layer and head, input shares from gradient x input (kinematic features,
  history features, EEG bands), and the exact parts of each output (K = K_base x gain, F = relu(K + EEG correction)),
- a contrast against rest windows with a permutation test,
- two checks of the maps themselves: a **deletion** check (does removing the top-ranked inputs move the output
  more than removing random ones?) and a **weight-shuffle** check (does the map change when the learned weights
  are destroyed?).

**How to read the results**

- **Mixed precision** is chosen by measurement: the V2-BUDGET cell times a training step with mixed precision on
  and off and keeps the faster setting that fits in memory. On some GPUs mixed precision is the slower one.
- **The F verdict is one test over all muscles** (the mean over muscles against the swapped-EEG null). The
  per-muscle p-values are descriptive. Beating the nulls shows that the EEG is time-locked to the task;
  artefacts that are locked to the lift would pass the same test, so it does not prove a cortical origin.
- **Rollout and flow** show where information could have come from, and they are the same for all muscles of a
  stream. The gradient-weighted maps and gradient x input are specific to one muscle. The deletion and
  weight-shuffle checks say which view to trust.
"""

SETUP = """# V2-SETUP
import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, display

from main.dataloader import emg_channel_names, describe_split
from main.preprocessing_emg_kin import compute_muscle_edge_prior, kin_feature_names
from v2 import data as D
from v2.config import CONFIG_V2
from v2.evaluate import baseline_comparison, evaluate_subject
from v2.interp.report import interpret_subject
from v2.model import build_model, count_parameters
from v2.smoke import project_runtime
from v2.train import fit, weighted_stream_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""

CONFIG = """# V2-CONFIG
CONFIG = copy.deepcopy(CONFIG_V2)
subject_list = [1,2,3,4,5,6,7,8,9,10,11,12]                      # participants to run; each needs its cache built by main_1stage.ipynb
RESULTS = Path(CONFIG["outputs"]["results_dir"]); RESULTS.mkdir(parents=True, exist_ok=True)
CKPT = Path(CONFIG["outputs"]["checkpoint_dir"])
print(json.dumps({k: CONFIG[k] for k in ("tokens", "model", "loss", "training", "interp")}, indent=1))
"""

BUDGET = """# V2-BUDGET
# A few forward/backward steps on RANDOM data: peak GPU memory, seconds per step, projected time for max_epochs.
# Trains nothing real and saves nothing. Needs the GPU free. Adjusts batch size / accumulation if memory is tight.
# On the GPU it measures mixed precision on and off and keeps the faster setting that fits in memory.
budget = project_runtime(CONFIG, device, n_windows=1100)
CONFIG["training"]["batch_size"] = budget["batch_size"]
CONFIG["training"]["gradient_accumulation_steps"] = budget["gradient_accumulation_steps"]
CONFIG["training"]["use_amp"] = budget["use_amp"]
print("measured settings (empty = out of memory):")
print(pd.DataFrame(budget["candidates"]).round(3).to_string(index=False, na_rep=""))
print(f"chosen: mixed precision {'on' if budget['use_amp'] else 'off'}, batch size {budget['batch_size']}, "
      f"accumulation {budget['gradient_accumulation_steps']}, peak memory {budget['peak_gb']} of {budget['total_gb']} GB, "
      f"{budget['parameters']:,} parameters")
print(f"projected time for {CONFIG['training']['max_epochs']} epochs: {budget['projected_minutes']:.1f} min")
"""

DATA = """# V2-DATA
subject_data = {}
for p_id in subject_list:
    t0 = time.time(); subj = f"P{p_id}"
    cache = CONFIG["data"]["cache_dir"]
    tok = {s: [D.build_tokens(r, CONFIG) for r in D.load_split(cache, p_id, s)] for s in ("train", "val", "test")}
    norm = D.fit_norm(tok["train"])
    prior = compute_muscle_edge_prior([t["emg_full"] for t in tok["train"]])
    tc = CONFIG["tokens"]
    subject_data[subj] = dict(
        p_id=p_id, tokens=tok, norm=norm, muscle_prior=prior, names=emg_channel_names(p_id),
        train_ds=D.TokenWindows(tok["train"], norm, tc["window"], tc["stride"], random_offset=True, seed=CONFIG["training"]["seed"]),
        val_ds=D.TokenWindows(tok["val"], norm, tc["window"], tc["stride"]),
    )
    sd = subject_data[subj]
    print(f"{subj}: train windows {len(sd['train_ds'])}, val windows {len(sd['val_ds'])}, "
          f"holds per train series {[int(t['hold_id'].max()) for t in tok['train']]}, {time.time() - t0:.0f} s")
    print(describe_split(p_id))
"""

TRAIN = """# V2-TRAIN
results = {}
for subj, sd in subject_data.items():
    t0 = time.time()
    torch.manual_seed(CONFIG["training"]["seed"])
    model = build_model(CONFIG, sd["muscle_prior"]).to(device)
    print(f"\\n=== {subj}: {count_parameters(model):,} parameters ===")
    history = fit(model, sd["train_ds"], sd["val_ds"], CONFIG, device, CKPT / subj)
    ep_k, ep_e = history["best_epoch_kin"], history["best_epoch_eeg"]
    print(f"{subj}: {history['epochs_run']} epochs, best K epoch {ep_k}, best E epoch {ep_e}, {(time.time() - t0) / 60:.1f} min")
    # E and the fusion are saved at the epoch with the lowest smoothed E + F loss, which need not be E's own best epoch
    lc = CONFIG["loss"]
    e_alone = [weighted_stream_loss(c, "E", lc) for c in history["components"]]
    ep_alone = int(np.argmin(e_alone)) + 1
    print(f"{subj}: validation loss of E alone is lowest at epoch {ep_alone} ({e_alone[ep_alone - 1]:.4f}); "
          f"E is saved from epoch {ep_e} ({e_alone[ep_e - 1] if ep_e else float('nan'):.4f}), K from epoch {ep_k}")
    vc = history["val_combined"]                     # the saved model joins each stream's own best epoch
    l_k, l_f = weighted_stream_loss(vc["components"], "K", lc), weighted_stream_loss(vc["components"], "F", lc)
    differ = ep_k != ep_e
    fusion_checked = not (differ and l_f > l_k)      # False: this K and this fusion never met in training, and F lost
    print(f"{subj}: combined best model, validation loss E + F {vc['eeg']:.4f}, K alone {l_k:.4f}, F alone {l_f:.4f} "
          f"-> F can be interpreted: "
          + ("yes" if fusion_checked else "NO (fusion and K come from different epochs and F is worse than K on validation)"))
    results[subj] = dict(model=model, history=history, fusion_checked=fusion_checked)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(history["val_kin"], color="#eb6834"); ax[0].set_title(f"{subj}: validation loss, K"); ax[0].set_xlabel("epoch")
    ax[1].plot(history["val_eeg"], color="#2a78d6"); ax[1].set_title(f"{subj}: validation loss, E + F"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(RESULTS / f"history_{subj}.png", dpi=130); plt.show()
"""

EVAL = """# V2-EVAL
tables = {k: [] for k in ("standard", "magnitude", "by_load", "ceiling", "nulls", "history")}
verdicts, n_pass = [], {"E": 0, "F": 0}
for subj, sd in subject_data.items():
    res = evaluate_subject(results[subj]["model"], sd["tokens"]["test"], sd["norm"], CONFIG, device, sd["p_id"], sd["names"])
    results[subj]["eval"] = res
    for k in tables:
        tables[k].append(res[k].assign(subject=subj))
    std = res["standard"]
    print(f"\\n=== {subj} ===")
    print(std.pivot(index="muscle", columns="output", values="vaf").loc[sd["names"], ["K", "E", "F"]].round(1).to_string())
    print("mean r:", std.groupby("output").r.mean().round(3).to_dict(), "| mean VAF:", std.groupby("output").vaf.mean().round(1).to_dict())
    print(res["ceiling"].round(1).to_string(index=False))
    bl = res["by_load"]                              # acceptance criterion 2 for K: one row per test series and weight
    bl = bl[(bl["output"] == "K") & (bl["grouping"] == "CurW")].assign(value=lambda d: d["value"].astype(str))
    print("criterion 2, K: height ratio in the hold phase per test series and weight (value 1 = light, 2 = middle, 4 = heavy; "
          "0.95-1.05 wanted for light and middle):")
    print(bl.pivot(index=["series", "value", "n_lifts"], columns="muscle", values="height_ratio")[sd["names"]].round(3).to_string())
    print("criterion 2, K: correlation over lifts of the predicted and the real hold level at one weight (NaN below 4 lifts):")
    print(bl.pivot(index=["series", "value", "n_lifts"], columns="muscle", values="r_lift")[sd["names"]].round(3).to_string())
    n = res["nulls"]
    print("per muscle against the nulls (descriptive; the verdict below is one test over all muscles):")
    print(n[["muscle", "E_r", "E_r_shuffle_mean", "E_r_swap_mean", "E_p_swap", "dF_vaf", "dF_vaf_swap_mean", "dF_p_swap", "gate"]].round(3).to_string(index=False))
    v = res["verdict"]                               # mean over muscles against the per-draw means of the swap null
    f_beats_k = bool(v["dF_vaf_mean"] > 0 and np.isfinite(v["dF_p_swap"]) and v["dF_p_swap"] <= 0.05)
    f_verdict = "EEG adds information beyond kinematics" if f_beats_k else "EEG adds nothing beyond kinematics"
    fusion_checked = results[subj]["fusion_checked"]         # set by V2-TRAIN
    if not fusion_checked:
        f_verdict = "not interpretable (fusion and K come from different epochs and F is worse than K on validation)"
    n_pass["E"] += bool(np.isfinite(v["E_p_swap"]) and v["E_p_swap"] <= 0.05)
    n_pass["F"] += bool(f_beats_k and fusion_checked)
    print(f"E verdict: mean r {v['E_r_mean']:.3f}, p against the swap null {v['E_p_swap']:.3f}")
    print(f"F verdict: mean VAF gain over K {v['dF_vaf_mean']:.2f}, p against the swap null {v['dF_p_swap']:.3f} -> {f_verdict}")
    print(f"  ({v['n_null']} null draws, {v['n_distinct']} distinct, {v['n_blocks']} blocks; no p below 1 / blocks is possible)")
    verdicts.append({"subject": subj, **v, "F_verdict": f_verdict, "fusion_checked": fusion_checked})
    print("history check (K with the history of another test series, when there is one):")
    print(res["history"].round(1).to_string(index=False))
for k, parts in tables.items():
    pd.concat(parts, ignore_index=True).to_csv(RESULTS / f"{k}.csv", index=False)
pd.DataFrame(verdicts).to_csv(RESULTS / "verdict.csv", index=False)
print(f"\\n{n_pass['F']} of {len(verdicts)} participants pass the F verdict, {n_pass['E']} of {len(verdicts)} pass E "
      f"(p against the swap null <= 0.05). Without any effect about 5 % pass by chance.")
print("saved tables to", RESULTS)
"""

COMPARE = """# V2-COMPARE
rows = []
for subj in subject_data:
    cmp_ = baseline_comparison(results[subj]["eval"]["standard"], subj, "results/magnitude_report_1stage.csv")
    if cmp_ is None:
        print(f"{subj}: no baseline row found"); continue
    rows.append(cmp_.assign(subject=subj))
    print(f"\\n{subj}: baseline vs v2 K"); print(cmp_.round(3).to_string(index=False))
if rows:
    allc = pd.concat(rows, ignore_index=True)
    allc.to_csv(RESULTS / "baseline_comparison.csv", index=False)
    print("\\nmean over participants and muscles: baseline VAF %.1f, v2 K VAF %.1f, baseline r %.3f, v2 K r %.3f"
          % (allc.baseline_vaf.mean(), allc.v2_K_vaf.mean(), allc.baseline_r.mean(), allc.v2_K_r.mean()))
    print(f"criterion 1 (K: mean VAF > 75.5 and mean r > 0.869), computed for {allc.subject.nunique()} of {len(subject_data)} "
          f"participants run here; the references 75.5 / 0.869 are means over all 12 participants:",
          bool(allc.v2_K_vaf.mean() > 75.5 and allc.v2_K_r.mean() > 0.869))
"""

INTERP = """# V2-INTERP
# Where each output comes from, through every layer (see the Interpretability section of the first cell).
# One 8 s window is centred on every lift event of the test series; events are CONFIG["interp"]["events"].
# Reads the trained models of V2-TRAIN; writes tables and figures under CONFIG["outputs"]["interpretability_dir"].
INTERP_DIR = Path(CONFIG["outputs"]["interpretability_dir"])
kin_names = kin_feature_names(include_velocity=True, drop_indices=[12, 25], gated_force=True)
for subj, sd in subject_data.items():
    t0 = time.time()
    rep = interpret_subject(results[subj]["model"], sd["tokens"]["test"], sd["norm"], CONFIG, device, sd["p_id"],
                            sd["names"], out_dir=INTERP_DIR / subj, kin_names=kin_names)
    results[subj]["interp"] = rep
    tb = rep["tables"]
    print(f"\\n=== {subj}: {len(rep.get('files', []))} figures and {len(tb)} tables in {INTERP_DIR / subj}, "
          f"{(time.time() - t0) / 60:.1f} min ===")
    if len(tb["parts"]):
        print("exact parts of the outputs at the event token (mean over lifts):")
        print(tb["parts"].round(3).to_string(index=False))
    if len(tb["faithfulness"]):
        f = tb["faithfulness"].assign(gain=lambda d: d["top"] - d["random"])
        print("deletion check: mean (top - random) change of the output. Above 0 = the view points at inputs the model uses.")
        print(f.groupby(["output", "view"])["gain"].mean().unstack().round(4).to_string())
        print("weight-shuffle check: Spearman between the maps of the trained and the shuffled model. Near 1 = the view ignores what was learned.")
        s = tb["sanity"]                             # spearman: mean over the events that have a map; n_nan = events without one
        print(s.groupby(["output", "view"])["spearman"].mean().unstack().round(2).to_string())
        if int(s["n_nan"].sum()):
            print(f"  {int(s['n_nan'].sum())} of {int(s['n'].sum())} map comparisons had no map (NaN), for example a closed fusion gate:")
            print(s[s["n_nan"] > 0].to_string(index=False))
    if len(tb["rest_contrast"]):
        rc = tb["rest_contrast"]                     # paired test: each lift against the rest after it
        testable = np.isfinite(rc["p"])              # p is NaN for an electrode that cannot be tested
        n_sig = int((testable & (rc["p"] < 0.05)).sum())
        print(f"rest contrast (E), {int(rc['n_pairs'].min())}-{int(rc['n_pairs'].max())} lift-rest pairs: {n_sig} of {int(testable.sum())} "
              f"testable muscle-electrode cells differ from rest (corrected p < 0.05); {int((~testable).sum())} not testable")
    event = CONFIG["interp"]["events"][0]
    for name in (f"time_K_{event}", f"time_E_{event}", f"scalp_E_{event}", f"muscle_graph_{event}", "deletion"):
        png = INTERP_DIR / subj / f"{name}.png"
        if png.exists():
            display(Image(filename=str(png)))
"""

TRIPTYCH = """# V2-TRIPTYCH
# Triptych per test series, like the interpretability figure of main_1stage: 10 s stretches that lead into a lift.
#   Panel 1  actual EMG envelope against K, E and F
#   Panel 2  exact parts of the outputs: K_base, K = K_base x gain, and the EEG correction F_eeg inside F
#   Panel 3  muscle-graph attention of the kinematic stream, change from its rest mean
# Reads the trained models of V2-TRAIN; writes PNG files under CONFIG["outputs"]["interpretability_dir"].
from main.interp import lift_events
from v2.interp import plots as P
from v2.interp.triptych import plot_triptych, series_internals, stretch_starts

TRIPTYCH_STRETCHES, TRIPTYCH_SECONDS = 2, 10.0          # stretches per test series, length of a stretch
INTERP_DIR = Path(CONFIG["outputs"]["interpretability_dir"])
tc = CONFIG["tokens"]
for subj, sd in subject_data.items():
    out_dir = INTERP_DIR / subj
    out_dir.mkdir(parents=True, exist_ok=True)
    shown = False
    for tok in sd["tokens"]["test"]:
        internals = series_internals(results[subj]["model"], tok, sd["norm"], device, tc["window"], tc["hop"],
                                     clip=CONFIG["eeg"]["clip"], residual=CONFIG["interp"]["residual"])
        events = lift_events(sd["p_id"], tok["series"], fs=CONFIG["data"]["fs"], n_samples=tok["n_samples"])
        for start in stretch_starts(tok, TRIPTYCH_STRETCHES, TRIPTYCH_SECONDS):
            fig = plot_triptych(internals, tok, sd["norm"], sd["names"], start, TRIPTYCH_SECONDS, events=events,
                                title=f"{subj}, test series {tok['series']}, from {start / 50:.0f} s")
            png = out_dir / f"triptych_S{tok['series']}_{start}.png"
            with P.style():
                fig.savefig(png, dpi=130)
            plt.close(fig)
            if not shown:                               # one figure per participant in the notebook; all are on disk
                display(Image(filename=str(png)))
                shown = True
    print(f"{subj}: triptychs saved to {out_dir}")
"""


def _cell(kind, src):
    lines = src.rstrip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def build() -> dict:
    code = (SETUP, CONFIG, BUDGET, DATA, TRAIN, EVAL, COMPARE, INTERP, TRIPTYCH)
    cells = [_cell("markdown", TITLE)] + [_cell("code", s) for s in code]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    Path("main_v2.ipynb").write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote main_v2.ipynb")
