"""The whole v3 analysis as numbered steps, shared by main_v3.ipynb and the command line.

  step 0  setup               config, participants, seeds, scales, preset, folders
  step 1  data                tokens, task events and splits of every participant (no training), cached per
                              participant in step1_data/cache/ and rebuilt only when the config or v2 cache changes
  step 2  timing_baselines    causal_cue, events_causal (4 horizons), phase and cue (no training)
  step 3  b_vs_f              K, B, F at 0, 200, 500, 1000 ms from the v2 horizon checkpoints, with the EEG controls
                              and onset timing (no training): the primary result
  step 4  budget              GPU hours of steps 5 and 6 (random data)
  step 5  train_E0            the EEG-only student without the guide (guide.scale 0), every seed
  step 6  train_Eguided       the EEG-only student with the K teacher, every guided scale and seed
  step 7  evaluate_students   E0 and Eguided: metrics at 500 and 50 Hz, onset, swaps, residual over timing, templates
  step 8  verdict             the proposal's interpretation, one row per horizon
Every step prints one header line and the files it wrote, and writes only into results/v3/step{n}_{name}/ (or under the
preset's folder). Steps 2 and 3 need no training and take minutes; run them first. Run from the repository root:
    env/Scripts/python.exe -m v3.run --step b_vs_f --participants all
    env/Scripts/python.exe -m v3.run --step all --dry-run          # what would run and where it would write
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG_V3, preset_config
from .conditions import CONDITIONS

STEPS = [
    ("setup", "config, participants, seeds, scales and folders", False),
    ("data", "tokens, task events and splits of every participant", False),
    ("timing_baselines", "causal_cue and events_causal at every horizon, phase and cue", False),
    ("b_vs_f", "K, B and F at every horizon from the v2 horizon checkpoints, with EEG controls and onset timing", False),
    ("budget", "GPU hours of the two training steps, on random data", False),
    ("train_E0", "EEG-only student without the guide (guide.scale 0), every seed", True),
    ("train_Eguided", "EEG-only student with the K teacher, every guided scale and seed", True),
    ("evaluate_students", "E0 and Eguided: metrics, onset, swaps, residual over timing, templates", False),
    ("verdict", "the proposal's interpretation, one row per horizon", False),
]
STEP_NAMES = [s[0] for s in STEPS]
OUTPUTS = {
    "data": ["cache/P{p}.pkl"],
    "timing_baselines": ["timing_baselines.csv", "cache/"],
    "b_vs_f": ["b_vs_f.csv", "b_vs_f_summary.csv", "onset.csv", "onset_summary.csv", "b_vs_f.png", "runs/"],
    "train_E0": ["outputs/checkpoints_v3/E0/s{seed}/P{p}/"],
    "train_Eguided": ["outputs/checkpoints_v3/Eguided_g{scale}/s{seed}/P{p}/"],
    "evaluate_students": ["standard.csv", "nulls.csv", "onset.csv", "swap.csv", "phase_swap.csv",
                          "residual_over_timing.csv", "template_deviation.csv", "template_deviation_summary.csv"],
    "verdict": ["verdict.csv", "eguided_vs_e0.csv", "student_means.csv"],
}
V2_STANDARD = "results/v2/standard.csv"
PHASE_METRICS = "eeg_controls/results/phase/phase_metrics.csv"


class Context:
    """Everything the steps share: config, selections, folders, the loaded data and what the steps produced."""

    def __init__(self, participants="all", seeds=None, scales=None, preset=None, dry_run=False, device=None,
                 root=".", log=print, v2_standard=V2_STANDARD, phase_metrics=PHASE_METRICS, rebuild_data=False):
        from eeg_controls.common import parse_subjects
        self.cfg = copy.deepcopy(CONFIG_V3) if preset is None else preset_config(preset)
        self.preset, self.dry_run, self.log, self.rebuild_data = preset, bool(dry_run), log, bool(rebuild_data)
        self.participants = parse_subjects(participants)
        self.seeds = list(self.cfg["training"]["seeds"] if seeds is None else seeds)
        all_scales = self.cfg["guide"]["scales"] if scales is None else scales
        self.guided_scales = [float(s) for s in all_scales if float(s) > 0]
        root = Path(root)
        self.results = root / self.cfg["outputs"]["results_dir"] / (preset or "")
        self.checkpoints = root / self.cfg["outputs"]["checkpoint_dir"] / (preset or "")
        self.v2_standard, self.phase_metrics = root / v2_standard, root / phase_metrics
        self.root = root
        self._device = device
        self.data, self.timing, self.runs, self.written = None, {}, {}, []

    @property
    def device(self):
        import torch
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def folder(self, name: str) -> Path:
        return self.results / f"step{STEP_NAMES.index(name)}_{name}"

    def save(self, df, name: str, filename: str, **kw) -> Path:
        path = self.folder(name) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kw)
        self.written.append(path)
        return path

    def save_figure(self, fig, name: str, filename: str) -> Path:
        path = self.folder(name) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        self.written.append(path)
        return path


def _header(ctx: Context, name: str) -> bool:
    """Print the step's header; in a dry run also its outputs, and return False so the step stops there."""
    i = STEP_NAMES.index(name)
    _, text, trains = STEPS[i]
    ctx.log(f"Step {i}/{len(STEPS) - 1} {name}: {text} ({'trains' if trains else 'no training'})")
    if ctx.dry_run:
        for out in OUTPUTS.get(name, []):
            where = out if out.startswith("outputs/") else str(ctx.folder(name) / out)
            ctx.log(f"    would write {where}")
        return False
    ctx.written = []
    return True


def _footer(ctx: Context, t0: float):
    for p in ctx.written:
        ctx.log(f"    wrote {p}")
    ctx.log(f"    done in {(time.time() - t0) / 60:.1f} min")


def student_dir(root, scale: float, seed: int, subject: str) -> Path:
    """Checkpoint folder of one student run: root/E0/s{seed}/P{p} or root/Eguided_g{scale}/s{seed}/P{p}.

    Folders of the earlier layouts (g{scale}/s{seed}/P{p}, and g{scale}/P{p} as seed 0) are used when the new one does
    not exist and the old one holds a run.
    """
    from .train import run_dir
    name = "E0" if float(scale) == 0.0 else f"Eguided_g{scale}"
    new = Path(root) / name / f"s{int(seed)}" / subject
    if not new.exists():
        old = run_dir(root, scale, seed, subject)
        if any((old / f).exists() for f in ("best.pt", "last.pt")):
            return old
    return new


def condition_of(scale: float) -> str:
    return "E0" if float(scale) == 0.0 else "Eguided"


# ---- steps ---------------------------------------------------------------------------------------------------------

def step_setup(ctx: Context) -> dict:
    if not _header(ctx, "setup"):
        return {}
    ctx.log(f"    participants {ctx.participants}, seeds {ctx.seeds}, guided scales {ctx.guided_scales}, "
            f"preset {ctx.preset}, results {ctx.results}, checkpoints {ctx.checkpoints}")
    for name, c in CONDITIONS.items():
        ctx.log(f"    {name:>10}: {c['inputs']} ({c['role']})")
    return {}


def step_data(ctx: Context) -> dict:
    if not _header(ctx, "data"):
        return {}
    t0 = time.time()
    out = _load_data(ctx)
    _footer(ctx, t0)
    return out


DATA_CACHE_BLOCKS = ("data", "tokens", "eeg", "history")       # the config blocks the tokens depend on


def _build_participant(ctx: Context, p: int) -> dict:
    """The slow part of step 1 for one participant: tokens (band power, decimation), task events, norm, prior, names."""
    from eeg_controls.common import lifts
    from main.dataloader import emg_channel_names
    from main.preprocessing_emg_kin import compute_muscle_edge_prior
    from v2 import data as D
    cfg = ctx.cfg
    tok = {s: [D.build_tokens(r, cfg) for r in D.load_split(ctx.root / cfg["data"]["cache_dir"], p, s)]
           for s in ("train", "val", "test")}
    for s in tok:
        for t in tok[s]:
            t["lifts"] = lifts(p, t["series"], t["n_samples"])
    return {"tokens": tok, "norm": D.fit_norm(tok["train"]), "names": emg_channel_names(p),
            "muscle_prior": compute_muscle_edge_prior([t["emg_full"] for t in tok["train"]])}


def _data_key(ctx: Context, p: int) -> str:
    """What the cached data depends on: the token config and the size and time of the participant's v2 cache files."""
    import json
    files = sorted((ctx.root / ctx.cfg["data"]["cache_dir"]).glob(f"P{int(p)}_*.npz"))
    stats = [(f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in files]
    return json.dumps({"config": {b: ctx.cfg.get(b) for b in DATA_CACHE_BLOCKS}, "files": stats}, sort_keys=True,
                      default=str)


def data_cache_path(ctx: Context, p: int) -> Path:
    return ctx.folder("data") / "cache" / f"P{int(p)}.pkl"


def _participant_data(ctx: Context, p: int) -> dict:
    """_build_participant, read from results/v3/step1_data/cache/P{p}.pkl when its key still matches, else rebuilt
    and saved. The key (config blocks and the v2 cache files' size and time) changes when anything the tokens depend on
    changes, so a stale cache is never used. Context(rebuild_data=True) forces a rebuild."""
    import pickle
    path, key = data_cache_path(ctx, p), _data_key(ctx, p)
    if path.exists() and not ctx.rebuild_data:
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == key:
                ctx.log(f"    P{p}: loaded from the cache {path}")
                return cached["data"]
            ctx.log(f"    P{p}: the cache is out of date (config or v2 cache files changed): rebuilding")
        except Exception as err:                                     # a broken or partly written file
            ctx.log(f"    P{p}: the cache could not be read ({err}): rebuilding")
    t0 = time.time()
    data = _build_participant(ctx, p)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"key": key, "data": data}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)                                                # never leave a half-written cache behind
    ctx.written.append(path)
    ctx.log(f"    P{p}: built in {(time.time() - t0) / 60:.1f} min and cached")
    return data


def _load_data(ctx: Context) -> dict:
    from v2 import data as D
    cfg, out = ctx.cfg, {}
    for p in ctx.participants:
        d = _participant_data(ctx, p)
        tok, norm, tc = d["tokens"], d["norm"], cfg["tokens"]
        out[f"P{p}"] = dict(
            p_id=p, tokens=tok, norm=norm, names=d["names"], muscle_prior=d["muscle_prior"],
            split={s: [t["series"] for t in tok[s]] for s in tok}, lifts={s: [t["lifts"] for t in tok[s]] for s in tok},
            train_ds=D.TokenWindows(tok["train"], norm, tc["window"], tc["stride"], random_offset=True,
                                    seed=cfg["training"]["seed"]),
            val_ds=D.TokenWindows(tok["val"], norm, tc["window"], tc["stride"]))
        ctx.log(f"    P{p}: {len(tok['train'])} train, {len(tok['val'])} validation, {len(tok['test'])} test series, "
                f"test lifts {[len(l) for l in out[f'P{p}']['lifts']['test']]}")
    ctx.data = out
    return out


def _need_data(ctx: Context):
    """The data of step 1, loaded once per Context (a later step loads it quietly when step 1 was not run here)."""
    if ctx.data is None:
        ctx.log("    loading the data of step 1")
        _load_data(ctx)


def step_timing_baselines(ctx: Context) -> dict:
    if not _header(ctx, "timing_baselines"):
        return {}
    from .timing import CAUSAL, cached_timing, score_timing
    _need_data(ctx)
    t0, tcfg, rows = time.time(), ctx.cfg["timing"], []
    for subj, sd in ctx.data.items():
        for variant in tcfg["variants"]:
            for h in (tcfg["horizons_ms"] if variant in CAUSAL else [0]):
                fitted = cached_timing(ctx.folder("timing_baselines") / "cache", sd["p_id"], sd["tokens"], variant, h)
                ctx.timing[(subj, variant, h)] = fitted
                rows.append(score_timing(fitted, sd["tokens"]["test"], sd["names"]).assign(subject=subj))
    table = pd.concat(rows, ignore_index=True)
    ctx.save(table, "timing_baselines", "timing_baselines.csv")
    ctx.log("    mean r (columns: scoring rate in Hz):\n" +
            table.groupby(["variant", "horizon_ms", "rate"]).r.mean().unstack("rate").round(3).to_string())
    _footer(ctx, t0)
    return {"table": table}


def _horizon_scores(p: int, h: int, ckpt_root: Path, n_null: int, sources) -> tuple:
    """B vs F scores and onset of one participant and horizon, from one load of the causal v2 horizon model."""
    import torch
    from eeg_controls.run_horizon import horizon_config, subject_data
    from v2.model import build_model
    from v2.train import load_best
    from .contrast import onset_b_vs_f, score_b_vs_f
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = horizon_config(h)
    sd = subject_data(p, cfg)
    model = build_model(cfg, sd["prior"]).to(device)
    load_best(model, ckpt_root / f"h{h}ms" / f"P{p}", device)
    return (score_b_vs_f(model, sd, cfg, device, p, h, n_null=n_null, sources=sources),
            onset_b_vs_f(model, sd, cfg, device, p, h))


def step_b_vs_f(ctx: Context) -> dict:
    if not _header(ctx, "b_vs_f"):
        return {}
    import matplotlib.pyplot as plt
    from .onset import onset_summary
    from .contrast import has_sources, summarise_b_vs_f
    t0, bc = time.time(), ctx.cfg["b_vs_f"]
    ckpt, runs = ctx.root / bc["checkpoint_dir"], ctx.folder("b_vs_f") / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    scores, onsets = [], []
    for h in bc["horizons_ms"]:
        for p in ctx.participants:
            f_s, f_o = runs / f"h{h}ms_P{p}.csv", runs / f"onset_h{h}ms_P{p}.csv"
            if f_s.exists() and f_o.exists() and has_sources(pd.read_csv(f_s), bc["sources"]):
                s, o = pd.read_csv(f_s), pd.read_csv(f_o)
            elif not (ckpt / f"h{h}ms" / f"P{p}" / "best.pt").exists():
                ctx.log(f"    P{p} h={h} ms: no horizon checkpoint at {ckpt / f'h{h}ms' / f'P{p}'}, skipped")
                continue
            else:
                try:
                    s, o = _horizon_scores(p, h, ckpt, int(ctx.cfg["evaluation"]["n_null"]), tuple(bc["sources"]))
                except ValueError as err:                       # e.g. fewer than 2 trials for a swap
                    ctx.log(f"    P{p} h={h} ms: skipped ({err})")
                    continue
                s.to_csv(f_s, index=False); o.to_csv(f_o, index=False)
            scores.append(s); onsets.append(o)
            a = s[(s.segment == "all") & (s.source == "eeg")]
            if len(a):
                a = a.iloc[0]
                ctx.log(f"    P{p} h={h} ms: eeg_skill {a.eeg_skill:+.4f} (p {a.p:.3f}), VAF gain {a.vaf_gain:+.2f}")
    if not scores:
        ctx.log("    no horizon checkpoint found: nothing to summarise")
        return {}
    table, onset = pd.concat(scores, ignore_index=True), pd.concat(onsets, ignore_index=True)
    summary = summarise_b_vs_f(table)
    ctx.save(table, "b_vs_f", "b_vs_f.csv"); ctx.save(summary, "b_vs_f", "b_vs_f_summary.csv")
    ctx.save(onset, "b_vs_f", "onset.csv")
    ctx.save(onset_summary(onset, by=("horizon_ms", "condition", "muscle")), "b_vs_f", "onset_summary.csv")
    allrows = summary[summary.segment == "all"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for src, d in allrows.groupby("source"):
        ax.errorbar(d.horizon_ms, d.eeg_skill_mean, yerr=d.eeg_skill_sem, marker="o", capsize=3, label=src)
    ax.axhline(0, color="gray", lw=1); ax.set_xlabel("horizon (ms)"); ax.set_ylabel("eeg_skill = 1 - MSE(F) / MSE(B)")
    ax.legend(fontsize=8); fig.tight_layout()
    ctx.save_figure(fig, "b_vs_f", "b_vs_f.png"); plt.close(fig)
    ctx.log(allrows[["horizon_ms", "source", "n_subjects", "eeg_skill_mean", "p_eeg_skill_holm_horizons",
                     "vaf_gain_mean"]].round(4).to_string(index=False))
    _footer(ctx, t0)
    return {"table": table, "summary": summary, "onset": onset}


def step_budget(ctx: Context) -> dict:
    if not _header(ctx, "budget"):
        return {}
    from .smoke import run_budget
    t0 = time.time()
    b = run_budget(ctx.cfg, ctx.device, [0.0] + ctx.guided_scales, ctx.seeds, len(ctx.participants))
    chosen = b["settings"].get("guided", b["settings"].get("control"))
    for key in ("batch_size", "gradient_accumulation_steps", "use_amp"):
        ctx.cfg["training"][key] = chosen[key]
    ctx.log(f"    {b['n_runs']} runs, at most {b['total_gpu_hours']:.1f} hours on {ctx.device} at "
            f"{b['max_epochs']} epochs per run (guided runs last at least {b['min_epochs_guided']} epochs)")
    if ctx.device.type == "cpu":
        ctx.log("    measured on the CPU: CPU hours, not a GPU estimate")
    ctx.log("    not included: step 7, which runs many full inference passes per student (nulls, swaps, templates)")
    _footer(ctx, t0)
    return b


def _train(ctx: Context, name: str, scales) -> dict:
    if not _header(ctx, name):
        return {}
    import torch
    from .guide import build_heads
    from .model import build_student
    from .teacher import check_teacher_split, load_teacher, teacher_path
    from .train import fit_guided
    _need_data(ctx)
    t0 = time.time()
    for subj, sd in ctx.data.items():
        teacher = None
        if any(s > 0 for s in scales):
            ck = teacher_path(ctx.cfg, sd["p_id"])
            if not ck.is_file():
                ctx.log(f"    {subj}: no v2 K checkpoint at {ck}, skipped")
                continue
            check_teacher_split(ck, sd["p_id"], sd["split"])
            teacher = load_teacher(ctx.cfg, sd["muscle_prior"], ck, ctx.device)
        for seed in ctx.seeds:
            for scale in scales:
                cfg = copy.deepcopy(ctx.cfg)
                cfg["guide"]["scale"], cfg["training"]["seed"] = scale, seed
                torch.manual_seed(seed)
                student, heads = build_student(cfg, sd["muscle_prior"]).to(ctx.device), build_heads(cfg)
                folder = student_dir(ctx.checkpoints, scale, seed, subj)
                hist = fit_guided(student, teacher if scale > 0 else None, heads, sd["train_ds"], sd["val_ds"], cfg,
                                  ctx.device, folder, log=lambda *_: None)
                ctx.runs[(condition_of(scale), scale, seed, subj)] = dict(student=student, history=hist, cfg=cfg)
                ctx.log(f"    {subj} {condition_of(scale)} scale {scale} seed {seed}: {hist['epochs_run']} epochs, "
                        f"best epoch {hist['best_epoch']} -> {folder}")
    _footer(ctx, t0)
    return {"runs": len(ctx.runs)}


def step_train_E0(ctx: Context) -> dict:
    return _train(ctx, "train_E0", [0.0])


def step_train_Eguided(ctx: Context) -> dict:
    return _train(ctx, "train_Eguided", ctx.guided_scales)


def _load_runs(ctx: Context):
    """Students of steps 5 and 6 that finished in an earlier session: load their best.pt."""
    from .model import build_student
    from .train import load_student
    _need_data(ctx)
    for subj, sd in ctx.data.items():
        for seed in ctx.seeds:
            for scale in [0.0] + ctx.guided_scales:
                key = (condition_of(scale), scale, seed, subj)
                folder = student_dir(ctx.checkpoints, scale, seed, subj)
                if key in ctx.runs or not (folder / "best.pt").exists():
                    continue
                cfg = copy.deepcopy(ctx.cfg)
                cfg["guide"]["scale"], cfg["training"]["seed"] = scale, seed
                student = build_student(cfg, sd["muscle_prior"]).to(ctx.device)
                ctx.runs[key] = dict(student=student, history=load_student(student, folder, ctx.device), cfg=cfg)


def step_evaluate_students(ctx: Context) -> dict:
    if not _header(ctx, "evaluate_students"):
        return {}
    from .evaluate import evaluate_student, residual_over_timing
    from .infer import predict_eeg_series
    from .nulls import evaluate_phase_matched_swap, evaluate_swap
    from .onset import onset_error
    from .templates import fit_emg_templates, student_template_deviation, summarise_template_deviation
    from .timing import cached_timing
    t0 = time.time()
    _load_runs(ctx)
    if not ctx.runs:
        ctx.log("    no trained student found: run steps 5 and 6 first")
        return {}
    tables = {k: [] for k in ("standard", "nulls", "onset", "swap", "phase_swap", "residual_over_timing",
                              "template_deviation")}
    templates = {}
    for (cond, scale, seed, subj), run in sorted(ctx.runs.items(), key=lambda kv: (kv[0][3], kv[0][1], kv[0][2])):
        sd, cfg, model = ctx.data[subj], run["cfg"], run["student"]
        test, lifts_t, dev, names = sd["tokens"]["test"], sd["lifts"]["test"], ctx.device, sd["names"]
        tag = dict(subject=subj, condition=cond, scale=scale, seed=seed)
        res = evaluate_student(model, test, sd["norm"], cfg, dev, names)
        tables["standard"].append(res["standard"].assign(**tag)); tables["nulls"].append(res["nulls"].assign(**tag))
        kw = dict(window=cfg["tokens"]["window"], hop=cfg["tokens"]["hop"], clip=cfg["eeg"]["clip"])
        for tok, lf in zip(test, lifts_t):
            pred = predict_eeg_series(model, tok, sd["norm"], dev, **kw)
            tables["onset"].append(onset_error(pred, tok["emg_full"], lf, 500.0, names=names).assign(**tag))
        for key, fn in (("swap", evaluate_swap), ("phase_swap", evaluate_phase_matched_swap)):
            try:
                tables[key].append(fn(model, test, lifts_t, sd["norm"], cfg, dev, names).assign(**tag))
            except ValueError as err:                           # fewer than 2 test trials: no swap possible
                ctx.log(f"    {subj} {cond} scale {scale} seed {seed}: no {key} ({err})")
        for variant in ctx.cfg["timing"]["variants"]:
            fitted = ctx.timing.get((subj, variant, 0)) or cached_timing(
                ctx.folder("timing_baselines") / "cache", sd["p_id"], sd["tokens"], variant, 0)
            tables["residual_over_timing"].append(residual_over_timing(
                model, test, [t["u"] for t in fitted["test"]], lifts_t, sd["norm"], cfg, dev, names)
                .assign(timing=variant, **tag))
        if subj not in templates:
            templates[subj] = fit_emg_templates(sd["tokens"]["train"], sd["lifts"]["train"], sd["norm"])
        tables["template_deviation"].append(student_template_deviation(
            model, test, lifts_t, templates[subj], sd["norm"], cfg, dev, names).assign(**tag))
        ctx.log(f"    {subj} {cond} scale {scale} seed {seed}: mean r "
                f"{res['standard'][res['standard'].rate == 500].r.mean():.3f}")
    out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items() if v}
    for k, v in out.items():
        ctx.save(v, "evaluate_students", f"{k}.csv")
    ctx.save(summarise_template_deviation(out["template_deviation"]), "evaluate_students",
             "template_deviation_summary.csv")
    _footer(ctx, t0)
    return out


def step_verdict(ctx: Context) -> dict:
    if not _header(ctx, "verdict"):
        return {}
    from .evaluate import subject_means
    from .verdict import eguided_vs_e0, proposal_verdict
    t0 = time.time()
    b_dir, s_dir = ctx.folder("b_vs_f"), ctx.folder("evaluate_students")
    if not (b_dir / "b_vs_f_summary.csv").exists():
        ctx.log("    step 3 (b_vs_f) has no summary yet: run it first")
        return {}
    summary = pd.read_csv(b_dir / "b_vs_f_summary.csv")
    onset = pd.read_csv(b_dir / "onset.csv") if (b_dir / "onset.csv").exists() else None
    student, tmpl = None, None
    if (s_dir / "standard.csv").exists():
        means = subject_means(pd.read_csv(s_dir / "standard.csv"), ctx.v2_standard, ctx.phase_metrics)
        ctx.save(means, "verdict", "student_means.csv")
        if means.scale.nunique() > 1:
            student = eguided_vs_e0(means)
            ctx.save(student, "verdict", "eguided_vs_e0.csv")
    if (s_dir / "template_deviation_summary.csv").exists():
        tmpl = pd.read_csv(s_dir / "template_deviation_summary.csv")
    verdict = proposal_verdict(summary, student, onset, tmpl)
    ctx.save(verdict, "verdict", "verdict.csv")
    for _, v in verdict.iterrows():
        ctx.log(f"    h = {int(v.horizon_ms)} ms: eeg_skill {v.eeg_skill:+.4f} (Holm p {v.p_eeg_skill_holm:.3g}), "
                f"MSE B {v.mse_B:.4f} F {v.mse_F:.4f}, onset error B {v.onset_abs_ms_B:.0f} ms F {v.onset_abs_ms_F:.0f} ms")
        ctx.log(f"      F beats B {v.F_beats_B}, survives the swap {v.F_survives_swap}, not peripheral "
                f"{v.F_not_peripheral}, Eguided beats E0 {v.Eguided_beats_E0} -> {v.interpretation}")
    _footer(ctx, t0)
    return {"verdict": verdict}


def load_table(ctx: Context, step: str, filename: str):
    """A table a step saved (None when the step has not written it yet), so a notebook cell works after a restart."""
    path = ctx.folder(step) / filename
    return pd.read_csv(path) if path.exists() else None


def load_histories(ctx: Context) -> pd.DataFrame:
    """Training history of every E0 and Eguided run found on disk: one row per run and epoch."""
    import json
    rows = []
    for p in ctx.participants:
        for seed in ctx.seeds:
            for scale in [0.0] + ctx.guided_scales:
                f = student_dir(ctx.checkpoints, scale, seed, f"P{p}") / "history.json"
                if not f.exists():
                    continue
                h = json.loads(f.read_text(encoding="utf-8"))
                n = len(h.get("val_e", []))
                rows.append(pd.DataFrame({"subject": f"P{p}", "condition": condition_of(scale), "scale": scale,
                                          "seed": seed, "epoch": np.arange(1, n + 1), "val_e": h["val_e"],
                                          "w": h.get("w", [np.nan] * n), "grad_share": h.get("grad_share", [np.nan] * n),
                                          "best_epoch": h.get("best_epoch", 0)}))
    cols = ["subject", "condition", "scale", "seed", "epoch", "val_e", "w", "grad_share", "best_epoch"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


STEP_FUNCTIONS = {"setup": step_setup, "data": step_data, "timing_baselines": step_timing_baselines,
                  "b_vs_f": step_b_vs_f, "budget": step_budget, "train_E0": step_train_E0,
                  "train_Eguided": step_train_Eguided, "evaluate_students": step_evaluate_students,
                  "verdict": step_verdict}


def run_steps(ctx: Context, steps) -> dict:
    steps = STEP_NAMES if steps in ("all", ["all"]) else list(steps)
    unknown = [s for s in steps if s not in STEP_FUNCTIONS]
    if unknown:
        raise ValueError(f"unknown step(s) {unknown}; the steps are {', '.join(STEP_NAMES)} or all")
    return {s: STEP_FUNCTIONS[s](ctx) for s in steps}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="v3 analysis steps (see the module docstring)")
    ap.add_argument("--step", default="all", help=f"one or more of {', '.join(STEP_NAMES)}, comma separated, or all")
    ap.add_argument("--participants", default="all")
    ap.add_argument("--seeds", default=None, help="comma separated, default config training.seeds")
    ap.add_argument("--scales", default=None, help="comma separated guide scales, default config guide.scales")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the steps and their outputs, run and write nothing")
    ap.add_argument("--rebuild-data", action="store_true", help="rebuild the step 1 data cache even when it is current")
    a = ap.parse_args(argv)
    ctx = Context(participants=a.participants, preset=a.preset, dry_run=a.dry_run, rebuild_data=a.rebuild_data,
                  seeds=None if a.seeds is None else [int(s) for s in a.seeds.split(",")],
                  scales=None if a.scales is None else [float(s) for s in a.scales.split(",")])
    return run_steps(ctx, "all" if a.step == "all" else a.step.split(","))


if __name__ == "__main__":
    main()
