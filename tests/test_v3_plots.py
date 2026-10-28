"""Every figure of the v3 notebook draws from tables shaped like the saved ones, and from empty input."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from v3 import plots
from v3.contrast import METRICS

R = np.random.default_rng(0)
SUBJ, MUS, HOR = ["P1", "P2", "P3"], ["AD", "BR", "ED", "FD", "FDI"], [0, 200, 500, 1000]
SOURCES = ["eeg", "phase_swap", "swap", "bandpower", "peripheral"]


def _summary():
    rows = []
    for h in HOR:
        for seg in plots.PHASES:
            for src in SOURCES:
                row = {"horizon_ms": h, "segment": seg, "source": src, "n_subjects": 3}
                for m in METRICS:
                    row.update({f"{m}_mean": R.normal(0.1, 0.02), f"{m}_sem": 0.01})
                row.update(p_eeg_skill_holm_horizons=R.uniform(), p_eeg_skill_holm_phases=R.uniform())
                rows.append(row)
    return pd.DataFrame(rows)


def _close(fig):
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_step_2_and_3_figures():
    timing = pd.DataFrame([dict(variant=v, horizon_ms=h, rate=rate, subject=s, muscle=m, r=R.uniform())
                           for v in ("phase", "cue", "cue_causal") for h in ([0] if v in ("phase", "cue") else HOR)
                           for rate in (500, 50) for s in SUBJ for m in MUS])
    _close(plots.timing_baselines(timing))
    s = _summary()
    for f in (plots.b_vs_f_horizon(s), plots.b_vs_f_phases(s, "eeg"), plots.b_vs_f_phases(s, "nope"),
              plots.b_vs_f_mse_ladder(s)):
        _close(f)
    table = pd.DataFrame([dict(subject=su, horizon_ms=h, source="eeg", segment="all", eeg_skill=R.normal())
                          for su in SUBJ for h in HOR])
    _close(plots.b_vs_f_participants(table))
    onset = pd.DataFrame([dict(subject=su, horizon_ms=h, condition=c, error_ms=R.normal(0, 50)) for su in SUBJ
                          for h in HOR for c in ("K", "B", "F", "causal_cue") for _ in range(4)])
    _close(plots.onset_errors(onset))


def test_training_and_student_figures():
    hist = pd.DataFrame([dict(subject=su, condition=c, scale=sc, seed=se, epoch=e, val_e=1 / e, w=0.1, grad_share=0.2)
                         for su in SUBJ for c, sc in (("E0", 0.0), ("Eguided", 0.1)) for se in (0, 1) for e in range(1, 6)])
    _close(plots.training_curves(hist))
    std = pd.DataFrame([dict(subject=su, condition=c, scale=sc, seed=se, rate=rate, muscle=m, r=R.uniform())
                        for su in SUBJ for c, sc in (("E0", 0.0), ("Eguided", 0.1)) for se in (0, 1)
                        for rate in (500, 50) for m in MUS])
    _close(plots.student_accuracy(std))
    sw = std[std.rate == 500].assign(r_true=lambda d: d.r, r_swap=lambda d: d.r - 0.05)
    _close(plots.swap_scatter(sw, sw))
    _close(plots.swap_scatter(None, sw))
    res = pd.DataFrame([dict(subject=su, condition=c, scale=sc, seed=0, timing="cue_causal", segment=p, skill=R.normal())
                        for su in SUBJ for c, sc in (("E0", 0.0), ("Eguided", 0.1)) for p in plots.PHASES])
    _close(plots.residual_phases(res))
    tmpl = pd.DataFrame([dict(scale=sc, source=src, segment="all", muscle=m, skill_mean=R.normal())
                         for sc in (0.0, 0.1) for src in ("student", "phase_swap") for m in MUS])
    _close(plots.template_skill(tmpl))


def test_verdict_and_empty_inputs():
    v = pd.DataFrame([dict(horizon_ms=h, eeg_skill=0.01, eeg_skill_sem=0.005, phase_swap_skill=0.0, F_beats_B=True,
                           F_survives_swap=False, F_not_peripheral=True) for h in HOR])
    _close(plots.verdict_overview(v))
    for f in (plots.timing_baselines(pd.DataFrame({"rate": []})), plots.b_vs_f_horizon(pd.DataFrame({"segment": []})),
              plots.onset_errors(None), plots.training_curves(None), plots.student_accuracy(None),
              plots.residual_phases(None), plots.template_skill(None), plots.verdict_overview(None)):
        _close(f)
