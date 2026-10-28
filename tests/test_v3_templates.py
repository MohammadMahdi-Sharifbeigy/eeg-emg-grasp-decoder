"""Template deviation test (v3.templates): EEG beyond the event-locked average trial."""
import copy
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_controls.common import EVENTS
from v2.tokens import Q, to_samples
from v3 import templates as T
from v3.config import CONFIG_V3, preset_config

MUSCLES = ["AD", "BR", "ED", "FD", "FDI"]
NB = 20
OFFS = (0, 20, 45, 50, 70, 150, 165, 190)                          # tokens after the LED, one per event


def _lift(t0, stretch=1.0, **kw):
    return {"led_on": t0 * Q, **{e: int(round((t0 + o * stretch) * Q)) for e, o in zip(EVENTS[1:], OFFS[1:])},
            "CurW": 1, "CurS": 1, **kw}


def test_bins_follow_event_warped_time():
    lifts = [_lift(100), _lift(500, stretch=1.2)]
    b = T.template_bins(lifts, 800, NB)
    rest = T.n_template_bins(NB) - 1
    assert b[0] == rest and b[99] == rest                          # before the first LED
    assert b[100] == 0 and b[110] == NB // 2                       # halfway from led_on to hand_start
    assert b[100 + 20] == NB and b[100 + 190] == 7 * NB            # hand_start, hand_stop
    assert b[100 + 190 + 5] == 7 * NB + 1                          # 0.1 s bins after hand_stop
    assert b[499] == rest                                          # more than 3 s after hand_stop
    assert b[500 + 12] == NB // 2                                  # the stretched trial: 12 of its 24 tokens
    assert T.n_template_bins(NB) == 7 * NB + 30 + 1


def test_trials_with_missing_or_unordered_events_get_no_bin():
    b = T.template_bins([_lift(100), _lift(400, lift_off=None)], 800, NB)
    assert (b[400:] == -1).all() and (b[100:400] >= 0).all()
    b = T.template_bins([_lift(100, replace=60 * Q + 100 * Q)], 800, NB)          # replace before lift_off
    assert (b[100:] == -1).all()
    b = T.template_bins([_lift(100, both_touch=145 * Q)], 800, NB)               # both_touch == first_touch is fine
    assert (b[100:] >= 0).all() and not np.isin(2 * NB + np.arange(NB), b).any()


def _base(bins):
    """The average trial in warped time: a bump over load and hold, a low rest level."""
    x = np.where(bins >= 0, bins, 0) / (7 * NB)
    return np.where(bins >= 7 * NB, 0.1, 0.1 + np.exp(-0.5 * ((x - 0.55) / 0.12) ** 2))


def _world(r, n_series, informative=True, n_tok=2600):
    """Per trial and muscle an amplitude around 1; EEG electrodes 0-4 (band 0) carry the trial's deviation."""
    toks, lifts = [], []
    for s in range(n_series):
        l = [_lift(t, stretch=r.uniform(0.8, 1.2)) for t in range(30, n_tok - 300, 280)]
        bins = T.template_bins(l, n_tok, NB)
        base = _base(bins)
        amp = np.ones((n_tok, 5))
        led = sorted(x["led_on"] // Q for x in l)
        for a, b in zip([0] + led[1:], led[1:] + [n_tok]):
            amp[a:b] = 1 + r.uniform(-0.4, 0.4, size=5)
        u = base[:, None] * amp + 0.01 * r.normal(size=(n_tok, 5))
        eeg = np.zeros((n_tok, 32, 4), np.float32)
        eeg[:, :5, 0] = base[:, None] * (amp - 1) if informative else r.normal(size=(n_tok, 5))
        emg = (np.maximum(u, 0) ** 2).astype(np.float32)
        toks.append({"series": s, "eeg": eeg, "emg": emg, "emg_full": to_samples(emg, n_tok * Q), "n_samples": n_tok * Q})
        lifts.append(l)
    return toks, lifts


NORM = {"eeg_mean": np.zeros((32, 4), np.float32), "eeg_std": np.ones((32, 4), np.float32),
        "emg_scale": np.ones(5, np.float32)}


class _Stub(torch.nn.Module):
    causal = False

    def forward(self, eeg):
        x = eeg[:, :, :5, 0]
        return {"E": torch.zeros(x.shape), "fin": torch.cat([x, torch.ones_like(x[..., :1])], -1)}


def _cfg():
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["evaluation"].update(n_null=19, seed=0)
    return cfg


def test_templates_recover_the_average_trial():
    r = np.random.default_rng(0)
    train, lifts = _world(r, 4)
    tm = T.fit_emg_templates(train, lifts, NORM, n_bins=NB)
    assert tm["template"].shape == (T.n_template_bins(NB), 5) and tm["count"].sum() > 0
    seen = tm["count"] > 50
    expected = _base(np.arange(T.n_template_bins(NB)))
    assert np.abs(tm["template"][seen] - expected[seen, None]).max() < 0.08
    dev, bins = T.template_deviation(tm, train[0], lifts[0], NORM)
    assert dev.shape == (len(bins), 5) and np.isfinite(dev[bins >= 0]).all()


def test_student_predicts_deviations_and_the_swap_does_not():
    r = np.random.default_rng(1)
    train, tl = _world(r, 4)
    test, sl = _world(r, 2)
    tm = T.fit_emg_templates(train, tl, NORM, n_bins=NB)
    df = T.student_template_deviation(_Stub(), test, sl, tm, NORM, _cfg(), torch.device("cpu"), MUSCLES)
    assert list(df.columns) == ["source"] + T.COLUMNS + ["var_template", "n_trials"]
    assert set(df.source) == set(T.SOURCES)
    st = df[(df.source == "student") & (df.segment == "all")].set_index("muscle")
    assert list(st.index) == MUSCLES + ["MEAN"]
    assert (st.skill > 0.5).all() and (st.p <= 0.05).all() and (st.r > 0.7).all()
    assert df[(df.source == "student") & (df.segment == "hold")].skill.gt(0.5).all()
    sw = df[(df.source == "swap") & (df.segment == "all")].set_index("muscle")
    assert (sw.skill < st.skill - 0.4).all()
    assert df.var_template.iloc[0] > 0.5 and df.n_trials.iloc[0] == 18


def test_uninformative_eeg_has_no_skill():
    r = np.random.default_rng(2)
    train, tl = _world(r, 4)
    test, sl = _world(r, 2, informative=False)
    tm = T.fit_emg_templates(train, tl, NORM, n_bins=NB)
    df = T.student_template_deviation(_Stub(), test, sl, tm, NORM, _cfg(), torch.device("cpu"), MUSCLES,
                                      sources=("student",))
    assert df[(df.segment == "all") & (df.muscle == "MEAN")].skill.iloc[0] < 0.05


def test_peripheral_is_skipped_when_the_student_lacks_those_channels():
    r = np.random.default_rng(3)
    train, tl = _world(r, 3)
    test, sl = _world(r, 2)
    for t in train + test:
        t["eeg"] = t["eeg"][:, :11]                                 # the central preset has 11 electrodes
    norm = dict(NORM, eeg_mean=np.zeros((11, 4), np.float32), eeg_std=np.ones((11, 4), np.float32))
    cfg = preset_config("central", _cfg())
    tm = T.fit_emg_templates(train, tl, norm, n_bins=NB)
    with pytest.warns(UserWarning, match="peripheral"):
        df = T.student_template_deviation(_Stub(), test, sl, tm, norm, cfg, torch.device("cpu"), MUSCLES)
    assert "peripheral" not in set(df.source)


def test_summarise_template_deviation_gap_and_holm():
    rows = []
    j = np.linspace(-0.004, 0.004, 12)
    for src, base in (("student", 0.05), ("swap", 0.01)):
        for seg in ("reach", "hold", "all"):
            for m in MUSCLES + ["MEAN"]:
                for i in range(12):
                    for seed in (0, 1):
                        rows.append({"subject": f"P{i + 1}", "scale": 0.0, "seed": seed, "source": src, "segment": seg,
                                     "muscle": m, "skill": base + j[i], "r": 0.2})
    g = T.summarise_template_deviation(pd.DataFrame(rows))
    st = g[(g.source == "student") & (g.muscle == "AD") & (g.segment == "hold")].iloc[0]
    assert st.skill_mean == pytest.approx(0.05) and st.p_skill < 0.01 and np.isnan(st.gap_mean)
    sw = g[(g.source == "swap") & (g.muscle == "AD") & (g.segment == "hold")].iloc[0]
    assert sw.gap_mean == pytest.approx(0.04) and sw.p_gap < 0.01
    fam = g[(g.source == "student") & (g.segment != "all") & (g.muscle != "MEAN")]
    assert len(fam) == 10 and fam.p_skill_holm.to_numpy() == pytest.approx(np.minimum(1, 10 * fam.p_skill.to_numpy()))
    assert g[g.muscle == "MEAN"].p_skill_holm.isna().all()


def test_config_has_the_templates_block():
    tc = CONFIG_V3["templates"]
    assert tc["n_bins"] == 20 and tc["swap_match_condition"] is True and tuple(tc["sources"]) == T.SOURCES
