import copy

import numpy as np
import pandas as pd
import pytest
import torch

from v2.tokens import Q, to_samples
from v3.config import CONFIG_V3
from v3.nulls import cue_aligned_swap, derangement, evaluate_swap, swap_group_test, swap_series, trial_bounds

MUSCLES = ["AD", "BR", "ED", "FD", "FDI"]


def _lifts(n_tok, period=250, first=30, w=None):
    return [{"lift": k, "CurW": (w[k % len(w)] if w else 1), "CurS": 1, "led_on": t * Q}
            for k, t in enumerate(range(first, n_tok - period + 1, period))]


def test_trial_bounds_cover_the_series():
    b = trial_bounds(_lifts(1000), 1000)
    assert b[0][0] == 0 and b[-1][1] == 1000
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
    assert all(a <= led < e for a, e, led in b)
    assert trial_bounds([{"led_on": None}], 100) == []


def test_derangement_has_no_fixed_point_and_respects_groups():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10):
        d = derangement(n, rng)
        assert (d != np.arange(n)).all() and sorted(d) == list(range(n))
    g = ["a", "a", "b", "b", "b", "c"]
    d = derangement(6, np.random.default_rng(1), g)
    assert (d != np.arange(6)).all()
    assert all(g[j] == g[i] for i, j in enumerate(d) if g.count(g[i]) > 1)
    assert np.array_equal(derangement(10, np.random.default_rng(5)), derangement(10, np.random.default_rng(5)))
    with pytest.raises(ValueError):
        derangement(1, rng)


def test_swap_aligns_the_cue_and_keeps_lengths():
    trials = [np.arange(10.0)[:, None] * 10 + k for k in range(3)]
    leds = [2, 5, 3]
    trials[1] = np.arange(14.0)[:, None] * 10 + 1
    out, donors = cue_aligned_swap(trials, leds, np.random.default_rng(0), return_donors=True)
    for i, (x, j) in enumerate(zip(out, donors)):
        assert len(x) == len(trials[i]) and j != i
        assert x[leds[i], 0] == trials[j][leds[j], 0]                    # the donor's cue lands on this trial's cue
    big = cue_aligned_swap([np.full((5, 1), 9.0), np.full((40, 1), -9.0)], [0, 30], np.random.default_rng(0), clip=5)
    assert all(np.abs(x).max() <= 5 for x in big)


def test_swap_series_replaces_every_trial():
    eeg = [np.random.default_rng(i).normal(size=(1000, 32, 4)).astype(np.float32) for i in range(2)]
    lifts = [_lifts(1000), _lifts(1000)]
    out, donors = swap_series(eeg, lifts, np.random.default_rng(0))
    assert [o.shape for o in out] == [e.shape for e in eeg]
    n_trials = sum(len(trial_bounds(l, 1000)) for l in lifts)
    assert len(donors) == n_trials and not any(np.array_equal(o, e) for o, e in zip(out, eeg))


def test_swap_series_can_match_condition():
    lifts = [_lifts(2000, w=[1, 2]), _lifts(2000, w=[1, 2])]
    eeg = [np.zeros((2000, 2, 1), np.float32) for _ in lifts]
    _, donors = swap_series(eeg, lifts, np.random.default_rng(0), match_condition=True)
    cond = [lift["CurW"] for l in lifts for lift in l]
    assert all(cond[i] == cond[j] for i, j in enumerate(donors))


class _Stub(torch.nn.Module):
    causal = False

    def forward(self, eeg):
        return {"E": torch.relu(eeg[:, :, 0, :1]).repeat(1, 1, 5).sqrt(), "fin": eeg[:, :, 0, :]}


def _world(trial_specific: bool, n_tok=2000, seed=0):
    r = np.random.default_rng(seed)
    toks, lifts = [], []
    for s in range(2):
        l = _lifts(n_tok)
        template = np.zeros(n_tok)
        amp = np.ones(n_tok)
        for a, b, led in trial_bounds(l, n_tok):
            t = np.arange(a, b) - led
            template[a:b] = np.exp(-0.5 * ((t - 40) / 15.0) ** 2)
            amp[a:b] = r.uniform(0.2, 2.0)
        drive = template * amp
        eeg = np.zeros((n_tok, 32, 4), np.float32)
        eeg[:, 0, 0] = drive if trial_specific else template
        emg = np.repeat(drive[:, None], 5, 1).astype(np.float32)
        toks.append({"series": s, "eeg": eeg, "emg": emg, "emg_full": to_samples(emg, n_tok * Q), "n_samples": n_tok * Q})
        lifts.append(l)
    norm = {"eeg_mean": np.zeros((32, 4), np.float32), "eeg_std": np.ones((32, 4), np.float32),
            "emg_scale": np.ones(5, np.float32)}
    return toks, lifts, norm


def _cfg():
    cfg = copy.deepcopy(CONFIG_V3)
    cfg["evaluation"].update(n_swap=3, seed=0)
    return cfg


def test_timing_only_model_keeps_its_score_under_the_swap():
    toks, lifts, norm = _world(trial_specific=False)
    df = evaluate_swap(_Stub(), toks, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)
    assert (df.r_true > 0.5).all() and (df.delta.abs() < 0.02).all()


def test_trial_specific_model_loses_its_score_under_the_swap():
    toks, lifts, norm = _world(trial_specific=True)
    df = evaluate_swap(_Stub(), toks, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)
    assert (df.r_true > 0.99).all() and (df.delta > 0.2).all()
    assert list(df.columns) == ["muscle", "r_true", "r_swap", "r_swap_sd", "delta", "n_draws"]


def test_swap_group_test():
    rows = []
    for i in range(12):
        for scale, d in ((0.0, 0.05), (0.1, 0.10)):
            rows.append({"subject": f"P{i}", "scale": scale, "seed": 0, "muscle": "AD", "r_true": 0.6,
                         "r_swap": 0.6 - d - 0.001 * i, "delta": d + 0.001 * i})
    g = swap_group_test(pd.DataFrame(rows)).set_index("scale")
    assert g.loc[0.1, "p_delta"] < 0.05 and g.loc[0.1, "mean_delta_gain"] == pytest.approx(0.05)
    assert g.loc[0.1, "p_delta_gain"] < 0.05 and np.isnan(g.loc[0.0, "p_delta_gain"])


# ---- phase-matched within-condition swap -------------------------------------------------------------------------
from v3.nulls import _warped, event_anchors, evaluate_phase_matched_swap, phase_matched_swap

EVENTS_ORDER = ("led_on", "hand_start", "first_touch", "lift_off", "replace", "release", "hand_stop")


def _jittered_lifts(n_tok, seed, period=300, first=30, w=(1, 2), s=(1,)):
    """Lifts whose intervals between events vary from lift to lift (so only a phase-matched swap keeps them aligned)."""
    r = np.random.default_rng(seed)
    out = []
    for k, t0 in enumerate(range(first, n_tok - period + 1, period)):
        gaps = np.cumsum([0] + list(r.integers(10, 40, size=6)))
        out.append({"lift": k, "CurW": w[k % len(w)], "CurS": s[k % len(s)],
                    **{e: int((t0 + g) * Q) for e, g in zip(EVENTS_ORDER, gaps)}})
    return out


def test_event_anchors_drop_missing_and_out_of_order_events():
    lift = {"led_on": 100 * Q, "hand_start": None, "first_touch": 130 * Q, "lift_off": 120 * Q, "replace": 150 * Q}
    assert event_anchors(lift, 90, 400) == {"led_on": 10, "first_touch": 40, "replace": 60}
    assert event_anchors(lift, 90, 140) == {"led_on": 10, "first_touch": 40}


def test_warp_moves_every_shared_event_onto_the_recipients():
    src = np.arange(100.0)[:, None]
    src_anchors = {"led_on": 10, "first_touch": 30, "lift_off": 70}
    anchors = {"led_on": 5, "first_touch": 40, "lift_off": 60, "replace": 80}
    x, n = _warped(src, src_anchors, 90, anchors)
    assert n == 3 and len(x) == 90
    for ev in src_anchors:
        assert x[anchors[ev], 0] == src[src_anchors[ev], 0]
    assert x[0, 0] == 5.0 and x[-1, 0] == 99.0                      # own speed outside: 10 - 5 = 5 and 70 + (89 - 60) = 99
    assert np.all(np.diff(x[:, 0]) >= 0)


def test_phase_matched_donors_share_the_condition_and_report_fallbacks():
    lifts = [_jittered_lifts(3000, 0, w=(1, 2, 4))]
    lifts[0][-1]["CurW"], lifts[0][-1]["CurS"] = 8, 9                 # a condition with a single trial
    eeg = [np.zeros((3000, 2, 1), np.float32)]
    _, table = phase_matched_swap(eeg, lifts, np.random.default_rng(0))
    cond = [(l["CurW"], l["CurS"]) for l in lifts[0]]
    same = table[table.match_level == "same_condition"]
    assert all(cond[r] == cond[d] for r, d in zip(same.recipient, same.donor))
    assert (table.recipient != table.donor).all()
    assert table.match_level.iloc[-1] == "any" and table.n_shared_anchors.max() == 7


class _Stub2(torch.nn.Module):
    causal = False

    def forward(self, eeg):
        return {"E": torch.relu(eeg[:, :, 0, :1]).repeat(1, 1, 5).sqrt(), "fin": eeg[:, :, 0, :]}


def _lift_off_world(trial_specific: bool, n_tok=3000):
    r = np.random.default_rng(1)
    toks, lifts = [], []
    for s in range(2):
        l = _jittered_lifts(n_tok, seed=10 + s, w=(1,))
        drive = np.zeros(n_tok)
        for a, b, led in trial_bounds(l, n_tok):
            lift = next(x for x in l if x["led_on"] // Q == led)
            t = np.arange(a, b) - lift["lift_off"] // Q
            drive[a:b] = np.exp(-0.5 * (t / 8.0) ** 2) * (r.uniform(0.2, 2.0) if trial_specific else 1.0)
        target = drive if trial_specific else drive * r.uniform(0.5, 1.5)
        eeg = np.zeros((n_tok, 32, 4), np.float32)
        eeg[:, 0, 0] = drive
        emg = np.repeat(target[:, None], 5, 1).astype(np.float32)
        toks.append({"series": s, "eeg": eeg, "emg": emg, "emg_full": to_samples(emg, n_tok * Q), "n_samples": n_tok * Q})
        lifts.append(l)
    norm = {"eeg_mean": np.zeros((32, 4), np.float32), "eeg_std": np.ones((32, 4), np.float32),
            "emg_scale": np.ones(5, np.float32)}
    return toks, lifts, norm


def test_phase_matched_swap_keeps_a_lift_off_locked_model_and_the_cue_swap_does_not():
    toks, lifts, norm = _lift_off_world(trial_specific=False)
    pm = evaluate_phase_matched_swap(_Stub2(), toks, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)
    cue = evaluate_swap(_Stub2(), toks, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)
    assert (pm.delta.abs() < 0.1).all() and (cue.delta > pm.delta + 0.3).all()      # warping stretches the bump a little
    assert {"mse_true", "mse_swap"} <= set(pm.columns)


def test_phase_matched_swap_removes_trial_specific_information():
    toks, lifts, norm = _lift_off_world(trial_specific=True)
    pm = evaluate_phase_matched_swap(_Stub2(), toks, lifts, norm, _cfg(), torch.device("cpu"), MUSCLES)
    assert (pm.r_true > 0.99).all() and (pm.delta > 0.15).all() and (pm.mse_swap > pm.mse_true).all()


def test_unknown_conditions_are_never_pooled():
    lifts = [_jittered_lifts(3000, 0)]
    for l in lifts[0]:
        l["CurW"] = None
    _, table = phase_matched_swap([np.zeros((3000, 2, 1), np.float32)], lifts, np.random.default_rng(0))
    assert (table.match_level == "any").all()
