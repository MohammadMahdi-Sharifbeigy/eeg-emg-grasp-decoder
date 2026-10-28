import numpy as np
import pandas as pd
import pytest

from v3.verdict import INTERPRETATION, TASK_LOCKED, eguided_vs_e0, proposal_verdict

SUBJECTS = [f"P{i}" for i in range(1, 13)]


def _summary(eeg=(0.05, 0.001), swap_gap=(0.04, 0.001), swap_skill_p=0.5, bandpower=0.01, peripheral=0.001,
             peripheral_p=0.5, horizons=(0, 500)):
    rows = []
    for h in horizons:
        common = dict(horizon_ms=h, segment="all", n_subjects=12, eeg_skill_sem=0.01, vaf_gain_mean=1.0, ccc_B_mean=0.8,
                      ccc_F_mean=0.82, mse_causal_cue_mean=0.5, mse_K_mean=0.3, mse_B_mean=0.2, mse_F_mean=0.19)
        rows += [dict(common, source="eeg", eeg_skill_mean=eeg[0], p_eeg_skill_holm_horizons=eeg[1], gap_mean=np.nan,
                      p_gap_holm_horizons=np.nan),
                 dict(common, source="phase_swap", eeg_skill_mean=eeg[0] - swap_gap[0], p_eeg_skill_holm_horizons=swap_skill_p,
                      gap_mean=swap_gap[0], p_gap_holm_horizons=swap_gap[1]),
                 dict(common, source="bandpower", eeg_skill_mean=bandpower, p_eeg_skill_holm_horizons=0.2,
                      gap_mean=0.0, p_gap_holm_horizons=0.5),
                 dict(common, source="peripheral", eeg_skill_mean=peripheral, p_eeg_skill_holm_horizons=peripheral_p,
                      gap_mean=0.0, p_gap_holm_horizons=0.5)]
        rows.append(dict(common, source="eeg", segment="hold", eeg_skill_mean=0.5, p_eeg_skill_holm_horizons=0.001))
    return pd.DataFrame(rows)


def _means(gain):
    j = np.linspace(-0.01, 0.01, 12)
    rows = [dict(subject=s, scale=0.0, student_r=0.6 + j[i]) for i, s in enumerate(SUBJECTS)]
    rows += [dict(subject=s, scale=0.1, student_r=0.6 + j[i] + gain + 0.001 * i) for i, s in enumerate(SUBJECTS)]
    return pd.DataFrame(rows)


def test_eguided_vs_e0():
    s = eguided_vs_e0(_means(0.05)).set_index("scale")
    assert bool(s.loc[0.1, "Eguided_beats_E0"]) and s.loc[0.1, "gain_r"] == pytest.approx(0.05 + 0.0055)
    assert not eguided_vs_e0(_means(-0.05)).Eguided_beats_E0.any()


@pytest.mark.parametrize("summary_kw, gain, expected", [
    ({}, 0.05, INTERPRETATION[(True, True)]),
    ({}, -0.05, INTERPRETATION[(True, False)]),
    ({"eeg": (0.0, 0.4)}, 0.05, INTERPRETATION[(False, True)]),
    ({"eeg": (0.0, 0.4)}, -0.05, INTERPRETATION[(False, False)]),
    ({"swap_gap": (0.0, 0.6)}, 0.05, TASK_LOCKED),
    ({"swap_skill_p": 0.001}, 0.05, TASK_LOCKED),
    ({"peripheral": 0.05, "peripheral_p": 0.001}, 0.05, TASK_LOCKED),
    ({"peripheral": 0.05, "peripheral_p": 0.5}, 0.05, INTERPRETATION[(True, True)]),     # above, but not significant
    ({"peripheral": float("nan")}, 0.05, TASK_LOCKED),                                     # unknown fails
])
def test_interpretation_matrix(summary_kw, gain, expected):
    v = proposal_verdict(_summary(**summary_kw), eguided_vs_e0(_means(gain)))
    assert list(v.horizon_ms) == [0, 500] and (v.interpretation == expected).all()


def test_rows_carry_the_metrics_behind_the_flags_and_onset():
    onset = pd.DataFrame({"horizon_ms": [0, 0, 0, 500], "condition": ["B", "B", "F", "F"],
                          "error_ms": [40.0, -60.0, 20.0, np.nan]})
    v = proposal_verdict(_summary(), eguided_vs_e0(_means(0.05)), onset=onset).set_index("horizon_ms")
    assert v.loc[0, "onset_abs_ms_B"] == 50.0 and v.loc[0, "onset_abs_ms_F"] == 20.0 and np.isnan(v.loc[500, "onset_abs_ms_F"])
    for col in ("mse_B", "mse_F", "mse_K", "mse_causal_cue", "vaf_gain", "ccc_B", "ccc_F", "eeg_skill", "swap_gap"):
        assert np.isfinite(v.loc[0, col]), col
    assert v.loc[0, "best_guided_scale"] == 0.1


def test_open_secondary_question_and_templates():
    v = proposal_verdict(_summary(eeg=(0.0, 0.4)))
    assert v.interpretation.str.contains("not been evaluated yet").all() and v.Eguided_beats_E0.isna().all()
    tmpl = pd.DataFrame([dict(scale=sc, source="student", segment="all", muscle=m, skill_mean=sk, p_skill_holm=p)
                         for sc, sk, p in ((0.0, 0.01, 0.5), (0.1, 0.05, 0.01)) for m in ("AD", "MEAN")])
    v = proposal_verdict(_summary(), eguided_vs_e0(_means(0.05)), template_summary=tmpl)
    assert (v.templates_E0 == False).all() and (v.templates_Eguided == True).all()   # noqa: E712
