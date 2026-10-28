import numpy as np
import pytest

from v3.onset import onset_error, onset_summary, onset_times

FS = 500.0


def _env(n=20000, lifts=((2000, 0.4), (8000, 0.6), (14000, 0.5)), shift_s=0.0, flat_muscle=None):
    env = np.zeros((n, 3))
    for led, t_on in lifts:
        a = led + int((t_on + shift_s) * FS)
        ramp = np.linspace(0, 1, int(0.1 * FS))
        for m in range(3):
            env[a:a + len(ramp), m] = ramp
            env[a + len(ramp):a + int(1.5 * FS), m] = 1.0
    if flat_muscle is not None:
        env[:, flat_muscle] = 0.3
    return env


def _lifts():
    return [{"lift": i, "led_on": led, "hand_start": led + 150} for i, led in enumerate((2000, 8000, 14000))]


def test_onset_of_a_ramp():
    o = onset_times(_env(), _lifts(), FS)
    first = o[(o.lift == 0)].onset_s.values
    assert np.allclose(first, 0.4 + 0.02, atol=1 / FS)                 # 20 % up a 0.1 s ramp
    assert len(o) == 9


def test_shifted_prediction_gives_the_shift():
    err = onset_error(_env(shift_s=0.06), _env(), _lifts(), FS)
    assert np.allclose(err.error_ms, 60.0, atol=2.0)
    s = onset_summary(err)
    assert np.allclose(s.median_abs_error_ms, 60.0, atol=2.0) and (s.n_no_onset == 0).all()


def test_flat_muscle_is_nan_and_missing_led_is_skipped():
    lifts = _lifts() + [{"lift": 9, "led_on": None, "hand_start": 100}]
    err = onset_error(_env(flat_muscle=1), _env(), lifts, FS)
    assert set(err.lift) == {0, 1, 2}
    assert err[err.muscle == 1].error_ms.isna().all() and err[err.muscle == 0].error_ms.notna().all()
    s = onset_summary(err).set_index("muscle")
    assert s.loc[1, "n_no_onset"] == 3 and np.isnan(s.loc[1, "median_abs_error_ms"])


def test_token_rate_and_names():
    env50 = _env()[::10]
    o = onset_times(env50, _lifts(), fs=50.0, names=["AD", "BR", "ED"])
    assert set(o.muscle) == {"AD", "BR", "ED"}
    assert o[o.lift == 0].onset_s.values == pytest.approx([0.42] * 3, abs=0.02)
