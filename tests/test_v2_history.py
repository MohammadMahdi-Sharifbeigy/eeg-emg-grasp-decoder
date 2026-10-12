import numpy as np

from v2.history import N_HIST, detect_holds, history_features, hold_ids

FS = 500.0


def _series():
    load = np.zeros(10000); grip = np.zeros(10000)
    load[1000:2500] = 4.0; grip[1000:2500] = 6.0      # hold 1: 3 s
    load[4000:4100] = 5.0                             # 0.2 s blip: not a hold
    load[6000:7000] = 8.0; grip[6000:7000] = 10.0     # hold 2: 2 s
    return load, grip


def test_detect_holds_ignores_short_blips():
    load, _ = _series()
    assert detect_holds(load, FS) == [(1000, 2500), (6000, 7000)]


def test_history_values_on_a_hand_made_series():
    load, grip = _series()
    f, holds = history_features(load, grip, FS)
    assert f.shape == (10000, N_HIST) and f.dtype == np.float32 and len(holds) == 2
    assert np.all(f[:2500, [0, 1, 3, 4, 5, 6, 7]] == 0)            # nothing known before the first hold ends
    np.testing.assert_allclose(f[:, 2], np.arange(10000) / FS / 300.0, rtol=1e-6)
    a = f[3000]                                                    # 1 s after hold 1 ended
    np.testing.assert_allclose(a, [1, 1 / 40, 3000 / FS / 300, 1.0 / 30, 0.4, 0.6, 0.4, 0.6], rtol=1e-5)
    b = f[7500]                                                    # 1 s after hold 2 ended
    np.testing.assert_allclose(b, [1, 2 / 40, 7500 / FS / 300, 1.0 / 30, 0.8, 1.0, 0.6, 0.8], rtol=1e-5)
    assert f[6500, 1] == np.float32(1 / 40)                        # during hold 2 only hold 1 is known


def test_history_is_causal():
    load, grip = _series()
    f1, _ = history_features(load, grip, FS)
    load2, grip2 = load.copy(), grip.copy()
    load2[5000:] = 0.0; grip2[5000:] = 3.0                         # change the future
    f2, _ = history_features(load2, grip2, FS)
    np.testing.assert_array_equal(f1[:5000], f2[:5000])


def test_time_since_last_hold_is_capped():
    load = np.zeros(30000); grip = np.zeros(30000); load[100:600] = 3.0
    f, _ = history_features(load, grip, FS)
    assert f[-1, 3] == 1.0


def test_hold_ids():
    ids = hold_ids(10000, [(1000, 2500), (6000, 7000)])
    assert ids.dtype == np.int64 and ids[999] == 0 and ids[1000] == 1 and ids[2499] == 1 and ids[2500] == 0 and ids[6500] == 2
