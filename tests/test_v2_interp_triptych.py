import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.infer import predict_tokens
from v2.interp.triptych import PARTS, plot_triptych, series_internals, stretch_starts
from v2.model import build_model

NAMES = ["AD", "BR", "FD", "ED", "FDI"]


def _tok(T=9003, seed=0):
    rng = np.random.default_rng(seed)
    kin = rng.standard_normal((T, 26)).astype(np.float32)
    kin[:, 10:12] = 0
    for a in (1500, 5000):
        kin[a:a + 1200, 10] = 4.0                                        # two detected holds
    return D.build_tokens({"series": 4, "eeg": rng.standard_normal((T, 32)).astype(np.float32), "kin": kin,
                           "emg": np.abs(rng.standard_normal((T, 5))).astype(np.float32)}, CONFIG_V2)


def _model():
    torch.manual_seed(0)
    m = build_model(CONFIG_V2).eval()
    with torch.no_grad():
        m.fusion.gate.fill_(0.3)
        m.kin.gain[-1].bias.fill_(0.2)
    return m


def test_series_internals_equal_stitched_inference_and_the_exact_parts():
    tok = _tok(); norm = D.fit_norm([tok]); m = _model()
    it = series_internals(m, tok, norm, torch.device("cpu"))
    n = tok["kin"].shape[0]
    assert all(it[k].shape == (n, 5) for k in PARTS) and it["muscle_kin"].shape == it["muscle_eeg"].shape == (n, 5, 5)
    pt = predict_tokens(m, tok, norm, torch.device("cpu"))
    for k in ("K", "E", "F"):
        np.testing.assert_allclose(it[k], pt[k], atol=1e-6)
    np.testing.assert_allclose(it["K"], it["K_base"] * it["gain"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(it["F"], np.maximum(it["K"] + it["F_eeg"], 0.0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(it["muscle_kin"].sum(-1), 1.0, atol=1e-4)          # rows are shares of the source muscles
    assert all(mod.tape is None for mod in m.modules() if hasattr(mod, "tape"))


def test_stretch_starts_lead_into_detected_holds():
    tok = _tok()                                                        # holds start at tokens 150 and 500
    assert stretch_starts(tok, n=2, length_s=4.0, lead_s=1.0) == [100, 450]
    assert stretch_starts(tok, n=1, length_s=4.0, lead_s=1.0) == [100]
    flat = dict(tok, hold_id=np.zeros_like(tok["hold_id"]))
    assert stretch_starts(flat, n=2) == [0]


def test_triptych_has_three_panels_per_muscle():
    tok = _tok(); norm = D.fit_norm([tok]); m = _model()
    it = series_internals(m, tok, norm, torch.device("cpu"))
    events = [{"first_touch": 1400, "lift_off": 1600, "replace": 2600, "release": None, "led_on": 900}]
    fig = plot_triptych(it, tok, norm, NAMES, start=100, length_s=6.0, events=events, title="P1 series 4")
    labels = [ax.get_ylabel() for ax in fig.axes]
    assert labels.count("AD") == 2 and labels.count("AD  <-") == 1 and len(fig.axes) == 3 * 5 + 2 * 5 + 1
    top = fig.axes[0]
    assert len(top.lines) >= 4 + 3 and top.get_xlim()[0] == 2.0                  # actual, K, E, F and three event lines
    assert [t.get_text() for t in top.get_legend().get_texts()][:4] == ["actual", "K (kinematics-led)", "E (EEG only)", "F (fused)"]
    plt.close(fig)
    fig = plot_triptych(it, tok, norm, NAMES, start=0, length_s=4.0, stream="eeg")
    assert "EEG stream" in [ax.get_title(loc="left") for ax in fig.axes if "Panel 3" in ax.get_title(loc="left")][0]
    plt.close(fig)
