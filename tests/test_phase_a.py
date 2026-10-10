"""Package 2 — Phase A (corrections): real EMG channel names, one inference path,
gain-corrected magnitude diagnostic, early-stop patience and the EEG-vs-kin
ablation cell in both notebooks.

No model is trained and no notebook cell is executed except the CONFIG cell
(a plain dict literal).
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main.dataloader import DEFAULT_RAW_DIR, emg_channel_names, short_emg_label
from main.inference import predict_series_stitched, stitch_windows, window_batch, window_starts
from main.magnitude_eval import (
    MAGNITUDE_COLUMNS,
    format_gain_table,
    format_magnitude_table,
    magnitude_report,
    magnitude_report_by_series,
    predict_series,
)
from main.model import KGGTModel

TRUE_LONG = ["Anterior Deltoid", "Brachoradial", "Flexor Digitorum",
             "Common Extensor Digitorum", "First Dorsal Interosseus"]
TRUE_SHORT = ["AD", "BR", "FD", "ED", "FDI"]
WRONG = ["FDI", "APB", "ADM", "ECR", "FCR"]
NOTEBOOKS = ["main.ipynb", "main_1stage.ipynb"]


# ════════════════════════════════════════════════════════════════════════════
# 1. EMG channel names
# ════════════════════════════════════════════════════════════════════════════

def _write_hs(path: Path, emg_names: list[str] | None) -> None:
    hs = {"eeg": {"sig": np.zeros((4, 2)), "names": np.array(["a", "b"], dtype=object),
                  "samplingrate": 500}}
    if emg_names is not None:
        hs["emg"] = {"sig": np.zeros((8, len(emg_names))), "samplingrate": 4000,
                     "names": np.array(emg_names, dtype=object)}
    sio.savemat(str(path), {"hs": hs})


def _fake_raw(tmp_path: Path, participant: int, names_per_series: dict) -> Path:
    p_dir = tmp_path / f"P{participant}"
    p_dir.mkdir(parents=True)
    for sid, names in names_per_series.items():
        _write_hs(p_dir / f"HS_P{participant}_S{sid}.mat", names)
    return tmp_path


def test_short_emg_label_map_and_fallback():
    assert [short_emg_label(n) for n in TRUE_LONG] == TRUE_SHORT
    assert short_emg_label("Brachioradial") == "BR"
    assert short_emg_label("  first dorsal interosseus ") == "FDI"
    assert short_emg_label("Some Other Muscle") == "Some Other Muscle"


def test_emg_channel_names_reads_the_participants_hs_files(tmp_path):
    root = _fake_raw(tmp_path, 3, {1: TRUE_LONG, 2: TRUE_LONG, "T": None})   # ST has no EMG
    assert emg_channel_names(3, root) == TRUE_SHORT
    assert emg_channel_names(3, root, short=False) == TRUE_LONG
    assert emg_channel_names("P3", root) == TRUE_SHORT
    assert emg_channel_names("LOSO_P3", str(root)) == TRUE_SHORT
    out = emg_channel_names(3, root)
    out.append("x")                              # callers get their own list
    assert emg_channel_names(3, root) == TRUE_SHORT


def test_emg_channel_names_unknown_name_falls_back_to_raw_string(tmp_path):
    names = ["Anterior Deltoid", "Mystery Muscle"]
    root = _fake_raw(tmp_path, 4, {1: names})
    assert emg_channel_names(4, root) == ["AD", "Mystery Muscle"]


def test_emg_channel_names_raises_when_series_disagree(tmp_path):
    root = _fake_raw(tmp_path, 5, {1: TRUE_LONG, 2: TRUE_LONG[::-1]})
    with pytest.raises(ValueError, match="P5"):
        emg_channel_names(5, root)


def test_emg_channel_names_missing_participant(tmp_path):
    with pytest.raises(FileNotFoundError):
        emg_channel_names(9, tmp_path)


# Loading the raw HS files takes ~5 s per participant: P1 by default, all 12 with
# WAYEEG_ALL_PARTICIPANTS=1 (P1..P12 were verified identical on 2026-10-10).
_REAL_PARTICIPANTS = range(1, 13) if os.environ.get("WAYEEG_ALL_PARTICIPANTS") else (1,)


@pytest.mark.parametrize("participant", _REAL_PARTICIPANTS)
def test_emg_channel_names_real_data(participant):
    if not list((DEFAULT_RAW_DIR / f"P{participant}").glob(f"HS_P{participant}_S*.mat")):
        pytest.skip(f"raw WAY-EEG-GAL data for P{participant} not available")
    assert emg_channel_names(participant, short=False) == TRUE_LONG
    assert emg_channel_names(participant) == TRUE_SHORT


def test_library_has_no_wrong_default_name_list():
    pat = re.compile(r"""["']FDI["']\s*,\s*["']APB["']""")
    for f in sorted((ROOT / "main").glob("*.py")):
        assert not pat.search(f.read_text(encoding="utf-8")), f.name


# ── notebooks ───────────────────────────────────────────────────────────────

def _nb(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell_src(nb, needle: str) -> str:
    hits = [s for s in _code_cells(nb) if needle in s]
    assert len(hits) == 1, (needle, len(hits))
    return hits[0]


def _config(name: str) -> dict:
    ns = {"copy": copy}
    exec(_cell_src(_nb(name), "CONFIG = {"), ns)
    return ns["CONFIG"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_hard_coded_emg_name_list(name):
    wrong_only = re.compile(r"""["'](APB|ADM|ECR|FCR)["']""")
    literal_list = re.compile(r"(EMG_NAMES|MUSCLE_NAMES)\s*=\s*\[\s*[\"']")
    for i, s in enumerate(_code_cells(_nb(name))):
        assert not wrong_only.search(s), (name, i, wrong_only.search(s).group(0))
        assert not literal_list.search(s), (name, i)
        assert '"FDI"' not in s and "'FDI'" not in s, (name, i)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_cells_take_names_from_the_helper(name):
    nb = _nb(name)
    for needle in ("PER-SUBJECT PRE-PROCESSING LOOP", "# EVALUATION — Full metric suite",
                   "MAGNITUDE EVALUATION", "HOW MUCH DOES THE MODEL USE EEG vs KINEMATICS?",
                   "INTERPRETABILITY — Full-Signal Triptych", "[OPTIONAL] Automatic LOSOCV",
                   "DIAGNOSTIC: Residual Decomposition",
                   "STATIC EDGE BIAS + MUSCLE SYNERGY MATRIX",
                   "plot_neural_mechanical_latency_lag,"):
        assert "emg_channel_names(" in _cell_src(nb, needle), needle
    # EMG illustration cell: names of the very file it loads
    assert 'short_emg_label(n) for n in hs_data["emg_names"]' in _cell_src(nb, "MUSCLE_NAMES =")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_documents_true_channel_order(name):
    md = [_src(c) for c in _nb(name)["cells"] if c["cell_type"] == "markdown"]
    note = [m for m in md if "EMG channel names" in m]
    assert len(note) == 1
    for long_name, short, old in zip(TRUE_LONG, TRUE_SHORT, WRONG):
        assert long_name in note[0] and f"| {short} |" in note[0] and f'"{old}"' in note[0]
    # right after the CONFIG cell
    cells = _nb(name)["cells"]
    i = next(k for k, c in enumerate(cells) if c["cell_type"] == "code" and "CONFIG = {" in _src(c))
    assert "EMG channel names" in _src(cells[i + 1])


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_code_cells_compile(name):
    for i, s in enumerate(_code_cells(_nb(name))):
        code = "\n".join("pass" if re.match(r"^\s*[%!]", ln) else ln for ln in s.splitlines())
        compile(code, f"{name}[code cell {i}]", "exec")


# ════════════════════════════════════════════════════════════════════════════
# 2. One inference path
# ════════════════════════════════════════════════════════════════════════════

WIN = 50
_TRANSFORMER_CFG = {
    "n_layers": 2, "n_heads": 2, "d_model": 16, "d_k": 8, "d_v": 8,
    "ffn_dim": 32, "dropout": 0.0, "chunk_size": WIN,
}
EEG_DIM, KIN_RAW, KIN_DIM, N_CH, GAT_HEADS = 6, 9, 8, 5, 2


def _tiny_model() -> KGGTModel:
    torch.manual_seed(0)
    m = KGGTModel(
        transformer_cfg=_TRANSFORMER_CFG, input_dim=EEG_DIM, node_dim=16, gat_hidden_dim=16,
        gat_heads=GAT_HEADS, out_channels=N_CH, kin_dim=KIN_DIM, use_kinematic_guidance=True,
        gat_kin_dim_override=4,
    )
    with torch.no_grad():                     # zero-initialised branches -> give them weights
        m.kin_skip_proj[-1].weight.normal_(0.0, 0.2)
        m.gat.kin_edge_linear.weight.normal_(0.0, 0.5)
    return m.eval()


def _series(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, EEG_DIM)).astype(np.float32),
            rng.standard_normal((T, KIN_RAW)).astype(np.float32),
            rng.random((T, N_CH)).astype(np.float32))


EMG_MEAN = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
EMG_STD = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8])
KIN_MEAN = torch.linspace(-0.5, 0.5, KIN_RAW)
KIN_STD = torch.linspace(0.5, 2.0, KIN_RAW)


def _pb(eeg, kin, emg):
    """Same shape as the notebook's local_prepare_batch_factory closure
    (kin z-score + column drop, EMG z-score)."""
    kn = ((kin - KIN_MEAN) / KIN_STD)[..., [i for i in range(KIN_RAW) if i != 3]]
    return {"eeg": eeg, "kin": kn}, (emg - EMG_MEAN) / EMG_STD


def _direct(model, eeg, kin, emg, starts):
    """Reference: one model call per window, exactly like a test-loader batch."""
    out = []
    with torch.no_grad():
        for s in starts:
            inp, _ = _pb(torch.as_tensor(eeg[s:s + WIN])[None], torch.as_tensor(kin[s:s + WIN])[None],
                         torch.as_tensor(emg[s:s + WIN])[None])
            out.append(model(inp["eeg"], inp["kin"])[0].numpy())
    return out


def test_window_starts():
    assert window_starts(150, 50) == [0, 50, 100]
    assert window_starts(170, 50) == [0, 50, 100, 120]        # last window aligned to the end
    assert window_starts(170, 50, tail="drop") == [0, 50, 100]
    assert window_starts(50, 50) == [0]
    with pytest.raises(ValueError):
        window_starts(49, 50)
    with pytest.raises(ValueError):
        window_starts(100, 50, tail="nope")


def test_stitch_windows_keeps_the_earlier_window_on_overlap():
    starts = [0, 4, 6]                                         # n = 10, window 4
    x = np.stack([np.full(4, 1.0), np.full(4, 2.0), np.full(4, 3.0)])
    np.testing.assert_array_equal(stitch_windows(x, starts, 10), [1, 1, 1, 1, 2, 2, 2, 2, 3, 3])


def test_stitcher_equals_direct_windowed_forward():
    model = _tiny_model()
    eeg, kin, emg = _series(4 * WIN)
    pred = predict_series_stitched(model, eeg, kin, _pb, window=WIN,
                                   device=torch.device("cpu"), batch_size=3)
    assert pred.shape == (4 * WIN, N_CH) and pred.dtype == np.float32
    ref = np.concatenate(_direct(model, eeg, kin, emg, [0, WIN, 2 * WIN, 3 * WIN]))
    np.testing.assert_allclose(pred, ref, rtol=1e-5, atol=1e-6)
    # NOT the same as one long eval-mode forward (the encoder does not chunk in eval mode)
    with torch.no_grad():
        inp, _ = _pb(torch.as_tensor(eeg)[None], torch.as_tensor(kin)[None], torch.as_tensor(emg)[None])
        long = model(inp["eeg"], inp["kin"])[0].numpy()
    assert np.abs(long - pred).max() > 1e-3


def test_stitcher_length_not_divisible_by_window():
    model = _tiny_model()
    T = 3 * WIN + 17
    eeg, kin, emg = _series(T, seed=1)
    pred, target = predict_series_stitched(model, eeg, kin, _pb, window=WIN,
                                           device=torch.device("cpu"), batch_size=2, emg=emg)
    assert pred.shape == (T, N_CH) and target.shape == (T, N_CH)
    wins = _direct(model, eeg, kin, emg, [0, WIN, 2 * WIN, T - WIN])
    # whole windows = the evaluation path; the tail comes from the end-aligned window
    np.testing.assert_allclose(pred[:3 * WIN], np.concatenate(wins[:3]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pred[3 * WIN:], wins[3][-17:], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(target, (emg - EMG_MEAN.numpy()) / EMG_STD.numpy(), rtol=1e-5, atol=1e-6)
    # tail="drop" reproduces the test loader (stride = window, incomplete tail dropped)
    dropped = predict_series_stitched(model, eeg, kin, _pb, window=WIN, device=torch.device("cpu"),
                                      batch_size=2, tail="drop")
    np.testing.assert_allclose(dropped, pred[:3 * WIN], rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):
        predict_series_stitched(model, eeg[:WIN - 1], kin[:WIN - 1], _pb, window=WIN,
                                device=torch.device("cpu"))


def test_stitcher_is_deterministic_and_independent_of_batch_size_and_mode():
    model = _tiny_model()
    eeg, kin, _ = _series(3 * WIN + 5, seed=2)
    kw = dict(window=WIN, device=torch.device("cpu"))
    a = predict_series_stitched(model, eeg, kin, _pb, batch_size=1, **kw)
    b = predict_series_stitched(model, eeg, kin, _pb, batch_size=1, **kw)
    np.testing.assert_array_equal(a, b)
    c = predict_series_stitched(model, eeg, kin, _pb, batch_size=64, **kw)
    np.testing.assert_allclose(a, c, rtol=1e-5, atol=1e-6)
    model.train()                                  # the stitcher switches to eval itself
    d = predict_series_stitched(model, eeg, kin, _pb, batch_size=1, **kw)
    np.testing.assert_array_equal(a, d)
    assert not model.training


def test_stitcher_internals_shapes_and_content():
    model = _tiny_model()
    T = 3 * WIN + 17
    eeg, kin, emg = _series(T, seed=3)
    pred, internals = predict_series_stitched(
        model, eeg, kin, _pb, window=WIN, device=torch.device("cpu"), batch_size=2,
        return_internals=True, full_attention=True)
    n_win, L, H = 4, _TRANSFORMER_CFG["n_layers"], _TRANSFORMER_CFG["n_heads"]
    assert internals["window"] == WIN
    np.testing.assert_array_equal(internals["starts"], [0, WIN, 2 * WIN, T - WIN])
    assert internals["encoder_attn"].shape == (n_win, L, H, WIN, WIN)
    assert internals["encoder_key_attn"].shape == (n_win, L, H, WIN)
    assert internals["gat_attn"].shape == (n_win, WIN, GAT_HEADS, N_CH, N_CH)
    assert internals["kin_edge_bias"].shape == (n_win, WIN, GAT_HEADS, N_CH, N_CH)
    assert internals["encoder_key_attn_stitched"].shape == (T, L, H)
    assert internals["gat_attn_stitched"].shape == (T, GAT_HEADS, N_CH, N_CH)
    assert internals["kin_edge_bias_stitched"].shape == (T, GAT_HEADS, N_CH, N_CH)
    # attention rows are distributions over the keys of the SAME window
    np.testing.assert_allclose(internals["encoder_attn"].sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(internals["encoder_key_attn"],
                               internals["encoder_attn"].mean(axis=-2), atol=1e-6)
    np.testing.assert_allclose(internals["encoder_key_attn"].sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(internals["gat_attn"].sum(-1), 1.0, atol=1e-5)
    # equals what a direct forward of window 1 leaves on the model
    with torch.no_grad():
        inp, _ = _pb(torch.as_tensor(eeg[WIN:2 * WIN])[None], torch.as_tensor(kin[WIN:2 * WIN])[None],
                     torch.as_tensor(emg[WIN:2 * WIN])[None])
        model(inp["eeg"], inp["kin"])
    for layer_idx, layer in enumerate(model.encoder.layers):
        np.testing.assert_allclose(internals["encoder_attn"][1, layer_idx],
                                   layer.mhsa.last_attn_weights[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(internals["kin_edge_bias"][1],
                               model.gat.last_kin_edge_bias[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(internals["kin_edge_bias_stitched"][WIN:2 * WIN],
                               internals["kin_edge_bias"][1], atol=0)
    # stitched tail = last 17 steps of the end-aligned window
    np.testing.assert_allclose(internals["encoder_key_attn_stitched"][3 * WIN:, 1],
                               internals["encoder_key_attn"][3, 1].T[-17:], atol=0)
    # default: the (W, W) matrices are not kept, and the prediction is unchanged
    pred2, light = predict_series_stitched(
        model, eeg, kin, _pb, window=WIN, device=torch.device("cpu"), batch_size=2,
        return_internals=True)
    assert "encoder_attn" not in light and light["encoder_key_attn"].shape == (n_win, L, H, WIN)
    np.testing.assert_array_equal(pred, pred2)
    # nothing big is left on the model
    assert all(layer.mhsa.last_attn_weights is None for layer in model.encoder.layers)


def test_window_batch_matches_prepare_batch_per_window():
    eeg, kin, emg = _series(2 * WIN + 9, seed=4)
    inputs, y = window_batch(eeg, kin, _pb, window=WIN, emg=emg, starts=[0, WIN])
    assert inputs["eeg"].shape == (2, WIN, EEG_DIM) and inputs["kin"].shape == (2, WIN, KIN_DIM)
    ref, yref = _pb(torch.as_tensor(eeg[WIN:2 * WIN])[None], torch.as_tensor(kin[WIN:2 * WIN])[None],
                    torch.as_tensor(emg[WIN:2 * WIN])[None])
    assert torch.equal(inputs["kin"][1], ref["kin"][0]) and torch.equal(y[1], yref[0])


def test_predict_series_delegates_to_the_stitcher():
    model = _tiny_model()
    T = 3 * WIN + 17
    eeg, kin, emg = _series(T, seed=5)
    dev = torch.device("cpu")
    pred, target = predict_series(model, eeg, kin, emg, _pb, EMG_MEAN, EMG_STD,
                                  window=WIN, device=dev, batch_size=2)
    z = predict_series_stitched(model, eeg, kin, _pb, window=WIN, device=dev, batch_size=2)
    n = 3 * WIN                                   # default: tail dropped, as before
    assert pred.shape == (n, N_CH) and target.shape == (n, N_CH) and pred.dtype == np.float64
    # (window batches differ between the two calls -> float32 round-off only)
    np.testing.assert_allclose(pred, z[:n].astype(np.float64) * EMG_STD.numpy() + EMG_MEAN.numpy(),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(target, emg[:n], rtol=1e-5, atol=1e-6)
    full, full_t = predict_series(model, eeg, kin, emg, _pb, EMG_MEAN, EMG_STD,
                                  window=WIN, device=dev, batch_size=2, tail="align_end")
    assert full.shape == (T, N_CH) and full_t.shape == (T, N_CH)
    np.testing.assert_allclose(full[:n], pred, rtol=1e-5, atol=1e-6)


def test_diagnostic_plots_accept_stacked_model_windows():
    """The decomposition plots take (n_windows, window, ·) batches, so the notebook
    can feed them 1 s model windows instead of one long window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.plots import (plot_kin_skip_over_time, plot_neural_mechanical_latency_lag,
                            plot_residual_decomposition, plot_spectral_power_decomposition)

    model = _tiny_model()
    eeg, kin, emg = _series(4 * WIN, seed=6)
    inputs, _ = window_batch(eeg, kin, _pb, window=WIN, emg=emg)
    kw = dict(emg_mean=EMG_MEAN, emg_std=EMG_STD, fs=50)
    for fn, extra in ((plot_residual_decomposition, {"smooth_hz": None}),
                      (plot_kin_skip_over_time, {"smooth_hz": None}),
                      (plot_neural_mechanical_latency_lag, {"max_lag_ms": 100}),
                      (plot_spectral_power_decomposition, {})):
        fig = fn(model, inputs["eeg"], inputs["kin"], emg, **kw, **extra)
        lines = fig.axes[0].get_lines()
        if fn in (plot_residual_decomposition, plot_kin_skip_over_time):
            assert len(lines[0].get_xdata()) == 4 * WIN        # the whole stitched stretch
        plt.close(fig)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_interpretability_cell_uses_the_one_inference_path(name):
    nb = _nb(name)
    s = _cell_src(nb, "INTERPRETABILITY — Full-Signal Triptych")
    assert "predict_series_stitched(" in s and 'window=_cfg["dataset"]["window_size"]' in s
    for gone in ("ATTN_TEMP", "ATTN_SIGMA", "attention_temperature", "gaussian_filter1d"):
        assert gone not in s, gone
    assert "_bm(" not in s                       # no direct forward on a long window
    loso = _cell_src(nb, "[OPTIONAL] Automatic LOSOCV")
    assert "predict_series_stitched(" in loso
    for gone in ("attention_temperature", "gaussian_filter1d"):
        assert gone not in loso, gone
    assert "pred_b = _bm(eeg_b, kin_b)" not in loso
    # no cell sets an attention temperature or smooths the attention any more
    for c in _code_cells(nb):
        assert "attention_temperature" not in c


# ════════════════════════════════════════════════════════════════════════════
# 5. Gain-corrected metric
# ════════════════════════════════════════════════════════════════════════════

def _envelope(T=4000, C=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 500.0
    out = np.empty((T, C))
    for c in range(C):
        bursts = np.clip(np.sin(2 * np.pi * (0.3 + 0.1 * c) * t + rng.uniform(0, 6)), 0, None) ** 4
        out[:, c] = 0.01 + 0.05 * (c + 1) * bursts + 0.001 * rng.random(T)
    return out


def test_gain_columns_exist():
    for col in ("gain", "vaf_gain", "rmse_gain"):
        assert col in MAGNITUDE_COLUMNS


def test_gain_of_a_scaled_prediction():
    y = _envelope()
    df = magnitude_report(0.6 * y, y, channel_names=["a", "b", "c"]).set_index("muscle")
    m = df.loc[["a", "b", "c"]]
    np.testing.assert_allclose(m["gain"], 1.0 / 0.6, rtol=1e-12)
    np.testing.assert_allclose(m["vaf_gain"], 100.0, atol=1e-9)
    np.testing.assert_allclose(m["rmse_gain"], 0.0, atol=1e-12)
    assert (m["vaf"] < 100.0).all() and (m["rmse"] > 0).all()      # uncorrected metrics unchanged
    assert df.loc["MEAN", "gain"] == pytest.approx(1.0 / 0.6)


def test_gain_of_a_perfect_prediction_is_one():
    y = _envelope(seed=1)
    m = magnitude_report(y.copy(), y).iloc[:-1]
    np.testing.assert_allclose(m["gain"], 1.0, rtol=1e-12)
    np.testing.assert_allclose(m["vaf_gain"], m["vaf"], atol=1e-9)
    np.testing.assert_allclose(m["rmse_gain"], 0.0, atol=1e-12)


def test_gain_is_the_least_squares_scalar_and_never_hurts_rmse():
    rng = np.random.default_rng(2)
    y = _envelope(seed=2)
    p = 0.8 * y + 0.004 * rng.standard_normal(y.shape) + 0.002
    m = magnitude_report(p, y).iloc[:-1]
    g = (p * y).sum(0) / (p * p).sum(0)
    np.testing.assert_allclose(m["gain"], g, rtol=1e-12)
    np.testing.assert_allclose(m["rmse_gain"], np.sqrt(((g * p - y) ** 2).mean(0)), rtol=1e-12)
    assert (m["rmse_gain"].to_numpy() <= m["rmse"].to_numpy() + 1e-15).all()
    zero = magnitude_report(np.zeros_like(y), y).iloc[:-1]           # no prediction -> no gain
    assert zero["gain"].isna().all() and zero["vaf_gain"].isna().all()


def test_gain_tables():
    ys = {4: _envelope(T=3000, seed=4), 5: _envelope(T=3000, seed=5)}
    series = {4: (0.5 * ys[4], ys[4]), 5: (1.25 * ys[5], ys[5])}
    df = magnitude_report_by_series(series, {4: "W", 5: "B"}, channel_names=["a", "b", "c"])
    assert {"gain", "vaf_gain", "rmse_gain"} <= set(df.columns)
    full = format_magnitude_table(df)
    assert "gain" in full and "VAF%_gain" in full and "rmse_gain" in full
    txt = format_gain_table(df)
    lines = txt.splitlines()
    assert "series 4 (W)" in txt and "series 5 (B)" in txt and "all test" not in txt
    row4 = next(ln for ln in lines if ln.startswith("series 4 (W)"))
    row5 = next(ln for ln in lines if ln.startswith("series 5 (B)"))
    assert row4.split()[-4:] == ["2.000", "2.000", "2.000", "2.000"]      # a, b, c, MEAN
    assert row5.split()[-4:] == ["0.800", "0.800", "0.800", "0.800"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_magnitude_cell_prints_the_gain_table_as_an_oracle_diagnostic(name):
    s = _cell_src(_nb(name), "MAGNITUDE EVALUATION")
    assert "format_gain_table(" in s
    low = s.lower()
    assert "oracle" in low and "not a model result" in low and "test target" in low


# ════════════════════════════════════════════════════════════════════════════
# 3. Ablation cell placement / 4. early-stop patience
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,suffix", [("main.ipynb", ""), ("main_1stage.ipynb", "_1stage")])
def test_ablation_cell_right_after_magnitude(name, suffix):
    cells = _nb(name)["cells"]
    i = next(k for k, c in enumerate(cells) if "# EVALUATION — Full metric suite" in _src(c))
    assert "MAGNITUDE EVALUATION" in _src(cells[i + 1])
    abl = _src(cells[i + 2])
    assert cells[i + 2]["cell_type"] == "code"
    assert "HOW MUCH DOES THE MODEL USE EEG vs KINEMATICS?" in abl and "input_ablation(" in abl
    # Phase B: results go to RESULTS_DIR = results/<run tag>/ (tests/test_phase_b.py)
    assert f'f"{{RESULTS_DIR}}/input_ablation{suffix}.png"' in abl
    assert f'f"{{RESULTS_DIR}}/input_ablation{suffix}.csv"' in abl
    other = "_1stage" if not suffix else ""
    assert f'/input_ablation{other}.csv"' not in abl
    # Phase D: a markdown note on the interpretability panels sits between the two
    assert cells[i + 3]["cell_type"] == "markdown" and "what each panel is" in _src(cells[i + 3])
    assert "INTERPRETABILITY — Full-Signal Triptych" in _src(cells[i + 4])
    assert sum("input_ablation(" in s for s in _code_cells(_nb(name))) == 1


def test_1stage_ablation_cell_is_inherited_from_main():
    a = _cell_src(_nb("main.ipynb"), "input_ablation(")
    b = _cell_src(_nb("main_1stage.ipynb"), "input_ablation(")
    a_lines, b_lines = a.splitlines(), b.splitlines()
    assert len(a_lines) == len(b_lines)
    diff = [(x, y) for x, y in zip(a_lines, b_lines) if x != y]
    assert 0 < len(diff) <= 6
    for x, y in diff:                             # only output names / the title differ
        assert "1stage" in y or "single-stage" in y, (x, y)
    gen = (ROOT / "tools" / "build_main_1stage_notebook.py").read_text(encoding="utf-8")
    assert "ABLATION_CELL" not in gen             # no own copy in the generator any more


def test_main_stage1_has_its_own_patience():
    cfg = _config("main.ipynb")["training"]
    assert cfg["stage1_early_stop_patience"] == 10
    assert cfg["stage2_early_stop_patience"] == 25
    assert cfg["early_stop_patience"] == 50
    nb = _nb("main.ipynb")
    train = _cell_src(nb, "PER-SUBJECT TRAINING LOOP  (Stage 1 + Stage 2)")
    assert '_cfg_s1.early_stop_patience = _cfg["training"].get("stage1_early_stop_patience"' in train
    assert '_cfg_s2.early_stop_patience = _cfg["training"].get("stage2_early_stop_patience"' in train
    assert "smoothed" in train.lower() or "moving average" in train.lower()
    loso = _cell_src(nb, "[OPTIONAL] Automatic LOSOCV")
    assert '_s1_cfg.early_stop_patience = notebook_cfg["training"].get("stage1_early_stop_patience"' in loso


def test_1stage_patience_is_20_and_used_by_train_single_stage():
    cfg = _config("main_1stage.ipynb")["training"]
    assert cfg["single_stage"] == {"lr": 1e-3, "epochs": 200, "early_stop_patience": 20}
    assert cfg["early_stop_patience"] == 50       # shared key stays for other uses
    assert cfg["stage1_early_stop_patience"] == 10 and cfg["stage2_early_stop_patience"] == 25
    helper = _cell_src(_nb("main_1stage.ipynb"), "def train_single_stage(")
    assert 'tc.early_stop_patience = ss.get("early_stop_patience", tc.early_stop_patience)' in helper
    assert "smoothed" in helper.lower() or "moving average" in helper.lower()


def test_both_configs_identical_outside_training_extras():
    a, b = _config("main.ipynb"), _config("main_1stage.ipynb")
    for k in ("data", "preprocessing", "dataset", "model", "outputs"):
        assert a[k] == b[k], k
    assert a["training"]["loss"] == b["training"]["loss"]
    for k, v in a["training"].items():
        if k != "checkpoint_dir":                 # Phase B: each notebook has its own tagged folder
            assert b["training"][k] == v, k
    assert a["training"]["checkpoint_dir"] != b["training"]["checkpoint_dir"]
