import pytest
import torch

from v2.config import CONFIG_V2
from v2.interp.explain import explain, trace
from v2.model import build_model

T = 48


def _inputs(seed=0, b=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, T, 24, generator=g), torch.rand(b, T, 8, generator=g), torch.randn(b, T, 32, 4, generator=g)


def _model(gate=0.5):
    torch.manual_seed(0)
    m = build_model(CONFIG_V2).eval()
    with torch.no_grad():
        m.fusion.gate.fill_(gate)
    return m


def _column_bias(n, j):
    bias = torch.zeros(n, n)
    bias[:, j] = 1e4
    return bias


def test_trace_runs_in_eval_mode_and_matches_a_plain_forward():
    m = _model().train()
    out, tape = trace(m, *_inputs())
    assert not m.training and len(tape.attn) == 13 and not out["E"].requires_grad
    with torch.no_grad():
        plain = m(*_inputs())
    assert all(torch.equal(out[k], plain[k]) for k in ("K", "E", "F"))


@pytest.mark.parametrize("output,n_layers", [("K", 3), ("E", 2), ("F", 2)])
def test_views_have_the_documented_shapes(output, n_layers):
    res = explain(_model(), *_inputs(b=2), output=output, muscle=1, t=20, flow=True, bin_size=4)
    ro, re, la, fl = res["rollout"], res["relevance"], res["layers"], res["flow"]
    assert res["target"].shape == (2,) and set(res["parts"]) == {"K", "E", "F", "K_base", "gain", "F_eeg"}
    assert ro["time"].shape == re["time"].shape == (2, T) and la["time_layers"].shape == (2, n_layers, 4, T)
    torch.testing.assert_close(ro["time"].sum(-1), torch.ones(2), atol=1e-4, rtol=0)
    assert (re["time"] >= 0).all() and (re["time"].sum(-1) <= 1 + 1e-4).all()
    assert fl["time"].shape == (T // 4,) and (fl["time"] <= 1 + 1e-5).all()
    assert {k: tuple(v.shape) for k, v in res["saliency"].items()} == {"kin": (2, T, 24), "hist": (2, T, 8), "eeg": (2, T, 32, 4)}
    if output == "K":
        assert set(ro) == {"time", "muscle"} and set(fl) == {"time"}
    else:
        assert ro["electrode"].shape == re["electrode"].shape == ro["input"].shape == re["input"].shape == (2, T, 32)
        torch.testing.assert_close(ro["electrode"].sum(-1), torch.ones(2, T), atol=1e-4, rtol=0)
        torch.testing.assert_close(ro["input"].sum((-1, -2)), torch.ones(2), atol=1e-4, rtol=0)
        assert fl["electrode"].shape == fl["input"].shape == (T // 4, 32) and (fl["input"] <= fl["electrode"] + 1e-9).all()
        assert la["electrode_layers"].shape == (2, 2, 4, 32, 32) and la["pool"].shape == (2, 32)
    if output == "F":
        assert "muscle" not in ro and "muscle_layers" not in la and la["cross"].shape == (2, 4, T)
    else:
        assert ro["muscle"].shape == (2, 5, 5) and re["muscle"].shape == (2, 5) and la["muscle_layers"].shape == (2, 2, 4, 5, 5)
    assert all(torch.isfinite(v).all() for v in list(ro.values()) + list(re.values()))


def test_rollout_is_shared_by_the_muscles_and_relevance_is_not():
    m = _model()
    a, b = (explain(m, *_inputs(), output="E", muscle=mu, t=20) for mu in (0, 3))
    assert torch.equal(a["rollout"]["time"], b["rollout"]["time"]) and torch.equal(a["rollout"]["input"], b["rollout"]["input"])
    assert not torch.allclose(a["relevance"]["input"], b["relevance"]["input"], atol=1e-6)
    assert float(a["relevance"]["time"][0, 20]) >= 0.25 - 1e-5           # the skip path of two layers: 0.5 * 0.5


def test_forced_attention_shows_up_in_rollout_flow_relevance_and_saliency():
    m = _model()
    for name in m.attention_layers()["eeg"]["temporal"]:
        att = m.get_submodule(name)                                       # every EEG token reads token 7
        att.forward = lambda q, context=None, bias=None, _f=att.forward: _f(q, context, _column_bias(q.shape[-2], 7))
    res = explain(m, *_inputs(), output="E", muscle=0, t=20, flow=True, bin_size=1)
    time = res["rollout"]["time"][0]
    assert float(time[7]) == pytest.approx(0.75, abs=1e-4) and float(time[20]) == pytest.approx(0.25, abs=1e-4)
    flow = res["flow"]["time"]
    assert flow[7] == pytest.approx(1.0, abs=1e-4) and flow[20] == pytest.approx(0.5, abs=1e-4)
    assert flow[[0, 30]].max() < 1e-4
    rel = res["relevance"]["time"][0]
    assert float(rel[7] + rel[20]) == pytest.approx(float(rel.sum()), abs=1e-5) and float(rel[20]) >= 0.25 - 1e-5
    sal = res["saliency"]["eeg"][0].abs().sum((-1, -2))                   # (T,)
    assert float(sal[[7, 20]].sum()) == pytest.approx(float(sal.sum()), rel=1e-4) and float(sal.sum()) > 0


def test_k_has_no_eeg_saliency_and_a_closed_gate_gives_zero_maps():
    closed = _model(gate=0.0)
    k = explain(closed, *_inputs(), output="K", muscle=2, t=20)
    assert float(k["saliency"]["eeg"].abs().sum()) == 0 and float(k["saliency"]["kin"].abs().sum()) > 0
    for output in ("F", "F_eeg"):
        f = explain(closed, *_inputs(), output=output, muscle=2, t=20)
        assert float(f["relevance"]["time"].abs().sum()) == 0 and float(f["relevance"]["input"].abs().sum()) == 0
        assert float(f["saliency"]["eeg"].abs().sum()) == 0 and all(torch.isfinite(v).all() for v in f["relevance"].values())
    opened = explain(_model(gate=0.5), *_inputs(), output="F_eeg", muscle=2, t=20)
    assert float(opened["relevance"]["input"].sum()) > 0 and float(opened["saliency"]["eeg"].abs().sum()) > 0
    assert float(opened["parts"]["F_eeg"].abs()) > 0 and float(opened["saliency"]["kin"].abs().sum()) == 0


def test_unknown_output_is_rejected():
    with pytest.raises(ValueError, match="output must be one of"):
        explain(_model(), *_inputs(), output="G")
