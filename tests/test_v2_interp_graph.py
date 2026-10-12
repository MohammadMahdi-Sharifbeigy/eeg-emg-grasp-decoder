import numpy as np
import pytest
import torch

from v2.interp.flow import attention_flow, bin_matrix
from v2.interp.rollout import head_mean, layer_matrix, relevance_layer, relevance_matrix, rollout, with_residual


def _stochastic(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g).softmax(-1)


def test_layer_matrix_and_rollout_by_hand():
    a1 = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])          # one head: both nodes read node 0
    a2 = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])          # one head: both nodes read node 1
    m1, m2 = layer_matrix(a1, 0.5), layer_matrix(a2, 0.5)
    torch.testing.assert_close(m1, torch.tensor([[1.0, 0.0], [0.5, 0.5]]))
    torch.testing.assert_close(m2, torch.tensor([[0.5, 0.5], [0.0, 1.0]]))
    torch.testing.assert_close(rollout([m1, m2]), torch.tensor([[0.75, 0.25], [0.5, 0.5]]))
    torch.testing.assert_close(layer_matrix(a2, 0.0), a2[0])
    torch.testing.assert_close(head_mean(torch.stack([a1[0], a2[0]])), torch.full((2, 2), 0.5))


def test_rollout_matches_the_baseline_implementation():
    from main.interp import attention_rollout
    a = _stochastic(2, 3, 4, 20, 20)                        # (batch, layers, heads, T, T)
    ours = rollout([layer_matrix(a[:, i], 0.5) for i in range(3)])
    torch.testing.assert_close(ours, attention_rollout(a, residual=0.5), atol=1e-5, rtol=0)
    torch.testing.assert_close(ours.sum(-1), torch.ones(2, 20), atol=1e-5, rtol=0)


def test_residual_needs_a_square_map():
    with pytest.raises(ValueError, match="square"):
        with_residual(torch.rand(5, 7))


def test_relevance_matrix_and_layer_by_hand():
    a = torch.tensor([[[0.2, 0.8], [0.5, 0.5]]])            # (heads = 1, 2, 2)
    g = torch.tensor([[[1.0, 0.5], [-1.0, -2.0]]])
    torch.testing.assert_close(relevance_matrix(a, g), torch.tensor([[1 / 3, 2 / 3], [0.0, 0.0]]))
    torch.testing.assert_close(relevance_layer(a, g, 0.5), torch.tensor([[2 / 3, 1 / 3], [0.0, 0.5]]))
    zero = relevance_matrix(a, torch.zeros_like(a))         # no gradient: all zero, never NaN
    assert float(zero.abs().sum()) == 0 and torch.isfinite(zero).all()


def test_flow_of_one_layer_is_the_matrix():
    m = _stochastic(4, 6).numpy()
    np.testing.assert_allclose(attention_flow([m]), m, atol=2e-6)


def test_flow_by_hand_with_a_bottleneck():
    m1 = np.array([[0.5, 0.5], [0.5, 0.5]])
    m2 = np.array([[1.0, 0.0], [0.2, 0.8]])                 # sink 0 is fed by node 0 only
    np.testing.assert_allclose(attention_flow([m1, m2]), [[0.5, 0.5], [0.7, 0.7]], atol=2e-6)
    np.testing.assert_allclose(attention_flow([m1, m2], sinks=[1]), [[0.7, 0.7]], atol=2e-6)


def test_flow_is_never_below_rollout_for_row_stochastic_layers():
    mats = [layer_matrix(_stochastic(2, 12, 12, seed=s), 0.5).numpy() for s in range(3)]
    flow = attention_flow(mats)
    roll = rollout([torch.from_numpy(m) for m in mats]).numpy()
    assert flow.shape == (12, 12) and (flow >= roll - 1e-5).all() and (flow <= 1 + 1e-5).all()


def test_flow_rejects_layers_that_do_not_chain():
    with pytest.raises(ValueError, match="chain"):
        attention_flow([np.ones((3, 4)), np.ones((2, 5))])


def test_bin_matrix_keeps_rows_stochastic():
    m = _stochastic(10, 10).numpy()
    b = bin_matrix(m, 4)
    assert b.shape == (3, 3)
    np.testing.assert_allclose(b.sum(1), 1.0, atol=1e-6)
    np.testing.assert_allclose(b[0, 0], m[:4, :4].sum() / 4, atol=1e-6)
    np.testing.assert_allclose(b[2, 2], m[8:, 8:].sum() / 2, atol=1e-6)        # the last bin holds 2 positions
    np.testing.assert_allclose(bin_matrix(m, 1), m, atol=1e-7)


def test_flow_rejects_a_sink_outside_the_last_layer():
    m1 = np.array([[0.5, 0.5], [0.5, 0.5]])
    m2 = np.array([[1.0, 0.0], [0.2, 0.8]])
    for bad in (-1, 2, 5):                                  # no negative indexing, no node of another layer
        with pytest.raises(ValueError, match="sink"):
            attention_flow([m1, m2], sinks=[bad])
    with pytest.raises(ValueError, match="sink"):
        attention_flow([m1], sinks=[2])


def test_flow_rejects_an_empty_list_of_layers():
    with pytest.raises(ValueError, match="at least one layer"):
        attention_flow([])


def test_flow_rejects_capacities_that_are_not_attention_weights():
    ok = np.array([[0.5, 0.5], [0.5, 0.5]])
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            attention_flow([ok, np.array([[bad, 0.5], [0.5, 0.5]])])
    for bad in (3000.0, 1.001):                             # 3000 used to overflow int32 and answer 0
        with pytest.raises(ValueError, match="above 1"):
            attention_flow([np.array([[bad, 0.5], [0.5, 0.5]]), ok])
    np.testing.assert_allclose(attention_flow([np.array([[1.0 + 5e-7, 0.0], [0.5, 0.5]])])[0, 0], 1.0, atol=2e-6)


def test_bin_matrix_rejects_a_size_below_one():
    for bad in (0, -3):
        with pytest.raises(ValueError, match="size"):
            bin_matrix(np.ones((4, 4)) / 4, bad)


def test_rollout_rejects_an_empty_list():
    with pytest.raises(ValueError, match="at least one layer"):
        rollout([])


def test_with_residual_returns_a_detached_float32_tensor():
    x = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
    out = with_residual(x, 0.5)
    assert out.dtype == torch.float32 and not out.requires_grad
    torch.testing.assert_close(out, (0.5 * x + 0.5 * torch.eye(3, dtype=torch.float64)).detach().float())
