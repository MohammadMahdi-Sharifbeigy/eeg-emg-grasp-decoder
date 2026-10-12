"""Attention flow (Abnar & Zuidema 2020): maximum flow through the layered attention graph.

Rollout multiplies the layers, so a path counts by the product of its weights. Flow asks how much
can be routed from an input node to an output node when every attention weight is a pipe capacity;
for row-stochastic layers it is never below rollout.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

_SCALE = 1_000_000          # scipy's maximum_flow needs integer capacities


def bin_matrix(m, size: int) -> np.ndarray:
    """Coarsen (n_q, n_s) by `size` on both axes: source positions summed, query positions averaged."""
    m = np.asarray(m, dtype=np.float64)
    if size < 1:
        raise ValueError(f"the bin size must be at least 1, got {size}")
    if size == 1:
        return m
    q_edges, s_edges = np.arange(0, m.shape[0], size), np.arange(0, m.shape[1], size)
    out = np.add.reduceat(np.add.reduceat(m, s_edges, axis=1), q_edges, axis=0)
    return out / np.diff(np.append(q_edges, m.shape[0]))[:, None]


def attention_flow(mats, sinks=None) -> np.ndarray:
    """Maximum flow from every input node to each sink node.

    mats:  forward-ordered list (at least one layer) of (n_{l+1}, n_l) matrices; entry [i, j] is the
           capacity of the edge from node j of layer l to node i of layer l + 1. The function is defined
           for attention-like weights in [0, 1]: every entry must be finite and at most 1 (+ 1e-6 for
           rounding), otherwise ValueError; negative entries count as 0.
    sinks: indices into the last layer, each in [0, n_last) (default: all of them); negative indices
           are not accepted.
    Returns (n_sinks, n_0): flow[k, j] = maximum flow from input node j to sink k.
    """
    mats = [np.asarray(m, dtype=np.float64) for m in mats]
    if not mats:
        raise ValueError("attention_flow needs at least one layer, got an empty list")
    sizes = [mats[0].shape[-1]] + [m.shape[0] for m in mats]
    for i, m in enumerate(mats):
        if m.ndim != 2 or m.shape[1] != sizes[i]:
            raise ValueError(f"layers do not chain: layer {i} has shape {m.shape}, expected (*, {sizes[i]})")
        if not np.isfinite(m).all():
            raise ValueError(f"layer {i} has capacities that are not finite")
        if m.size and m.max() > 1.0 + 1e-6:
            raise ValueError(f"layer {i} has a capacity above 1 ({m.max():g}); expected weights in [0, 1]")
    mats = [np.clip(m, 0.0, None) for m in mats]
    sinks = list(range(sizes[-1])) if sinks is None else [int(s) for s in sinks]
    for s in sinks:
        if not 0 <= s < sizes[-1]:
            raise ValueError(f"sink index {s} is outside the last layer [0, {sizes[-1]})")
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)        # first node id of every layer
    rows, cols, caps = [], [], []
    for i, m in enumerate(mats):
        c = np.rint(m * _SCALE).astype(np.int32)
        dst, src = np.nonzero(c)
        rows.append(offs[i] + src)
        cols.append(offs[i + 1] + dst)
        caps.append(c[dst, src])
    n = int(offs[-1])
    graph = csr_matrix((np.concatenate(caps), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n), dtype=np.int32)
    out = np.zeros((len(sinks), sizes[0]))
    for k, s in enumerate(sinks):
        for j in range(sizes[0]):
            out[k, j] = maximum_flow(graph, j, int(offs[-2]) + s).flow_value / _SCALE
    return out
