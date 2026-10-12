"""Where one output value of KG-GT v2 came from: one output, one muscle, one token t of one window.

Views, each covering every layer between the inputs and that value:
  rollout    plain attention rollout (the same for every muscle)
  flow       attention flow (maximum flow), on time bins
  relevance  gradient-weighted rollout, specific to the muscle and the token
  layers     the attention of every single layer and head at token t, not composed
  saliency   gradient x input on the three model inputs (reaches the layers that have no attention:
             the stem, the band embedding, the history gain)
plus the exact parts of the output (K = K_base * gain, F = relu(K + F_eeg)).

Stages:  K, K_base : stem -> kin temporal layers -> kin muscle graph
         E         : electrode graph -> pooling -> eeg temporal layers -> eeg muscle graph
         F, F_eeg  : electrode graph -> pooling -> eeg temporal layers -> cross-attention
"""
import numpy as np
import torch

from ..attention import recording
from .flow import attention_flow, bin_matrix
from .rollout import head_mean, layer_matrix, relevance_layer, relevance_matrix, rollout

STAGES = {"K": "K", "K_base": "K", "E": "E", "F": "F", "F_eeg": "F"}
PARTS = ("K", "E", "F", "K_base", "gain", "F_eeg")


def trace(model, kin, hist, eeg, grad: bool = False):
    """One recorded forward pass in eval mode -> (outputs, tape)."""
    model.eval()
    with recording(model, grad) as tape, torch.set_grad_enabled(grad):
        out = model(kin, hist, eeg)
    return out, tape


def _stack(tape, names, residual, grads=None):
    """Layer matrices of one stage of residual layers: plain, or gradient-weighted when grads is given."""
    if grads is None:
        return [layer_matrix(tape.attn[n], residual) for n in names]
    return [relevance_layer(tape.attn[n], grads[n], residual) for n in names]


def _single(tape, name, grads=None):
    """A layer without a skip connection (electrode pooling, cross-attention)."""
    return head_mean(tape.attn[name]) if grads is None else relevance_matrix(tape.attn[name], grads[name])


def attention_maps(model, tape, kind, t, residual=0.5, grads=None, muscle=None) -> dict:
    """Maps into token t through every layer of the stages of `kind` ("K", "E" or "F").

    grads=None: plain rollout. With grads ({layer name: d target / d attention}): gradient-weighted
    rollout for that target; its maps sum to at most 1 and are all zero when the target has no gradient.
      time      (B, T)      share of each input token
      electrode (B, T, E)   share of each electrode in the pooled EEG token of each time step     [E, F]
      input     (B, T, E)   time x electrode                                                      [E, F]
      muscle    (B, M, M)   row m = share of each muscle node in muscle m's node at token t       [K, E]
                            ((B, M), the row of the explained muscle, when `muscle` is given)
    """
    L = model.attention_layers()
    maps = {}
    if kind == "K":
        r = rollout(_stack(tape, L["kin"]["temporal"], residual, grads))[:, t]              # (B, T)
        maps["time"] = r @ model.kin.stem_matrix(r.shape[-1]).to(r)
    else:
        r = rollout(_stack(tape, L["eeg"]["temporal"], residual, grads))                    # (B, T, T)
        if kind == "F":
            r = _single(tape, L["fusion"]["cross"], grads) @ r
        maps["time"] = r[:, t]
        graph = rollout(_stack(tape, L["eeg"]["electrode"], residual, grads))               # (B, T, E, E)
        maps["electrode"] = (_single(tape, L["eeg"]["pool"], grads) @ graph).squeeze(-2)    # (B, T, E)
        maps["input"] = maps["time"][..., None] * maps["electrode"]
    if kind != "F":
        m = rollout(_stack(tape, L["kin" if kind == "K" else "eeg"]["muscle"], residual, grads))[:, t]
        maps["muscle"] = m if muscle is None else m[:, muscle]
    return maps


def layer_views(model, tape, kind, t) -> dict:
    """What every single layer and head did at token t (nothing composed).

      time_layers      (B, L, H, T)      attention of token t in each temporal layer
      electrode_layers (B, L, H, E, E)   electrode-graph attention at token t                     [E, F]
      pool             (B, E)            pooling weights at token t                               [E, F]
      cross            (B, H, T)         cross-attention of token t                               [F]
      muscle_layers    (B, L, H, M, M)   muscle-graph attention at token t                        [K, E]
    """
    L = model.attention_layers()
    a = {n: v.detach().float() for n, v in tape.attn.items()}
    s = "kin" if kind == "K" else "eeg"
    out = {"time_layers": torch.stack([a[n][:, :, t] for n in L[s]["temporal"]], dim=1)}
    if kind != "K":
        out["electrode_layers"] = torch.stack([a[n][:, t] for n in L["eeg"]["electrode"]], dim=1)
        out["pool"] = a[L["eeg"]["pool"]][:, t, 0, 0]
    if kind == "F":
        out["cross"] = a[L["fusion"]["cross"]][:, :, t]
    else:
        out["muscle_layers"] = torch.stack([a[n][:, t] for n in L[s]["muscle"]], dim=1)
    return out


def _bin_time(x: np.ndarray, size: int) -> np.ndarray:
    """Mean over consecutive time bins along axis 0 (a last partial bin is kept)."""
    edges = np.arange(0, x.shape[0], size)
    counts = np.diff(np.append(edges, x.shape[0])).reshape(-1, *([1] * (x.ndim - 1)))
    return np.add.reduceat(x, edges, axis=0) / counts


def flow_maps(model, tape, kind, t, residual=0.5, bin_size=4, b=0) -> dict:
    """Attention flow into token t for batch item b, on time bins of bin_size tokens (NumPy).

      time      (bins,)     maximum flow from each time bin
      electrode (bins, E)   maximum flow from each electrode into the pooled token of that bin    [E, F]
      input     (bins, E)   min(time, electrode): the pooled token is the only way out of a time
                            step's electrode graph, so the flow through it is the smaller of the two
    """
    L = model.attention_layers()
    item = lambda x: x[b].cpu().numpy().astype(np.float64)
    sink = [int(t) // bin_size]
    if kind == "K":
        mats = [item(m) for m in _stack(tape, L["kin"]["temporal"], residual)]
        mats.insert(0, model.kin.stem_matrix(mats[0].shape[-1]).numpy())
        return {"time": attention_flow([bin_matrix(m, bin_size) for m in mats], sink)[0]}
    mats = [item(m) for m in _stack(tape, L["eeg"]["temporal"], residual)]
    if kind == "F":
        mats.append(item(_single(tape, L["fusion"]["cross"])))
    time = attention_flow([bin_matrix(m, bin_size) for m in mats], sink)[0]
    graph = [_bin_time(item(m), bin_size) for m in _stack(tape, L["eeg"]["electrode"], residual)]   # each (bins, E, E)
    pool = _bin_time(item(_single(tape, L["eeg"]["pool"])), bin_size)                               # (bins, 1, E)
    electrode = np.stack([attention_flow([m[s] for m in graph] + [pool[s]])[0] for s in range(len(pool))])
    return {"time": time, "electrode": electrode, "input": np.minimum(time[:, None], electrode)}


def explain(model, kin, hist, eeg, output="E", muscle=0, t=None, residual=0.5, flow=False, bin_size=4) -> dict:
    """Every view for one target: out[output][:, t, muscle]. t=None is the window centre."""
    if output not in STAGES:
        raise ValueError(f"output must be one of {sorted(STAGES)}, got {output!r}")
    kind = STAGES[output]
    kin, hist, eeg = (x.detach().clone().requires_grad_(True) for x in (kin, hist, eeg))
    out, tape = trace(model, kin, hist, eeg, grad=True)
    t = kin.shape[1] // 2 if t is None else int(t)
    target = out[output][:, t, muscle]
    names = list(tape.attn)
    g = torch.autograd.grad(target.sum(), [tape.attn[n] for n in names] + [kin, hist, eeg], allow_unused=True)
    filled = lambda gx, x: torch.zeros_like(x) if gx is None else gx          # not on the path to the target
    grads = {n: filled(gx, tape.attn[n]) for n, gx in zip(names, g)}
    res = {
        "target": target.detach(),
        "parts": {k: out[k][:, t, muscle].detach() for k in PARTS},
        "rollout": attention_maps(model, tape, kind, t, residual),
        "relevance": attention_maps(model, tape, kind, t, residual, grads, muscle),
        "saliency": {k: (x * filled(gx, x)).detach()
                     for k, x, gx in zip(("kin", "hist", "eeg"), (kin, hist, eeg), g[len(names):])},
        "layers": layer_views(model, tape, kind, t),
    }
    if flow:
        res["flow"] = flow_maps(model, tape, kind, t, residual, bin_size)
    return res
