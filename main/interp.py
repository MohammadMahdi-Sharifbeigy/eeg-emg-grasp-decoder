"""main/interp.py — interpretability that says what the model actually does.

The KG-GT output is

    out = lag( decoder( gat(nodes(EEG), kin_gat) + kin_skip(kin_skip_features) ) )

and both the per-muscle decoder (a linear layer on the node embedding plus a linear
"slope" layer on its time difference) and the lag convolution (no bias) are linear.
So the output splits EXACTLY into two additive paths:

    EEG / graph path   lag(decoder(gat_out))                    (holds the decoder bias)
    kinematic path     lag(decoder(kin_skip) − decoder(0))

``decompose_prediction`` returns both (they sum to the model output up to float
round-off) plus three OCCLUSION effects, which are not exact shares:

    graph_no_kin_bias     the graph path recomputed with the kinematic edge bias of the
                          GAT set to 0 (the graph path still depends on kinematics
                          through that bias; this variant is the EEG-only part)
    kin_force_effect      kin path − kin path with the force / force-rate inputs replaced
    kin_velocity_effect   kin path − kin path with the pose-velocity inputs replaced
    (the kin skip MLP is non-linear, so the two effects need not add up to the kin path)

Other tools:
    lift_events / event_samples / extract_event_windows / phase_levels   task events
    attention_rollout / encoder_attention / rollout_rows                 within-window rollout
    eeg_occlusion_attribution / deletion_curve / eeg_effect_size         does EEG matter?
    event_locked_windows / concat_event_locked / event_locked_summary    all test lifts

Attention exists only WITHIN one model window (the training window, 1 s); nothing here
crosses a window border.  All model inputs are the NORMALISED tensors returned by the
notebook's ``prepare_batch`` (``main.inference.window_batch``).
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from .ablation import shuffle_time
from .inference import stitch_windows, window_batch, window_starts

__all__ = [
    "EVENT_COLUMNS", "EVENT_LABELS", "lift_events", "event_samples",
    "extract_event_windows", "phase_levels",
    "kin_column_groups", "raw_zero_kin_baseline", "series_gate", "eeg_channel_names",
    "decompose_prediction", "decompose_series", "DECOMPOSITION_KEYS",
    "attention_rollout", "encoder_attention", "rollout_rows",
    "eeg_occlusion_attribution", "deletion_curve", "eeg_effect_size",
    "event_locked_windows", "concat_event_locked", "event_locked_summary",
]


# ════════════════════════════════════════════════════════════════════════════
# 1. Task events (P{p}_AllLifts.mat)
# ════════════════════════════════════════════════════════════════════════════

# event key -> AllLifts column (times in seconds on the lift's own clock)
EVENT_COLUMNS: dict[str, str] = {
    "led_on": "LEDOn",
    "hand_start": "tHandStart",
    "first_touch": "tFirstDigitTouch",
    "both_touch": "tBothDigitTouch",
    "lift_off": "tLiftOff",
    "replace": "tReplace",
    "release": "tBothReleased",
    "hand_stop": "tHandStop",
    "led_off": "LEDOff",
}

EVENT_LABELS: dict[str, str] = {
    "led_on": "LED on",
    "hand_start": "hand start",
    "first_touch": "first digit touch",
    "both_touch": "both digits touch",
    "lift_off": "lift-off",
    "replace": "replace",
    "release": "release",
    "hand_stop": "hand stop",
    "led_off": "LED off",
}


def lift_events(
    participant: int,
    series: int,
    fs: float = 500.0,
    data_dir=None,
    n_samples: int | None = None,
) -> list[dict]:
    """Task events of every lift of one series, as sample indices in series time.

    Time base (the one validated with ``main.magnitude_eval.lift_hold_segments``):
        t0            = StartTime − LEDOn
        t(event)      = t0 + t_event              (seconds from the start of the series)
        sample(event) = round(t(event) · fs)

    Args:
        participant, series: e.g. 1, 4.
        fs:        sampling rate of the arrays the samples index (500 Hz for the cache).
        data_dir:  raw data root holding ``P{p}/P{p}_AllLifts.mat`` (default: repo data/way-eeg/raw).
        n_samples: series length; events outside [0, n_samples) become None.

    Returns:
        One dict per lift, in AllLifts order:
            {"lift", "CurW", "CurS", "t0",
             "led_on", "hand_start", "first_touch", "both_touch", "lift_off",
             "replace", "release", "hand_stop", "led_off"}
        Event values are int sample indices, or None when the time is missing (NaN),
        the column does not exist, or the sample falls outside the series.
        CurW ∈ {1, 2, 4} (weight), CurS ∈ {1, 2, 3} (surface).
    """
    from .dataloader import _alllifts_path, _mat_load, _names_to_list

    path = _alllifts_path(participant, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"AllLifts file not found: {path}")
    P = _mat_load(path)["P"]
    cols = _names_to_list(P.ColNames)
    lifts = np.atleast_2d(np.asarray(P.AllLifts, dtype=float))
    for required in ("Run", "Lift", "StartTime", "LEDOn"):
        if required not in cols:
            raise KeyError(f"{path.name}: column {required!r} missing")
    col = {n: cols.index(n) for n in cols}

    def _int(row, name):
        v = row[col[name]] if name in col else np.nan
        return int(v) if np.isfinite(v) else None

    out = []
    for row in lifts[lifts[:, col["Run"]].astype(int) == int(series)]:
        t0 = row[col["StartTime"]] - row[col["LEDOn"]]
        ev = {"lift": _int(row, "Lift"), "CurW": _int(row, "CurW"), "CurS": _int(row, "CurS"),
              "t0": float(t0)}
        for key, name in EVENT_COLUMNS.items():
            t = t0 + row[col[name]] if name in col else np.nan
            s = int(round(float(t) * fs)) if np.isfinite(t) else None
            if s is not None and (s < 0 or (n_samples is not None and s >= n_samples)):
                s = None
            ev[key] = s
        out.append(ev)
    return out


def event_samples(events: Sequence[Mapping], name: str) -> list:
    """Sample index of event ``name`` for every lift (None where missing)."""
    if name not in EVENT_COLUMNS:
        raise KeyError(f"unknown event {name!r}; choose from {sorted(EVENT_COLUMNS)}")
    return [e.get(name) for e in events]


def extract_event_windows(x: np.ndarray, samples: Sequence, pre: int = 250, post: int = 250):
    """Cut ``x[s − pre : s + post]`` around every event sample ``s``.

    Events that are None or whose window does not fit inside the array are skipped.

    Returns:
        (windows, kept): ``windows`` (n_kept, pre + post, …) — index ``pre`` is the event
        sample; ``kept`` (n_kept,) int64 positions into ``samples``.
    """
    x = np.asarray(x)
    pre, post = int(pre), int(post)
    kept = [i for i, s in enumerate(samples)
            if s is not None and s - pre >= 0 and s + post <= len(x)]
    if kept:
        windows = np.stack([x[samples[i] - pre: samples[i] + post] for i in kept])
    else:
        windows = np.empty((0, pre + post) + x.shape[1:], dtype=x.dtype)
    return windows, np.asarray(kept, dtype=np.int64)


def phase_levels(env: np.ndarray, events: Sequence[Mapping], fs: float = 500.0) -> dict:
    """Mean level of an envelope in the task phases of every lift (sanity check, no model).

    Phases (per lift, series time):
        rest_before   LED on − 1 s … LED on            (hand parked, before the cue)
        reach         hand start … first digit touch
        load          first digit touch … lift-off
        hold          lift-off … replace
        release       replace … both digits released
        rest_after    release + 0.5 s … release + 1.5 s

    Args:
        env:    (T,) or (T, C) non-negative envelope sampled at ``fs``.
        events: output of ``lift_events`` with samples at the same ``fs``.

    Returns:
        {phase: (n_lifts, C) array}, NaN where a phase is missing or empty.
    """
    env = np.asarray(env, dtype=np.float64)
    if env.ndim == 1:
        env = env[:, None]
    one = int(round(fs))
    half = int(round(0.5 * fs))

    def _bounds(e):
        led, hs, ft = e.get("led_on"), e.get("hand_start"), e.get("first_touch")
        lo, rp, rl = e.get("lift_off"), e.get("replace"), e.get("release")
        return {
            "rest_before": (None if led is None else led - one, led),
            "reach": (hs, ft),
            "load": (ft, lo),
            "hold": (lo, rp),
            "release": (rp, rl),
            "rest_after": (None if rl is None else rl + half, None if rl is None else rl + half + one),
        }

    names = ("rest_before", "reach", "load", "hold", "release", "rest_after")
    out = {n: np.full((len(events), env.shape[1]), np.nan) for n in names}
    for i, e in enumerate(events):
        for n, (a, b) in _bounds(e).items():
            if a is None or b is None or a < 0 or b > len(env) or b - a < 1:
                continue
            out[n][i] = env[a:b].mean(axis=0)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Small helpers around the notebook objects
# ════════════════════════════════════════════════════════════════════════════

def kin_column_groups(names: Sequence[str]) -> dict[str, list[int]]:
    """Indices of the force and pose-velocity inputs in the model's kin vector.

    ``names`` = ``main.preprocessing_emg_kin.kin_feature_names(...)`` of the config.
        force     gated forces and their rates  (names containing F_L / F_G)
        velocity  pose velocities               (names starting with v_, not a force)
    """
    names = list(names)
    force = [i for i, n in enumerate(names) if "F_L" in n or "F_G" in n]
    velocity = [i for i, n in enumerate(names) if n.startswith("v_") and i not in force]
    return {"force": force, "velocity": velocity}


def raw_zero_kin_baseline(prepare_batch: Callable, n_eeg: int, n_kin_raw: int, n_emg: int) -> torch.Tensor:
    """Model-input (normalised) value of a kinematic feature that is 0 in physical units.

    A zero raw kin row is sent through ``prepare_batch``; the result is the (K,) vector
    ``(0 − kin_mean) / kin_std`` after the notebook's column drop.  Used as the occlusion
    baseline "no force / not moving" (0 N, 0 cm/s), which is not 0 in z-space.
    """
    inputs, _ = prepare_batch(torch.zeros(1, 1, n_eeg), torch.zeros(1, 1, n_kin_raw),
                              torch.zeros(1, 1, n_emg))
    return inputs["kin"][0, 0].detach().float().cpu()


def series_gate(participant: int, series, data_dir=None, fs: float = 500.0,
                n_samples: int | None = None, gate_kwargs: Mapping | None = None):
    """Hand-near-object gate (T,) in [0, 1] of one series, from the RAW kin of its HS file.

    Returns None when the HS file is not there (the figures then omit the shading).
    ``gate_kwargs`` = CONFIG["preprocessing"]["kinematics"]["gated_force"] (dict / True / None).
    """
    from .dataloader import DEFAULT_RAW_DIR, load_hs
    from .preprocessing_emg_kin import hand_near_object_gate

    root = Path(data_dir) if data_dir is not None else DEFAULT_RAW_DIR
    path = root / f"P{int(participant)}" / f"HS_P{int(participant)}_S{series}.mat"
    if not path.exists():
        return None
    kw = dict(gate_kwargs) if isinstance(gate_kwargs, Mapping) else {}
    gate = hand_near_object_gate(load_hs(path)["kin"], fs, **kw)[:, 0]
    if n_samples is not None:
        if len(gate) < n_samples:
            gate = np.concatenate([gate, np.zeros(n_samples - len(gate), dtype=gate.dtype)])
        gate = gate[:n_samples]
    return gate


@functools.lru_cache(maxsize=None)
def _eeg_names_cached(participant: int, data_dir_str: str) -> tuple[str, ...]:
    from .dataloader import _mat_load, _names_to_list

    p_dir = Path(data_dir_str) / f"P{participant}"
    files = sorted(p_dir.glob(f"HS_P{participant}_S*.mat"))
    if not files:
        raise FileNotFoundError(f"No HS files found for P{participant} in {p_dir}")
    return tuple(_names_to_list(_mat_load(files[0])["hs"].eeg.names))


def eeg_channel_names(participant, data_dir=None) -> list[str]:
    """EEG electrode names of a participant (``hs.eeg.names`` = ``load_hs(...)["eeg_names"]``)."""
    from .dataloader import DEFAULT_RAW_DIR, _participant_number

    key = str(data_dir if data_dir is not None else DEFAULT_RAW_DIR)
    return list(_eeg_names_cached(_participant_number(participant), key))


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _as_batch(x, device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32).to(device)
    return t.unsqueeze(0) if t.dim() == 2 else t


def _call(model, eeg, kin):
    return model(eeg=eeg, kin=kin).float()


# ════════════════════════════════════════════════════════════════════════════
# 2. Exact path decomposition
# ════════════════════════════════════════════════════════════════════════════

DECOMPOSITION_KEYS = ("total", "graph", "kin", "graph_no_kin_bias",
                      "kin_force_effect", "kin_velocity_effect")


@contextlib.contextmanager
def _zero_kin_edge_bias(gat):
    """Inside the block the kinematic edge bias of the GAT is 0 (weights untouched)."""
    lin = getattr(gat, "kin_edge_linear", None)
    if lin is None:
        yield
        return
    handle = lin.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out))
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def decompose_prediction(
    model: torch.nn.Module,
    eeg,
    kin=None,
    *,
    force_cols: Sequence[int] | None = None,
    vel_cols: Sequence[int] | None = None,
    kin_baseline=None,
    as_numpy: bool = True,
) -> dict:
    """Split the model output of one batch of windows into its two additive paths.

    Args:
        model: a ``KGGTModel`` (with or without kinematic guidance); put in eval mode.
        eeg:   (B, W, C_eeg) or (W, C_eeg) normalised model input, W = model window.
        kin:   (B, W, K) normalised model kin input (ignored without kinematic guidance).
        force_cols / vel_cols: indices into the K kin inputs (``kin_column_groups``);
               None / empty -> the corresponding effect is 0.
        kin_baseline: (K,) value the occluded columns are set to (default 0 = the
               per-feature training mean in z-space; ``raw_zero_kin_baseline`` gives
               "0 N / not moving").
        as_numpy: return float32 numpy arrays (default) or CPU tensors.

    Returns (all AFTER lag alignment, (B, W, C), normalised output units):
        "total"               model output                    = graph + kin   (exact)
        "graph"               EEG / graph path  lag(decoder(gat_out)), holds the decoder bias
        "kin"                 kinematic skip path  lag(decoder(kin_skip) − decoder(0))
        "graph_no_kin_bias"   OCCLUSION: graph path with the GAT's kinematic edge bias = 0
        "kin_force_effect"    OCCLUSION: kin − kin(force inputs -> baseline)
        "kin_velocity_effect" OCCLUSION: kin − kin(pose-velocity inputs -> baseline)
        "decoder_bias"        lag(decoder(0)), the constant inside "graph"
        "gat_attn"            (B, W, N, N) muscle-graph attention AFTER softmax of the
                              unmodified forward pass, heads averaged ([receiving, source]);
                              None if the model does not expose it
    Without kinematic guidance: "kin" and both effects are 0 and
    "graph_no_kin_bias" = "graph" = "total".
    """
    for attr in ("encoder", "node_projection", "gat", "decoder"):
        if not hasattr(model, attr):
            raise TypeError(f"decompose_prediction needs a KGGTModel-like model (no .{attr})")
    model.eval()
    dev = _device_of(model)
    eeg = _as_batch(eeg, dev)
    B, W = eeg.shape[:2]
    lag = getattr(model, "lag_align", None)

    def _lag(x):
        return lag(x) if lag is not None else x

    def _dec(x):
        return model.decoder(x).squeeze(-1)

    nodes = model.node_projection(model.encoder(eeg))
    guided = bool(getattr(model, "use_kinematic_guidance", False))
    gat_attn = None

    if guided:
        if kin is None:
            raise ValueError("kin is required when use_kinematic_guidance=True")
        kin = _as_batch(kin, dev)
        gat_kin = model.get_gat_kin(kin)
        gat_out = model.gat(nodes, gat_kin)
        raw_attn = getattr(model.gat, "last_attn_weights", None)
        raw_bias = getattr(model.gat, "last_kin_edge_bias", None)
        if raw_attn is not None:
            gat_attn = raw_attn.float().reshape(B, W, *raw_attn.shape[1:]).mean(dim=2)

        def _skip(k):
            return model.kin_skip_proj(model.get_skip_kin(k)).view(B, W, model.out_channels, -1)

        kin_skip = _skip(kin)
        bias = _dec(torch.zeros_like(kin_skip))
        total = _lag(_dec(gat_out + kin_skip))
        graph = _lag(_dec(gat_out))
        kin_part = _lag(_dec(kin_skip) - bias)

        with _zero_kin_edge_bias(model.gat):
            graph0 = _lag(_dec(model.gat(nodes, gat_kin)))
        # leave the model's stored internals as the unmodified forward pass left them
        if raw_attn is not None:
            model.gat.last_attn_weights = raw_attn
        if raw_bias is not None:
            model.gat.last_kin_edge_bias = raw_bias

        if kin_baseline is None:
            base = torch.zeros(kin.shape[-1], device=dev)
        else:
            base = torch.as_tensor(kin_baseline, dtype=torch.float32).to(dev)

        def _effect(cols):
            cols = [int(c) for c in (cols or [])]
            if not cols:
                return torch.zeros_like(kin_part)
            k2 = kin.clone()
            k2[..., cols] = base[cols]
            return kin_part - _lag(_dec(_skip(k2)) - bias)

        force_eff, vel_eff = _effect(force_cols), _effect(vel_cols)
        dec_bias = _lag(bias)
    else:
        gat_out = model.gat(nodes)
        last = getattr(model.gat, "layers", None)
        raw_attn = getattr(last[-1], "last_attn_weights", None) if last else None
        if raw_attn is not None:
            gat_attn = raw_attn.float().reshape(B, W, *raw_attn.shape[1:]).mean(dim=2)
        total = _lag(_dec(gat_out))
        graph = total
        graph0 = total
        kin_part = torch.zeros_like(total)
        force_eff = torch.zeros_like(total)
        vel_eff = torch.zeros_like(total)
        dec_bias = _lag(_dec(torch.zeros_like(gat_out)))

    out = {
        "total": total, "graph": graph, "kin": kin_part, "graph_no_kin_bias": graph0,
        "kin_force_effect": force_eff, "kin_velocity_effect": vel_eff,
        "decoder_bias": dec_bias, "gat_attn": gat_attn,
    }
    for k, v in out.items():
        if v is None:
            continue
        v = v.float().cpu()
        out[k] = v.numpy() if as_numpy else v
    return out


def decompose_series(
    model: torch.nn.Module,
    eeg: np.ndarray,
    kin: np.ndarray,
    prepare_batch: Callable,
    window: int,
    batch_size: int = 16,
    tail: str = "align_end",
    **decompose_kwargs,
) -> dict:
    """``decompose_prediction`` over a whole series, in windows of the training size.

    Same windows as ``main.inference.predict_series_stitched`` (consecutive,
    non-overlapping, a trailing partial window covered by one end-aligned window),
    stitched back in time.

    Returns:
        {"total", "graph", "kin", "graph_no_kin_bias", "kin_force_effect",
         "kin_velocity_effect", "decoder_bias"}: (T', C) float32, normalised units;
        "gat_attn": (T', N, N) or None;  "starts": window start samples;  "window".
    """
    window = int(window)
    eeg, kin = np.asarray(eeg), np.asarray(kin)
    starts = window_starts(len(eeg), window, tail=tail)
    n_out = len(eeg) if tail == "align_end" else (len(eeg) // window) * window
    chunks: dict[str, list] = {}
    for b0 in range(0, len(starts), int(batch_size)):
        inputs, _ = window_batch(eeg, kin, prepare_batch, window, starts=starts[b0:b0 + int(batch_size)])
        parts = decompose_prediction(model, inputs["eeg"], inputs.get("kin"), **decompose_kwargs)
        for k, v in parts.items():
            if v is not None:
                chunks.setdefault(k, []).append(v)
    out: dict = {k: stitch_windows(np.concatenate(v), starts, n_out) for k, v in chunks.items()}
    out.setdefault("gat_attn", None)
    out["starts"] = np.asarray(starts, dtype=np.int64)
    out["window"] = window
    return out


# ════════════════════════════════════════════════════════════════════════════
# 3. Attention rollout (within one model window)
# ════════════════════════════════════════════════════════════════════════════

def attention_rollout(attn_per_layer, residual: float = 0.5, heads_averaged: bool = False):
    """Attention rollout (Abnar & Zuidema 2020) of a stack of self-attention layers.

    Per layer: heads averaged, residual connection added
    (``(1 − residual)·A + residual·I``, rows renormalised to sum 1); the layer matrices
    are multiplied from the first layer (right) to the last (left):

        R = Â_L · Â_{L-1} · … · Â_1          R[q, k] = share of input time step k in
                                             the last-layer state of query time step q

    Args:
        attn_per_layer: (…, L, H, T, T) attention weights [query, key] — the weights
                        used in the forward pass, no temperature, no per-head scaling.
                        With ``heads_averaged=True``: (…, L, T, T).
        residual:       weight of the identity (0.5 = attention and skip count equally).

    Returns:
        (…, T, T), rows sum to 1; numpy in -> numpy out, tensor in -> tensor out.
    """
    is_numpy = not torch.is_tensor(attn_per_layer)
    a = torch.as_tensor(np.asarray(attn_per_layer) if is_numpy else attn_per_layer)
    a = a.to(torch.float64) if is_numpy else a.float()
    if not heads_averaged:
        if a.dim() < 4:
            raise ValueError(f"expected (..., L, H, T, T), got {tuple(a.shape)}")
        a = a.mean(dim=-3)
    if a.dim() < 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError(f"expected (..., L, T, T) after head averaging, got {tuple(a.shape)}")
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    a = (1.0 - residual) * a + residual * eye
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    roll = a[..., 0, :, :]
    for layer in range(1, a.shape[-3]):
        roll = a[..., layer, :, :] @ roll
    return roll.numpy() if is_numpy else roll


def _mhsa_layers(model) -> list:
    layers = getattr(getattr(model, "encoder", None), "layers", None) or []
    return [layer.mhsa for layer in layers if hasattr(layer, "mhsa")]


@contextlib.contextmanager
def _true_attention(model):
    """Make ``mhsa.last_attn_weights`` the weights USED in the forward pass.

    In eval mode ``MultiHeadSelfAttention`` stores softmax(scores / attention_temperature)
    but computes its output with the fused kernel at temperature 1.  Inside this block
    the temperature attribute is 1, so stored == used; it is restored afterwards and the
    stored (B, H, T, T) tensors are dropped.
    """
    mhsas = _mhsa_layers(model)
    saved = [getattr(m, "attention_temperature", 1.0) for m in mhsas]
    for m in mhsas:
        m.attention_temperature = 1.0
    try:
        yield mhsas
    finally:
        for m, t in zip(mhsas, saved):
            m.attention_temperature = t
            m.last_attn_weights = None


@torch.no_grad()
def encoder_attention(model: torch.nn.Module, eeg, average_heads: bool = False) -> torch.Tensor:
    """True encoder self-attention of one batch of model windows.

    Returns a CPU tensor (B, L, H, W, W) [query, key] — or (B, L, W, W) with
    ``average_heads=True`` — softmax of the scaled dot products, i.e. exactly the weights
    the forward pass applies (no temperature).  Large: use small batches.
    """
    model.eval()
    eeg = _as_batch(eeg, _device_of(model))
    with _true_attention(model) as mhsas:
        if not mhsas:
            raise TypeError("model has no encoder.layers[*].mhsa to read attention from")
        model.encoder(eeg)
        per_layer = []
        for m in mhsas:
            a = m.last_attn_weights.float()
            per_layer.append((a.mean(dim=1) if average_heads else a).cpu())
    return torch.stack(per_layer, dim=1)


@torch.no_grad()
def rollout_rows(model: torch.nn.Module, eeg, query: int, batch_size: int = 4,
                 residual: float = 0.5) -> np.ndarray:
    """Rollout row of one query time step for every window: (B, W) over key time.

    ``out[b, k]`` = share of input time step k of window b in the last-layer encoder
    state at time step ``query`` (rows sum to 1).
    """
    eeg = torch.as_tensor(eeg, dtype=torch.float32)
    if eeg.dim() == 2:
        eeg = eeg.unsqueeze(0)
    rows = []
    for b0 in range(0, eeg.shape[0], int(batch_size)):
        attn = encoder_attention(model, eeg[b0:b0 + int(batch_size)], average_heads=True)
        rows.append(attention_rollout(attn, residual=residual, heads_averaged=True)[:, int(query)].numpy())
    return np.concatenate(rows).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# 4. Does the EEG matter?  occlusion attribution, deletion curve, effect size
# ════════════════════════════════════════════════════════════════════════════

def _time_bins(window: int, time_bin: int) -> list[tuple[int, int]]:
    time_bin = int(time_bin)
    if time_bin <= 0:
        raise ValueError(f"time_bin must be positive, got {time_bin}")
    return [(a, min(a + time_bin, window)) for a in range(0, window, time_bin)]


def _electrode_groups(n_electrodes: int, groups) -> tuple[list, list[list[int]]]:
    if groups is None:
        return list(range(n_electrodes)), [[i] for i in range(n_electrodes)]
    if isinstance(groups, Mapping):
        return list(groups.keys()), [[int(i) for i in v] for v in groups.values()]
    idx = [[int(i) for i in (g if isinstance(g, (list, tuple, np.ndarray)) else [g])] for g in groups]
    return list(range(len(idx))), idx


@torch.no_grad()
def eeg_occlusion_attribution(
    model: torch.nn.Module,
    eeg,
    kin=None,
    time_bin: int = 25,
    electrode_groups=None,
    query: int | None = None,
) -> dict:
    """Occlusion attribution of the EEG input: what changes when a piece is removed?

    Each time bin (all electrodes) and, separately, each electrode / electrode group
    (all time steps) is replaced by the window mean of the electrode (0 for a z-scored
    window), and the change of every muscle's output is recorded:

        Δ = output(original) − output(occluded)                        (B, W, C)

    Args:
        model:   called as ``model(eeg=…, kin=…)`` -> (B, W, C); put in eval mode.
        eeg:     (B, W, E) normalised EEG windows.   kin: (B, W, K) or None.
        time_bin: bin width in samples (25 = 50 ms at 500 Hz).
        electrode_groups: None (every electrode alone), a dict name -> indices, or a
                 list of index lists.
        query:   sample at which the signed change is read (default: window centre).

    Returns (numpy float32):
        "time_abs"         (B, n_bins, C)  mean |Δ| over the window
        "time_signed"      (B, n_bins, C)  Δ at the query sample
        "electrode_abs"    (B, G, C)       "electrode_signed" (B, G, C)
        "bins"             (n_bins, 2) [start, stop) samples;  "bin_centers" (n_bins,)
        "electrode_labels" group names / indices;  "query";  "output" (B, W, C)
    """
    model.eval()
    dev = _device_of(model)
    eeg = _as_batch(eeg, dev)
    kin = _as_batch(kin, dev) if kin is not None else None
    B, W, E = eeg.shape
    q = W // 2 if query is None else int(query)
    if not 0 <= q < W:
        raise ValueError(f"query {q} outside the window of {W} samples")
    base = eeg.mean(dim=1, keepdim=True).expand_as(eeg)
    out = _call(model, eeg, kin)

    def _delta(mask_fn):
        x = eeg.clone()
        mask_fn(x)
        d = out - _call(model, x, kin)
        return d.abs().mean(dim=1), d[:, q]

    bins = _time_bins(W, time_bin)
    t_abs, t_sig = [], []
    for a, b in bins:
        def _mask(x, a=a, b=b):
            x[:, a:b] = base[:, a:b]
        d_abs, d_sig = _delta(_mask)
        t_abs.append(d_abs)
        t_sig.append(d_sig)

    labels, groups = _electrode_groups(E, electrode_groups)
    e_abs, e_sig = [], []
    for g in groups:
        def _mask(x, g=g):
            x[:, :, g] = base[:, :, g]
        d_abs, d_sig = _delta(_mask)
        e_abs.append(d_abs)
        e_sig.append(d_sig)

    def _np(lst):
        return torch.stack(lst, dim=1).float().cpu().numpy()

    bins_np = np.asarray(bins, dtype=np.int64)
    return {
        "time_abs": _np(t_abs), "time_signed": _np(t_sig),
        "electrode_abs": _np(e_abs), "electrode_signed": _np(e_sig),
        "bins": bins_np, "bin_centers": (bins_np[:, 0] + bins_np[:, 1] - 1) / 2.0,
        "electrode_labels": labels, "query": q, "output": out.cpu().numpy(),
    }


def _deletion_summary(fractions, top_w, rand_w, full_w) -> dict:
    """Aggregate per-window deletion results (n, F) / (n,) into curves and a score."""
    fr = np.concatenate([[0.0], np.asarray(fractions, dtype=np.float64)])
    top = np.concatenate([[0.0], np.asarray(top_w, dtype=np.float64).mean(axis=0)])
    rnd = np.concatenate([[0.0], np.asarray(rand_w, dtype=np.float64).mean(axis=0)])
    full = float(np.mean(full_w))
    diff = top - rnd
    area = float(np.sum(0.5 * (diff[1:] + diff[:-1]) * np.diff(fr)))
    score = area / (full * fr[-1]) if full > 1e-12 else 0.0
    return {"fractions": fr, "top": top, "random": rnd, "full": full,
            "area": area, "score": float(score),
            "top_per_window": np.asarray(top_w), "random_per_window": np.asarray(rand_w),
            "full_per_window": np.asarray(full_w)}


@torch.no_grad()
def deletion_curve(
    model: torch.nn.Module,
    eeg,
    kin=None,
    attribution=None,
    time_bin: int = 25,
    fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    n_random: int = 3,
    seed: int = 0,
) -> dict:
    """Faithfulness of the time-bin attribution: delete the top-attributed bins vs random bins.

    For every fraction f the k = round(f · n_bins) EEG time bins with the highest
    attribution of each window are replaced by the electrode's window mean and the
    output change (mean |Δ| over time and muscles) is measured; the same is done for k
    randomly chosen bins (``n_random`` seeded draws).  If the attribution ranks the bins
    the model really uses, the "top" curve lies above the "random" curve.

    Args:
        attribution: (B, n_bins) score per window and bin (higher = more important);
                     None -> ``eeg_occlusion_attribution(...)["time_abs"]`` averaged
                     over muscles.

    Returns:
        "fractions" (F+1,) starting at 0;  "top", "random" (F+1,) mean output change;
        "full"   output change with ALL bins replaced;
        "area"   ∫ (top − random) d fraction  (output units);
        "score"  area / (full · max fraction): 0 = no better than random, > 0 = the
                 attribution finds the bins that matter (0 when the EEG has no effect);
        "*_per_window" arrays for pooling several calls.
    """
    model.eval()
    dev = _device_of(model)
    eeg = _as_batch(eeg, dev)
    kin = _as_batch(kin, dev) if kin is not None else None
    B, W, _ = eeg.shape
    bins = _time_bins(W, time_bin)
    n_bins = len(bins)
    if attribution is None:
        attribution = eeg_occlusion_attribution(model, eeg, kin, time_bin=time_bin)["time_abs"].mean(axis=-1)
    attribution = np.asarray(attribution, dtype=np.float64)
    if attribution.shape != (B, n_bins):
        raise ValueError(f"attribution must be {(B, n_bins)}, got {attribution.shape}")
    order = np.argsort(-attribution, axis=1, kind="stable")           # best bin first
    base = eeg.mean(dim=1, keepdim=True).expand_as(eeg)
    out = _call(model, eeg, kin)
    rng = np.random.default_rng(int(seed))

    def _change(chosen) -> np.ndarray:
        """chosen: per window, the bin indices to delete -> (B,) mean |Δ|."""
        x = eeg.clone()
        for b in range(B):
            for j in chosen[b]:
                a, e = bins[int(j)]
                x[b, a:e] = base[b, a:e]
        return (out - _call(model, x, kin)).abs().mean(dim=(1, 2)).cpu().numpy()

    top_w, rand_w = [], []
    for f in fractions:
        k = int(min(n_bins, max(1, round(float(f) * n_bins))))
        top_w.append(_change([order[b, :k] for b in range(B)]))
        draws = [_change([rng.permutation(n_bins)[:k] for _ in range(B)]) for _ in range(int(n_random))]
        rand_w.append(np.mean(draws, axis=0))
    full_w = _change([range(n_bins)] * B)
    return _deletion_summary(fractions, np.stack(top_w, axis=1), np.stack(rand_w, axis=1), full_w)


def _iter_inputs(windows) -> Iterable:
    """Accept (eeg, kin), {"eeg", "kin"} or an iterable of such batches."""
    if isinstance(windows, Mapping):
        yield windows["eeg"], windows.get("kin")
    elif isinstance(windows, (tuple, list)) and len(windows) == 2 and (
            torch.is_tensor(windows[0]) or isinstance(windows[0], np.ndarray)):
        yield windows[0], windows[1]
    else:
        for batch in windows:
            if isinstance(batch, Mapping):
                yield batch["eeg"], batch.get("kin")
            else:
                yield batch[0], batch[1]


@torch.no_grad()
def eeg_effect_size(model: torch.nn.Module, windows, seed: int = 0, per_channel: bool = False):
    """How much does the output change when the EEG is time-shuffled within the window?

        effect_c = mean |output − output(EEG shuffled)|  /  SD(output)      per muscle c

    over all given windows (same seeded permutation for every window, kinematics
    untouched).  0 = the model ignores the EEG; ~1 = shuffling the EEG moves the output
    as much as the output varies by itself.

    Args:
        windows: normalised model inputs — ``(eeg, kin)`` tensors (N, W, ·), a dict
                 {"eeg", "kin"}, or an iterable of such batches, e.g.
                 ``(prepare_batch(e, k, m)[0] for e, k, m in test_loader)``.

    Returns:
        float (mean over muscles), or the (C,) array with ``per_channel=True``.
    """
    model.eval()
    dev = _device_of(model)
    n = 0
    s_abs = s1 = s2 = None
    for eeg, kin in _iter_inputs(windows):
        eeg = _as_batch(eeg, dev)
        kin = _as_batch(kin, dev) if kin is not None else None
        out = _call(model, eeg, kin).double()
        shuf = _call(model, shuffle_time(eeg, seed), kin).double()
        C = out.shape[-1]
        d = (out - shuf).abs().reshape(-1, C).sum(0).cpu().numpy()
        o = out.reshape(-1, C)
        a1, a2 = o.sum(0).cpu().numpy(), (o ** 2).sum(0).cpu().numpy()
        s_abs = d if s_abs is None else s_abs + d
        s1 = a1 if s1 is None else s1 + a1
        s2 = a2 if s2 is None else s2 + a2
        n += o.shape[0]
    if n == 0:
        raise ValueError("eeg_effect_size: no windows given")
    mean_abs = s_abs / n
    std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    effect = np.where(std > 1e-12, mean_abs / np.where(std > 1e-12, std, 1.0), 0.0)
    return effect if per_channel else float(effect.mean())


# ════════════════════════════════════════════════════════════════════════════
# 5. Event-locked analysis (all lifts)
# ════════════════════════════════════════════════════════════════════════════

_PER_LIFT_KEYS = (
    "pred", "target", "pred_z", "target_z",
    "graph", "kin", "graph_no_kin_bias", "kin_force_effect", "kin_velocity_effect",
    "rollout", "occ_time_abs", "occ_time_signed", "occ_electrode_abs", "occ_electrode_signed",
)


def _is_decomposable(model) -> bool:
    return all(hasattr(model, a) for a in ("encoder", "node_projection", "gat", "decoder"))


def event_locked_windows(
    model: torch.nn.Module,
    eeg: np.ndarray,
    kin: np.ndarray,
    emg: np.ndarray,
    samples: Sequence,
    prepare_batch: Callable,
    pre: int = 250,
    post: int = 250,
    *,
    meta: Sequence[Mapping] | None = None,
    emg_mean=None,
    emg_std=None,
    force_cols: Sequence[int] | None = None,
    vel_cols: Sequence[int] | None = None,
    kin_baseline=None,
    time_bin: int = 25,
    electrode_groups=None,
    rollout: bool = True,
    occlusion: bool = True,
    deletion: bool = True,
    deletion_kwargs: Mapping | None = None,
    batch_size: int = 8,
    seed: int = 0,
) -> dict:
    """Run the model on one window per lift, centred on an event, and explain each.

    The window of lift i is ``[s_i − pre, s_i + post)`` of the series (s_i = event
    sample); ``pre + post`` must be the training window (500 = 1 s), so the model sees
    exactly the kind of window it was trained on, with the event at index ``pre``.
    Lifts whose event is None or whose window leaves the series are skipped.

    Args:
        model:         trained model (eval mode is set here).
        eeg, kin, emg: (T, ·) arrays of ONE series as stored in the dataset (un-normalised).
        samples:       event sample per lift (``event_samples(lift_events(...), "lift_off")``).
        prepare_batch: the notebook's normalisation closure.
        meta:          one dict per lift (e.g. the ``lift_events`` dicts: CurW, CurS, lift).
        emg_mean, emg_std: (C,) to return the prediction in original EMG units.
        force_cols, vel_cols, kin_baseline: see ``decompose_prediction``.
        time_bin, electrode_groups: see ``eeg_occlusion_attribution``.
        rollout / occlusion / deletion: switch the EEG evidence parts off.

    Returns (n = kept lifts, W = pre + post, C muscles, E electrodes / groups):
        "pred", "target"           (n, W, C) original EMG units (z units without emg_mean/std)
        "pred_z", "target_z"       (n, W, C) normalised units
        "graph", "kin", "graph_no_kin_bias", "kin_force_effect", "kin_velocity_effect"
                                   (n, W, C) normalised units (absent for a model that
                                   cannot be decomposed)
        "rollout"                  (n, W) rollout row of the query at the event sample
        "occ_time_abs", "occ_time_signed"            (n, n_bins, C)
        "occ_electrode_abs", "occ_electrode_signed"  (n, E, C)
        "bin_centers"              (n_bins,) samples RELATIVE to the event
        "deletion"                 ``deletion_curve`` result pooled over the lifts
        "kept"  (n,) positions into ``samples``;  "event_samples" (n,);  "meta" list of n dicts
        "pre", "post", "window", "query" (= pre), "electrode_labels"
    """
    pre, post = int(pre), int(post)
    window = pre + post
    eeg, kin, emg = np.asarray(eeg), np.asarray(kin), np.asarray(emg)
    target, kept = extract_event_windows(emg, samples, pre, post)
    starts = [int(samples[i]) - pre for i in kept]
    model.eval()
    decomposable = _is_decomposable(model)
    has_attn = bool(_mhsa_layers(model))
    del_kw = dict(deletion_kwargs or {})

    acc: dict[str, list] = {}
    del_parts = {"top": [], "random": [], "full": []}
    del_fractions = None
    bin_centers, electrode_labels = None, None
    for bi, b0 in enumerate(range(0, len(starts), int(batch_size))):
        b_starts = starts[b0:b0 + int(batch_size)]
        inputs, y = window_batch(eeg, kin, prepare_batch, window, emg=emg, starts=b_starts)
        e_t, k_t = inputs["eeg"], inputs.get("kin")
        acc.setdefault("target_z", []).append(y.float().cpu().numpy())
        if decomposable:
            parts = decompose_prediction(model, e_t, k_t, force_cols=force_cols,
                                         vel_cols=vel_cols, kin_baseline=kin_baseline)
            acc.setdefault("pred_z", []).append(parts["total"])
            for k in DECOMPOSITION_KEYS[1:]:
                acc.setdefault(k, []).append(parts[k])
        else:
            with torch.no_grad():
                acc.setdefault("pred_z", []).append(_call(model, e_t, k_t).cpu().numpy())
        if rollout and has_attn:
            acc.setdefault("rollout", []).append(rollout_rows(model, e_t, query=pre))
        occ = None
        if occlusion or deletion:
            occ = eeg_occlusion_attribution(model, e_t, k_t, time_bin=time_bin,
                                            electrode_groups=electrode_groups, query=pre)
            bin_centers = occ["bin_centers"] - pre
            electrode_labels = occ["electrode_labels"]
        if occlusion:
            for k in ("time_abs", "time_signed", "electrode_abs", "electrode_signed"):
                acc.setdefault("occ_" + k, []).append(occ[k])
        if deletion:
            d = deletion_curve(model, e_t, k_t, attribution=occ["time_abs"].mean(axis=-1),
                               time_bin=time_bin, seed=int(seed) + bi, **del_kw)
            del_fractions = d["fractions"][1:]
            del_parts["top"].append(d["top_per_window"])
            del_parts["random"].append(d["random_per_window"])
            del_parts["full"].append(d["full_per_window"])

    out: dict = {k: np.concatenate(v) for k, v in acc.items()}
    n = len(kept)
    C = emg.shape[1]
    if n == 0:
        out["pred_z"] = np.empty((0, window, C), dtype=np.float32)
        out["target_z"] = np.empty((0, window, C), dtype=np.float32)
    if emg_mean is not None and emg_std is not None:
        m = np.asarray(emg_mean.detach().cpu() if torch.is_tensor(emg_mean) else emg_mean, dtype=np.float32)
        s = np.asarray(emg_std.detach().cpu() if torch.is_tensor(emg_std) else emg_std, dtype=np.float32)
        out["pred"] = out["pred_z"] * s + m
        out["target"] = target.astype(np.float32)
        out["units"] = "original EMG units"
    else:
        out["pred"], out["target"] = out["pred_z"], out["target_z"]
        out["units"] = "normalised units (z)"
    out["deletion"] = (_deletion_summary(del_fractions, np.concatenate(del_parts["top"]),
                                         np.concatenate(del_parts["random"]),
                                         np.concatenate(del_parts["full"]))
                       if del_parts["top"] else None)
    out["bin_centers"] = bin_centers
    out["electrode_labels"] = electrode_labels
    out["kept"] = kept
    out["event_samples"] = np.asarray([samples[i] for i in kept], dtype=np.int64)
    out["meta"] = [dict(meta[i]) for i in kept] if meta is not None else [{} for _ in kept]
    out.update(pre=pre, post=post, window=window, query=pre)
    return out


def concat_event_locked(parts: Sequence[Mapping]) -> dict:
    """Pool the lifts of several ``event_locked_windows`` results (e.g. all test series).

    All parts must use the same pre / post.  ``kept`` / ``event_samples`` stay
    per-series indices; ``meta`` is concatenated in the same order.
    """
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("concat_event_locked: nothing to concatenate")
    if len({(p["pre"], p["post"]) for p in parts}) != 1:
        raise ValueError("concat_event_locked: parts use different pre / post")
    nonempty = [p for p in parts if len(p["kept"])] or parts[:1]
    ref = nonempty[0]
    out: dict = {}
    for k in _PER_LIFT_KEYS:
        if all(k in p for p in nonempty):
            out[k] = np.concatenate([p[k] for p in nonempty])
    for k in ("kept", "event_samples"):
        out[k] = np.concatenate([p[k] for p in nonempty])
    out["meta"] = [m for p in nonempty for m in p["meta"]]
    dels = [p["deletion"] for p in nonempty if p.get("deletion") is not None]
    out["deletion"] = (_deletion_summary(dels[0]["fractions"][1:],
                                         np.concatenate([d["top_per_window"] for d in dels]),
                                         np.concatenate([d["random_per_window"] for d in dels]),
                                         np.concatenate([d["full_per_window"] for d in dels]))
                       if dels else None)
    for k in ("bin_centers", "electrode_labels", "pre", "post", "window", "query", "units"):
        out[k] = ref.get(k)
    return out


def _summarise(ev: Mapping, idx: np.ndarray) -> dict:
    out: dict = {"n": int(len(idx))}
    for k in _PER_LIFT_KEYS:
        if k not in ev:
            continue
        x = np.asarray(ev[k], dtype=np.float64)[idx]
        if len(idx):
            out[k + "_mean"], out[k + "_sd"] = x.mean(axis=0), x.std(axis=0)
        else:
            out[k + "_mean"] = np.full(x.shape[1:], np.nan)
            out[k + "_sd"] = np.full(x.shape[1:], np.nan)
    return out


def event_locked_summary(ev: Mapping, group_by: str | None = None) -> dict:
    """Mean ± SD across lifts of everything ``event_locked_windows`` returned.

    Args:
        ev:       result of ``event_locked_windows`` / ``concat_event_locked``.
        group_by: a ``meta`` key (e.g. "CurW" = weight) -> additionally one summary
                  per value of that key.

    Returns:
        {"n", "<key>_mean", "<key>_sd" for every per-lift array (lift axis removed),
         "t" (W,) samples relative to the event, "bin_centers", "electrode_labels",
         "deletion", "pre", "post", "units",
         "group_by", "groups": {value: {"n", "<key>_mean", "<key>_sd"}}}
    """
    n = len(ev["kept"])
    out = _summarise(ev, np.arange(n))
    out["t"] = np.arange(-int(ev["pre"]), int(ev["post"]))
    for k in ("bin_centers", "electrode_labels", "deletion", "pre", "post", "units"):
        out[k] = ev.get(k)
    out["group_by"] = group_by
    out["groups"] = {}
    if group_by is not None:
        values = [m.get(group_by) for m in ev["meta"]]
        for v in sorted({v for v in values if v is not None}):
            out["groups"][v] = _summarise(ev, np.asarray([i for i, x in enumerate(values) if x == v],
                                                         dtype=np.int64))
    return out
