"""ONE inference path for a whole series: windows of the training window size.

The model is trained on windows of ``cfg["dataset"]["window_size"]`` samples
(500 = 1 s).  In eval mode ``TransformerEncoder`` does not chunk, so a longer
input (e.g. a 10 s window) is attended over as a whole — a regime the model
never saw.  Everything that needs a prediction for a full series (evaluation
tables, magnitude report, interpretability plots, diagnostics) should therefore
go through ``predict_series_stitched``: consecutive NON-overlapping windows of
exactly the training window size, through the notebook's ``prepare_batch``
(normalisation + kin column drop), stitched back in time.

    pred_z = predict_series_stitched(model, eeg, kin, prepare_batch,
                                     window=cfg["dataset"]["window_size"], device=device)
    pred = pred_z * emg_std + emg_mean                    # original EMG units

The first ``(T // window) * window`` samples are exactly what a test loader with
stride = window yields (the evaluation path).  A series length that is not a
multiple of the window gets one extra window aligned to the series end; in the
overlap the samples of the EARLIER window are kept, so only the last
``T % window`` samples come from that extra window.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

__all__ = ["window_starts", "window_batch", "stitch_windows", "predict_series_stitched"]

_TAILS = ("align_end", "drop")


def window_starts(n_samples: int, window: int, tail: str = "align_end") -> list[int]:
    """Start samples of the model windows covering a series of ``n_samples``.

    Consecutive non-overlapping windows 0, window, 2·window, …; if a tail shorter
    than one window is left, ``tail="align_end"`` adds a final window starting at
    ``n_samples − window`` and ``tail="drop"`` ignores the tail (test-loader behaviour).
    """
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
    window = int(window)
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if n_samples < window:
        raise ValueError(f"series of {n_samples} samples is shorter than one window ({window})")
    starts = list(range(0, n_samples - window + 1, window))
    if tail == "align_end" and n_samples % window:
        starts.append(n_samples - window)
    return starts


def window_batch(
    eeg: np.ndarray,
    kin: np.ndarray,
    prepare_batch: Callable,
    window: int,
    emg: np.ndarray | None = None,
    starts: Sequence[int] | None = None,
):
    """Stack model windows of a series and send them through ``prepare_batch``.

    Args:
        eeg, kin:      (T, C_eeg), (T, K_raw) arrays of one series, as stored in the dataset.
        prepare_batch: the notebook closure ``(eeg, kin, emg) -> ({"eeg", "kin"}, emg_z)``.
        window:        model window in samples (training window size).
        emg:           optional (T, C) EMG; without it a zero placeholder is passed and
                       the returned target is None.
        starts:        window start samples (default: ``window_starts(T, window, "drop")``,
                       i.e. whole windows only).

    Returns:
        (inputs, target_z): ``inputs`` is what ``prepare_batch`` returns, with tensors of
        shape (n_windows, window, ·); ``target_z`` is (n_windows, window, C) or None.
    """
    window = int(window)
    if starts is None:
        starts = window_starts(len(eeg), window, tail="drop")
    e = torch.stack([torch.as_tensor(eeg[s:s + window]) for s in starts])
    k = torch.stack([torch.as_tensor(kin[s:s + window]) for s in starts])
    if emg is None:
        m = torch.zeros(len(starts), window, 1, dtype=e.dtype)   # broadcasts against (C,) stats
    else:
        m = torch.stack([torch.as_tensor(emg[s:s + window]) for s in starts])
    inputs, y = prepare_batch(e, k, m)
    return inputs, (y if emg is not None else None)


def stitch_windows(per_window: np.ndarray, starts: Sequence[int], n_samples: int | None = None) -> np.ndarray:
    """(n_windows, window, …) -> (n_samples, …) along time.

    Windows are written in order; where a window overlaps samples that are already
    covered (the end-aligned last window) the earlier window's samples are kept.
    """
    per_window = np.asarray(per_window)
    window = per_window.shape[1]
    starts = [int(s) for s in starts]
    if n_samples is None:
        n_samples = max(starts) + window
    out = np.empty((n_samples,) + per_window.shape[2:], dtype=per_window.dtype)
    covered = 0
    for w, s in enumerate(starts):
        if s > covered:
            raise ValueError(f"windows leave a gap before sample {s} (covered up to {covered})")
        skip = covered - s                       # samples already written by earlier windows
        if skip < window:
            out[covered:s + window] = per_window[w, skip:]
            covered = s + window
    if covered < n_samples:
        raise ValueError(f"windows cover {covered} of {n_samples} samples")
    return out


def _forward(model: torch.nn.Module, inputs) -> torch.Tensor:
    return model(**inputs) if isinstance(inputs, dict) else model(inputs)


@torch.no_grad()
def predict_series_stitched(
    model: torch.nn.Module,
    eeg: np.ndarray,
    kin: np.ndarray,
    prepare_batch: Callable,
    window: int,
    device: torch.device | None = None,
    batch_size: int = 32,
    return_internals: bool = False,
    *,
    emg: np.ndarray | None = None,
    use_amp: bool = False,
    tail: str = "align_end",
    full_attention: bool = False,
):
    """Predict one full series in windows of the training window size (z-space).

    Args:
        model:            trained model called as ``model(eeg=…, kin=…)`` (put in eval mode here).
        eeg, kin:         (T, C_eeg), (T, K_raw) arrays of ONE series, un-normalised, as in
                          the dataset (``test_ds._windows``).
        prepare_batch:    the notebook's normalisation closure (moves the batch to its device).
        window:           ``cfg["dataset"]["window_size"]`` (default config: 500).
        device:           device of the model; only used to decide whether AMP applies.
        batch_size:       windows per forward pass (does not change the result).
        return_internals: also return the attention / edge-bias internals (see below).
        emg:              optional (T, C) EMG of the series -> the normalised target is
                          returned too.
        use_amp:          autocast on CUDA, as in ``collect_predictions``.
        tail:             "align_end" (default) covers a trailing partial window with one
                          window aligned to the series end; "drop" leaves it out, which is
                          exactly the test loader with stride = window.
        full_attention:   with ``return_internals``, also keep the full (window × window)
                          encoder attention matrices (large: only for short stretches).

    Returns:
        ``pred``                              (T', C) float32, normalised (z) units, where
                                              T' = T ("align_end") or (T // window)·window ("drop")
        ``(pred, target)``                    if ``emg`` is given
        ``(pred, internals)`` / ``(pred, target, internals)`` with ``return_internals=True``.

        ``internals`` (numpy float32; n_w windows, W = window, L encoder layers, H encoder
        heads, Hg GAT heads, N muscles; a key is absent when the model does not expose it):
            "window"                     int W
            "starts"                     (n_w,) int64 start sample of every window
            "encoder_key_attn"           (n_w, L, H, W)   attention each KEY time step of the
                                         window receives, mean over the window's queries
            "encoder_attn"               (n_w, L, H, W, W) full matrices [query, key]
                                         (only with ``full_attention=True``)
            "gat_attn"                   (n_w, W, Hg, N, N) muscle-graph attention per time step
            "kin_edge_bias"              (n_w, W, Hg, N, N) kinematic edge bias per time step
            "encoder_key_attn_stitched"  (T', L, H)       \
            "gat_attn_stitched"          (T', Hg, N, N)    > the same along series time: each
            "kin_edge_bias_stitched"     (T', Hg, N, N)   /  time step from ITS OWN window
        Encoder attention never crosses a window border (block-diagonal per window).
    """
    model.eval()
    window = int(window)
    eeg = np.asarray(eeg)
    kin = np.asarray(kin)
    if len(eeg) != len(kin) or (emg is not None and len(emg) != len(eeg)):
        raise ValueError("eeg, kin (and emg) must have the same number of samples")
    starts = window_starts(len(eeg), window, tail=tail)
    n_out = len(eeg) if tail == "align_end" else (len(eeg) // window) * window
    on_cuda = device is not None and torch.device(device).type == "cuda"
    amp = bool(use_amp) and on_cuda

    encoder_layers = list(getattr(getattr(model, "encoder", None), "layers", []) or [])
    gat = getattr(model, "gat", None)

    preds, targets = [], []
    key_attn, full_attn, gat_attn, kin_bias = [], [], [], []
    for b0 in range(0, len(starts), int(batch_size)):
        batch_starts = starts[b0:b0 + int(batch_size)]
        inputs, y = window_batch(eeg, kin, prepare_batch, window, emg=emg, starts=batch_starts)
        with torch.autocast(device_type="cuda" if on_cuda else "cpu", enabled=amp):
            p = _forward(model, inputs)
        n_b = len(batch_starts)
        preds.append(p.float().cpu().numpy().reshape(n_b, window, -1))
        if y is not None:
            targets.append(y.float().cpu().numpy().reshape(n_b, window, -1))

        if return_internals:
            per_layer = [getattr(getattr(layer, "mhsa", None), "last_attn_weights", None)
                         for layer in encoder_layers]
            if per_layer and all(a is not None and a.shape[0] == n_b for a in per_layer):
                # mean over the queries of the window, layer by layer (no big copies)
                key_attn.append(torch.stack(
                    [a.mean(dim=-2, dtype=torch.float32) for a in per_layer], dim=1,
                ).cpu().numpy())                                                # (n_b, L, H, W)
                if full_attention:
                    full_attn.append(np.stack(
                        [a.float().cpu().numpy() for a in per_layer], axis=1))  # (n_b, L, H, W, W)
            ga = getattr(gat, "last_attn_weights", None)                        # (n_b·W, Hg, N, N)
            if ga is not None and ga.shape[0] == n_b * window:
                gat_attn.append(ga.float().reshape(n_b, window, *ga.shape[1:]).cpu().numpy())
            kb = getattr(gat, "last_kin_edge_bias", None)                       # (n_b, W, Hg, N, N)
            if kb is not None and kb.shape[0] == n_b:
                kin_bias.append(kb.float().cpu().numpy())

    # do not leave (B, H, W, W) attention tensors on the model
    for layer in encoder_layers:
        if getattr(getattr(layer, "mhsa", None), "last_attn_weights", None) is not None:
            layer.mhsa.last_attn_weights = None

    pred = stitch_windows(np.concatenate(preds), starts, n_out).astype(np.float32, copy=False)
    out = [pred]
    if emg is not None:
        out.append(stitch_windows(np.concatenate(targets), starts, n_out).astype(np.float32, copy=False))
    if return_internals:
        internals: dict = {"window": window, "starts": np.asarray(starts, dtype=np.int64)}
        if key_attn:
            ka = np.concatenate(key_attn)                                       # (n_w, L, H, W)
            internals["encoder_key_attn"] = ka
            internals["encoder_key_attn_stitched"] = stitch_windows(
                np.moveaxis(ka, -1, 1), starts, n_out)                          # (T', L, H)
        if full_attn:
            internals["encoder_attn"] = np.concatenate(full_attn)
        if gat_attn:
            internals["gat_attn"] = np.concatenate(gat_attn)
            internals["gat_attn_stitched"] = stitch_windows(internals["gat_attn"], starts, n_out)
        if kin_bias:
            internals["kin_edge_bias"] = np.concatenate(kin_bias)
            internals["kin_edge_bias_stitched"] = stitch_windows(internals["kin_edge_bias"], starts, n_out)
        out.append(internals)
    return out[0] if len(out) == 1 else tuple(out)
