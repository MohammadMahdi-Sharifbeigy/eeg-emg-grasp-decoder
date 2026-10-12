"""Can a map be trusted? Deletion curves and a weight-shuffle check."""
import copy

import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def deletion_curve(model, kin, hist, eeg, score, output="E", muscle=0, t=None, which="eeg",
                   fractions=(0.05, 0.1, 0.2, 0.4), n_random=10, seed=0, chunk=8) -> dict:
    """Set the highest-scored cells of one input to 0 (the training mean) and measure how far the target
    out[output][0, t, muscle] moves, against deleting the same number of random cells.

    which="eeg": score (T, E) or (T,); a cell is one electrode at one token, all bands.
    which="kin": score (T,); a cell is all kinematic features of one token.
    Inputs are (1, T, ...) tensors on the model's device.

    The score is used as given: the highest values are deleted first. A signed map (gradient x input)
    must be passed as its absolute value, or its strongly negative cells are deleted last. Equal scores
    are taken in order of position (earliest token first). A (T,) score on the EEG input is repeated
    over the electrodes, so it deletes whole tokens, while the random baseline deletes single cells.
    The deleted copies go through the model `chunk` at a time (memory); the result does not depend on it.
    """
    if which not in ("eeg", "kin"):
        raise ValueError(f"which must be 'eeg' or 'kin', got {which!r}")
    if int(chunk) < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk!r}")
    model.eval()
    score = np.asarray(score, dtype=np.float64)
    if which == "eeg":
        if score.ndim == 1:
            score = np.repeat(score[:, None], eeg.shape[2], axis=1)
        shape = tuple(eeg.shape[1:3])
    else:
        shape = (kin.shape[1],)
    if score.shape != shape:
        raise ValueError(f"score has shape {score.shape}, expected {shape}")
    t = kin.shape[1] // 2 if t is None else int(t)
    n = score.size
    order = np.argsort(-score.ravel(), kind="stable")
    rng = np.random.default_rng(seed)
    keep = []                                                          # one mask per deletion; False = deleted
    for f in fractions:
        k = max(int(round(f * n)), 1)
        for picked in [order[:k]] + [rng.permutation(n)[:k] for _ in range(n_random)]:
            mask = np.ones(n, dtype=bool)
            mask[picked] = False
            keep.append(mask.reshape(shape))
    keep = torch.from_numpy(np.stack(keep)).to(kin.device)[..., None]  # (deletions, T[, E], 1)
    values = []
    for i in range(0, keep.shape[0], int(chunk)):
        part = keep[i:i + int(chunk)]
        rep = lambda a: a.expand(part.shape[0], *a.shape[1:])
        if which == "eeg":
            out = model(rep(kin), rep(hist), eeg * part)
        else:
            out = model(kin * part, rep(hist), rep(eeg))
        values.append(out[output][:, t, muscle].float())
    full = float(model(kin, hist, eeg)[output][0, t, muscle])
    change = (torch.cat(values) - full).abs().cpu().numpy().reshape(len(fractions), 1 + n_random)
    rand = change[:, 1:]
    rand_mean = rand.mean(1) if n_random else np.zeros(len(fractions))
    rand_sd = rand.std(1) if n_random else np.zeros(len(fractions))
    return {"fractions": np.asarray(fractions, dtype=np.float64), "full": full, "top": change[:, 0],
            "random_mean": rand_mean, "random_sd": rand_sd, "aopc": float((change[:, 0] - rand_mean).mean())}


def shuffled_weights(model, seed: int = 0):
    """A copy of the model with the entries of every weight tensor (2 or more dimensions) shuffled.

    Biases and normalisation parameters stay, so the copy still runs in the same range. A map that
    looks the same for this copy does not depend on what the model learned."""
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                perm = torch.randperm(p.numel(), generator=g).to(p.device)
                p.copy_(p.flatten()[perm].view_as(p))
    return m


def map_similarity(a, b) -> float:
    """Spearman rank correlation of two maps (1 = the inputs are ranked the same way)."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])
