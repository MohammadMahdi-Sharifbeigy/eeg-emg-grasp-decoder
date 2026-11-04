"""Statistics of the pilot: weight-matched contrasts, group cluster tests, the cross-validated decoder, partial rank
correlation. Every group test treats the participant as the unit."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_TRIALS = 3


def matched_contrast(x: np.ndarray, table: pd.DataFrame, kind: str) -> tuple[np.ndarray | None, int]:
    """Mean of `kind` ("heavier"/"lighter") lifts minus mean of expected lifts with the same current weight.

    The difference is taken within each current weight and averaged with the number of unexpected lifts as weight,
    so the physical object is the same on both sides and only the expectation differs.
    Returns (difference with the shape of x[0], number of unexpected lifts used) or (None, 0).
    """
    ok = ~table["rejected"].to_numpy(bool)
    change, w = table["change"].to_numpy(object), table["CurW"].to_numpy(float)
    diffs, counts = [], []
    for cw in np.unique(w):
        u = ok & (change == kind) & (w == cw)
        e = ok & (change == "expected") & (w == cw)
        if u.sum() >= MIN_TRIALS and e.sum() >= MIN_TRIALS:
            diffs.append(x[u].mean(0) - x[e].mean(0))
            counts.append(int(u.sum()))
    if not counts:
        return None, 0
    return np.average(np.stack(diffs), axis=0, weights=counts), int(sum(counts))


def cluster_test(X: np.ndarray, adjacency=None, n_permutations: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Sign-flip cluster test over participants. X: (n_participants, n_times, n_channels).

    One row per cluster: first and last time index, channels, cluster mass and corrected p (two-sided).
    """
    from mne.stats import permutation_cluster_1samp_test
    t_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
        X, adjacency=adjacency, n_permutations=n_permutations, seed=seed, out_type="mask", verbose=False, tail=0)
    rows = []
    for mask, p in zip(clusters, p_values):
        ti, ci = np.nonzero(mask)
        rows.append({"t_first": int(ti.min()), "t_last": int(ti.max()), "channels": sorted(set(ci.tolist())),
                     "mass": float(t_obs[mask].sum()), "p": float(p)})
    return pd.DataFrame(rows, columns=["t_first", "t_last", "channels", "mass", "p"]).sort_values("p",
                                                                                                    ignore_index=True)


def channel_adjacency(ch_names: list):
    import mne
    info = mne.create_info(list(ch_names), 500.0, "eeg")
    info.set_montage("standard_1005")
    adj, _ = mne.channels.find_ch_adjacency(info, "eeg")
    return adj


def time_adjacency_only(n_channels: int):
    """Adjacency with no neighbours between signals (each EMG / force signal is tested on its own over time)."""
    from scipy.sparse import csr_matrix
    return csr_matrix((n_channels, n_channels))


def _oof_scores(X, y, groups):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    s = np.full(len(y), np.nan)
    for g in np.unique(groups):
        test = groups == g
        if len(np.unique(y[~test])) < 2:
            continue
        s[test] = model.fit(X[~test], y[~test]).decision_function(X[test])
    return s


def decode(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_shuffles: int = 200, seed: int = 0) -> dict:
    """Leave-one-series-out logistic decoder; AUC of the pooled out-of-fold scores and a label-shuffle null.

    Labels are shuffled within each series, so the null keeps the class balance of every fold.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)

    def auc(labels):
        s = _oof_scores(X, labels, groups)
        ok = np.isfinite(s)
        return roc_auc_score(labels[ok], s[ok]) if len(np.unique(labels[ok])) == 2 else np.nan

    obs = auc(y)
    null = []
    for _ in range(n_shuffles):
        yy = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            yy[idx] = rng.permutation(yy[idx])
        null.append(auc(yy))
    null = np.asarray(null)
    return {"auc": float(obs), "null_mean": float(np.nanmean(null)),
            "p": float((1 + np.sum(null >= obs)) / (1 + np.isfinite(null).sum())), "n": int(len(y))}


def partial_spearman(x, y, covariates: np.ndarray | None = None) -> float:
    """Spearman correlation of x and y after removing a linear fit of the covariates from the ranks of both."""
    x, y = pd.Series(x).rank().to_numpy(float), pd.Series(y).rank().to_numpy(float)
    if covariates is not None and np.size(covariates):
        C = np.column_stack([np.ones(len(x)), pd.DataFrame(covariates).rank().to_numpy(float)])
        x = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
        y = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    return float(spearmanr(x, y).statistic)


def group_wilcoxon(values, center: float = 0.0, alternative: str = "two-sided") -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return {"n": len(v), "mean": float(np.mean(v)) if len(v) else np.nan, "p": np.nan, "n_positive": 0}
    return {"n": len(v), "mean": float(v.mean()), "n_positive": int((v > center).sum()),
            "p": float(wilcoxon(v - center, alternative=alternative).pvalue)}
