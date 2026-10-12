"""Event-locked interpretability of one participant: maps over all lifts -> tables -> figures.

Every figure has its numbers in a table, so nothing has to be read from colour.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..electrodes import EEG_NAMES
from ..tokens import FS_TOK
from . import plots as P
from .events import REST, concat_events, event_locked, event_tokens, event_windows, mean_sem, paired_contrast
from .explain import PARTS, explain
from .faithfulness import deletion_curve, map_similarity, shuffled_weights

OUTPUTS = ("K", "E", "F_eeg")
OUTPUT_LABELS = {"K": "K (kinematics-led)", "E": "E (EEG only)", "F_eeg": "EEG correction inside F"}
TABLES = ("time", "electrodes", "muscle_graph", "inputs", "parts", "rest_contrast", "faithfulness", "sanity")
HIST_NAMES = ["has history", "hold count", "time in series", "time since hold", "previous load", "previous grip",
              "mean load", "mean grip"]


def _share(x):
    """Rescale each event's non-negative map to sum to 1 (over all axes but the first); an all-zero map stays zero."""
    x = np.asarray(x, dtype=np.float64)
    s = x.sum(axis=tuple(range(1, x.ndim)), keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def _pairs(lift_meta, rest_meta):
    """Row numbers (in the lift stack, in the rest stack) of every lift that has its own rest: the rest row
    with lift = k follows lift k of the same series. Lifts without a kept rest window and rests without a
    kept lift window are dropped."""
    if not len(lift_meta) or not len(rest_meta):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    a = lift_meta[["series", "lift"]].assign(i=np.arange(len(lift_meta)))
    b = rest_meta[["series", "lift"]].assign(j=np.arange(len(rest_meta)))
    m = a.merge(b, on=["series", "lift"], how="inner", validate="one_to_one")
    return m["i"].to_numpy(), m["j"].to_numpy()


def _scores(res, which, key):
    """The maps of one explanation as deletion scores: (T, E) for the EEG input, (T,) for the kinematic input."""
    return {"rollout": res["rollout"][key][0].cpu().numpy(), "relevance": res["relevance"][key][0].cpu().numpy(),
            "saliency": res["saliency"][which][0].abs().sum(-1).cpu().numpy()}


def _faithfulness(model, test_tokens, norm, ev_tokens, cfg, device, names):
    """Deletion curves and the weight-shuffle check on the first event type, for E (EEG cells) and K (kinematic tokens)."""
    ic = cfg["interp"]
    window, clip, bin_size, event = int(cfg["tokens"]["window"]), cfg["eeg"]["clip"], int(ic["flow_bin"]), ic["events"][0]
    shuffled = shuffled_weights(model, ic["seed"])
    to_tokens = lambda m: np.repeat(m, bin_size, axis=0)[:window]                       # flow bins -> tokens
    curves, sanity = [], []
    for output, which, key in (("E", "eeg", "input"), ("K", "kin", "time")):
        for tok in test_tokens:
            events = ev_tokens[(tok["series"], event)][: ic["max_events"]]
            for _, x in event_windows(tok, norm, events, device, window, clip):
                def ex(net, mi, flow=False):
                    return explain(net, *x, output=output, muscle=mi, t=window // 2, residual=ic["residual"],
                                   flow=flow, bin_size=bin_size)
                flows = [to_tokens(ex(net, 0, True)["flow"][key]) for net in (model, shuffled)]   # same for every muscle
                for mi, mname in enumerate(names):
                    ours, other = (_scores(ex(net, mi), which, key) for net in (model, shuffled))
                    ours["flow"], other["flow"] = flows
                    for view, score in ours.items():
                        c = deletion_curve(model, *x, score, output=output, muscle=mi, t=window // 2, which=which,
                                           fractions=ic["deletion_fractions"], n_random=ic["n_random"], seed=ic["seed"])
                        curves.append(pd.DataFrame({"output": output, "muscle": mname, "view": view,
                                                    "fraction": c["fractions"], "top": c["top"], "random": c["random_mean"]}))
                        sanity.append({"output": output, "muscle": mname, "view": view,
                                       "spearman": map_similarity(score, other[view])})
    return curves, sanity


def interpret_subject(model, test_tokens, norm, cfg, device, participant, names, out_dir=None, data_dir=None,
                      kin_names=None, eeg_names=EEG_NAMES, log=print) -> dict:
    """Event-locked interpretability of one participant's test series (see the module docstring of explain.py
    for the views). Returns {"tables", "figures", "arrays"}; with out_dir it also writes them and closes the figures."""
    ic, names, eeg_names = cfg["interp"], list(names), list(eeg_names)
    window, clip, bin_size = int(cfg["tokens"]["window"]), cfg["eeg"]["clip"], int(ic["flow_bin"])
    kw = dict(window=window, clip=clip, residual=ic["residual"], bin_size=bin_size)
    ev_tokens = {(tok["series"], e): event_tokens(participant, tok["series"], tok["n_samples"], e, data_dir)
                 for tok in test_tokens for e in list(ic["events"]) + [REST]}

    def stack(output, muscle, event, flow=False, limit=None):
        return concat_events([event_locked(model, tok, norm, ev_tokens[(tok["series"], event)][:limit], device,
                                           output=output, muscle=muscle, flow=flow, **kw) for tok in test_tokens])

    n_kin = test_tokens[0]["kin"].shape[1]
    if kin_names is None or len(kin_names) != n_kin:
        kin_names = [f"kin {i}" for i in range(n_kin)]
    band_names = [f"{lo}-{hi} Hz" for lo, hi in cfg["eeg"]["bands"]]
    t_tok = (np.arange(window) - window // 2) / FS_TOK
    t_bin = (np.arange(0, window, bin_size) + (bin_size - 1) / 2 - window // 2) / FS_TOK
    tables, figures, arrays, rest_stacks = {k: [] for k in TABLES}, {}, {}, {}
    done = False

    for event in ic["events"]:
        graphs, contrast_maps, contrast_masks, input_groups = {}, {}, {}, {}
        for output in OUTPUTS:
            stacks = [stack(output, mi, event) for mi in range(len(names))]
            if not len(stacks[0]["meta"]):
                log(f"  {output} / {event}: no event window fits inside the test series; skipped")
                continue
            done = True
            label = OUTPUT_LABELS[output]
            fl = stack(output, 0, event, flow=True, limit=ic["max_events"])           # flow is the same for every muscle
            curves = {"rollout": ([], []), "relevance": ([], [])}
            profiles, shares = {}, []
            for mi, (mname, st) in enumerate(zip(names, stacks)):
                tag = {"output": output, "muscle": mname, "event": event}
                for view in ("rollout", "relevance"):
                    mean, sem = mean_sem(_share(st[f"{view}/time"]))
                    curves[view][0].append(mean)
                    curves[view][1].append(sem)
                    tables["time"].append(pd.DataFrame({**tag, "view": view, "time_s": t_tok, "mean": mean, "sem": sem}))
                    if output != "K":
                        mean, sem = mean_sem(_share(st[f"{view}/input"].sum(1)))        # (events, E) -> share per electrode
                        tables["electrodes"].append(pd.DataFrame({**tag, "view": view, "electrode": eeg_names,
                                                                  "mean": mean, "sem": sem}))
                        profiles["rollout" if view == "rollout" else f"relevance: {mname}"] = mean
                    if output != "F_eeg":
                        row = st[f"{view}/muscle"].mean(0)                              # rollout (M, M), relevance (M,)
                        row = row[mi] if view == "rollout" else row
                        tables["muscle_graph"].append(pd.DataFrame({**tag, "view": view, "source": names, "share": row}))
                        graphs.setdefault(f"{output} {view}", np.zeros((len(names), len(names))))[mi] = row
                if output == "K":
                    feat = np.concatenate([np.abs(st["saliency/kin"]).sum(1), np.abs(st["saliency/hist"]).sum(1)], axis=1)
                    feat_names = list(kin_names) + HIST_NAMES
                    tables["parts"].append(pd.DataFrame([{"muscle": mname, "event": event, "n": len(st["meta"]),
                                                          **st["meta"][list(PARTS)].mean().to_dict()}]))
                else:
                    feat, feat_names = np.abs(st["saliency/eeg"]).sum((1, 2)), band_names   # (events, bands)
                mean, sem = mean_sem(_share(feat))
                tables["inputs"].append(pd.DataFrame({**tag, "feature": feat_names, "mean": mean, "sem": sem}))
                shares.append(mean)
                if output == "E":
                    if mi not in rest_stacks:
                        rest_stacks[mi] = stack("E", mi, REST)
                    rest = rest_stacks[mi]
                    i, j = _pairs(st["meta"], rest["meta"])                             # each lift with the rest after it
                    if len(i) >= 3:
                        c = paired_contrast(_share(st["relevance/input"][i].sum(1)),
                                            _share(rest["relevance/input"][j].sum(1)), ic["n_perm"], ic["seed"])
                        tables["rest_contrast"].append(pd.DataFrame({**tag, "electrode": eeg_names, "diff": c["diff"],
                                                                     "t": c["t"], "p": c["p"], "n_pairs": len(i)}))
                        contrast_maps[mname] = np.nan_to_num(c["t"], nan=0.0)           # not testable: drawn as 0
                        contrast_masks[mname] = np.isfinite(c["p"]) & (c["p"] < 0.05)
                    elif mi == 0:
                        log(f"  E / {event}: {len(i)} lift-rest pairs, the rest contrast needs 3; skipped")
            for k in ("time_layers", "electrode_layers", "pool", "cross", "muscle_layers"):   # attention only: any muscle
                if f"layers/{k}" in stacks[0]:
                    arrays[f"{output}__{event}__{k}"] = stacks[0][f"layers/{k}"].mean(0)
            input_groups[f"{label}: {'features' if output == 'K' else 'frequency bands'}"] = (feat_names, np.mean(shares, axis=0))

            views = {"rollout": (t_tok, np.stack(curves["rollout"][0]), np.stack(curves["rollout"][1]))}
            if len(fl["meta"]):
                mean, sem = mean_sem(_share(fl["flow/time"]) / bin_size)                # share per token, like the others
                tables["time"].append(pd.DataFrame({"output": output, "muscle": "all", "event": event, "view": "flow",
                                                    "time_s": t_bin, "mean": mean, "sem": sem}))
                views["flow"] = (t_bin, np.tile(mean, (len(names), 1)), np.tile(sem, (len(names), 1)))
                if output != "K":
                    mean, sem = mean_sem(_share(fl["flow/input"].sum(1)))
                    tables["electrodes"].append(pd.DataFrame({"output": output, "muscle": "all", "event": event,
                                                              "view": "flow", "electrode": eeg_names, "mean": mean, "sem": sem}))
                    profiles = {"rollout": profiles["rollout"], "attention flow": mean,
                                **{k: v for k, v in profiles.items() if k != "rollout"}}
            views["relevance"] = (t_tok, np.stack(curves["relevance"][0]), np.stack(curves["relevance"][1]))
            figures[f"time_{output}_{event}"] = P.plot_time_views(
                views, names, f"{label}: where in time, around {event} (mean and SEM over lifts)", event)
            figures[f"layers_{output}_{event}"] = P.plot_layer_heads(
                arrays[f"{output}__{event}__time_layers"], t_tok,
                f"{label}: attention of the {event} token, per temporal layer and head", event)
            if output != "K":
                figures[f"scalp_{output}_{event}"] = P.plot_scalp(
                    profiles, eeg_names, f"{label}: which electrodes, around {event} (share per electrode)")
            if output == "E":
                layers = arrays[f"E__{event}__electrode_layers"].mean(1)                 # (layers, E, E), heads averaged
                figures[f"electrode_layers_{event}"] = P.plot_matrices(
                    {f"electrode layer {i + 1}": m for i, m in enumerate(layers)},
                    f"Electrode graph attention at {event} (row = receiving electrode)", eeg_names, eeg_names)
        if graphs:
            figures[f"muscle_graph_{event}"] = P.plot_matrices(
                graphs, f"Muscle graph at {event} (row = receiving muscle, column = source muscle)", names, names)
        if contrast_maps:
            figures[f"rest_contrast_{event}"] = P.plot_scalp(
                contrast_maps, eeg_names, f"E: {event} minus the rest after it, paired t per electrode (marked: corrected p < 0.05)",
                signed=True, masks=contrast_masks)
        if input_groups:
            figures[f"inputs_{event}"] = P.plot_shares(input_groups, f"Input shares at {event} (mean over muscles)")

    if done:
        curves, sanity = _faithfulness(model, test_tokens, norm, ev_tokens, cfg, device, names)
        if curves:
            faith = pd.concat(curves, ignore_index=True)
            tables["faithfulness"].append(faith.groupby(["output", "muscle", "view", "fraction"], as_index=False).mean())
            tables["sanity"].append(pd.DataFrame(sanity).groupby(["output", "muscle", "view"], as_index=False).mean())
            g = faith.groupby(["output", "view", "fraction"], as_index=False)[["top", "random"]].mean()
            panels = {OUTPUT_LABELS[o]: {v: (d["fraction"].to_numpy(), d["top"].to_numpy(), d["random"].to_numpy())
                                         for v, d in g[g["output"] == o].groupby("view")} for o in ("E", "K")}
            figures["deletion"] = P.plot_deletion(panels, "Deletion check: removing what each view ranks highest")

    result = {"tables": {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in tables.items()},
              "figures": figures, "arrays": arrays}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, df in result["tables"].items():
            df.to_csv(out_dir / f"{k}.csv", index=False)
        np.savez_compressed(out_dir / "layers.npz", **arrays)
        for k, fig in figures.items():
            fig.savefig(out_dir / f"{k}.png", dpi=130)
            plt.close(fig)
        result["files"] = sorted(str(p) for p in out_dir.glob("*.png"))
    return result
