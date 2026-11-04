"""Run the prediction-error pilot. No training: every step reads the raw WS files and the lift tables.

    env/Scripts/python.exe -m pe_pilot.run                         # all steps, all participants
    env/Scripts/python.exe -m pe_pilot.run --step epochs --participants 1-3
    env/Scripts/python.exe -m pe_pilot.run --step error --results-dir <folder>

Steps: epochs (cache per participant), behaviour, error, expectation, learning. Each writes to
results/pe_pilot/step{n}_{name}/.
"""
import argparse
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from pe_pilot import analysis, plots
from pe_pilot.config import CONFIG
from pe_pilot.data import load_epochs

STEPS = ["epochs", "behaviour", "error", "expectation", "learning"]


def step_dir(results_dir, name) -> Path:
    d = Path(results_dir) / f"step{STEPS.index(name) + 1}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(steps, participants, data_dir=CONFIG["data_dir"], results_dir=CONFIG["results_dir"], rebuild=False) -> dict:
    out = {}
    t0 = time.time()
    epochs = {}
    for p in participants:
        epochs[p] = load_epochs(p, data_dir, results_dir, rebuild=rebuild)
        t = epochs[p]["table"]
        print(f"[epochs] P{p}: {len(t)} lifts, {int(t['rejected'].sum())} rejected, "
              f"{dict(t['change'].value_counts())}  ({time.time() - t0:.0f} s)")
    if "epochs" in steps:
        counts = pd.DataFrame([{"participant": p, "lifts": len(e["table"]), "rejected": int(e["table"]["rejected"].sum()),
                                **e["table"].loc[~e["table"]["rejected"], "change"].value_counts().to_dict()}
                               for p, e in epochs.items()])
        counts.to_csv(step_dir(results_dir, "epochs") / "counts.csv", index=False)
        out["counts"] = counts
    if "behaviour" in steps:
        d = step_dir(results_dir, "behaviour")
        per, group = analysis.behaviour(epochs)
        per.to_csv(d / "behaviour_participants.csv", index=False)
        group.to_csv(d / "behaviour_group.csv", index=False)
        _save(plots.behaviour_figure(per), d / "behaviour.png")
        out["behaviour"] = group
        print("[behaviour]\n", group.round(4).to_string(index=False))
    if "error" in steps:
        d = step_dir(results_dir, "error")
        summary, detail = analysis.error(epochs)
        summary.to_csv(d / "error_summary.csv", index=False)
        for (kind, signal), v in detail["contrasts"].items():
            v["clusters"].to_csv(d / f"clusters_{kind}_{signal}.csv", index=False)
        for kind in ("heavier", "lighter"):
            _save(plots.error_figure(detail, kind), d / f"error_{kind}.png")
        out["error"], out["error_detail"] = summary, detail
        print("[error]\n", summary.round(4).to_string(index=False))
    if "expectation" in steps:
        d = step_dir(results_dir, "expectation")
        per, group = analysis.expectation(epochs)
        per.to_csv(d / "expectation_participants.csv", index=False)
        group.to_csv(d / "expectation_group.csv", index=False)
        _save(plots.expectation_figure(per), d / "expectation.png")
        out["expectation"], out["expectation_per"] = group, per
        print("[expectation]\n", group.round(4).to_string(index=False))
    if "learning" in steps:
        d = step_dir(results_dir, "learning")
        per, group = analysis.learning(epochs)
        per.to_csv(d / "learning_participants.csv", index=False)
        group.to_csv(d / "learning_group.csv", index=False)
        out["learning"], out["learning_per"] = group, per
        print("[learning]\n", group.round(4).to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return out


def _participants(s):
    if s == "all":
        return CONFIG["participants"]
    if "-" in s:
        a, b = map(int, s.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default=",".join(STEPS))
    ap.add_argument("--participants", default="all")
    ap.add_argument("--data-dir", default=CONFIG["data_dir"])
    ap.add_argument("--results-dir", default=CONFIG["results_dir"])
    ap.add_argument("--rebuild", action="store_true", help="recompute the epoch cache")
    a = ap.parse_args(argv)
    steps = a.step.split(",")
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps {sorted(unknown)}; choose from {STEPS}")
    matplotlib.use("Agg")
    run(steps, _participants(a.participants), a.data_dir, a.results_dir, a.rebuild)


if __name__ == "__main__":
    main()
