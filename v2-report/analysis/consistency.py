"""How alike are the rollout and attention-flow maps of the 12 participants?

Reads outputs/interpretability_v2/P*/ (read only); writes v2-report/tables/cons_*.tex,
v2-report/figures/cons_*.png and v2-report/analysis/consistency.csv. No model is run.

Run from the repository root:  env/Scripts/python.exe v2-report/analysis/consistency.py
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v2.electrodes import EEG_NAMES          # noqa: E402
from v2.interp import plots as P             # noqa: E402

IV2 = ROOT / "outputs" / "interpretability_v2"
REP = ROOT / "v2-report"
FIG, TAB = REP / "figures", REP / "tables"
SUBJ = [f"P{i}" for i in range(1, 13)]
MUSCLES = ["AD", "BR", "FD", "ED", "FDI"]
EVENTS = ["first_touch", "lift_off"]
RNG = np.random.default_rng(0)
N_PERM, N_SPLIT = 2000, 500


def load(name):
    out = []
    for s in SUBJ:
        d = pd.read_csv(IV2 / s / f"{name}.csv")
        d["subject"] = s
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ agreement metrics on a subjects x features matrix
def pairwise(X, kind="spearman"):
    vals = []
    for i, j in combinations(range(len(X)), 2):
        a, b = X[i], X[j]
        vals.append(spearmanr(a, b)[0] if kind == "spearman" else np.corrcoef(a, b)[0, 1])
    return np.array(vals)


def loo(X, kind="spearman"):
    out = []
    for i in range(len(X)):
        rest = np.delete(X, i, axis=0).mean(0)
        out.append(spearmanr(X[i], rest)[0] if kind == "spearman" else np.corrcoef(X[i], rest)[0, 1])
    return np.array(out)


def kendall_w(X):
    """Kendall's coefficient of concordance of m raters (rows) ranking n items (columns), with chi-square p."""
    m, n = X.shape
    R = np.vstack([rankdata(x) for x in X])
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    stat = m * (n - 1) * W
    return W, float(chi2.sf(stat, n - 1))


def split_half(X, kind="spearman"):
    """Mean over random 6/6 splits of the correlation of the two half-means, Spearman-Brown corrected."""
    m = len(X)
    rs = []
    for _ in range(N_SPLIT):
        idx = RNG.permutation(m)
        a, b = X[idx[: m // 2]].mean(0), X[idx[m // 2:]].mean(0)
        r = spearmanr(a, b)[0] if kind == "spearman" else np.corrcoef(a, b)[0, 1]
        rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs))


def null_pairwise(X, kind="spearman", circular=False):
    """Null of the mean pairwise correlation: features permuted (or circularly shifted) within every subject."""
    vals = []
    for _ in range(N_PERM // 4):          # pairwise is O(m^2); 500 draws are enough for p >= 0.002
        Y = np.vstack([np.roll(x, RNG.integers(len(x))) if circular else RNG.permutation(x) for x in X])
        vals.append(np.nanmean(pairwise(Y, kind)))
    return np.array(vals)


def summary(X, label, kind="spearman", circular=False):
    pw, lo = pairwise(X, kind), loo(X, kind)
    null = null_pairwise(X, kind, circular)
    obs = float(np.nanmean(pw))
    W, pW = kendall_w(X)
    return {"map": label, "n_features": X.shape[1], "mean_pairwise": obs, "min_pairwise": float(np.nanmin(pw)),
            "loo_mean": float(np.nanmean(lo)), "loo_min": float(np.nanmin(lo)),
            "split_half": split_half(X, kind), "kendall_W": W, "p_W": pW,
            "null_mean": float(null.mean()), "null_95": float(np.quantile(null, 0.95)),
            "p_perm": float((1 + (null >= obs).sum()) / (1 + len(null)))}


rows = []

# ------------------------------------------------------------------ 1. electrode maps
el = load("electrodes")


def elec_matrix(output, event, view, muscle=None):
    d = el[(el.output == output) & (el.event == event) & (el.view == view)]
    if muscle is not None:
        d = d[d.muscle == muscle]
    elif view == "relevance":                                     # muscle-specific: mean over the five muscles
        d = d.groupby(["subject", "electrode"], as_index=False)["mean"].mean()
    else:                                                         # rollout and flow are the same for every muscle
        d = d.groupby(["subject", "electrode"], as_index=False)["mean"].first()
    return np.vstack([d[d.subject == s].set_index("electrode").loc[EEG_NAMES, "mean"].to_numpy() for s in SUBJ])


ELEC = {}
for output in ("E", "F_eeg"):
    for event in EVENTS:
        for view in ("rollout", "relevance", "flow"):
            X = elec_matrix(output, event, view)
            ELEC[(output, event, view)] = X
            rows.append({"family": "electrodes", "output": output, "event": event, "view": view,
                         **summary(X, f"{output}/{event}/{view}")})
        for m in MUSCLES:                                          # gradient-weighted rollout per muscle
            X = elec_matrix(output, event, "relevance", m)
            rows.append({"family": "electrodes_muscle", "output": output, "event": event, "view": f"relevance {m}",
                         **summary(X, f"{output}/{event}/relevance/{m}")})

# top-3 electrodes per subject and group test against the uniform share 1/32
top3 = {}
for key in [("E", "lift_off", "rollout"), ("E", "lift_off", "relevance"), ("E", "first_touch", "relevance")]:
    X = ELEC[key]
    cnt = pd.Series(0, index=EEG_NAMES)
    for x in X:
        for e in np.array(EEG_NAMES)[np.argsort(-x)[:3]]:
            cnt[e] += 1
    d = X - 1 / 32
    t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))
    mx = []
    for _ in range(N_PERM):
        f = RNG.choice([-1, 1], size=(len(d), 1)) * d
        mx.append(np.abs(f.mean(0) / (f.std(0, ddof=1) / np.sqrt(len(f)))).max())
    mx = np.array(mx)
    p = np.array([(1 + (mx >= abs(v)).sum()) / (1 + len(mx)) for v in t])
    top3[key] = pd.DataFrame({"electrode": EEG_NAMES, "group_mean": X.mean(0), "t": t, "p_maxT": p,
                              "n_top3": cnt.values}).sort_values("group_mean", ascending=False)

# ------------------------------------------------------------------ 2. time profiles
tm = load("time")
TIME = {}
for output in ("K", "E"):
    for event in EVENTS:
        for view in ("rollout", "relevance"):
            d = tm[(tm.output == output) & (tm.event == event) & (tm.view == view)]
            d = d.groupby(["subject", "time_s"], as_index=False)["mean"].mean()
            t = np.sort(d.time_s.unique())
            keep = np.abs(t) > 0.05                                     # the explained token and its stem footprint left out
            X = np.vstack([d[d.subject == s].set_index("time_s").loc[t, "mean"].to_numpy()[keep] for s in SUBJ])
            X = X / X.sum(1, keepdims=True)
            TIME[(output, event, view)] = (t[keep], X)
            r = summary(X, f"{output}/{event}/{view}", kind="pearson", circular=True)
            inner = np.abs(t[keep]) <= 3.0                              # the window edges (beyond 3 s) rise in every map
            pk = t[keep][inner][X[:, inner].argmax(1)]
            r.update(peak_mean=float(pk.mean()), peak_sd=float(pk.std(ddof=1)))
            rows.append({"family": "time", "output": output, "event": event, "view": view, **r})
    for event in EVENTS:
        d = tm[(tm.output == output) & (tm.event == event) & (tm.view == "flow_excess") & (tm.muscle == "all")]
        t = np.sort(d.time_s.unique())
        X = np.vstack([d[d.subject == s].set_index("time_s").loc[t, "mean"].to_numpy() for s in SUBJ])
        keep = np.isfinite(X).all(0) & (np.abs(t) > 0.05)
        X = X[:, keep]
        TIME[(output, event, "flow_excess")] = (t[keep], X)
        rows.append({"family": "time", "output": output, "event": event, "view": "flow_excess",
                     **summary(X, f"{output}/{event}/flow_excess", kind="pearson", circular=True)})

# ------------------------------------------------------------------ 3. muscle graphs (off-diagonal entries)
mg = load("muscle_graph")
off = ~np.eye(5, dtype=bool)
for output in ("K", "E"):
    for event in EVENTS:
        for view in ("rollout", "relevance"):
            mats = []
            for s in SUBJ:
                d = mg[(mg.subject == s) & (mg.output == output) & (mg.event == event) & (mg.view == view)]
                M = d.pivot_table(index="muscle", columns="source", values="share").loc[MUSCLES, MUSCLES].to_numpy()
                mats.append(M[off])
            X = np.vstack(mats)
            rows.append({"family": "muscle_graph", "output": output, "event": event, "view": view,
                         **summary(X, f"{output}/{event}/{view}")})

# ------------------------------------------------------------------ 4. EEG band shares (gradient x input)
inp = load("inputs")
BANDS = None
for event in EVENTS:
    d = inp[(inp.output == "E") & (inp.event == event)].groupby(["subject", "feature"], as_index=False)["mean"].mean()
    BANDS = list(dict.fromkeys(d.feature))
    X = np.vstack([d[d.subject == s].set_index("feature").loc[BANDS, "mean"].to_numpy() for s in SUBJ])
    W, pW = kendall_w(X)
    rows.append({"family": "bands", "output": "E", "event": event, "view": "gradient x input", "map": f"E/{event}/bands",
                 "n_features": 4, "kendall_W": W, "p_W": pW, "mean_pairwise": float(np.nanmean(pairwise(X))),
                 "loo_mean": float(np.nanmean(loo(X)))})
    if event == "lift_off":
        BAND_X = X

# ------------------------------------------------------------------ 5. event-against-rest contrast (t maps)
rc = load("rest_contrast")
for event in EVENTS:
    d = rc[(rc.event == event)].groupby(["subject", "electrode"], as_index=False)["t"].mean()
    X = np.vstack([d[d.subject == s].set_index("electrode").loc[EEG_NAMES, "t"].fillna(0).to_numpy() for s in SUBJ])
    rows.append({"family": "rest_contrast", "output": "E", "event": event, "view": "t map (mean over muscles)",
                 **summary(X, f"E/{event}/rest t")})
    sig = rc[(rc.event == event) & (rc.p < 0.05)].groupby("subject").electrode.nunique().reindex(SUBJ, fill_value=0)
    rows[-1]["n_sig_mean"] = float(sig.mean())

res = pd.DataFrame(rows)
res.to_csv(REP / "analysis" / "consistency.csv", index=False)
for k, v in top3.items():
    v.to_csv(REP / "analysis" / f"group_{'_'.join(k)}.csv", index=False)

# ------------------------------------------------------------------ figures
VIEW_LBL = {"rollout": "rollout", "relevance": "gradient-weighted rollout", "flow": "attention flow",
            "flow_excess": "flow - uniform"}

# (a) agreement of the electrode maps against the null
sel = res[(res.family == "electrodes")]
with P.style():
    fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
    x = np.arange(len(sel))
    ax.bar(x - 0.2, sel.mean_pairwise, 0.4, color=P.VIEW_COLORS["rollout"], label="mean pairwise Spearman $\\rho$")
    ax.bar(x + 0.2, sel.loo_mean, 0.4, color=P.VIEW_COLORS["relevance"], label="leave-one-out $\\rho$")
    ax.scatter(x, sel.null_95, marker="_", s=400, color="black", label="null, 95th percentile", zorder=3)
    ax.axhline(0, color=P.MUTED, linewidth=0.8)
    ax.set_xticks(x, [f"{o}\n{e}\n{VIEW_LBL[v]}" for o, e, v in zip(sel.output, sel.event, sel.view)], fontsize=6.5)
    ax.set_ylabel("agreement across participants")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    ax.set_title("Electrode maps: how alike are the 12 participants?", loc="left", fontweight="bold")
    fig.savefig(FIG / "cons_electrodes.png", dpi=150)
    plt.close(fig)

# (b) group mean scalp maps and top-3 counts
maps, tight = {}, []
for key, lbl in [(("E", "lift_off", "rollout"), "E rollout, lift-off"),
                 (("E", "lift_off", "relevance"), "E grad.-weighted, lift-off"),
                 (("E", "first_touch", "relevance"), "E grad.-weighted, first touch")]:
    g = top3[key].set_index("electrode").loc[EEG_NAMES]
    maps[f"{lbl}\ngroup mean"] = g.group_mean.to_numpy()
    maps[f"{lbl}\nparticipants with it in top 3"] = g.n_top3.to_numpy().astype(float)
fig = P.plot_scalp(maps, EEG_NAMES, "Group maps over 12 participants")
fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] + 0.35)
with P.style():
    fig.savefig(FIG / "cons_scalp.png", dpi=150)
plt.close(fig)

# (c) time profiles of every participant
with P.style():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained", sharex=True)
    for row, output in enumerate(("K", "E")):
        for col, view in enumerate(("rollout", "relevance", "flow_excess")):
            ax = axes[row, col]
            t, X = TIME[(output, "lift_off", view)]
            for x in X:
                ax.plot(t, x, color=P.CONTEXT, linewidth=0.7)
            ax.plot(t, X.mean(0), color=P.VIEW_COLORS["flow" if view == "flow_excess" else view], linewidth=2)
            r = res[(res.family == "time") & (res.output == output) & (res.event == "lift_off") & (res.view == view)].iloc[0]
            ax.set_title(f"{output}, {VIEW_LBL[view]}: pairwise r = {r.mean_pairwise:.2f}, LOO r = {r.loo_mean:.2f}",
                         loc="left", fontsize=8, fontweight="bold")
            ax.axvline(0, color=P.MUTED, linewidth=0.8)
            if row == 1:
                ax.set_xlabel("time from lift_off (s)")
    axes[0, 0].set_ylabel("share (K)"); axes[1, 0].set_ylabel("share (E)")
    fig.suptitle("Time profiles at lift_off: every participant (grey) and the mean (colour)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(FIG / "cons_time.png", dpi=150)
    plt.close(fig)

# (d) participant x participant similarity of the E gradient-weighted electrode map
X = ELEC[("E", "lift_off", "relevance")]
S = np.array([[spearmanr(a, b)[0] for b in X] for a in X])
with P.style():
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    im = ax.imshow(S, cmap=P.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(12), SUBJ, fontsize=7, rotation=90); ax.set_yticks(range(12), SUBJ, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman $\\rho$")
    ax.set_title("E, gradient-weighted rollout, lift_off:\nelectrode map similarity", loc="left", fontsize=8, fontweight="bold")
    fig.savefig(FIG / "cons_similarity.png", dpi=150)
    plt.close(fig)

# (e) band shares per participant
with P.style():
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    bottom = np.zeros(12)
    for k, b in enumerate(BANDS):
        ax.bar(SUBJ, BAND_X[:, k], bottom=bottom, label=b, color=plt.get_cmap("viridis")(k / 3))
        bottom += BAND_X[:, k]
    ax.set_ylabel("share of |gradient x input|")
    ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("E at lift_off: share of each EEG band, per participant", loc="left", fontweight="bold")
    fig.savefig(FIG / "cons_bands.png", dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------ tables (LaTeX, Persian captions, English terms)
_FA = str.maketrans("0123456789.-", "۰۱۲۳۴۵۶۷۸۹٫−")


def fa(x, nd=2):
    if x is None or not np.isfinite(x):
        return "--"
    return rf"\lr{{{f'{x:.{nd}f}'.translate(_FA)}}}"


def pfa(p):
    return r"\lr{<۰٫۰۰۱}" if p < 0.001 else fa(p, 3)


def lr(s):
    return rf"\lr{{{str(s).replace('_', '-')}}}"


def table(name, header, body, caption, label, align=None):
    align = align or "l" + "c" * (len(header) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


HEAD = ["نقشه", r"\lr{pairwise $\rho$}", r"\lr{LOO $\rho$}", r"\lr{split-half}", r"\lr{Kendall's W}", r"\lr{null 95\%}", r"\lr{p}"]


def body(d, name_fn):
    return [[name_fn(r), fa(r.mean_pairwise), fa(r.loo_mean), fa(r.split_half), fa(r.kendall_W), fa(r.null_95), pfa(r.p_perm)]
            for r in d.itertuples()]


d = res[res.family == "electrodes"]
table("cons_electrodes", HEAD, body(d, lambda r: lr(f"{r.output}, {r.event}, {VIEW_LBL[r.view]}")),
      r"یکسانی نقشه‌ی الکترودها بین ۱۲ شرکت‌کننده. \lr{pairwise $\rho$}: میانگین همبستگی اسپیرمن روی ۶۶ جفت؛ \lr{LOO}: هر نفر در برابر میانگین ۱۱ نفر دیگر؛ \lr{split-half}: همبستگی میانگین دو نیمه‌ی تصادفی با تصحیح اسپیرمن-براون؛ \lr{null 95\%} و \lr{p}: در برابر جابه‌جایی برچسب الکترودها در هر شرکت‌کننده.",
      "tab:cons-elec", align="lcccccc")
d = res[res.family == "electrodes_muscle"]
d = d[d.output == "E"]
table("cons_electrodes_muscle", HEAD, body(d, lambda r: lr(f"E, {r.event}, {r.view.split()[1]}")),
      r"همان معیارها برای نقشه‌ی \lr{gradient-weighted rollout} هر عضله در خروجی \lr{E}.", "tab:cons-elec-musc", align="lcccccc")
d = res[res.family == "time"]
table("cons_time", HEAD + [r"\lr{peak (s)}"],
      [b + [fa(r.peak_mean) + r" $\pm$ " + fa(r.peak_sd) if np.isfinite(r.peak_mean if hasattr(r, 'peak_mean') else np.nan) else "--"]
       for b, r in zip(body(d, lambda r: lr(f"{r.output}, {r.event}, {VIEW_LBL[r.view]}")), d.itertuples())],
      r"یکسانی منحنی‌های زمانی بین شرکت‌کننده‌ها (همبستگی پیرسون؛ توکن خودِ رخداد کنار گذاشته شده). \lr{null}: جابه‌جایی دایره‌ای منحنی هر شرکت‌کننده. ستون آخر: زمان قله نسبت به رخداد، میانگین $\pm$ انحراف معیار بین شرکت‌کننده‌ها.",
      "tab:cons-time", align="lccccccc")
d = res[res.family == "muscle_graph"]
table("cons_muscle", HEAD, body(d, lambda r: lr(f"{r.output}, {r.event}, {VIEW_LBL[r.view]}")),
      r"یکسانی \lr{muscle graph} (۲۰ درایه‌ی غیرقطری ماتریس ۵×۵) بین شرکت‌کننده‌ها.", "tab:cons-muscle", align="lcccccc")
d = res[res.family == "rest_contrast"]
table("cons_rest", ["نقشه", r"\lr{pairwise $\rho$}", r"\lr{LOO $\rho$}", r"\lr{Kendall's W}", r"\lr{null 95\%}", r"\lr{p}", "الکترودهای معنادار (میانگین)"],
      [[lr(f"E, {r.event}, event - rest t"), fa(r.mean_pairwise), fa(r.loo_mean), fa(r.kendall_W), fa(r.null_95), pfa(r.p_perm), fa(r.n_sig_mean, 1)] for r in d.itertuples()],
      r"یکسانی نقشه‌های $t$ مقایسه‌ی رخداد با استراحت بین شرکت‌کننده‌ها.", "tab:cons-rest", align="lcccccc")
g = top3[("E", "lift_off", "relevance")].head(8)
table("cons_top", [r"\lr{electrode}", "میانگین گروه (درصد)", r"\lr{t} در برابر یکنواخت", r"\lr{p (max-T)}", "شمار افراد با آن در سه الکترود اول"],
      [[lr(r.electrode), fa(100 * r.group_mean), fa(r.t), pfa(r.p_maxT), lr(f"{r.n_top3} / 12")] for r in g.itertuples()],
      r"هشت الکترود با بیشترین سهم میانگین در \lr{gradient-weighted rollout} خروجی \lr{E} هنگام \lr{lift-off}. آزمون $t$ تک‌نمونه‌ای سهم هر الکترود در برابر سهم یکنواخت (\lr{۳٫۱۲۵} درصد) با تصحیح \lr{max-T} روی ۳۲ الکترود.",
      "tab:cons-top", align="lcccc")
d = res[res.family == "bands"]
bm = BAND_X.mean(0); bs = BAND_X.std(0, ddof=1)
dom = pd.Series([BANDS[i] for i in BAND_X.argmax(1)]).value_counts()
table("cons_bands", [r"\lr{band}", "سهم میانگین (درصد)", "انحراف معیار", "شمار افراد با بیشترین سهم"],
      [[lr(b), fa(100 * bm[k], 1), fa(100 * bs[k], 1), lr(f"{dom.get(b, 0)} / 12")] for k, b in enumerate(BANDS)],
      rf"سهم هر باند \lr{{EEG}} در خروجی \lr{{E}} هنگام \lr{{lift-off}} (\lr{{gradient × input}}). \lr{{Kendall's W}} روی ترتیب باندها: {fa(d[d.event == 'lift_off'].kendall_W.iloc[0])} (\lr{{p}} = {pfa(d[d.event == 'lift_off'].p_W.iloc[0])}).",
      "tab:cons-bands", align="lccc")

pd.set_option("display.width", 220)
print(res[["family", "output", "event", "view", "mean_pairwise", "loo_mean", "split_half", "kendall_W", "p_W", "null_95", "p_perm"]].round(3).to_string())
for k, v in top3.items():
    print(k); print(v.head(8).round(4).to_string(index=False))
print("bands", BANDS, (100 * bm).round(1), (100 * bs).round(1), dom.to_dict())
for k in TIME:
    pass
print(res[res.family == "time"][["output", "event", "view", "peak_mean", "peak_sd"]].round(2).to_string())
