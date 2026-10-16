"""Tables and figures of the EEG control experiments (phase baseline, reference maps, forecast horizon) for the report.

Reads eeg_controls/results/ (phase, reference, figures), results/v2_horizon/ and outputs/interpretability_v2/ (all read
only); writes v2-report/tables/ctrl_*.tex and copies the figures to v2-report/figures/ctrl_*.png. No model is run.

Run from the repository root:  env/Scripts/python.exe v2-report/analysis/eeg_controls_results.py
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eeg_controls.reference_maps import rollout_group   # noqa: E402
from v2.electrodes import EEG_NAMES             # noqa: E402

CTRL = ROOT / "eeg_controls" / "results"
HORIZON = ROOT / "results" / "v2_horizon"
REP = ROOT / "v2-report"
FIG, TAB = REP / "figures", REP / "tables"

_FA = str.maketrans("0123456789.-", "۰۱۲۳۴۵۶۷۸۹٫−")


def fa(x, nd=2, sign=False):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return rf"\lr{{{s.translate(_FA)}}}"


def lr(s):
    return rf"\lr{{{str(s).replace('_', '-')}}}"


def groups(spec):
    """Header row over column groups: spec = [(title, n_columns), ...] after the first (row label) column."""
    cells, rules, col = [""], [], 2
    for title, n in spec:
        cells.append(rf"\multicolumn{{{n}}}{{c}}{{{title}}}")
        rules.append(rf"\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    return " & ".join(cells) + r" \\ " + " ".join(rules)


def table(name, header, body, caption, label, align=None, wide=False, top=None):
    align = align or "l" + "c" * (len(header) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             *([r"\resizebox{\textwidth}{!}{%"] if wide else []),
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", *([groups(top)] if top else []),
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [(" & ".join(r) + r" \\") if r != "MID" else r"\midrule" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if wide else []) + [r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def frac(k, n):
    return rf"\lr{{{int(k)}/{int(n)}}}".translate(_FA)


# ------------------------------------------------------------------ figures
for f in sorted((CTRL / "figures").glob("*.png")):
    shutil.copy(f, FIG / f"ctrl_{f.name}")

# ------------------------------------------------------------------ A: phase baseline
cmp = pd.read_csv(CTRL / "phase" / "phase_vs_E.csv")
models = ["cue", "coarse", "phase", "phase_weight", "E", "K"]
subj = cmp.groupby("subject")[[f"{m}_r" for m in models] + [f"{m}_vaf" for m in models]].mean()
body = [[lr(s)] + [fa(row[f"{m}_r"], 3) for m in models] + [fa(row[f"{m}_vaf"], 1) for m in models]
        for s, row in subj.iterrows()]
mean = subj.mean()
body += ["MID", [r"\textbf{میانگین}"] + [fa(mean[f"{m}_r"], 3) for m in models] + [fa(mean[f"{m}_vaf"], 1) for m in models]]
names = [r"\lr{cue}", r"\lr{coarse}", r"\lr{phase}", r"\lr{+weight}", r"\lr{E}", r"\lr{K}"]
table("ctrl_phase", ["شرکت‌کننده"] + names + names, body,
      r"مدل‌های زمان‌بندی تکلیف در برابر \lr{E} و \lr{K} روی سری‌های آزمون. شش ستون اول $r$ و شش ستون بعد \lr{VAF} (درصد)، هر دو میانگین روی پنج عضله. \lr{cue}: فقط روشن شدن \lr{LED}؛ \lr{coarse}: هشت رخداد با گام نیم‌ثانیه؛ \lr{phase}: هشت رخداد با گام ریز و مرحله‌های تکلیف؛ \lr{+weight}: همان به‌علاوه‌ی وزن جسم. \lr{E} و \lr{K} از مدل غیرعلّی فصل‌های پیش‌اند.",
      "tab:ctrl-phase", wide=True, align="l" + "c" * 12, top=[(r"$r$", 6), (r"\lr{VAF} (درصد)", 6)])

# ------------------------------------------------------------------ B: reference maps
ref = pd.read_csv(CTRL / "reference" / "reference_maps.csv")
grp = pd.read_csv(CTRL / "reference" / "group.csv")
subjects = sorted(ref.subject.unique(), key=lambda s: int(s[1:]))
nums = [int(s[1:]) for s in subjects]
erd = ref[(ref.kind == "erd")]
beta_lo = erd[(erd.band == "beta") & (erd.window == "lift_off")].groupby("electrode")["value"].mean()
alpha_lo = erd[(erd.band == "alpha") & (erd.window == "lift_off")].groupby("electrode")["value"].mean()
cmc = ref[ref.kind == "cmc"].groupby("electrode")["value"].mean()
roll = pd.Series(rollout_group(nums, "lift_off", "rollout"), index=EEG_NAMES)
neg_beta = erd[(erd.band == "beta") & (erd.window == "lift_off")].pivot(index="electrode", columns="subject", values="value")

order = beta_lo.sort_values().index[:8]
body = [[lr(e), fa(beta_lo[e], 1), fa(alpha_lo[e], 1), frac((neg_beta.loc[e] < 0).sum(), len(subjects)),
         fa(100 * roll[e], 2), rf"\lr{{{int(roll.rank(ascending=False)[e])}}}".translate(_FA)] for e in order]
table("ctrl_erd_top", [r"\lr{electrode}", r"\lr{ERD} بتا (درصد)", r"\lr{ERD} آلفا (درصد)", "افراد با کاهش بتا",
                        r"سهم \lr{rollout} \lr{E} (درصد)", r"رتبه در \lr{rollout}"], body,
      r"هشت الکترود با بیشترین کاهش توان بتا در ثانیه‌ی پیش از \lr{lift-off}، میانگین گروه. عدد منفی یعنی توان نسبت به استراحت پیش از حرکت کم شده است. دو ستون آخر سهم همان الکترود در \lr{rollout} خروجی \lr{E} هنگام \lr{lift-off} (سهم یکنواخت \lr{۳٫۱۲۵} درصد) و رتبه‌ی آن بین ۳۲ الکترود.",
      "tab:ctrl-erd-top")

order = cmc.sort_values(ascending=False).index[:8]
lvl = ref[ref.kind == "cmc"].cmc_sig_level.mean()
body = [[lr(e), fa(cmc[e], 4), fa(beta_lo[e], 1), fa(100 * roll[e], 2)] for e in order]
table("ctrl_cmc_top", [r"\lr{electrode}", r"\lr{CMC} (میانگین عضله‌ها)", r"\lr{ERD} بتا (درصد)", r"سهم \lr{rollout} \lr{E} (درصد)"], body,
      rf"هشت الکترود با بیشترین \lr{{CMC}} بتا (۱۵ تا ۳۰ هرتز) در نگه‌داشتن، میانگین پنج عضله و {len(subjects)} نفر. سطح معناداری ۹۵ درصد برای یک نفر حدود {fa(lvl, 4)} است.",
      "tab:ctrl-cmc-top")

refs = ["-ERD alpha lift_off", "-ERD beta lift_off", "-ERD alpha hold", "-ERD beta hold", "CMC beta hold"]
ref_fa = {"-ERD alpha lift_off": r"\lr{$-$ERD} آلفا، \lr{lift-off}", "-ERD beta lift_off": r"\lr{$-$ERD} بتا، \lr{lift-off}",
          "-ERD alpha hold": r"\lr{$-$ERD} آلفا، نگه‌داشتن", "-ERD beta hold": r"\lr{$-$ERD} بتا، نگه‌داشتن",
          "CMC beta hold": r"\lr{CMC} بتا، نگه‌داشتن"}
part = grp[(grp.level == "participants") & (grp.muscle == "mean") & (grp.event == "lift_off")]
gmap = grp[(grp.level == "group map") & (grp.event == "lift_off")]
body = []
for r_ in refs:
    row = [ref_fa[r_]]
    for view in ("rollout", "relevance"):
        a = part[(part.view == view) & (part.reference == r_)].iloc[0]
        g = gmap[(gmap.view == view) & (gmap.reference == r_)].iloc[0]
        row += [fa(a.mean_rho, 2, True), frac(a.n_positive, a.n), fa(a.p, 2), fa(g.rho, 2, True), fa(g.p, 3)]
    body.append(row)
h = [r"$\bar\rho$", "مثبت", r"\lr{p}", r"$\rho$ گروه", r"\lr{p}"]
table("ctrl_ref_rho", ["نقشه‌ی مرجع"] + h + h, body,
      r"همبستگی اسپیرمن نقشه‌ی الکترود خروجی \lr{E} هنگام \lr{lift-off} با نقشه‌های مرجع، روی ۳۲ الکترود. پنج ستون اول \lr{rollout} و پنج ستون بعد \lr{gradient-weighted rollout}. $\bar\rho$: میانگین روی افراد؛ «مثبت»: شمار افراد با $\rho>0$؛ \lr{p} بعد از آن: \lr{sign-flip} روی افراد. $\rho$ گروه: نقشه‌ی میانگین گروه در برابر نقشه‌ی مرجع میانگین گروه؛ \lr{p} آن: جابه‌جایی برچسب الکترودها. علامت \lr{ERD} برعکس شده تا عدد مثبت یعنی «جایی که توان بیشتر کم شده، \lr{rollout} هم بیشتر است».",
      "tab:ctrl-ref-rho", wide=True, align="l" + "c" * 10,
      top=[(r"\lr{rollout}", 5), (r"\lr{gradient-weighted rollout}", 5)])

# ------------------------------------------------------------------ C: forecast horizon
summ = pd.read_csv(HORIZON / "summary.csv")
summ["res_sig"] = summ.res_p_tokens <= 0.05
summ["pre_sig"] = (summ.pre_move_p_ridge <= 0.05) & (summ.pre_move_skill_ridge > 0)
n_subj = summ.subject.nunique()


def wil(x, alternative="greater"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x != 0)]
    return float(wilcoxon(x, alternative=alternative).pvalue) if len(x) >= 5 else np.nan


p_fk = {h: wil(d.F_minus_K_vaf) for h, d in summ.groupby("horizon_ms")}
p_pre = {h: wil(d.pre_move_skill_ridge) for h, d in summ.groupby("horizon_ms")}
body = []
for h_ms, d in summ.groupby("horizon_ms"):
    sem = lambda c: d[c].std(ddof=1) / np.sqrt(len(d))
    body.append([rf"\lr{{{h_ms}}}".translate(_FA),
                 fa(d.K_vaf.mean(), 1) + r"\,$\pm$\," + fa(sem("K_vaf"), 1),
                 fa(d.K_r.mean(), 3), fa(d.E_r.mean(), 3),
                 fa(d.F_minus_K_vaf.mean(), 2, True) + r"\,$\pm$\," + fa(sem("F_minus_K_vaf"), 2), fa(p_fk[h_ms], 3),
                 fa(d.res_r_tokens.mean(), 3), frac(d.res_sig.sum(), len(d)),
                 fa(d.pre_move_skill_ridge.mean(), 3, True), frac(d.pre_sig.sum(), len(d)), fa(p_pre[h_ms], 2),
                 fa(d.pre_move_skill_F.mean(), 3, True)])
table("ctrl_horizon", [r"افق (\lr{ms})", r"\lr{VAF} \lr{K}", r"$r$ \lr{K}", r"$r$ \lr{E}", r"\lr{F}$-$\lr{K} (\lr{VAF})", r"\lr{p}",
                        r"$r$ باقی‌مانده", "معنادار", r"مهارت \lr{ridge} پیش از حرکت", "معنادار", r"\lr{p}", r"مهارت \lr{F} پیش از حرکت"], body,
      rf"مدل علّی با افق‌های پیش‌بینی مختلف، میانگین {n_subj} نفر ($\pm$ خطای معیار). ستون‌های \lr{{p}}: آزمون \lr{{Wilcoxon}} یک‌طرفه روی افراد، برای اینکه \lr{{F}}$-$\lr{{K}} یا مهارت بزرگ‌تر از صفر باشد. $r$ باقی‌مانده: \lr{{residual test}} با \lr{{tokens}} \lr{{EEG}} (میانگین عضله‌ها)؛ «معنادار»: شمار افراد با $p\le0.05$. مهارت پیش از حرکت: $1-\mathrm{{MSE}}(K+\mathrm{{EEG}})/\mathrm{{MSE}}(K)$ در ثانیه‌ی پیش از شروع حرکت دست؛ «معنادار»: مهارت مثبت و $p\le0.05$ در برابر جابه‌جایی زمانی.",
      "tab:ctrl-horizon", wide=True)

hs = sorted(summ.horizon_ms.unique())
wide_s = summ.pivot(index="subject", columns="horizon_ms")
order = sorted(wide_s.index, key=lambda s: int(s[1:]))
body = []
for s in order:
    row = [lr(s)]
    for h_ms in hs:
        star = r"$^{*}$" if wide_s.loc[s, ("res_p_tokens", h_ms)] <= 0.05 else ""
        row += [fa(wide_s.loc[s, ("K_vaf", h_ms)], 1), fa(wide_s.loc[s, ("F_minus_K_vaf", h_ms)], 2, True),
                fa(wide_s.loc[s, ("res_r_tokens", h_ms)], 3) + star]
    body.append(row)
body += ["MID", [r"\textbf{میانگین}"] + sum([[fa(wide_s[("K_vaf", h)].mean(), 1), fa(wide_s[("F_minus_K_vaf", h)].mean(), 2, True),
                                               fa(wide_s[("res_r_tokens", h)].mean(), 3)] for h in hs], [])]
table("ctrl_horizon_subj", ["شرکت‌کننده"] + [r"\lr{VAF} \lr{K}", r"\lr{F}$-$\lr{K}", r"$r$ باقی‌مانده"] * len(hs), body,
      r"جزئیات جدول \ref{tab:ctrl-horizon} برای هر شرکت‌کننده. ستاره: $r$ باقی‌مانده با $p\le0.05$ در برابر جابه‌جایی زمانی.",
      "tab:ctrl-horizon-subj", wide=True, align="l" + "ccc" * len(hs),
      top=[(rf"افق \lr{{{h}}} میلی‌ثانیه".translate(_FA), 3) for h in hs])

seg = pd.concat([pd.read_csv(f) for f in HORIZON.glob("h*ms/P*/segments.csv")])
seg["sig"] = (seg.p_skill_ridge <= 0.05) & (seg.skill_ridge > 0)
seg_fa = {"pre_move": "پیش از حرکت", "reach": "رساندن دست", "load": "بارگذاری", "hold": "نگه‌داشتن",
          "release": "رها کردن", "other": "بقیه", "all": "همه"}
body = []
seg_stats = {}
for name in seg_fa:
    d = seg[seg.segment == name]
    row = [seg_fa[name]]
    for h_ms in hs:
        x = d[d.horizon_ms == h_ms]
        row += [fa(x.r_residual_ridge.mean(), 3), fa(x.skill_ridge.mean(), 3, True), frac(x.sig.sum(), len(x))]
    rr = d.pivot(index="subject", columns="horizon_ms", values="r_residual_ridge")
    sk = d.pivot(index="subject", columns="horizon_ms", values="skill_ridge")
    seg_stats[name] = {"p_growth": wil(rr[hs[-1]] - rr[hs[0]]),
                       "p_skill": {h: wil(sk[h]) for h in hs}, "n_up": int((rr[hs[-1]] > rr[hs[0]]).sum())}
    row += [frac(seg_stats[name]["n_up"], len(rr)), fa(seg_stats[name]["p_growth"], 3)]
    body.append(row)
head = ["مرحله"]
for h_ms in hs:
    head += [r"$r$", "مهارت", "معنادار"]
head += ["افزایش", r"\lr{p}"]
table("ctrl_segments", head, body,
      rf"\lr{{residual test}} جدا برای هر مرحله‌ی تکلیف و هر افق، میانگین {n_subj} نفر. $r$: همبستگی باقی‌مانده‌ی پیش‌بینی‌شده‌ی \lr{{ridge}} با باقی‌مانده‌ی واقعی \lr{{K}} در همان مرحله. مهارت: $1-\mathrm{{MSE}}(K+\mathrm{{EEG}})/\mathrm{{MSE}}(K)$؛ «معنادار»: شمار افراد با مهارت مثبت و $p\le0.05$ در برابر جابه‌جایی زمانی. دو ستون آخر: شمار افرادی که $r$ آن‌ها از افق صفر تا بلندترین افق بالا رفته و \lr{{p}} آزمون \lr{{Wilcoxon}} یک‌طرفه برای این افزایش.",
      "tab:ctrl-segments", wide=True, align="l" + "ccc" * len(hs) + "cc",
      top=[(rf"افق \lr{{{h_ms}}} میلی‌ثانیه".translate(_FA), 3) for h_ms in hs] + [(r"رشد $r$", 2)])
print("p F-K > 0:", {h: round(v, 4) for h, v in p_fk.items()}, "p pre skill > 0:", {h: round(v, 3) for h, v in p_pre.items()})
for k, v in seg_stats.items():
    print(k, "n_up", v["n_up"], "p_growth", round(v["p_growth"], 4), "p_skill", {h: round(x, 3) for h, x in v["p_skill"].items()})

print("tables:",sorted(p.name for p in TAB.glob("ctrl_*.tex")))
print("figures:", len(list(FIG.glob("ctrl_*.png"))))
print(summ.groupby("horizon_ms")[["K_vaf", "E_r", "F_minus_K_vaf", "res_r_tokens", "res_sig", "pre_move_skill_ridge", "pre_sig", "pre_move_skill_F"]].agg(["mean"]).round(3))
print("beta ERD top:", beta_lo.sort_values().head(8).round(1).to_dict())
print("CMC top:", cmc.sort_values(ascending=False).head(8).round(4).to_dict(), "level", round(lvl, 4))
print("rollout top:", (100 * roll).sort_values(ascending=False).head(8).round(2).to_dict())
print(neg_beta.loc[["C3", "C4", "CP1", "CP5", "CP6", "Cz"]].round(1))
