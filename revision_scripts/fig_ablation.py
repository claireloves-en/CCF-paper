"""
Paper figure for the J1 encoding ablation (Revision_Phase6_Plan_V01.md §6).

Top panel: test AUC-PR of each model under the four configurations, with 95% paired-bootstrap
intervals as error bars (uncertainty_check.py). Bottom panel: the encoding's contribution,
3 - 1 and 4 - 2, per model with the interval on the difference -- the quantity the assessment's
priority-1 question actually turns on, which a reader should not have to subtract by eye.

Configuration colours use the first four slots of the validated palette shared with Figs 3 and
5-7, and each configuration also has its own hatch, for greyscale and colour-blind readers.

Writes result_csv/figures/fig_ablation.png (300 dpi).
--phase7: the corrected-data runs (Revision_Phase7_Plan_V01.md) -> fig_ablation_V02.png, from
ablation_V02.csv and the stratified Phase 7 bootstrap.
--v11: Fig 2 of CCF_MM_V11 -> fig_ablation_V03.png. As --phase7, with the manuscript's labels
(R, R + H, E, E + H) and k-NN taken from knn_scaled.py (training-standardised features, k chosen
on validation; knn_scaled_V01.csv), which is the k-NN comparison of Table 4. knn_scaled.py gives
no per-configuration interval, so the k-NN bars carry no error bar; its differences do.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "result_csv"
FIGS = RES / "figures"; FIGS.mkdir(parents=True, exist_ok=True)

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
          "CatBoost", "XGBoost", "LightGBM", "K-NN"]
SHORT = {"Logistic Regression": "Logistic\nregression", "Decision Tree": "Decision\ntree",
         "Random Forest": "Random\nforest", "AdaBoost": "AdaBoost", "CatBoost": "CatBoost",
         "XGBoost": "XGBoost", "LightGBM": "LightGBM", "K-NN": "k-NN"}
CONFIGS = [("notime_raw", "1  Raw variables", "#8a8984", ""),
           ("notime_raw_client", "2  Raw + client history", "#2a78d6", "///"),
           ("notime", "3  Encoded", "#eb6834", ""),
           ("notime_client", "4  Encoded + client history", "#1baf7a", "///")]
DIFFS = [("3 - 1  encoding, without client history", "3 − 1  without client history", "#eb6834", "o"),
         ("4 - 2  encoding, with client history", "4 − 2  with client history", "#1baf7a", "s")]
INK, GRID = "#0b0b0b", "#e6e5e1"
V11 = "--v11" in sys.argv
PHASE7 = "--phase7" in sys.argv or V11
if PHASE7:
    CONFIGS = [("p7_" + c, l, col, h) for c, l, col, h in CONFIGS]
    DIFFS = [("P1  " + k, l, col, mk) for k, l, col, mk in DIFFS[:1]] + [("P2  " + k, l, col, mk) for k, l, col, mk in DIFFS[1:]]
    abl = pd.read_csv(RES / "ablation_V02.csv").set_index(["config", "model"])
    boot = pd.read_csv(RES / "test_bootstrap_V02_stratified.csv").set_index(["config", "model"])
    diffs = pd.read_csv(RES / "test_bootstrap_diffs_V02_stratified.csv")
    if V11:
        CONFIGS = [(c, l, col, h) for (c, _l, col, h), l in zip(CONFIGS, [
            "R   Conventional", "R + H   Conventional with history",
            "E   Theory-informed encoding", "E + H   Encoding with history"])]
        DIFFS = [(k, l, col, mk) for (k, _l, col, mk), l in zip(DIFFS, ["E − R", "(E + H) − (R + H)"])]
        SHORT["K-NN"] = "k-NN*"
        kn = pd.read_csv(RES / "knn_scaled_V01.csv")
        lev = kn[kn.config.isin(["R", "R+H", "E", "E+H"])].set_index("config")
        for cfg, key in zip([c for c, *_ in CONFIGS], ["R", "R+H", "E", "E+H"]):
            abl.loc[(cfg, "K-NN"), "test_aucpr"] = lev.loc[key, "test_aucpr_scaled"]
            boot = boot.drop(index=(cfg, "K-NN"), errors="ignore")
        sc = kn[kn.kind == "scaled"].reset_index(drop=True)
        assert len(sc) == 2 and sc.config[0] == "difference E - R"
        diffs = diffs[diffs.a != "K-NN"]
        for (kind, *_), row in zip(DIFFS, sc.itertuples()):
            diffs = pd.concat([diffs, pd.DataFrame([{"kind": kind, "a": "K-NN", "diff": row.diff,
                                                     "lo": row.lo, "hi": row.hi}])])
else:
    abl = pd.read_csv(RES / "ablation_V01.csv").set_index(["config", "model"])
    boot = pd.read_csv(RES / "test_bootstrap_V01.csv").set_index(["config", "model"])
    diffs = pd.read_csv(RES / "test_bootstrap_diffs_V01.csv")

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.9, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
x = np.arange(len(MODELS))
w = 0.2

for i, (cfg, label, colour, hatch) in enumerate(CONFIGS):
    off = (i - 1.5) * w
    vals = [abl.loc[(cfg, m), "test_aucpr"] if (cfg, m) in abl.index else np.nan for m in MODELS]
    lo = [boot.loc[(cfg, m), "aucpr_lo"] if (cfg, m) in boot.index else np.nan for m in MODELS]
    hi = [boot.loc[(cfg, m), "aucpr_hi"] if (cfg, m) in boot.index else np.nan for m in MODELS]
    err = np.array([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
    a1.bar(x + off, vals, w * 0.92, color=colour, hatch=hatch, edgecolor="white", linewidth=0.5,
           label=label, zorder=3)
    a1.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.7,
                capsize=1.6, capthick=0.7, zorder=4)

a1.set_ylabel("Test AUC-PR")
a1.set_ylim(0, 1.0)
a1.set_yticks(np.arange(0, 1.01, 0.2))
a1.grid(axis="y", color=GRID, lw=0.7, zorder=0)
a1.set_axisbelow(True)
# above the axes, not inside: at ncol=2 inside, the fourth entry sat on the tree bars
a1.legend(frameon=False, ncol=2 if V11 else 4, loc="lower left", bbox_to_anchor=(0, 1.01, 0.75 if V11 else 1, 0.1),
          mode="expand", fontsize=7.5, handlelength=1.5, columnspacing=1.0,
          borderaxespad=0)
for sp in ("top", "right"):
    a1.spines[sp].set_visible(False)

a2.axhline(0, color=INK, lw=0.9, zorder=2)
for j, (kind, label, colour, marker) in enumerate(DIFFS):
    sub = diffs[diffs.kind == kind].set_index("a")
    off = (j - 0.5) * 0.18
    v = [sub.loc[m, "diff"] if m in sub.index else np.nan for m in MODELS]
    lo = [sub.loc[m, "lo"] if m in sub.index else np.nan for m in MODELS]
    hi = [sub.loc[m, "hi"] if m in sub.index else np.nan for m in MODELS]
    err = np.array([np.array(v) - np.array(lo), np.array(hi) - np.array(v)])
    a2.errorbar(x + off, v, yerr=err, fmt=marker, ms=4.2, color=colour, mec="white", mew=0.7,
                elinewidth=1.0, capsize=2.0, capthick=0.9, label=label, zorder=3)

a2.set_ylabel("Difference in test AUC-PR" if V11 else "Encoding's contribution\n(difference in test AUC-PR)")
a2.set_xticks(x)
a2.set_xticklabels([SHORT[m] for m in MODELS])
a2.grid(axis="y", color=GRID, lw=0.7, zorder=0)
a2.set_axisbelow(True)
a2.legend(frameon=False, loc="upper left", fontsize=7.5, handlelength=1.2, borderaxespad=0.3)
for sp in ("top", "right"):
    a2.spines[sp].set_visible(False)
# headroom so the direction labels clear k-NN, the only point above zero
a2.set_ylim(-0.105, 0.185 if V11 else 0.125)
a2.annotate("encoding ahead", xy=(0.995, 0.96), xycoords="axes fraction", ha="right",
            va="top", fontsize=7, color="#5f5e5a")
a2.annotate("baseline ahead", xy=(0.995, 0.04), xycoords="axes fraction", ha="right",
            va="bottom", fontsize=7, color="#5f5e5a")

if V11:
    fig.text(0.01, -0.01, "* k-NN on features standardised on the training period (knn_scaled.py); its bars have "
             "no interval.\nError bars: unadjusted 95% paired stratified bootstrap intervals.",
             fontsize=6.5, color="#5f5e5a", ha="left", va="top")
fig.tight_layout(h_pad=1.0)
out = FIGS / ("fig_ablation_V03.png" if V11 else "fig_ablation_V02.png" if PHASE7 else "fig_ablation.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
print("\nencoding's contribution, ensemble means:")
ens = ["Random Forest", "CatBoost", "XGBoost", "LightGBM"]
for kind, label, _c, _m in DIFFS:
    sub = diffs[(diffs.kind == kind) & (diffs.a.isin(ens))]
    print(f"  {label}: {sub['diff'].mean():+.4f}")
