"""
Paper Fig 3: mean validation AUC-PR of the eight models (top) and their combined
training-and-prediction time (bottom) at each resampling method and target class
ratio, with no resampling as the reference line. From the trans_time-free sweep
(output/d7_sweep_rows_notime.csv, default hyperparameters, chronological split).

Replaces the submitted figure, which put mean MCC and time on two y-axes of one
panel; here they are two panels on a shared x-axis. Method colours are the first
three slots of the validated palette used in the other figures, and each method
also has its own line style and marker, for greyscale and colour-blind readers.
Writes result_csv/figures/fig3_sampling_notime.png (300 dpi).
--phase7: the sweep on the corrected encoded file (d7_sweep_rows_p7_notime.csv) -> fig3_sampling_notime_V02.png.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "result_csv"
RATIOS = ["1:150", "1:100", "1:50", "10:90", "20:80", "30:70", "40:60", "50:50(1:1)"]
TICKS = ["1:150", "1:100", "1:50", "10:90", "20:80", "30:70", "40:60", "1:1"]
METHODS = [("RUS", "#2a78d6", "-", "o"), ("ROS", "#eb6834", "--", "s"),
           ("SMOTE", "#1baf7a", "-.", "^")]
REF, INK, GRID = "#8a8984", "#0b0b0b", "#e6e5e1"

PHASE7 = "--phase7" in sys.argv
rows = pd.read_csv(HERE / "output" / ("d7_sweep_rows_p7_notime.csv" if PHASE7 else "d7_sweep_rows_notime.csv"))
mean = rows.groupby(["method", "ratio"]).aucpr.mean()
secs = rows.groupby(["method", "ratio"]).seconds.first() / 60
ns_ap, ns_min = mean[("No Sampling", "No Sampling (1:173)")], secs[("No Sampling", "No Sampling (1:173)")]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
x = range(len(RATIOS))
for meth, c, ls, mk in METHODS:
    a1.plot(x, [mean[(meth, r)] for r in RATIOS], color=c, ls=ls, lw=1.6, marker=mk, ms=4.5,
            mec="white", mew=0.8, label=meth)
    a2.plot(x, [secs[(meth, r)] for r in RATIOS], color=c, ls=ls, lw=1.6, marker=mk, ms=4.5,
            mec="white", mew=0.8)
# time label at the left end: at the right it sat on the ROS line in the corrected sweep
for ax, v, fmt, lx, ha in [(a1, ns_ap, f"No resampling {ns_ap:.4f}", len(RATIOS) - 1, "right"),
                          (a2, ns_min, f"No resampling {ns_min:.1f} min", 0, "left")]:
    ax.axhline(v, color=REF, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.annotate(fmt, (lx, v), xytext=(0, 3), textcoords="offset points",
                ha=ha, va="bottom", fontsize=7, color="#52514e")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
a1.set_ylabel("Mean AUC-PR (validation)")
a2.set_ylabel("Time, 8 models (min)")
a2.set_xticks(list(x)); a2.set_xticklabels(TICKS)
a2.set_xlabel("Target minority to majority class ratio after resampling")
a2.set_ylim(bottom=0)
a1.legend(loc="lower left", frameon=False, ncol=3, fontsize=7.5, handlelength=2.8)
fig.tight_layout()
out = RES / "figures" / ("fig3_sampling_notime_V02.png" if PHASE7 else "fig3_sampling_notime.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
print("no resampling", round(ns_ap, 4), round(ns_min, 1), "min")
