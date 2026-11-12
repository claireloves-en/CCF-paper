"""
Paper figure: test-set precision-recall curves, encoded features only (left) vs
encoded + client-level features (right), from the curve points model_V25 saves
(result_csv/pr_curves_test_<TAG>.csv). Same colours and line styles as the
per-run figures: the dataviz reference palette in the paper's fixed model order,
validated; line style as a secondary cue for colour-blind and greyscale readers.
Writes result_csv/figures/pr_curves_test_paired.png (300 dpi).
--phase7: the four corrected-data runs of model_V26 (raw, raw + client, encoded, encoded + client)
in a 2 x 2 grid -> pr_curves_test_V02.png.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "result_csv"
ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
         "CatBoost", "XGBoost", "LightGBM", "K-NN"]
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STYLES = [(0, (1, 1.5)), "-", "-", "--", "--", "-.", ":", "-."]
PANELS = [("temporal_notime", "Encoded features"),
          ("temporal_notime_client", "Encoded + client-level features")]
FRAUD_RATE = 1538 / 259284          # test set
PHASE7 = "--phase7" in sys.argv
if PHASE7:
    PANELS = [("temporal_p7_notime_raw", "1  Raw variables"),
              ("temporal_p7_notime_raw_client", "2  Raw + client-level features"),
              ("temporal_p7_notime", "3  Encoded features"),
              ("temporal_p7_notime_client", "4  Encoded + client-level features")]

metrics = {tag: pd.read_csv(RES / f"test_metric_result_V01_{tag}.csv").set_index("metric\\model")
           for tag, _ in PANELS}
# drawn at its final printed size (16 cm wide), so the type prints at the sizes set here
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
if PHASE7:
    fig, grid = plt.subplots(2, 2, figsize=(6.3, 6.6), sharex=True, sharey=True)
    axes = grid.ravel()
else:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.7), sharey=True)
for ax, (tag, title) in zip(axes, PANELS):
    d = pd.read_csv(RES / f"pr_curves_test_{tag}.csv")
    for m, c, s in zip(ORDER, COLOURS, STYLES):
        cur = d[(d.model == m) & (d.point == "curve")]
        op = d[(d.model == m) & (d.point != "curve")]
        ax.plot(cur.recall, cur.precision, color=c, ls=s, lw=1.3, label=m)
        ax.plot(op.recall, op.precision, "o", ms=5, color=c, mec="white", mew=1.0, zorder=5)
    ax.axhline(FRAUD_RATE, color="#8a8984", lw=1, ls=(0, (2, 2)))
    ax.set_title(title, fontsize=8.5, loc="left", color="#0b0b0b")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    if not PHASE7 or ax in axes[2:]:
        ax.set_xlabel("Recall")
    ax.grid(color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for ax in (axes[::2] if PHASE7 else axes[:1]):
    ax.set_ylabel("Precision")
# one shared legend below the panels (model identity only; AUC-PR values are in Table 4):
# in-panel legends sat on the weak models' curves, and a legend at the side shrank the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.01), handlelength=2.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.06 if PHASE7 else 0.1, 1, 1))
out = RES / "figures" / ("pr_curves_test_V02.png" if PHASE7 else "pr_curves_test_paired.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
