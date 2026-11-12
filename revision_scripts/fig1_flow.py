"""
Paper Fig 1: the analysis flow, redrawn for the revised design (chronological
split, client-level features, hyperparameter search and operating threshold on
the validation set, one-time test evaluation, 'LightGBM' spelled correctly).
Keeps the submitted figure's monochrome rounded-box style and top-to-bottom layout.
Writes result_csv/figures/fig1_flow.png (300 dpi).
--phase7: the design of Revision_Phase7_Plan_V01.md -- four feature representations compared, the
test window as the comparison period, the later period scored once, and SHAP, permutation
importance and LIME -> fig1_flow_V02.png.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PHASE7 = "--phase7" in sys.argv
OUT = Path(__file__).resolve().parent.parent / "result_csv" / "figures" / ("fig1_flow_V02.png" if PHASE7 else "fig1_flow.png")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.weight": "bold"})
fig, ax = plt.subplots(figsize=(6.3, 7.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 11.8); ax.axis("off")
LW, FS = 1.1, 7.2


def box(x0, y0, x1, y1, text="", r=0.18, fs=FS, lw=LW, square=False, label=None):
    style = "square,pad=0" if square else f"round,pad=0,rounding_size={r}"
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=style, fc="white", ec="black", lw=lw))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)
    if label:
        ax.text((x0 + x1) / 2, y0 - 0.2, label, ha="center", va="center", fontsize=FS)
    return (x0, y0, x1, y1)


def arrow(p, q, conn="arc3"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=LW, color="black",
                                                    shrinkA=0, shrinkB=0, mutation_scale=9,
                                                    connectionstyle=conn))


def line(*pts):
    xs, ys = zip(*pts); ax.plot(xs, ys, color="black", lw=LW, solid_capstyle="butt")


# --- data and feature engineering -------------------------------------------------
FS = 6.4
box(0.05, 10.6, 2.35, 11.3, "Credit card transaction\ndataset (1,296,675\ntransactions)", square=True, fs=6.2)
box(2.7, 9.95, 5.9, 11.75, r=0.25)
if PHASE7:
    ax.text(4.3, 9.75, "Feature representations: R, E, R + H, E + H\n(+ R + E hybrid and one-change ablations)",
            ha="center", va="center", fontsize=FS - 0.4)
    box(2.85, 11.15, 5.75, 11.6, "Raw variables, conventional (R)", fs=FS)
    box(2.85, 10.6, 5.75, 11.05, "Theory-informed encoding (E)", fs=FS)
    box(2.85, 10.1, 5.75, 10.55, "Client-level behavioural features (H)", fs=FS - 0.4)
else:
    ax.text(4.3, 9.75, "Feature engineering", ha="center", va="center", fontsize=FS)
    box(2.85, 11.15, 5.75, 11.6, "Feature selection", fs=FS)
    box(2.85, 10.6, 5.75, 11.05, "Feature encoding", fs=FS)
    box(2.85, 10.1, 5.75, 10.55, "Client-level behavioural features", fs=FS)
arrow((2.35, 10.95), (2.7, 10.95))
box(6.3, 10.65, 9.6, 11.25, "Chronological split", fs=6.8)
arrow((5.9, 10.95), (6.3, 10.95))

# --- the three partitions ------------------------------------------------------------
Y0, Y1 = 8.6, 9.15
box(2.9, Y0, 6.5, Y1, "Training data (64%), to 10 Dec 2019", square=True, fs=6.0)
box(6.5, Y0, 8.25, Y1, "Validation (16%)\nto 6 Mar 2020", square=True, fs=6.0)
box(8.25, Y0, 10.0, Y1, "Test (20%)\nto 21 Jun 2020", square=True, fs=6.0)
line((7.95, 10.65), (7.95, 9.4)); line((4.7, 9.4), (9.125, 9.4))
for x in (4.7, 7.375, 9.125):
    arrow((x, 9.4), (x, Y1))

# --- training -----------------------------------------------------------------------
box(2.9, 7.55, 6.5, 8.15, "Class-imbalance handling\n(resampling compared; class weighting)", fs=6.0)
arrow((4.7, Y0), (4.7, 8.15))
box(0.1, 5.55, 7.2, 7.1, r=0.25)
ax.text(3.65, 5.35, "Machine learning model training", ha="center", va="center", fontsize=FS)
names = [["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost"],
         ["CatBoost", "XGBoost", "LightGBM", "k-NN"]]
w = (7.2 - 0.1 - 0.2) / 4
for i, row in enumerate(names):
    for j, n in enumerate(row):
        x0 = 0.2 + j * w
        yy = 6.4 - i * 0.7
        box(x0, yy, x0 + w - 0.1, yy + 0.55, n, fs=6.0)
arrow((4.7, 7.55), (4.7, 7.1))

# --- validation: tuning and threshold -------------------------------------------------
box(7.6, 5.95, 9.3, 7.0, "Hyperparameter\nsearch\n(AUC-PR)", fs=6.2)
line((7.375, Y0), (7.375, 7.3)); line((7.375, 7.3), (8.45, 7.3)); arrow((8.45, 7.3), (8.45, 7.0))
arrow((7.6, 6.475), (7.2, 6.475))
box(1.6, 4.35, 5.9, 4.95, "Validation performance and operating\nthreshold (0.5% alert budget)", fs=6.2)
arrow((3.75, 5.2), (3.75, 4.95))
line((8.45, 5.95), (8.45, 4.65)); arrow((8.45, 4.65), (5.9, 4.65))

# --- test, comparison ------------------------------------------------------------------
box(1.6, 3.25, 5.9, 3.85, ("Test-window evaluation: representations and\nmodels compared (paired bootstrap)" if PHASE7
                          else "One-time test-set evaluation\nand model comparison"), fs=6.2)
arrow((3.75, 4.35), (3.75, 3.85))
line((9.7, Y0), (9.7, 3.55)); arrow((9.7, 3.55), (5.9, 3.55))

# --- interpretation ------------------------------------------------------------------
box(1.6, 1.95, 5.9, 2.8, r=0.25)
ax.text(3.75, 1.75, "Model interpretation", ha="center", va="center", fontsize=FS)
if PHASE7:
    box(1.8, 2.15, 2.95, 2.6, "SHAP", fs=6.4); box(3.05, 2.15, 4.45, 2.6, "Permutation", fs=6.4)
    box(4.55, 2.15, 5.7, 2.6, "LIME", fs=6.4)
    box(6.15, 2.45, 9.95, 3.1, "Confirmation on a later period\n(21 Jun-31 Dec 2020), scored once", fs=6.2)
    arrow((8.05, 3.55), (8.05, 3.1))
else:
    box(2.3, 2.15, 3.6, 2.6, "SHAP", fs=6.4); box(3.9, 2.15, 5.2, 2.6, "LIME", fs=6.4)
arrow((3.75, 3.25), (3.75, 2.8))
box(1.6, 0.9, 5.9, 1.4, "Post-modelling exploratory visualisations", fs=6.2)
arrow((3.75, 1.6), (3.75, 1.4))
box(6.5, 1.35, 8.7, 1.85, "Verification of results", fs=6.2)
line((5.9, 1.15), (7.6, 1.15)); line((7.6, 1.15), (7.6, 1.35))
line((7.6, 1.85), (7.6, 2.375)); arrow((7.6, 2.375), (5.9, 2.375))
box(1.6, 0.05, 5.9, 0.55, "Cross-disciplinary analysis", fs=6.4)
arrow((3.75, 0.9), (3.75, 0.55))

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
