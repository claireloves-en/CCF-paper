"""
Paper figure for the Phase 7 diagnostic comparisons (Revision_Phase7_Plan_V01.md §4.2): what each
one-change arm, the hybrid and the native categorical handling do to test AUC-PR, for random forest
and CatBoost, with 95% paired stratified-bootstrap intervals (uncertainty_check.py --scheme stratified).

One row per comparison, grouped: adding the encoding to the raw features; changing one feature family
of E (at E's hyperparameters); the reduced amount + category + time pair; CatBoost's own categorical
handling. Random forest is a blue circle and CatBoost an orange square (shape as well as colour, for
greyscale and colour-blind readers); the pair passes the palette validator (CVD dE 24.7).

Writes result_csv/figures/fig_targeted_V01.png (300 dpi).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "result_csv"
FIGS = RES / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#5f5e5a", "#e6e5e1"
SERIES = [("Random Forest", "Random forest", "#2a78d6", "o"), ("CatBoost", "CatBoost", "#eb6834", "s")]
ROWS = [  # (group, bootstrap kind, label)
    ("Encoding added to the raw features", "hybrid - raw", "R + E  vs  R"),
    ("Encoding added to the raw features", "hybrid + client - raw + client", "R + E + H  vs  R + H"),
    ("One change to the encoded features (vs E)", "ablation abl_hour - E", "3 time bands → hour of day"),
    ("One change to the encoded features (vs E)", "ablation abl_cat_onehot - E", "category features → one-hot"),
    ("One change to the encoded features (vs E)", "ablation abl_plus_onehot - E", "+ one-hot category"),
    ("One change to the encoded features (vs E)", "ablation abl_plus_dow - E", "+ day of week"),
    ("One change to the encoded features (vs E)", "ablation abl_no_interactions - E", "− interaction terms"),
    ("One change to the encoded features (vs E)", "ablation abl_no_external - E", "− external enrichment"),
    ("One change to the encoded features (vs E)", "ablation abl_no_geo_occ - E", "− all 14 geographic/occupational"),
    ("Amount, category and time only", "reduced: encoded - raw", "encoded  vs  raw"),
    ("CatBoost native categorical handling", "catnative - raw", "native  vs  R"),
    ("CatBoost native categorical handling", "catnative + client - raw + client", "native + H  vs  R + H"),
]

d = pd.read_csv(RES / "test_bootstrap_diffs_V02_stratified.csv")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

# y positions with a gap between groups
ys, labels, groups, y, prev = [], [], [], 0.0, None
for g, kind, lab in ROWS:
    if prev is not None and g != prev:
        y += 0.8
        groups.append((g, y))
    elif prev is None:
        groups.append((g, y))
    ys.append(y); labels.append(lab); prev = g; y += 1.0
ys = np.array(ys)

fig, ax = plt.subplots(figsize=(6.9, 4.9))
ax.axvline(0, color=INK, lw=0.9, zorder=2)
for j, (model, name, colour, marker) in enumerate(SERIES):
    off = (j - 0.5) * 0.28
    for yi, (_g, kind, _l) in zip(ys, ROWS):
        r = d[(d.kind == kind) & (d.a == model)]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.errorbar(r["diff"], yi + off, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt=marker,
                    ms=4.6, color=colour, mec="white", mew=0.7, elinewidth=1.1, capsize=2.0, capthick=0.9,
                    zorder=3, label=name if yi == ys[0] or (model == "CatBoost" and kind.startswith("catnative") and yi == ys[-2]) else None)
ax.set_yticks(ys)
ax.set_yticklabels(labels)
ax.invert_yaxis()
for g, gy in groups:
    ax.annotate(g, xy=(0, gy - 0.62), xycoords=("axes fraction", "data"), fontsize=7.5,
                fontweight="bold", color=INK, ha="left", va="center")
ax.set_xlabel("Difference in test AUC-PR (positive: the arm is ahead)")
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
h, l = ax.get_legend_handles_labels()
seen = {}
for hh, ll in zip(h, l):
    seen.setdefault(ll, hh)
ax.legend(seen.values(), seen.keys(), frameon=False, loc="lower left", fontsize=7.5, handlelength=1.2)
fig.tight_layout()
out = FIGS / "fig_targeted_V01.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
