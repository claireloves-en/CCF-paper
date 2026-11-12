"""
Paper Figs 5-7 for the revised models (base configuration: encoded features,
trans_time dropped, chronological split, tuned hyperparameters).

  Fig 5  feature importance of the four interpreted models, from
         result_csv/feature_rank_table_V05_temporal_notime.csv (the numbers in
         Table 5). Four panels on one feature order (Table 5's, by average
         importance) and one x-axis, so importance can be compared across models.
  Fig 6  SHAP beeswarms of the four models, taken unchanged from the executed
         notebook of record (executed/model_V25_temporal_notime.ipynb) and set
         out as a 2 x 2 grid with panel letters.
  Fig 7  SHAP dependence of 'Amount spent' and 'Cardholder age' in CatBoost.
         model_V25 draws these only as an interactive force plot, so CatBoost is
         refitted here exactly as in the notebook (same TUNED parameters, seed,
         data and sample) and its validation AUC-PR is checked against the
         notebook's before the SHAP values are used.

Usage: python fig5_7_interpretation.py [--skip-fig7] [--phase7]
--phase7: the corrected-data run of record (executed/model_V26_temporal_p7_notime.ipynb, V03
hyperparameters). Fig 5 then has no cross-model average to sort by (Phase 7 plan §10, 4C): features are
ordered by CatBoost's validation permutation importance (explain_compare.py), the one consistently
defined measure. Outputs carry _V02.
Writes result_csv/figures/fig5_importance_notime.png, fig6_shap_beeswarm_notime.png,
       fig7_shap_dependence_catboost_notime.png (300 dpi)
"""
import base64, io, json, random, sys, textwrap, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PATH = HERE.parent
RES = PATH / "result_csv"
FIG = RES / "figures"
V04 = "--v04" in sys.argv          # CCF_MM_V04: renamed constructs (labels_v04.py); Fig 6 from SHAP, not notebook images
PHASE7 = "--phase7" in sys.argv or V04
TAG = "temporal_p7_notime" if PHASE7 else "temporal_notime"
SUF = "_V03" if V04 else "_V02" if PHASE7 else ""
from labels_v04 import relabel
MODELS = ["Decision Tree", "Random Forest", "CatBoost", "XGBoost"]
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})


def tidy(ax):
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------- Fig 5
TOP = 12
if PHASE7:
    imp = pd.read_csv(RES / f"feature_rank_table_V06_{TAG}.csv").set_index("feature\\model")
    perm = pd.read_csv(RES / "permutation_importance_V01.csv")
    perm = perm[(perm.config == "E") & (perm.model == "CatBoost") & (perm.kind == "feature")]
    order = perm.sort_values("aucpr_drop", ascending=False)["label"].tolist()
    top = imp.loc[[f for f in order if f in imp.index][:TOP]]
else:
    imp = pd.read_csv(RES / f"feature_rank_table_V05_{TAG}.csv").set_index("feature\\model")
    top = imp.sort_values("Average", ascending=False).head(TOP)
labels = ["\n".join(textwrap.wrap(relabel(f) if V04 else f, 34)) for f in top.index]
fig, axes = plt.subplots(1, 4, figsize=(6.3, 4.2), sharey=True, sharex=True)
y = np.arange(TOP)
for ax, m, letter in zip(axes, MODELS, "ABCD"):
    ax.barh(y, top[m].values, color=BLUE, height=0.7)
    ax.set_title(f"({letter}) {m}", fontsize=8, loc="left", color=INK)
    ax.set_xlim(0, 0.75); ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0", ".25", ".5", ".75"])
    tidy(ax)
axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=6.5); axes[0].invert_yaxis()
fig.supxlabel("Feature importance (normalised to sum to 1 within each model)", fontsize=7.5)
fig.tight_layout(w_pad=0.6)
fig.savefig(FIG / f"fig5_importance_notime{SUF}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("wrote fig5; top features", list(top.index[:5]))

# ---------------------------------------------------------------- Fig 6
def beeswarms_v04():
    """The notebook's four beeswarms (cells 61, 86, 130, 155) recomputed with the renamed labels: the same
    kept hyperparameters, the same balanced 2 x 1000 validation sample (lime_table_a3.prepare reproduces
    cell 60), the same explainers and plotting call. Guard: each refit's validation AUC-PR equals the
    notebook's run of record."""
    import shap
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.tree import DecisionTreeClassifier
    from lime_table_a3 import prepare
    train_df, valid_df, df_random, predictors, _ = prepare(TAG)
    kept = pd.read_csv(RES / "hyperparameter_tuning_V03_p7_notime.csv").set_index("model")
    nb_ap = pd.read_csv(RES / f"models_metric_result_V06_{TAG}.csv").set_index("metric\\model").loc["AUC-PR"]
    X, y = train_df[predictors], train_df.is_fraud.values
    out = []
    for m in MODELS:
        p = json.loads(kept.loc[m, "chosen_params"])
        if m in ("Decision Tree", "Random Forest"):
            p = {k: (None if k == "max_depth" and v == 0 else v) for k, v in p.items()}
            cls = DecisionTreeClassifier if m == "Decision Tree" else RandomForestClassifier
            extra = {} if m == "Decision Tree" else {"n_jobs": 4, "verbose": False}
            clf = cls(random_state=1200, **p, **extra).fit(X, y)
            pv = clf.predict_proba(valid_df[predictors])[:, 1]
            sv = shap.TreeExplainer(clf)(df_random[predictors])
            exp = shap.Explanation(sv.values[:, :, 1], sv.base_values[:, 1], data=df_random[predictors].values,
                                   feature_names=[relabel(c) for c in predictors])
        elif m == "CatBoost":
            clf = CatBoostClassifier(**p, eval_metric="AUC", random_seed=1200, od_type="Iter", od_wait=100, verbose=False)
            clf.fit(X, y, verbose=False)
            pv = clf.predict_proba(valid_df[predictors])[:, 1]
            sv = shap.Explainer(clf, link=shap.links.logit)(df_random[predictors])
            exp = shap.Explanation(sv.values, sv.base_values, data=sv.data, feature_names=[relabel(c) for c in predictors])
        else:
            params = {"objective": "binary:logistic", "eval_metric": "aucpr", "seed": 1200, **p}
            dtr, dva = xgb.DMatrix(X, y), xgb.DMatrix(valid_df[predictors], valid_df.is_fraud.values)
            clf = xgb.train(params, dtr, 1000, [(dtr, "train"), (dva, "valid")], early_stopping_rounds=50,
                            maximize=True, verbose_eval=False)
            pv = clf.predict(xgb.DMatrix(valid_df[predictors], feature_names=predictors))
            sv = shap.TreeExplainer(clf)(df_random[predictors])
            exp = shap.Explanation(sv.values, sv.base_values, data=sv.data, feature_names=[relabel(c) for c in predictors])
        ap = average_precision_score(valid_df.is_fraud.values, pv)
        assert abs(ap - nb_ap[m]) < 1e-6, f"{m}: refit validation AUC-PR {ap:.6f} vs notebook {nb_ap[m]:.6f}"
        plt.figure()
        shap.plots.beeswarm(exp, max_display=20, color=plt.get_cmap("rainbow"), show=False)
        buf = io.BytesIO(); plt.gcf().savefig(buf, format="png", dpi=100, bbox_inches="tight"); plt.close("all")
        out.append(Image.open(buf).convert("RGB"))
        print(f"fig6 {m}: validation AUC-PR {ap:.4f} (notebook {nb_ap[m]:.4f})")
    return out


nb = json.load(open(PATH / "executed" / f"model_{'V26' if PHASE7 else 'V25'}_{TAG}.ipynb"))
CELLS = {"Decision Tree": 61, "Random Forest": 86, "CatBoost": 130, "XGBoost": 155}
imgs = beeswarms_v04() if V04 else []
for m in ([] if V04 else MODELS):
    c = nb["cells"][CELLS[m]]
    assert "beeswarm" in "".join(c["source"]), (m, "cell moved")
    png = [o for o in c["outputs"] if "image/png" in o.get("data", {})]
    assert len(png) == 1, (m, len(png))
    imgs.append(Image.open(io.BytesIO(base64.b64decode(png[0]["data"]["image/png"]))).convert("RGB"))
W, H = max(i.width for i in imgs), max(i.height for i in imgs)
PAD, HEAD = 30, 70
sheet = Image.new("RGB", (2 * W + PAD, 2 * (H + HEAD) + PAD), "white")
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
except OSError:
    font = ImageFont.load_default()
for k, (im, m, letter) in enumerate(zip(imgs, MODELS, "ABCD")):
    x0, y0 = (k % 2) * (W + PAD), (k // 2) * (H + HEAD + PAD)
    sheet.paste(im, (x0 + (W - im.width), y0 + HEAD))       # right-align: the SHAP axes line up
    draw.text((x0 + 10, y0 + 12), f"({letter}) {m}", fill=INK, font=font)
sheet.save(FIG / f"fig6_shap_beeswarm_notime{SUF}.png", dpi=(300, 300))
print("wrote fig6", sheet.size)

if "--skip-fig7" in sys.argv:
    sys.exit()

# ---------------------------------------------------------------- Fig 7
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from lime_table_a3 import prepare
train_df, valid_df, df_random, predictors, _ = prepare(TAG)

kept = pd.read_csv(RES / ("hyperparameter_tuning_V03_p7_notime.csv" if PHASE7 else "hyperparameter_tuning_V02_notime.csv")).set_index("model")
params = json.loads(kept.loc["CatBoost", "chosen_params"])
clf = CatBoostClassifier(**params, eval_metric="AUC", random_seed=1200, od_type="Iter",
                         od_wait=100, verbose=False)
clf.fit(train_df[predictors], train_df.is_fraud.values, verbose=False)
ap = average_precision_score(valid_df.is_fraud.values, clf.predict_proba(valid_df[predictors])[:, 1])
nb_ap = pd.read_csv(RES / f"models_metric_result_V06_{TAG}.csv").set_index("metric\\model").loc["AUC-PR", "CatBoost"]
print(f"CatBoost refit: validation AUC-PR {ap:.4f} (notebook of record {nb_ap:.4f})")

sv = shap.Explainer(clf, link=shap.links.logit)(df_random[predictors])
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
for ax, feat, letter, xl in [(axes[0], "Amount spent (USD)", "A", "Amount spent (USD, log scale)"),
                             (axes[1], "Cardholder age", "B", "Cardholder age (years)")]:
    j = predictors.index(feat)
    x, s = df_random[feat].values, sv.values[:, j]
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.scatter(x, s, s=6, color=BLUE, alpha=0.45, lw=0)
    if letter == "A":
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["$1", "$10", "$100", "$1,000"])
    ax.set_xlabel(xl); ax.set_title(f"({letter}) {feat.replace(' (USD)', '')}", fontsize=8, loc="left", color=INK)
    tidy(ax); ax.grid(axis="y", color=GRID, lw=0.6)
axes[0].set_ylabel("SHAP value (log-odds of fraud)")
fig.tight_layout()
fig.savefig(FIG / f"fig7_shap_dependence_catboost_notime{SUF}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# numbers quoted in the text: where the amount's SHAP value changes sign
a, s = df_random["Amount spent (USD)"].values, sv.values[:, predictors.index("Amount spent (USD)")]
o = np.argsort(a); a, s = a[o], s[o]
sm = pd.Series(s).rolling(41, center=True, min_periods=10).median().values
cross = [(a[i], a[i + 1]) for i in range(len(a) - 1) if np.sign(sm[i]) != np.sign(sm[i + 1]) and not np.isnan(sm[i + 1])]
print("amount: rolling-median SHAP sign changes between", [(round(p, 2), round(q, 2)) for p, q in cross][:6])
age, sa = df_random["Cardholder age"].values, sv.values[:, predictors.index("Cardholder age")]
print(pd.DataFrame({"age": age, "shap": sa}).groupby(pd.cut(age, [0, 20, 30, 40, 45, 50, 55, 60, 70, 100])).shap.agg(["mean", "count"]).round(3).to_string())
print("wrote fig7")
