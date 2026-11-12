"""
Reshape d7_sweep.py's long-format rows into the result_csv tables.

Usage: python reshape_sweep.py [TAG]        e.g. TAG=_notime
Reads  output/d7_sweep_rows{TAG}.csv
Writes result_csv/class_ratio_{rus,ros,smote}_V06{TAG}.csv         (MCC)
       result_csv/class_ratio_{rus,ros,smote}_aucpr_V06{TAG}.csv   (AUC-PR)
       result_csv/best_sampling_per_model_V01{TAG}.csv
       result_csv/sampling_mean_by_setting_V01{TAG}.csv   (mean over the 8 models)

Layout matches the submitted class_ratio_*_V05.csv: one row per model plus Average
and Time Used, the shared no-sampling column first, then the eight ratios.
With no TAG it reproduces the committed _V06 / _V01 files (used as a check).
"""
import sys
from datetime import timedelta
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "result_csv"
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
DEST = RES if TAG else HERE / "output" / "reshape_check"
DEST.mkdir(exist_ok=True)

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
          "CatBoost", "XGBoost", "LightGBM", "K-NN"]
RATIOS = ["1:150", "1:100", "1:50", "10:90", "20:80", "30:70", "40:60", "50:50(1:1)"]
NS = "No Sampling (1:173)"

rows = pd.read_csv(HERE / "output" / f"d7_sweep_rows{TAG}.csv")
base = rows[rows.method == "No Sampling"].set_index("model")

for method in ["RUS", "ROS", "SMOTE"]:
    for metric, suffix in [("mcc", ""), ("aucpr", "_aucpr")]:
        w = rows[rows.method == method].pivot(index="model", columns="ratio", values=metric)
        w = w.reindex(index=MODELS, columns=RATIOS)
        w.insert(0, NS, base[metric].reindex(MODELS))
        w.loc["Average"] = w.mean()
        secs = rows[rows.method == method].groupby("ratio").seconds.first()
        w.loc["Time Used"] = [str(timedelta(seconds=float(base.seconds.iloc[0])))] + \
                             [str(timedelta(seconds=float(secs[r]))) for r in RATIOS]
        w.index.name = f"{method} Models"
        w.reset_index().to_csv(DEST / f"class_ratio_{method.lower()}{suffix}_V06{TAG}.csv")

best = []
for m in MODELS:
    r = rows[rows.model == m].sort_values("aucpr", ascending=False).iloc[0]
    ns = base.loc[m, "aucpr"]
    best.append(dict(model=m, method=r.method, ratio=r.ratio, aucpr=r.aucpr, mcc=r.mcc,
                     aucpr_no_sampling=ns, gain_over_no_sampling=r.aucpr - ns))
pd.DataFrame(best).to_csv(DEST / f"best_sampling_per_model_V01{TAG}.csv", index=False)

rows["setting"] = rows.method.where(rows.method == "No Sampling", rows.method + " " + rows.ratio)
mean = (rows.groupby("setting")[["aucpr", "mcc"]].mean()
        .sort_values("aucpr", ascending=False).rename(columns=lambda c: "mean_" + c))
mean.insert(0, "rank_aucpr", range(1, len(mean) + 1))
mean.to_csv(DEST / f"sampling_mean_by_setting_V01{TAG}.csv")
print(mean.head(8).round(4).to_string())
print(f"wrote to {DEST}")
