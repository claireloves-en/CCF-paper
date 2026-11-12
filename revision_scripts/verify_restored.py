"""
Check that the restored revision scripts reproduce the committed results.

Run after:
  python phase1_reproduce.py
  python phase2_splits.py --configs random
  python d7_sweep.py --only-no-sampling
  python d8_tuning.py --models "Logistic Regression,Decision Tree"

Each check compares output/ against the file in result_csv/ that the original
script produced, and prints the largest absolute difference.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT, RES = HERE / "output", HERE.parent / "result_csv"
TOL = 1e-9
ok = True


def report(label, diff):
    global ok
    status = "MATCH" if diff <= TOL else "DIFFERS"
    ok &= diff <= TOL
    print(f"  {status:8s} max|diff| = {diff:.3e}   {label}")


print("phase1_reproduce.py  vs  models_metric_result_V05_replica.csv")
a = pd.read_csv(OUT / "phase1_corrected.csv").set_index("metric\\model")
b = pd.read_csv(RES / "models_metric_result_V05_replica.csv").set_index("metric\\model")
report("all metrics, all models", float((a - b[a.columns]).abs().values.max()))

print("phase2_splits.py (random)  vs  split_comparison_V01.csv")
a = pd.read_csv(OUT / "phase2_split_comparison.csv").set_index(["config", "model"])
b = pd.read_csv(RES / "split_comparison_V01.csv").set_index(["config", "model"])
num = a.select_dtypes("number").columns
report("random config, all columns",
       float((a[num] - b.loc[a.index, num]).abs().values.max()))

print("d7_sweep.py (no sampling)  vs  class_ratio_*_V06.csv")
r = pd.read_csv(OUT / "d7_sweep_rows.csv").set_index("model")
for metric, suffix in [("mcc", ""), ("aucpr", "_aucpr")]:
    w = pd.read_csv(RES / f"class_ratio_ros{suffix}_V06.csv", index_col=0)
    w = w.set_index(w.columns[0])["No Sampling (1:173)"].drop(["Average", "Time Used"]).astype(float)
    report(f"No Sampling column, {metric}", float((r[metric] - w.loc[r.index]).abs().max()))

print("d8_tuning.py (LR, DT)  vs  hyperparameter_trials_V01.csv / _tuning_V01.csv")
a = pd.read_csv(OUT / "d8_trials.csv").set_index(["model", "trial"])["aucpr"]
b = pd.read_csv(RES / "hyperparameter_trials_V01.csv").set_index(["model", "trial"])["aucpr"]
report("per-trial AUC-PR, 80 trials", float((a - b.loc[a.index]).abs().max()))
a = pd.read_csv(OUT / "d8_final.csv").set_index("model")["aucpr_full"]
b = pd.read_csv(RES / "hyperparameter_tuning_V01.csv").set_index("model")["aucpr_tuned"]
# the committed tuning table rounds to 4 dp
d = float((a.round(4) - b.loc[a.index]).abs().max())
print(f"  {'MATCH' if d == 0 else 'DIFFERS':8s} max|diff| = {d:.1e}   full-train refit (4 dp, as committed)")
ok &= d == 0

print("\nALL MATCH" if ok else "\nSOME CHECKS DIFFER - see above")
