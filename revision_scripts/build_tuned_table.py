"""
Turn a finished hyperparameter search into the TUNED entry used by the model
notebook, applying the selection rule from Revision_Phase2_Results_V03.md:

    a model's tuned hyperparameters are kept only if their full-training-set
    refit beats that model's UNTUNED baseline on the same data (validation
    AUC-PR); otherwise the untuned configuration is kept, and marked as such.

Inputs (from revision_scripts/output/):
    d8_final<TAG>.csv, d8_trials<TAG>.csv          - d8_tuning.py
    d7_sweep_rows<TAG>.csv                          - d7_sweep.py --only-no-sampling
Outputs (to result_csv/):
    hyperparameter_tuning_V02<TAG>.csv              - per model: untuned vs tuned, kept
    hyperparameter_trials_V02<TAG>.csv              - every trial, with its parameters
and prints the TUNED entry as Python, for the model notebook.

Usage: python build_tuned_table.py _notime        (or _notime_client)
       python build_tuned_table.py _p7_notime --version V03 [--models "Random Forest,CatBoost"]
Phase 7 writes hyperparameter_tuning_V03<TAG>.csv; --models restricts the table to the
models a diagnostic configuration was searched for (search and baseline must both have them).
"""
import json, sys
from pathlib import Path
import pandas as pd

TAG = sys.argv[1]
VERSION = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "V02"
HERE = Path(__file__).resolve().parent
OUT, RES = HERE / "output", HERE.parent / "result_csv"

# The untuned configurations -- the V21 settings, as used by d7_sweep.py's
# ML_Models() -- written in the keyword form the model notebook consumes.
UNTUNED = {
    "Logistic Regression": {"class_weight": "balanced"},
    "Decision Tree": {"class_weight": "balanced"},
    "Random Forest": {"n_estimators": 100, "criterion": "gini", "class_weight": "balanced"},
    "AdaBoost": {"n_estimators": 100, "learning_rate": 0.8},
    "CatBoost": {"iterations": 500, "learning_rate": 0.02, "depth": 12, "bagging_temperature": 0.2},
    "XGBoost": {"eta": 0.039, "max_depth": 2, "subsample": 0.8, "colsample_bytree": 0.9,
                "eval_metric": "auc"},
    "LightGBM": {"learning_rate": 0.05, "num_leaves": 7, "max_depth": 4, "min_child_samples": 100,
                 "max_bin": 100, "bagging_fraction": 0.9, "feature_fraction": 0.7,
                 "min_child_weight": 0, "min_split_gain": 0, "scale_pos_weight": 150,
                 "metric": "auc"},
    "K-NN": {"n_neighbors": 1},
}
ORDER = list(UNTUNED)
if "--models" in sys.argv:
    _only = sys.argv[sys.argv.index("--models") + 1].split(",")
    assert set(_only) <= set(ORDER), _only
    ORDER = [m for m in ORDER if m in _only]

_topk = OUT / f"d8_final_top3{TAG}.csv"      # refit_top_k.py output, preferred when present
final = pd.read_csv(_topk if _topk.exists() else OUT / f"d8_final{TAG}.csv").set_index("model")
print(f"search results from {(_topk if _topk.exists() else OUT / f'd8_final{TAG}.csv').name}")
base = pd.read_csv(OUT / f"d7_sweep_rows{TAG}.csv")
base = base[base.method == "No Sampling"].set_index("model")["aucpr"]
assert set(ORDER) <= set(final.index) and set(ORDER) <= set(base.index), "search or baseline incomplete"

rows, entry = [], {}
for m in ORDER:
    tuned = json.loads(final.loc[m, "params"])
    if tuned.get("max_depth") == 0:          # d8_tuning.py encodes "unlimited" as 0
        tuned["max_depth"] = None
    u, t = float(base.loc[m]), float(final.loc[m, "aucpr_full"])
    keep_tuned = t > u
    entry[m] = tuned if keep_tuned else UNTUNED[m]
    rows.append(dict(model=m, aucpr_untuned=u, aucpr_tuned=t, gain=t - u,
                     auc_tuned=float(final.loc[m, "auc_full"]),
                     kept="tuned" if keep_tuned else "untuned (search did not beat it)",
                     chosen_params=json.dumps(entry[m])))

pd.DataFrame(rows).to_csv(RES / f"hyperparameter_tuning_{VERSION}{TAG}.csv", index=False)
pd.read_csv(OUT / f"d8_trials{TAG}.csv").to_csv(RES / f"hyperparameter_trials_{VERSION}{TAG}.csv", index=False)

print(pd.DataFrame(rows).drop(columns="chosen_params").round(4).to_string(index=False))
print(f"\nTUNED entry for CONFIG={TAG.lstrip('_')!r}:")
print("    %r: {" % TAG.lstrip("_"))
for m in ORDER:
    note = "" if rows[ORDER.index(m)]["kept"] == "tuned" else "   # untuned: search did not beat it"
    print(f"        {m!r}: {entry[m]!r},{note}")
print("    },")
