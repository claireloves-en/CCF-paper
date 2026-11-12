"""
Fit and evaluate one data configuration, for the J1 encoding ablation.
(Revision_Phase6_Plan_V01.md §5, Route A -- decided 2026-10-07.)

model_V25 is the notebook of record for the two encoded configurations, but its cost is SHAP,
LIME, the feature-importance table and the human-readable labels, none of which the paper needs
for a baseline it does not interpret. This runs the same protocol without them:

  chronological 64/16/20 cut on trans_time -> fit each of the 8 models on the full training
  set with its kept hyperparameters -> set each model's threshold on the VALIDATION set at a
  0.5% alert budget -> score the test set ONCE with nothing refitted and nothing re-tuned.

Hyperparameters come from result_csv/hyperparameter_tuning_V0*_<config>.csv, the same table
the notebook's TUNED dict was transcribed from (its `chosen_params`, which already holds the
untuned baseline where the search did not beat it). The fits mirror the notebook's cells,
including CatBoost's od_wait, XGBoost's early stopping on validation aucpr and LightGBM's
stopping_rounds=100 and positional feature names.

Predictor order is the data file's column order, which is what the notebook uses. (Note that
d8_tuning.py hardcodes a DIFFERENT order -- job_* before city_pop, cardholder_age last -- so
its reported validation AUC-PR for a given configuration is not bit-identical to the
notebook's. The chosen hyperparameters are unaffected as choices; only the search's own
scores were computed under that ordering.)

--check compares the result against the committed tables for the configuration, which is the
gate the plan puts in front of the new configurations: if this script reproduces
configuration `notime` exactly, it implements the notebook's protocol and the baselines it
produces are comparable to the published numbers.

Usage:
  python ablation_run.py --config notime --check      # the guard run
  python ablation_run.py --config notime_raw
Configs: notime, notime_client, notime_raw, notime_raw_client (Phase 6); p7_* (Phase 7, see _P7)
Writes output/test_metric_result_V01_<TAG>.csv, output/test_operating_point_V01_<TAG>.csv,
       output/test_predictions_<TAG>.csv.gz  (per-row test probabilities, for J4 and §8)
"""
import argparse, json, time, warnings
from datetime import datetime
from pathlib import Path
from data_paths import data_file as resolve  # encoded files: current, or archived via CCF_ENCODED_VERSION

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef, confusion_matrix)
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
RESULTS = PATH / "result_csv"
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)

RANDOM_STATE = 1200
ALERT_RATE = 0.005
MAX_ROUNDS, EARLY_STOP, NO_JOBS = 1000, 50, 4
TARGET = "is_fraud"
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
               "CatBoost", "XGBoost", "LightGBM", "K-NN"]

# config -> (data file, hyperparameter table, TAG)
CONFIGS = {
    "notime":           ("fraudTrain_encoded.csv",             "hyperparameter_tuning_V02_notime.csv",           "temporal_notime"),
    "notime_client":    ("fraudTrain_encoded_client.csv",      "hyperparameter_tuning_V02_notime_client.csv",    "temporal_notime_client"),
    "notime_raw":       ("fraudTrain_raw_baseline.csv",        "hyperparameter_tuning_V02_notime_raw.csv",       "temporal_notime_raw"),
    "notime_raw_client":("fraudTrain_raw_baseline_client.csv", "hyperparameter_tuning_V02_notime_raw_client.csv","temporal_notime_raw_client"),
}

# Phase 7 (Revision_Phase7_Plan_V01.md). Every configuration fitted on the corrected encoded
# files, plus the raw baselines re-selected with the same top-3 refit the encoded ones had.
# Hyperparameter tables are V03; result tags carry p7 so nothing above is overwritten.
# The targeted ablations and the reduced pair hold the hyperparameters at E's (p7_notime)
# kept set, so each arm differs from E in its features only.
_P7 = {
    "p7_notime":          ("fraudTrain_encoded.csv",             "p7_notime"),
    "p7_notime_client":   ("fraudTrain_encoded_client.csv",      "p7_notime_client"),
    "p7_notime_raw":      ("fraudTrain_raw_baseline.csv",        "notime_raw"),
    "p7_notime_raw_client": ("fraudTrain_raw_baseline_client.csv", "notime_raw_client"),
    "p7_hybrid":          ("ablation_data/fraudTrain_hybrid.csv",        "p7_hybrid"),
    "p7_hybrid_client":   ("ablation_data/fraudTrain_hybrid_client.csv", "p7_hybrid_client"),
}
_P7["p7_catnative"] = ("ablation_data/fraudTrain_catnative.csv", "p7_catnative")
_P7["p7_catnative_client"] = ("ablation_data/fraudTrain_catnative_client.csv", "p7_catnative_client")
for _arm in ("abl_cat_onehot", "abl_plus_onehot", "abl_hour", "abl_plus_dow", "abl_no_interactions",
             "abl_no_external", "abl_no_geo_occ", "red_raw", "red_enc"):
    _P7[f"p7_{_arm}"] = (f"ablation_data/fraudTrain_{_arm}.csv", "p7_notime")
# Location grid with training-split bounds (grid_bounds_check.py), at E's / E+H's kept settings
_P7["p7_grid_train"] = ("ablation_data/fraudTrain_grid_train.csv", "p7_notime")
_P7["p7_grid_train_client"] = ("ablation_data/fraudTrain_grid_train_client.csv", "p7_notime_client")
# Phase 8 C1 (Revision_Phase8_Plan_V01.md): E+, tuned on its own (random forest and CatBoost)
_P7["p7_eplus"] = ("ablation_data/fraudTrain_eplus.csv", "p7_eplus")
_P7["p7_eplus_client"] = ("ablation_data/fraudTrain_eplus_client.csv", "p7_eplus_client")
for _cfg, (_file, _hp) in _P7.items():
    CONFIGS[_cfg] = (_file, f"hyperparameter_tuning_V03_{_hp}.csv", f"temporal_{_cfg}")


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def budget_threshold(proba, alert_rate=ALERT_RATE):
    """model_V25 cell 28, unchanged: (threshold, achieved rate, usable)."""
    p = np.asarray(proba, dtype=float)
    k = max(1, int(round(len(p) * alert_rate)))
    thr = float(np.partition(p, -k)[-k])
    achieved = float((p >= thr).mean())
    return thr, achieved, achieved <= alert_rate * 2


# ------------------------------------------------------------------ fitters
# Each returns predict(DataFrame) -> fraud probability. They mirror the notebook's cells.
def fit_logreg(p, tr, va, predictors):
    scaler = StandardScaler().fit(tr[predictors])
    m = LogisticRegression(max_iter=1000, **p).fit(scaler.transform(tr[predictors]), tr[TARGET].values)
    return lambda d: m.predict_proba(scaler.transform(d[predictors]))[:, 1]


def _sk(cls, p, tr, predictors, **extra):
    p = {k: (None if k == "max_depth" and v == 0 else v) for k, v in p.items()}
    m = cls(random_state=RANDOM_STATE, **p, **extra).fit(tr[predictors], tr[TARGET].values)
    return lambda d: m.predict_proba(d[predictors])[:, 1]


def fit_tree(p, tr, va, predictors):
    return _sk(DecisionTreeClassifier, p, tr, predictors)


def fit_forest(p, tr, va, predictors):
    return _sk(RandomForestClassifier, p, tr, predictors, n_jobs=NO_JOBS, verbose=False)


def fit_ada(p, tr, va, predictors):
    y = tr[TARGET].values
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y), y=y)))
    m = AdaBoostClassifier(**p, algorithm="SAMME", random_state=RANDOM_STATE)
    m.fit(tr[predictors], y, sample_weight=np.array([cw[v] for v in y]))
    return lambda d: m.predict_proba(d[predictors])[:, 1]


def fit_cat(p, tr, va, predictors):
    m = CatBoostClassifier(**p, eval_metric="AUC", random_seed=RANDOM_STATE,
                           od_type="Iter", od_wait=100, verbose=False)
    cat = [c for c in predictors if tr[c].dtype == object]   # catnative only; [] elsewhere
    m.fit(tr[predictors], tr[TARGET].values, cat_features=cat or None, verbose=False)
    return lambda d: m.predict_proba(d[predictors])[:, 1]


def fit_xgb(p, tr, va, predictors):
    dtr = xgb.DMatrix(tr[predictors], tr[TARGET].values)
    dva = xgb.DMatrix(va[predictors], va[TARGET].values)
    params = {"objective": "binary:logistic", "eval_metric": "aucpr", "seed": RANDOM_STATE, **p}
    bst = xgb.train(params, dtr, MAX_ROUNDS, [(dtr, "train"), (dva, "valid")],
                    early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)
    return lambda d: bst.predict(xgb.DMatrix(d[predictors], feature_names=predictors))


def fit_lgb(p, tr, va, predictors):
    names = [f"f{i}" for i in range(len(predictors))]
    params = {"boosting_type": "gbdt", "objective": "binary", "metric": "average_precision",
              "bagging_freq": 1, "nthread": 8, "verbose": -1, "seed": RANDOM_STATE, **p}
    dtr = lgb.Dataset(tr[predictors].values, label=tr[TARGET].values, feature_name=names)
    dva = lgb.Dataset(va[predictors].values, label=va[TARGET].values, feature_name=names)
    bst = lgb.train(params, dtr, valid_sets=[dtr, dva], valid_names=["train", "valid"],
                    num_boost_round=MAX_ROUNDS,
                    callbacks=[lgb.early_stopping(stopping_rounds=2 * EARLY_STOP, verbose=False)])
    return lambda d: bst.predict(d[predictors].values, num_iteration=bst.best_iteration)


def fit_knn(p, tr, va, predictors):
    m = KNeighborsClassifier(**p, n_jobs=-1).fit(tr[predictors], tr[TARGET].values)
    return lambda d: m.predict_proba(d[predictors])[:, 1]


FITTERS = {"Logistic Regression": fit_logreg, "Decision Tree": fit_tree,
           "Random Forest": fit_forest, "AdaBoost": fit_ada, "CatBoost": fit_cat,
           "XGBoost": fit_xgb, "LightGBM": fit_lgb, "K-NN": fit_knn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--check", action="store_true",
                    help="compare the result against the committed tables for this config")
    ap.add_argument("--models", default="", help="comma-separated subset, for partial re-runs")
    args = ap.parse_args()
    data_file, tune_file, tag = CONFIGS[args.config]
    only = set(args.models.split(",")) if args.models else None

    df = pd.read_csv(resolve(data_file))
    predictors = [c for c in df.columns if c not in (TARGET, "trans_time")]
    t = df["trans_time"]
    c1, c2 = t.quantile(0.64), t.quantile(0.80)
    train, valid, test = df[t <= c1], df[(t > c1) & (t <= c2)], df[t > c2]
    log(f"{args.config}: {len(predictors)} predictors; train {len(train):,} / valid {len(valid):,} "
        f"/ test {len(test):,}; fraud {int(train[TARGET].sum())}/{int(valid[TARGET].sum())}/{int(test[TARGET].sum())}")

    tuned = pd.read_csv(RESULTS / tune_file).set_index("model")
    y_test = test[TARGET].values
    metrics_rows = {}
    budget_rows = {}
    preds_out = {}

    for name in MODEL_ORDER:
        if only and name not in only:
            continue
        params = json.loads(tuned.loc[name, "chosen_params"])
        s = time.time()
        predict = FITTERS[name](params, train, valid, predictors)
        p_valid, p_test = predict(valid), predict(test)
        preds_out[name] = p_test

        # threshold-0.5 metrics on test, kept for continuity with the published tables
        # strictly above 0.5, as the notebook (cell 213): a decision-tree leaf can score exactly 0.5
        # (the raw baseline's does); >= here made the threshold-0.5 metrics differ from the notebook's
        alerted = (p_test > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, alerted, labels=[0, 1]).ravel()
        recall = recall_score(y_test, alerted)
        specificity = tn / (tn + fp)
        metrics_rows[name] = [recall,
                              precision_score(y_test, alerted, zero_division=0),
                              specificity,
                              f1_score(y_test, alerted),
                              roc_auc_score(y_test, p_test),
                              average_precision_score(y_test, p_test),
                              matthews_corrcoef(y_test, alerted),
                              (recall + specificity) / 2]

        # operating point: threshold set on VALIDATION, applied unchanged to test
        thr, _, usable = budget_threshold(p_valid)
        if not usable:
            log(f"  {name}: scores too coarse for a {ALERT_RATE:.1%} operating point")
            budget_rows[name] = [thr] + [np.nan] * 8
        else:
            a = (p_test >= thr).astype(int)
            budget_rows[name] = [thr, float(a.mean()), recall_score(y_test, a),
                                 precision_score(y_test, a, zero_division=0),
                                 f1_score(y_test, a), matthews_corrcoef(y_test, a),
                                 # realised workload (Phase 7 §5): counts, not only rates
                                 int(a.sum()), int((a & y_test).sum()), int((a & (1 - y_test)).sum())]
        log(f"  {name}: AUC-PR {metrics_rows[name][5]:.4f}, "
            f"budget recall {budget_rows[name][2]:.4f} precision {budget_rows[name][3]:.4f} "
            f"({time.time() - s:.0f}s)")

    cols = [m for m in MODEL_ORDER if m in metrics_rows]
    met = pd.DataFrame({"metric\\model": ["Recall", "Precision", "Specificity", "F1 Score",
                                          "AUC", "AUC-PR", "MCC", "Balanced Classification Rate"],
                        **{m: metrics_rows[m] for m in cols}})
    bud = pd.DataFrame({"metric\\model": ["Operating threshold (from validation)", "Alert rate on test",
                                          "Recall (budget)", "Precision (budget)",
                                          "F1 (budget)", "MCC (budget)",
                                          "Alerts on test (count)", "TP (budget)", "FP (budget)"],
                        **{m: budget_rows[m] for m in cols}})
    met.to_csv(OUT / f"test_metric_result_V01_{tag}.csv", index=False)
    bud.to_csv(OUT / f"test_operating_point_V01_{tag}.csv", index=False)
    # `row` is the transaction's row number in fraudTrain.csv (every data file is row-aligned
    # with it), so later scripts can attach cc_num and timestamps without rebuilding the split.
    pd.DataFrame(preds_out).assign(**{TARGET: y_test, "row": test.index.values}).to_csv(
        OUT / f"test_predictions_{tag}.csv.gz", index=False, compression="gzip")
    log(f"wrote test_metric_result_V01_{tag}.csv, test_operating_point_V01_{tag}.csv, "
        f"test_predictions_{tag}.csv.gz")

    if args.check:
        ok = True
        for frame, fname in [(met, f"test_metric_result_V01_{tag}.csv"),
                             (bud, f"test_operating_point_V01_{tag}.csv")]:
            ref_path = RESULTS / fname
            if not ref_path.exists():
                log(f"CHECK skipped: {fname} is not committed"); continue
            ref = pd.read_csv(ref_path)
            b = ref.set_index("metric\\model")[cols]
            a = frame.set_index("metric\\model")[cols].loc[b.index]   # rows added since are not in the reference
            d = (a - b).abs()
            worst = float(np.nanmax(d.values))
            log(f"CHECK {fname}: max |diff| = {worst:.3e}")
            if worst > 1e-9:
                ok = False
                print(d[d > 1e-9].dropna(how="all").dropna(axis=1, how="all").to_string())
        print("\nGUARD " + ("PASSED -- this script reproduces the notebook's protocol"
                            if ok else "FAILED -- do not run the new configurations yet"))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
