"""
D7 (Revision_Plan_V02.md): sampling sweep, corrected and scored by AUC-PR.

Restored 2026-09-14. The original ran from a session scratchpad that was later
wiped; this is the same code with only the paths made relative and an
--only-no-sampling option added for a quick reproduction check. It produced the
long-format output/d7_sweep_rows.csv, which was reshaped into:

  result_csv/class_ratio_{rus,ros,smote}_V06.csv         (MCC)
  result_csv/class_ratio_{rus,ros,smote}_aucpr_V06.csv   (AUC-PR)
  result_csv/best_sampling_per_model_V01.csv

findratio_V10's ML_Models() is the notebook form of the same code.

Changes from the V08/V09 sweep:
  - every model scored on valid_df. V08 scored XGBoost and LightGBM on test_df,
    so class_ratio_*_V05.csv is not comparable across models.
  - probabilities retained; AUC-PR (average precision) recorded alongside MCC.
  - the split is re-derived before every ratio. V08 overwrote train_df with the
    resampled frame and carried it into the next iteration, compounding.
  - chronological 64/16/20 split on trans_time (the adopted primary design).

Predictor set is the full 28 the paper describes. Untuned V21 hyperparameters
(this predates D8).

~205 minutes for all 25 configurations; ~10 with --only-no-sampling.
"""
import gc, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
from data_paths import data_file  # encoded files: current, or archived via CCF_ENCODED_VERSION

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
ROWS = OUT / "d7_sweep_rows.csv"   # removed below, once the configuration (and so the file) is known

ONLY_NO_SAMPLING = "--only-no-sampling" in sys.argv
# Data configuration, matching d8_tuning.py. With --only-no-sampling these give the
# untuned baseline each tuned configuration is compared against.
DROP_TRANS_TIME = "--drop-trans-time" in sys.argv
CLIENT_FEATURES = "--client-features" in sys.argv
TAG = ("_notime" if DROP_TRANS_TIME else "") + ("_client" if CLIENT_FEATURES else "")
if TAG:
    ROWS = OUT / f"d7_sweep_rows{TAG}.csv"
    if ROWS.exists():
        ROWS.unlink()

RFC_METRIC, NUM_ESTIMATORS, NO_JOBS = 'gini', 100, 4
RANDOM_STATE = 1200
MAX_ROUNDS, EARLY_STOP = 1000, 50

target = 'is_fraud'
predictors = ['trans_time', 'trans_time_grp', 'category_1', 'category_2',
    'category_3', 'category_4a', 'category_4b', 'category_4c',
    'category_interaction_1', 'category_interaction_2', 'amt', 'gender',
    'city_1', 'city_2', 'city_3', 'city_4a', 'city_4b', 'state_x', 'state_y',
    'job_1', 'job_2', 'job_3', 'job_4', 'city_pop', 'loc_cluster_issue',
    'loc_cluster_transac', 'issue_transac_dist', 'cardholder_age']

CLIENT = ['card_amt_zscore', 'card_hours_since_prev', 'card_txn_1h', 'card_txn_24h',
          'card_txn_7d', 'card_dist_from_usual_km', 'card_new_category']
if DROP_TRANS_TIME:
    predictors = [p for p in predictors if p != 'trans_time']
if CLIENT_FEATURES:
    predictors = predictors + CLIENT

# J1 (Revision_Phase6_Plan_V01.md): the conventional baseline from the raw Sparkov
# columns, built by raw_baseline_features.py. Its predictors are taken from the data
# file's COLUMN ORDER rather than a hardcoded list -- that is the order the model
# notebook uses, and random forest's max_features and XGBoost's colsample_bytree sample
# features by index, so a different order gives different fits. The hardcoded list above
# is kept as it is for the encoded configurations: it is what the committed
# hyperparameter tables were searched under, and verify_restored.py checks them.
RAW_BASELINE = "--raw-baseline" in sys.argv
if RAW_BASELINE:
    DATA_FILE = ("fraudTrain_raw_baseline_client.csv" if CLIENT_FEATURES
                 else "fraudTrain_raw_baseline.csv")
    predictors = [c for c in pd.read_csv(data_file(DATA_FILE), nrows=1).columns
                  if c not in (target, 'trans_time')]
    TAG = "_notime_raw" + ("_client" if CLIENT_FEATURES else "")
    ROWS = OUT / f"d7_sweep_rows{TAG}.csv"
    if ROWS.exists():
        ROWS.unlink()

# Phase 7 (Revision_Phase7_Plan_V01.md): any prepared data file, predictors in its COLUMN
# ORDER (as the model notebook and ablation_run.py use them), outputs suffixed _<tag>.
# Used for every configuration fitted on the corrected encoded files, so none of them
# inherits the hardcoded order above.
#   --data fraudTrain_encoded.csv --tag notime_v02
GENERIC = "--data" in sys.argv
if GENERIC:
    DATA_FILE = sys.argv[sys.argv.index("--data") + 1]
    TAG = "_" + sys.argv[sys.argv.index("--tag") + 1]
    predictors = [c for c in pd.read_csv(data_file(DATA_FILE), nrows=1).columns
                  if c not in (target, 'trans_time')]
    ROWS = OUT / f"d7_sweep_rows{TAG}.csv"
    if ROWS.exists():
        ROWS.unlink()

if ROWS.exists():                 # fresh rows for a fresh run, for this configuration only
    ROWS.unlink()

ratio_list = [0.0066667, 0.01, 0.02, 0.111111, 0.25, 0.428571, 0.666666, 1]
ratio_names = ["1:150", "1:100", "1:50", "10:90", "20:80", "30:70", "40:60", "50:50(1:1)"]


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


DATA_FILE = (DATA_FILE if RAW_BASELINE or GENERIC else
             ("fraudTrain_encoded_client.csv" if CLIENT_FEATURES else "fraudTrain_encoded.csv"))
df_enc = pd.read_csv(data_file(DATA_FILE))
log(f"{len(predictors)} predictors{TAG or ' (as published)'}")
_t = df_enc['trans_time']
C1, C2 = _t.quantile(0.64), _t.quantile(0.80)
log(f"temporal cutoffs: train <= {C1:.0f}h, valid <= {C2:.0f}h, test > {C2:.0f}h")


def split():
    t = df_enc['trans_time']
    return df_enc[t <= C1], df_enc[(t > C1) & (t <= C2)]


def ML_Models(train_df, valid_df):
    """Eight models, trained on train_df, scored on valid_df. Returns per-model
    MCC (at 0.5) and AUC-PR, in the canonical model order."""
    yv = valid_df[target].values
    mcc, apr = [], []

    def score(name, proba):
        proba = np.asarray(proba)
        m = matthews_corrcoef(yv, (proba > 0.5).astype(int))
        a = average_precision_score(yv, proba)
        mcc.append(m); apr.append(a)
        log(f"      {name:20s} MCC={m:.4f} AUC-PR={a:.4f}")

    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(train_df[predictors]), sc.transform(valid_df[predictors])
    m = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, train_df[target])
    score("Logistic Regression", m.predict_proba(Xva)[:, 1])
    del m, Xtr, Xva; gc.collect()

    c = DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    c.fit(train_df[predictors], train_df[target].values)
    score("Decision Tree", c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    c = RandomForestClassifier(n_jobs=NO_JOBS, random_state=RANDOM_STATE,
                               criterion=RFC_METRIC, n_estimators=NUM_ESTIMATORS,
                               verbose=False, class_weight='balanced')
    c.fit(train_df[predictors], train_df[target].values)
    score("Random Forest", c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    c = AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME',
                           learning_rate=0.8, n_estimators=NUM_ESTIMATORS)
    cw = compute_class_weight('balanced', classes=np.unique(train_df[target].values),
                              y=train_df[target].values)
    cwd = dict(enumerate(cw))
    c.fit(train_df[predictors], train_df[target].values,
          sample_weight=np.array([cwd[v] for v in train_df[target].values]))
    score("AdaBoost", c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    c = CatBoostClassifier(iterations=500, learning_rate=0.02, depth=12,
                           eval_metric='AUC', random_seed=RANDOM_STATE,
                           bagging_temperature=0.2, od_type='Iter',
                           metric_period=50, od_wait=100)
    c.fit(train_df[predictors], train_df[target].values, verbose=False)
    score("CatBoost", c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
    dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
    p = {'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
         'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc',
         'seed': RANDOM_STATE}
    bst = xgb.train(p, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                    early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)
    score("XGBoost", bst.predict(dvalid))
    del dtrain, dvalid, bst; gc.collect()

    names = [f"f{i}" for i in range(len(predictors))]
    lp = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
          'learning_rate': 0.05, 'num_leaves': 7, 'max_depth': 4,
          'min_child_samples': 100, 'max_bin': 100, 'subsample': 0.9,
          'subsample_freq': 1, 'colsample_bytree': 0.7, 'min_child_weight': 0,
          'min_split_gain': 0, 'nthread': 8, 'verbose': -1,
          'scale_pos_weight': 150, 'seed': RANDOM_STATE}
    lt = lgb.Dataset(train_df[predictors].values, label=train_df[target].values, feature_name=names)
    lv = lgb.Dataset(valid_df[predictors].values, label=valid_df[target].values, feature_name=names)
    bst = lgb.train(lp, lt, valid_sets=[lt, lv], valid_names=['train', 'valid'],
                    num_boost_round=MAX_ROUNDS,
                    callbacks=[lgb.early_stopping(2 * EARLY_STOP, verbose=False)])
    score("LightGBM", bst.predict(valid_df[predictors].values))
    del lt, lv, bst; gc.collect()

    c = KNeighborsClassifier(1, n_jobs=-1)
    c.fit(train_df[predictors], train_df[target].values)
    score("K-NN", c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    return mcc, apr


MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
          "CatBoost", "XGBoost", "LightGBM", "K-NN"]


def emit(method, ratio_name, mcc, apr, secs):
    rows = [dict(method=method, ratio=ratio_name, model=m, mcc=mc, aucpr=ap,
                 seconds=secs) for m, mc, ap in zip(MODELS, mcc, apr)]
    pd.DataFrame(rows).to_csv(ROWS, mode='a', header=not ROWS.exists(), index=False)


t0 = time.time()

# --- no sampling (shared baseline column) --------------------------------
log("=== No Sampling (1:173) ===")
tr, va = split()
s = time.time(); mcc, apr = ML_Models(tr, va); emit("No Sampling", "No Sampling (1:173)", mcc, apr, time.time() - s)

# --- the three methods x eight ratios ------------------------------------
SAMPLERS = {
    "RUS":   lambda r: RandomUnderSampler(sampling_strategy=r, random_state=RANDOM_STATE),
    "ROS":   lambda r: RandomOverSampler(sampling_strategy=r, random_state=RANDOM_STATE),
    "SMOTE": lambda r: SMOTE(sampling_strategy=r, random_state=RANDOM_STATE),
}
if not ONLY_NO_SAMPLING:
    for method, make in SAMPLERS.items():
        for r, rname in zip(ratio_list, ratio_names):
            log(f"=== {method} {rname} ===")
            tr, va = split()                      # re-split every ratio: no compounding
            X, y = make(r).fit_resample(tr[predictors], tr[target])
            tr = pd.DataFrame(X, columns=predictors); tr[target] = y
            s = time.time()
            mcc, apr = ML_Models(tr, va)
            emit(method, rname, mcc, apr, time.time() - s)
            del tr, X, y; gc.collect()
            log(f"    elapsed so far {(time.time()-t0)/60:.1f} min")

log(f"TOTAL {(time.time()-t0)/60:.1f} min")
