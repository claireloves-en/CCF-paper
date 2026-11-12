"""
Phase 2 items 6 + 9 (Revision_Plan_V02.md): temporal vs random split, and a
budget-based operating point.

Restored 2026-09-14. The original ran from a session scratchpad that was later
wiped; this is the same code with only the paths made relative and a --configs
option added for partial re-runs. It produced:

  result_csv/split_comparison_V01.csv   (= output/phase2_split_comparison.csv)

Three configurations, each with the Phase 1 corrected metrics:
  random          - train_test_split(shuffle=True), the submitted design
  temporal        - chronological 64/16/20 by trans_time
  temporal_notime - chronological, with trans_time dropped from the predictors
                    (diagnostic: under a temporal split the model is asked to
                    extrapolate past every trans_time value it ever saw)

Per model, on the validation set:
  - metrics at threshold 0.5
  - ROC-AUC and AUC-PR (threshold-free)
  - budget operating point: threshold set so ~0.5% of transactions are alerted
  - F1-optimal threshold

Resampling (ROS 1:150) is applied to the training split only, in every config.
The untuned V21 hyperparameters are used throughout -- this predates D7 and D8.

Usage: python phase2_splits.py [--configs random,temporal,temporal_notime]
~10 minutes per configuration.
"""
import gc, sys, time, warnings
from datetime import datetime
from pathlib import Path
from data_paths import data_file  # encoded files: current, or archived via CCF_ENCODED_VERSION

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             roc_auc_score, matthews_corrcoef,
                             average_precision_score, precision_recall_curve)
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

RFC_METRIC, NUM_ESTIMATORS, NO_JOBS = 'gini', 100, 4
VALID_SIZE = TEST_SIZE = 0.20
RANDOM_STATE = 1200
MAX_ROUNDS, EARLY_STOP, VERBOSE_EVAL = 1000, 50, 50
ALERT_RATE = 0.005          # fraud-investigation budget: 0.5% of transactions

target = 'is_fraud'
ALL_PREDICTORS = ['trans_time', 'trans_time_grp', 'category_1', 'category_2',
    'category_3', 'category_4a', 'category_4b', 'category_4c',
    'category_interaction_1', 'category_interaction_2', 'amt', 'gender',
    'city_1', 'city_2', 'city_3', 'city_4a', 'city_4b', 'state_x', 'state_y',
    'job_1', 'job_2', 'job_3', 'job_4', 'city_pop', 'loc_cluster_issue',
    'loc_cluster_transac', 'issue_transac_dist', 'cardholder_age']

CONFIGS = ["random", "temporal", "temporal_notime"]
if "--configs" in sys.argv:
    CONFIGS = sys.argv[sys.argv.index("--configs") + 1].split(",")


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


df_enc = pd.read_csv(data_file("fraudTrain_encoded.csv"))
log("loaded", df_enc.shape)


def make_split(kind):
    if kind == "random":
        tr, te = train_test_split(df_enc, test_size=TEST_SIZE,
                                  random_state=RANDOM_STATE, shuffle=True)
        tr, va = train_test_split(tr, test_size=VALID_SIZE,
                                  random_state=RANDOM_STATE, shuffle=True)
        return tr, va, te
    # chronological: cut on trans_time (hours since the first transaction, a
    # monotone transform of unix_time) so no timestamp straddles a boundary.
    t = df_enc['trans_time']
    c1, c2 = t.quantile(0.64), t.quantile(0.80)
    tr = df_enc[t <= c1]
    va = df_enc[(t > c1) & (t <= c2)]
    te = df_enc[t > c2]
    log(f"  temporal cutoffs: train <= {c1:.0f}h, valid <= {c2:.0f}h, test > {c2:.0f}h")
    return tr, va, te


def operating_points(y, proba):
    """Budget-based and F1-optimal thresholds, on the validation set.

    Note: this first implementation has no guard for coarse scorers -- where
    more than (1 - ALERT_RATE) of rows tie at 0 it alerts everything. That
    degeneracy (B5) is recorded in alert_rate and handled in model_V23.
    """
    proba = np.asarray(proba)
    k = max(1, int(round(len(proba) * ALERT_RATE)))
    thr_budget = np.partition(proba, -k)[-k]        # k-th largest score
    alerted = proba >= thr_budget
    rec_b = recall_score(y, alerted.astype(int))
    prec_b = precision_score(y, alerted.astype(int), zero_division=0)

    prec, rec, thr = precision_recall_curve(y, proba)
    f1s = np.divide(2 * prec * rec, prec + rec,
                    out=np.zeros_like(prec), where=(prec + rec) > 0)
    j = int(np.argmax(f1s[:-1])) if len(thr) else 0
    thr_f1 = float(thr[j]) if len(thr) else 0.5
    pred_f1 = (proba >= thr_f1).astype(int)
    return dict(thr_budget=float(thr_budget), alert_rate=float(alerted.mean()),
                recall_budget=rec_b, precision_budget=prec_b,
                thr_f1=thr_f1, recall_f1=recall_score(y, pred_f1),
                precision_f1=precision_score(y, pred_f1, zero_division=0),
                f1_f1=f1_score(y, pred_f1), mcc_f1=matthews_corrcoef(y, pred_f1))


rows = []


def record(config, name, y, proba):
    proba = np.asarray(proba)
    binary = (proba > 0.5).astype(int)
    rec = recall_score(y, binary)
    spec = recall_score(y, binary, pos_label=0)
    op = operating_points(y, proba)
    rows.append(dict(config=config, model=name,
        recall=rec, precision=precision_score(y, binary, zero_division=0),
        specificity=spec, f1=f1_score(y, binary),
        auc=roc_auc_score(y, proba), aucpr=average_precision_score(y, proba),
        mcc=matthews_corrcoef(y, binary), bcr=(rec + spec) / 2, **op))
    r = rows[-1]
    log(f"    {name:20s} AUC={r['auc']:.4f} AUC-PR={r['aucpr']:.4f} "
        f"MCC={r['mcc']:.4f} | budget rec={r['recall_budget']:.3f} "
        f"prec={r['precision_budget']:.3f} @rate={r['alert_rate']*100:.2f}%")


def run_config(config, kind, predictors):
    log(f"=== {config} ({len(predictors)} predictors) ===")
    train_df, valid_df, test_df = make_split(kind)

    ros = RandomOverSampler(sampling_strategy=0.0066667, random_state=RANDOM_STATE)
    X, y = ros.fit_resample(train_df[predictors], train_df[target])
    train_df = pd.DataFrame(X, columns=predictors)
    train_df[target] = y
    yv = valid_df[target].values

    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(train_df[predictors]), sc.transform(valid_df[predictors])
    m = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, train_df[target])
    record(config, "Logistic Regression", yv, m.predict_proba(Xva)[:, 1])
    del m, Xtr, Xva; gc.collect()

    c = DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    c.fit(train_df[predictors], train_df[target].values)
    record(config, "Decision Tree", yv, c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    c = RandomForestClassifier(n_jobs=NO_JOBS, random_state=RANDOM_STATE,
                               criterion=RFC_METRIC, n_estimators=NUM_ESTIMATORS,
                               verbose=False, class_weight='balanced')
    c.fit(train_df[predictors], train_df[target].values)
    record(config, "Random Forest", yv, c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    c = AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME',
                           learning_rate=0.8, n_estimators=NUM_ESTIMATORS)
    cw = compute_class_weight('balanced', classes=np.unique(train_df[target].values),
                              y=train_df[target].values)
    cwd = dict(enumerate(cw))
    c.fit(train_df[predictors], train_df[target].values,
          sample_weight=np.array([cwd[v] for v in train_df[target].values]))
    record(config, "AdaBoost", yv, c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    c = CatBoostClassifier(iterations=500, learning_rate=0.02, depth=12,
                           eval_metric='AUC', random_seed=RANDOM_STATE,
                           bagging_temperature=0.2, od_type='Iter',
                           metric_period=VERBOSE_EVAL, od_wait=100)
    c.fit(train_df[predictors], train_df[target].values, verbose=False)
    record(config, "CatBoost", yv, c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()

    dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
    dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
    p = {'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
         'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc',
         'seed': RANDOM_STATE}
    bst = xgb.train(p, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                    early_stopping_rounds=EARLY_STOP, maximize=True, verbose_eval=False)
    record(config, "XGBoost", yv, bst.predict(dvalid))
    del dtrain, dvalid, bst; gc.collect()

    names = [f"f{i}" for i in range(len(predictors))]
    lp = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
          'learning_rate': 0.05, 'num_leaves': 7, 'max_depth': 4,
          'min_child_samples': 100, 'max_bin': 100, 'subsample': 0.9,
          'subsample_freq': 1, 'colsample_bytree': 0.7, 'min_child_weight': 0,
          'min_split_gain': 0, 'nthread': 8, 'verbose': -1, 'scale_pos_weight': 150,
          'seed': RANDOM_STATE}
    lt = lgb.Dataset(train_df[predictors].values, label=train_df[target].values, feature_name=names)
    lv = lgb.Dataset(valid_df[predictors].values, label=valid_df[target].values, feature_name=names)
    bst = lgb.train(lp, lt, valid_sets=[lt, lv], valid_names=['train', 'valid'],
                    num_boost_round=MAX_ROUNDS,
                    callbacks=[lgb.early_stopping(2 * EARLY_STOP, verbose=False)])
    record(config, "LightGBM", yv, bst.predict(valid_df[predictors].values))
    del lt, lv, bst; gc.collect()

    c = KNeighborsClassifier(1, n_jobs=-1)
    c.fit(train_df[predictors], train_df[target].values)
    record(config, "K-NN", yv, c.predict_proba(valid_df[predictors])[:, 1])
    del c; gc.collect()


t0 = time.time()
SPEC = {"random": ("random", ALL_PREDICTORS),
        "temporal": ("temporal", ALL_PREDICTORS),
        "temporal_notime": ("temporal", [p for p in ALL_PREDICTORS if p != 'trans_time'])}
for cfg in CONFIGS:
    run_config(cfg, *SPEC[cfg])
log(f"total {(time.time()-t0)/60:.1f} min")

pd.DataFrame(rows).to_csv(OUT / "phase2_split_comparison.csv", index=False)
