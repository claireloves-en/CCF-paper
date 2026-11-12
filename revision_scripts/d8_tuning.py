"""
D8 (Revision_Plan_V02.md / reviewer comment A5): randomised hyperparameter
search, selected on validation AUC-PR.

Restored 2026-09-14. The original ran from a session scratchpad that was later
wiped; this is the same search -- identical grids, identical RNG consumption --
with three changes that do not affect results:
  - paths made relative;
  - each trial's hyperparameters are written as one JSON column. The original
    appended rows whose columns differed per model under a single header, so the
    committed hyperparameter_trials_V01.csv could only keep scores, not params;
  - a --models option for partial re-runs. Skipped models still draw their 40
    samples, so the random stream -- and every later model's trials -- match a
    full run exactly.

It produced (after reshaping):
  result_csv/hyperparameter_tuning_V01.csv   (from output/d8_final.csv)
  result_csv/hyperparameter_trials_V01.csv   (from output/d8_trials.csv, scores only)

Design, all of it reportable in the appendix table:
  - chronological 64/16/20 split (D6); no resampling (D7 selected it).
  - 40 random draws per model from the grids below.
  - each draw is fitted on a class-stratified 25% subsample of the training
    set; the winning configuration is then refit on the FULL training set and
    that refit is the number reported.
  - selection metric is AUC-PR on the validation set.
  - XGBoost and LightGBM early-stop on the validation set, on AUC-PR.
  - the logistic-regression scaler is fitted on the full training set.

The validation set is used for early stopping, hyperparameter selection and
reporting, which is optimistic. test_df is untouched here; the honest numbers
come from the one-time test evaluation in model_V24.

Usage: python d8_tuning.py [--drop-trans-time] [--client-features] [--raw-baseline]
                           [--data FILE --tag NAME]   (Phase 7: any data file, file column order)
                           [--models "Logistic Regression,Decision Tree"]
~110 minutes for all eight models.
"""
import gc, json, sys, time, warnings
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
TRIALS_CSV, FINAL_CSV = OUT / "d8_trials.csv", OUT / "d8_final.csv"

RANDOM_STATE = 1200
N_TRIALS = 40
SUBSAMPLE = 0.25
target = 'is_fraud'
predictors = ['trans_time', 'trans_time_grp', 'category_1', 'category_2',
    'category_3', 'category_4a', 'category_4b', 'category_4c',
    'category_interaction_1', 'category_interaction_2', 'amt', 'gender',
    'city_1', 'city_2', 'city_3', 'city_4a', 'city_4b', 'state_x', 'state_y',
    'job_1', 'job_2', 'job_3', 'job_4', 'city_pop', 'loc_cluster_issue',
    'loc_cluster_transac', 'issue_transac_dist', 'cardholder_age']

# B4 (Revision_Plan_V03.md): the authors decided to drop trans_time. It is still
# used to cut the chronological split; this only removes it from the predictors.
DROP_TRANS_TIME = "--drop-trans-time" in sys.argv
# D10: add the seven client-level behavioural features, computed causally by
# MLProj01_fraudTrain_clientfeatures_V01.ipynb into fraudTrain_encoded_client.csv.
CLIENT_FEATURES = "--client-features" in sys.argv
CLIENT = ['card_amt_zscore', 'card_hours_since_prev', 'card_txn_1h', 'card_txn_24h',
          'card_txn_7d', 'card_dist_from_usual_km', 'card_new_category']
DATA_FILE = "fraudTrain_encoded_client.csv" if CLIENT_FEATURES else "fraudTrain_encoded.csv"
if DROP_TRANS_TIME:
    predictors = [p for p in predictors if p != 'trans_time']
if CLIENT_FEATURES:
    predictors = predictors + CLIENT
TAG = ("_notime" if DROP_TRANS_TIME else "") + ("_client" if CLIENT_FEATURES else "")

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
    if not DROP_TRANS_TIME:
        raise SystemExit("--raw-baseline has no trans_time predictor; pass --drop-trans-time too")

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

if TAG:
    TRIALS_CSV, FINAL_CSV = OUT / f"d8_trials{TAG}.csv", OUT / f"d8_final{TAG}.csv"

ONLY = None
if "--models" in sys.argv:
    ONLY = set(sys.argv[sys.argv.index("--models") + 1].split(","))

rng = np.random.default_rng(RANDOM_STATE)


def pick(seq):
    """Uniform choice that preserves the element's Python type.

    np.random.Generator.choice coerces through a numpy array, which turns a
    mixed list like ['balanced', None] into an object array whose elements have
    no .item(), and a numeric list into numpy scalars. Index instead.
    """
    return seq[int(rng.integers(len(seq)))]


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


df_enc = pd.read_csv(data_file(DATA_FILE))
t = df_enc['trans_time']
c1, c2 = t.quantile(0.64), t.quantile(0.80)
TRAIN, VALID = df_enc[t <= c1], df_enc[(t > c1) & (t <= c2)]
log(f"{len(predictors)} predictors{TAG or ' (as published)'}; train {len(TRAIN)} / valid {len(VALID)}; fraud {int(TRAIN[target].sum())}/{int(VALID[target].sum())}")

SUB, _ = train_test_split(TRAIN, train_size=SUBSAMPLE, random_state=RANDOM_STATE,
                          stratify=TRAIN[target])
log(f"search subsample {len(SUB)} rows, fraud {int(SUB[target].sum())}")

Xv, yv = VALID[predictors], VALID[target].values
# Phase 7 catnative: string columns are passed to CatBoost as cat_features. Only CatBoost can be
# searched on such a file (--models CatBoost); the other fitters need numeric input.
CAT_COLS = [c for c in predictors if TRAIN[c].dtype == object]
if CAT_COLS:
    log(f"categorical (CatBoost cat_features): {CAT_COLS}")
_scaler = None if CAT_COLS else StandardScaler().fit(TRAIN[predictors])


# ----------------------------------------------------------------- grids
# max_depth=0 is decoded to None (unlimited) in _sk().
def g_logreg():
    return dict(C=float(10 ** rng.uniform(-3, 2)),
                class_weight=pick(['balanced', None]),
                solver=pick(['lbfgs', 'liblinear']))


def g_tree():
    return dict(max_depth=int(pick([4, 6, 8, 10, 14, 20, 30, 0])),
                min_samples_split=int(pick([2, 10, 50, 200, 1000])),
                min_samples_leaf=int(pick([1, 5, 20, 100, 500])),
                criterion=pick(['gini', 'entropy']),
                class_weight=pick(['balanced', None]))


def g_forest():
    return dict(n_estimators=int(pick([100, 200, 400])),
                max_depth=int(pick([6, 10, 16, 24, 0])),
                min_samples_split=int(pick([2, 10, 50, 200])),
                min_samples_leaf=int(pick([1, 5, 20, 100])),
                max_features=pick(['sqrt', 'log2', 0.5]),
                class_weight=pick(['balanced', 'balanced_subsample', None]))


def g_ada():
    return dict(n_estimators=int(pick([50, 100, 200, 400])),
                learning_rate=float(10 ** rng.uniform(-2, 0.3)))


def g_cat():
    return dict(iterations=int(pick([300, 500, 1000])),
                depth=int(pick([4, 6, 8, 10, 12])),
                learning_rate=float(10 ** rng.uniform(-2.2, -0.7)),
                l2_leaf_reg=float(pick([1, 3, 10, 30])),
                bagging_temperature=float(rng.uniform(0, 1)))


def g_xgb():
    return dict(eta=float(10 ** rng.uniform(-2.2, -0.7)),
                max_depth=int(pick([2, 3, 4, 6, 8, 10])),
                subsample=float(rng.uniform(0.6, 1.0)),
                colsample_bytree=float(rng.uniform(0.5, 1.0)),
                min_child_weight=float(pick([1, 5, 20, 100])),
                gamma=float(pick([0, 0.5, 2, 5])),
                scale_pos_weight=float(pick([1, 10, 50, 150])))


def g_lgb():
    return dict(learning_rate=float(10 ** rng.uniform(-2.2, -0.7)),
                num_leaves=int(pick([7, 15, 31, 63, 127])),
                max_depth=int(pick([4, 6, 8, 12, -1])),
                min_child_samples=int(pick([20, 50, 100, 300])),
                feature_fraction=float(rng.uniform(0.5, 1.0)),
                bagging_fraction=float(rng.uniform(0.6, 1.0)),
                scale_pos_weight=float(pick([1, 10, 50, 150])))


def g_knn():
    return dict(n_neighbors=int(pick([1, 3, 5, 11, 25, 50])),
                weights=pick(['uniform', 'distance']),
                p=int(pick([1, 2])))


# ---------------------------------------------------------------- fitters
def f_logreg(p, tr):
    m = LogisticRegression(max_iter=1000, C=p['C'], class_weight=p['class_weight'],
                           solver=p['solver'])
    m.fit(_scaler.transform(tr[predictors]), tr[target])
    return m.predict_proba(_scaler.transform(Xv))[:, 1]


def _sk(cls, p, tr, **extra):
    p = {k: (None if k == 'max_depth' and v == 0 else v) for k, v in p.items()}
    m = cls(random_state=RANDOM_STATE, **p, **extra)
    m.fit(tr[predictors], tr[target].values)
    return m.predict_proba(Xv)[:, 1]


def f_tree(p, tr):
    return _sk(DecisionTreeClassifier, p, tr)


def f_forest(p, tr):
    return _sk(RandomForestClassifier, p, tr, n_jobs=4)


def f_ada(p, tr):
    cw = compute_class_weight('balanced', classes=np.unique(tr[target].values),
                              y=tr[target].values)
    d = dict(enumerate(cw))
    m = AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME', **p)
    m.fit(tr[predictors], tr[target].values,
          sample_weight=np.array([d[v] for v in tr[target].values]))
    return m.predict_proba(Xv)[:, 1]


def f_cat(p, tr):
    m = CatBoostClassifier(random_seed=RANDOM_STATE, eval_metric='AUC',
                           od_type='Iter', od_wait=100, verbose=False, **p)
    m.fit(tr[predictors], tr[target].values, cat_features=CAT_COLS or None, verbose=False)
    return m.predict_proba(Xv)[:, 1]


def f_xgb(p, tr):
    dtr = xgb.DMatrix(tr[predictors], tr[target].values)
    dva = xgb.DMatrix(Xv, yv)
    params = {'objective': 'binary:logistic', 'eval_metric': 'aucpr',
              'seed': RANDOM_STATE, **p}
    bst = xgb.train(params, dtr, 1000, [(dva, 'valid')],
                    early_stopping_rounds=50, verbose_eval=False)
    out = bst.predict(dva)
    del dtr, dva, bst
    return out


def f_lgb(p, tr):
    names = [f"f{i}" for i in range(len(predictors))]
    params = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'average_precision',
              'verbose': -1, 'nthread': 8, 'seed': RANDOM_STATE, 'bagging_freq': 1, **p}
    dtr = lgb.Dataset(tr[predictors].values, label=tr[target].values, feature_name=names)
    dva = lgb.Dataset(Xv.values, label=yv, feature_name=names)
    bst = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=1000,
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    out = bst.predict(Xv.values)
    del dtr, dva, bst
    return out


def f_knn(p, tr):
    m = KNeighborsClassifier(n_jobs=-1, **p)
    m.fit(tr[predictors], tr[target].values)
    return m.predict_proba(Xv)[:, 1]


MODELS = [
    ("Logistic Regression", g_logreg, f_logreg),
    ("Decision Tree",       g_tree,   f_tree),
    ("Random Forest",       g_forest, f_forest),
    ("AdaBoost",            g_ada,    f_ada),
    ("CatBoost",            g_cat,    f_cat),
    ("XGBoost",             g_xgb,    f_xgb),
    ("LightGBM",            g_lgb,    f_lgb),
    ("K-NN",                g_knn,    f_knn),
]


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)


def main():
    for _f in (TRIALS_CSV, FINAL_CSV):     # fresh outputs for a fresh search
        if _f.exists():
            _f.unlink()
    t0 = time.time()
    for name, grid, fit in MODELS:
        if ONLY is not None and name not in ONLY:
            for _ in range(N_TRIALS):     # keep the random stream aligned with a full run
                grid()
            continue
        log(f"=== {name}: {N_TRIALS} trials on {len(SUB)} rows ===")
        best = (None, -1.0)
        for k in range(N_TRIALS):
            p = grid()
            try:
                s = time.time()
                proba = fit(p, SUB)
                ap = average_precision_score(yv, proba)
                secs = time.time() - s
            except Exception as e:
                log(f"  trial {k:2d} FAILED: {type(e).__name__}: {str(e)[:90]}")
                append(TRIALS_CSV, dict(model=name, trial=k, aucpr=np.nan, seconds=np.nan,
                                        error=str(e)[:200], params=json.dumps(p)))
                continue
            append(TRIALS_CSV, dict(model=name, trial=k, aucpr=ap, seconds=secs,
                                    error="", params=json.dumps(p)))
            if ap > best[1]:
                best = (p, ap)
                log(f"  trial {k:2d} AUC-PR={ap:.4f}  <-- best  {p}")
            gc.collect()

        log(f"  refitting {name} on the full training set ({len(TRAIN)} rows)")
        proba = fit(best[0], TRAIN)
        row = dict(model=name,
                   aucpr_subsample=best[1],
                   aucpr_full=average_precision_score(yv, proba),
                   auc_full=roc_auc_score(yv, proba),
                   mcc_full_at_0p5=matthews_corrcoef(yv, (proba > 0.5).astype(int)),
                   params=json.dumps(best[0]))
        append(FINAL_CSV, row)
        log(f"  {name}: subsample {best[1]:.4f} -> full-train {row['aucpr_full']:.4f} "
            f"(AUC {row['auc_full']:.4f})")
        gc.collect()

    log(f"TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
