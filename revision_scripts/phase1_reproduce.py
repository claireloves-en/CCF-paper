"""
Phase 0 reproduction check + Phase 1 corrected metrics (Revision_Plan_V02.md).

Restored 2026-09-14. The original ran from a session scratchpad that was later
wiped; this is the same code, with only the input/output paths made relative
and the seeded / unseeded variants folded into one flag. It produced:

  default (seeded)  -> result_csv/models_metric_result_V05_replica.csv
                       (= output/phase1_corrected.csv)
  --unseeded        -> result_csv/models_metric_result_V04_repro_diff.csv
                       (= output/repro_diff.csv)

It replicates MLProj01_fraudTrain_model_V21.ipynb exactly, bugs included, so
the as-is table can be diffed against models_metric_result_V04.csv, and in the
same pass computes the corrected metrics Phase 1 calls for:
  - every model scored on valid_df
  - ROC-AUC from predicted probabilities, never from thresholded labels
  - AUC-PR (average precision) added

Seeded vs unseeded: V21 constructed the Decision Tree without random_state and
passed params['random_state'] to xgb.train, which reads 'seed' and silently
ignored it. --unseeded reproduces that; the default seeds both.

~10 minutes.
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
                             average_precision_score)
from imblearn.over_sampling import RandomOverSampler
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

SEEDED = "--unseeded" not in sys.argv

HERE = Path(__file__).resolve().parent
PATH = HERE.parent                      # MLProj01_US/
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# --- constants, verbatim from cell 0 -------------------------------------
RFC_METRIC = 'gini'
NUM_ESTIMATORS = 100
NO_JOBS = 4
VALID_SIZE = 0.20
TEST_SIZE = 0.20
RANDOM_STATE = 1200
MAX_ROUNDS = 1000
EARLY_STOP = 50
VERBOSE_EVAL = 50

column_names_dict = {
    'trans_time': 'Time difference from the first transaction',
    'trans_time_grp': 'Transaction time group',
    'category_1': 'Offline vs. online',
    'category_2': 'Non-essential vs. essential',
    'category_3': 'Average item cost',
    'category_4a': 'Purpose of purchase: household/wellbeing',
    'category_4b': 'Purpose of purchase: entertainment/lifestyle',
    'category_4c': 'Purpose of purchase: shopping-related',
    'category_interaction_1': 'Interaction 1: Non-essential vs. essential*Average item cost',
    'category_interaction_2': 'Interaction 2: Non-essential vs. essential*Purpose of purchase: household or wellbeing',
    'amt': 'Amount spent (USD)',
    'gender': 'Gender',
    'city_1': 'City socioeconomic status',
    'city_2': 'City social inequality level',
    'city_3': 'City Rural-Urban',
    'city_4a': 'City high-speed fixed broadband coverage',
    'city_4b': 'City high-speed mobile broadband',
    'state_x': 'State West vs East',
    'state_y': 'State North vs South',
    'city_pop': 'Population of the city',
    'job_1': 'Occupation income level',
    'job_2': 'Occupation education requirement',
    'job_3': 'Occupation career level',
    'job_4': 'Occupation cyber knowledge',
    'cardholder_age': 'Cardholder age',
    'loc_cluster_issue': 'Location Cluster Card Issue',
    'loc_cluster_transac': 'Location Cluster Transaction',
    'issue_transac_dist': 'Distance between where card was issued and transaction made',
}
target = 'is_fraud'
predictors = [column_names_dict[k] for k in [
    'trans_time', 'trans_time_grp', 'category_1', 'category_2', 'category_3',
    'category_4a', 'category_4b', 'category_4c',
    'category_interaction_1', 'category_interaction_2',
    'amt', 'gender', 'city_1', 'city_2', 'city_3', 'city_4a', 'city_4b',
    'state_x', 'state_y', 'job_1', 'job_2', 'job_3', 'job_4', 'city_pop',
    'loc_cluster_issue', 'loc_cluster_transac', 'issue_transac_dist',
    'cardholder_age']]


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


# --- data, split, resample (cells 3-20) ----------------------------------
log(f"reading fraudTrain_encoded.csv ({'seeded' if SEEDED else 'unseeded, as V21'})")
df_enc = pd.read_csv(data_file("fraudTrain_encoded.csv")).rename(columns=column_names_dict)

train_df, test_df = train_test_split(df_enc, test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE, shuffle=True)
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE,
                                      random_state=RANDOM_STATE, shuffle=True)
log("train/valid/test", len(train_df), len(valid_df), len(test_df))

ros = RandomOverSampler(sampling_strategy=0.0066667, random_state=RANDOM_STATE)
X, y = ros.fit_resample(train_df[predictors], train_df[target])
train_df = pd.DataFrame(X, columns=predictors)
train_df[target] = y

y_valid = valid_df[target].values
y_test = test_df[target].values

METRICS = ['Recall', 'Precision', 'Specificity', 'F1 Score', 'AUC', 'MCC',
           'Balanced Classification Rate']
asis = {}       # replicate V21 exactly
corrected = {}  # Phase 1


def score_asis(name, y_true, binary_preds, auc_value):
    """Metric row exactly as V21 builds it (AUC passed in as V21 computes it)."""
    recall = recall_score(y_true, binary_preds, average='binary')
    precision = precision_score(y_true, binary_preds, average='binary')
    specificity = recall_score(y_true, binary_preds, pos_label=0)
    f1 = f1_score(y_true, binary_preds)
    mcc = matthews_corrcoef(y_true, binary_preds)
    bcr = (recall + specificity) / 2
    asis[name] = [recall, precision, specificity, f1, auc_value, mcc, bcr]
    log(f"  as-is  {name}: AUC={auc_value:.6f} MCC={mcc:.6f}")


def score_corrected(name, proba, thr=0.5):
    """Phase 1: always valid_df, always AUC from probabilities, plus AUC-PR."""
    binary = (np.asarray(proba) > thr).astype(int)
    recall = recall_score(y_valid, binary, average='binary')
    precision = precision_score(y_valid, binary, average='binary')
    specificity = recall_score(y_valid, binary, pos_label=0)
    corrected[name] = [recall, precision, specificity, f1_score(y_valid, binary),
                       roc_auc_score(y_valid, proba),
                       average_precision_score(y_valid, proba),
                       matthews_corrcoef(y_valid, binary),
                       (recall + specificity) / 2]
    log(f"  FIXED  {name}: AUC={corrected[name][4]:.6f} AUC-PR={corrected[name][5]:.6f}")


t0 = time.time()

# --- Logistic Regression (cells 29-42) -----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[predictors])
X_valid = scaler.transform(valid_df[predictors])
m = LogisticRegression(max_iter=1000, class_weight='balanced')
m.fit(X_train, train_df[target])
preds = m.predict(X_valid)
proba = m.predict_proba(X_valid)[:, 1]
score_asis("Logistic Regression", y_valid, preds, roc_auc_score(y_valid, proba))
score_corrected("Logistic Regression", proba)
del m, X_train, X_valid; gc.collect()

# --- Decision Tree (cells 45-70) -----------------------------------------
clf = DecisionTreeClassifier(class_weight='balanced',
                             **({'random_state': RANDOM_STATE} if SEEDED else {}))
clf.fit(train_df[predictors], train_df[target].values)
preds = clf.predict(valid_df[predictors])
proba = clf.predict_proba(valid_df[predictors])[:, 1]
score_asis("Decision Tree", y_valid, preds, roc_auc_score(y_valid, preds))  # V21 bug: binary
score_corrected("Decision Tree", proba)
del clf; gc.collect()

# --- Random Forest (cells 72-93) -----------------------------------------
clf = RandomForestClassifier(n_jobs=NO_JOBS, random_state=RANDOM_STATE,
                             criterion=RFC_METRIC, n_estimators=NUM_ESTIMATORS,
                             verbose=False, class_weight='balanced')
clf.fit(train_df[predictors], train_df[target].values)
preds = clf.predict(valid_df[predictors])
proba = clf.predict_proba(valid_df[predictors])[:, 1]
score_asis("Random Forest", y_valid, preds, roc_auc_score(y_valid, preds))  # V21 bug
score_corrected("Random Forest", proba)
del clf; gc.collect()

# --- AdaBoost (cells 95-110) ---------------------------------------------
clf = AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME',
                         learning_rate=0.8, n_estimators=NUM_ESTIMATORS)
cw = compute_class_weight('balanced', classes=np.unique(train_df[target].values),
                          y=train_df[target].values)
cwd = dict(enumerate(cw))
sample_weight = np.array([cwd[c] for c in train_df[target].values])
clf.fit(train_df[predictors], train_df[target].values, sample_weight=sample_weight)
preds = clf.predict(valid_df[predictors])
proba = clf.predict_proba(valid_df[predictors])[:, 1]
score_asis("AdaBoost", y_valid, preds, roc_auc_score(y_valid, preds))  # V21 bug
score_corrected("AdaBoost", proba)
del clf; gc.collect()

# --- CatBoost (cells 112-131) --------------------------------------------
clf = CatBoostClassifier(iterations=500, learning_rate=0.02, depth=12,
                         eval_metric='AUC', random_seed=RANDOM_STATE,
                         bagging_temperature=0.2, od_type='Iter',
                         metric_period=VERBOSE_EVAL, od_wait=100)
clf.fit(train_df[predictors], train_df[target].values, verbose=False)
preds = clf.predict(valid_df[predictors])
proba = clf.predict_proba(valid_df[predictors])[:, 1]
score_asis("CatBoost", y_valid, preds, roc_auc_score(y_valid, preds))  # V21 bug
score_corrected("CatBoost", proba)
del clf; gc.collect()

# --- XGBoost (cells 133-150) ---------------------------------------------
# V21 scores XGBoost on test_df at threshold 0.175 (cells 141-149).
dtrain = xgb.DMatrix(train_df[predictors], train_df[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], test_df[target].values)
params = {'objective': 'binary:logistic', 'eta': 0.039, 'max_depth': 2,
          'subsample': 0.8, 'colsample_bytree': 0.9, 'eval_metric': 'auc',
          **({'seed': RANDOM_STATE} if SEEDED else {'random_state': RANDOM_STATE})}
xgb_model = xgb.train(params, dtrain, MAX_ROUNDS, [(dtrain, 'train'), (dvalid, 'valid')],
                      early_stopping_rounds=EARLY_STOP, maximize=True,
                      verbose_eval=False)
preds_test = xgb_model.predict(dtest)
binary_test = np.array([1 if p > 0.175 else 0 for p in preds_test])
score_asis("XGBoost", y_test, binary_test,
           roc_auc_score(y_test, binary_test))  # V21 bug: binary, on test, at 0.175
score_corrected("XGBoost", xgb_model.predict(dvalid))
del dtrain, dvalid, dtest, xgb_model; gc.collect()

# --- LightGBM (cells 153-168) --------------------------------------------
# V21 scores LightGBM on test_df at threshold 0.5 (cells 157-167).
safe_names = [f"f{i}" for i in range(len(predictors))]  # lgb rejects ':' '*' in names
lparams = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
           'learning_rate': 0.05, 'num_leaves': 7, 'max_depth': 4,
           'min_child_samples': 100, 'max_bin': 100, 'subsample': 0.9,
           'subsample_freq': 1, 'colsample_bytree': 0.7, 'min_child_weight': 0,
           'min_split_gain': 0, 'nthread': 8, 'verbose': 0,
           'scale_pos_weight': 150}
ltrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                     feature_name=safe_names)
lvalid = lgb.Dataset(valid_df[predictors].values, label=valid_df[target].values,
                     feature_name=safe_names)
lgb_model = lgb.train(lparams, ltrain, valid_sets=[ltrain, lvalid],
                      valid_names=['train', 'valid'], num_boost_round=MAX_ROUNDS,
                      callbacks=[lgb.early_stopping(stopping_rounds=2 * EARLY_STOP,
                                                    verbose=False)])
preds_test = lgb_model.predict(test_df[predictors].values)
binary_test = np.array([1 if p > 0.5 else 0 for p in preds_test])
score_asis("LightGDM", y_test, binary_test,
           roc_auc_score(y_test, preds_test))  # real AUC, but on test_df
score_corrected("LightGBM", lgb_model.predict(valid_df[predictors].values))
del ltrain, lvalid, lgb_model; gc.collect()

# --- k-NN (cells 171-183) ------------------------------------------------
clf = KNeighborsClassifier(1, n_jobs=-1)
clf.fit(train_df[predictors], train_df[target].values)
proba = clf.predict_proba(valid_df[predictors])[:, 1]
binary = np.array([1 if p > 0.5 else 0 for p in proba])
score_asis("K-NN", y_valid, binary, roc_auc_score(y_valid, proba))  # k=1: proba is 0/1
score_corrected("K-NN", proba)
del clf; gc.collect()

log(f"total runtime {(time.time()-t0)/60:.1f} min")

# --- write and diff ------------------------------------------------------
order = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
         "CatBoost", "XGBoost", "LightGDM", "K-NN"]
rep = pd.DataFrame({'metric\\model': METRICS})
for k in order:
    rep[k] = asis[k]
rep.to_csv(OUT / "repro_asis.csv", index=False)

corder = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
          "CatBoost", "XGBoost", "LightGBM", "K-NN"]
cor = pd.DataFrame({'metric\\model': ['Recall', 'Precision', 'Specificity', 'F1 Score',
                                      'AUC', 'AUC-PR', 'MCC',
                                      'Balanced Classification Rate']})
for k in corder:
    cor[k] = corrected[k]
cor.to_csv(OUT / "phase1_corrected.csv", index=False)

v04 = pd.read_csv(PATH / "result_csv" / "models_metric_result_V04.csv")
a = rep.set_index('metric\\model').astype(float)
b = v04.set_index(v04.columns[0]).astype(float)
b.columns = [c.strip() for c in b.columns]
diff = (a - b[a.columns]).abs()
diff.to_csv(OUT / "repro_diff.csv")
print("\nas-is vs models_metric_result_V04.csv, max abs diff per model:")
print(diff.max().round(6).to_string())
