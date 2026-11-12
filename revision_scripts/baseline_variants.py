"""
Two diagnostics that reframe the J1 ablation. (Revision_Phase6_Results_V01.md §8.)

Run after the first baseline tuning came out absurd: the conventional baseline scored 0.47-0.52
validation AUC-PR against the encoded set's 0.87. The encoded features are deterministic
functions of the same raw columns, so a flexible learner on the raw columns should roughly match
them; a gap that size had to be the baseline's fault.

A. HOW SHOULD city (894 levels) AND job (494) BE HANDLED?
   raw_baseline_features.py target-encoded them, out of fold within the training split. With a
   0.58% base rate and ~930 training rows per city, an out-of-fold estimate is almost pure noise
   at 894 distinct values -- maximally attractive to a greedy splitter, which fits it and does
   not generalise. Every other conventional treatment is compared here.

B. WHICH ENCODED FEATURES ACTUALLY CARRY SIGNAL?
   Drop each group of the 27 encoded features in turn and see what validation AUC-PR does.

Both use one fixed LightGBM setting (lr 0.05, 63 leaves, <=400 rounds, early stopping on
validation AUC-PR) rather than per-model tuning: the point is to compare feature sets on equal
terms cheaply. The numbers are indicative of direction and size, not the final tuned figures,
which come from d8_tuning.py + ablation_run.py.

Usage: python baseline_variants.py
Writes output/baseline_variants_V01.csv
"""
import sys
from pathlib import Path
from data_paths import data_file  # encoded files: current, or archived via CCF_ENCODED_VERSION

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
SEED, FOLDS = 1200, 5
PARAMS = {"objective": "binary", "metric": "average_precision", "verbose": -1,
          "seed": SEED, "learning_rate": 0.05, "num_leaves": 63}

enc = pd.read_csv(data_file("fraudTrain_encoded.csv"))
rawb = pd.read_csv(PATH / "fraudTrain_raw_baseline.csv")
raw = pd.read_csv(PATH / "fraudTrain.csv", usecols=["city", "job"])
t = enc["trans_time"]
tr = (t <= t.quantile(0.64)).values
va = ((t > t.quantile(0.64)) & (t <= t.quantile(0.80))).values
y = enc["is_fraud"].values
prior = y[tr].mean()
rows = []


def score(X, label, group):
    names = [f"f{i}" for i in range(X.shape[1])]
    m = lgb.train(PARAMS, lgb.Dataset(X[tr], label=y[tr], feature_name=names),
                  valid_sets=[lgb.Dataset(X[va], label=y[va], feature_name=names)],
                  num_boost_round=400,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    ap = average_precision_score(y[va], m.predict(X[va]))
    print(f"{label:46s} {X.shape[1]:4d} cols  AUC-PR {ap:.4f}", flush=True)
    rows.append(dict(group=group, variant=label, n_cols=X.shape[1], valid_aucpr=ap))
    return ap


def smoothed_oof(col, m):
    """The encoding raw_baseline_features.py uses, with the smoothing weight as a parameter."""
    rng = np.random.default_rng(SEED)
    out = np.full(len(col), prior)
    fold = np.full(len(col), -1); fold[tr] = rng.integers(FOLDS, size=tr.sum())

    def mapping(mask):
        s = pd.DataFrame({"k": col[mask], "y": y[mask]}).groupby("k")["y"].agg(["sum", "size"])
        return (s["sum"] + m * prior) / (s["size"] + m)

    out[~tr] = pd.Series(col[~tr]).map(mapping(tr)).fillna(prior).values
    for f in range(FOLDS):
        a = tr & (fold == f)
        out[a] = pd.Series(col[a]).map(mapping(tr & (fold != f))).fillna(prior).values
    return out


# ---------------------------------------------------------- A: city / job handling
print("A. conventional handling of city (894 levels) and job (494)\n")
# city and job are added below in each treatment, so the base drops whichever encoding the raw
# baseline file carries: target encoding when this was first run, ordinal codes since it was rebuilt
base = [c for c in rawb.columns if c not in ("is_fraud", "trans_time", "city_target_enc", "job_target_enc",
                                             "city_ordinal", "job_ordinal")]
assert len(base) == 75, len(base)
V = rawb[base]
score(V.values, "city/job dropped", "A")
for m in (20, 200, 2000):
    extra = np.column_stack([smoothed_oof(raw["city"].values, m), smoothed_oof(raw["job"].values, m)])
    score(np.column_stack([V.values, extra]), f"target encoding, smoothing {m}", "A")
freq = np.column_stack([raw[c].map(raw[c][tr].value_counts()).fillna(0).values for c in ("city", "job")])
score(np.column_stack([V.values, freq]), "frequency (count) encoding", "A")
ordn = np.column_stack([pd.Categorical(raw[c]).codes for c in ("city", "job")])
score(np.column_stack([V.values, ordn]), "ordinal / label encoding", "A")
oh = np.column_stack([pd.get_dummies(raw[c], dtype=np.int8).values for c in ("city", "job")])
score(np.column_stack([V.values, oh]), "one-hot city + job", "A")

# ---------------------------------------------------------- B: encoded feature groups
print("\nB. which of the 27 encoded features carry signal\n")
allp = [c for c in enc.columns if c not in ("is_fraud", "trans_time")]
CITY = [c for c in allp if c.startswith("city_") and c != "city_pop"]
JOB = [c for c in allp if c.startswith("job_")]
STATE = ["state_x", "state_y"]
LOC = ["loc_cluster_issue", "loc_cluster_transac", "issue_transac_dist"]
CAT = [c for c in allp if c.startswith("category_")]
score(enc[allp].values, "encoded, all 27", "B")
for drop, lab in [(CITY, "city_1..city_4b (5)"), (JOB, "job_1..job_4 (4)"),
                  (STATE, "state_x/state_y (2)"), (LOC, "location clusters + distance (3)"),
                  (CITY + JOB, "city + job (9)"),
                  (CITY + JOB + STATE + LOC, "all geographic + occupational (14)")]:
    score(enc[[c for c in allp if c not in drop]].values, f"minus {lab}", "B")
score(enc[CAT + ["amt", "trans_time_grp", "cardholder_age", "gender", "city_pop"]].values,
      "only category_* + amt + time + age + pop", "B")

# Phase 7: --out baseline_variants_V02.csv on the corrected encoded file (V01 is the pre-fix record)
_out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "baseline_variants_V01.csv"
pd.DataFrame(rows).to_csv(OUT / _out, index=False)
print(f"\nwrote {OUT / _out}")
