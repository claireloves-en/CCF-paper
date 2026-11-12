"""
The untuned CatBoost baseline for the catnative configurations (Revision_Phase7_Plan_V01.md §4.2).

build_tuned_table.py keeps a searched hyperparameter set only if it beats the model's untuned
baseline on the same data, which d7_sweep.py --only-no-sampling provides -- for all eight models.
On a file with string columns only CatBoost can run, so this fits d7_sweep.py's untuned CatBoost
(the same settings, with the string columns as cat_features) and writes its row in d7_sweep.py's
format.

  python catnative_baseline.py --data ablation_data/fraudTrain_catnative.csv --tag p7_catnative
  -> output/d7_sweep_rows_p7_catnative.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef

from data_paths import data_file

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
RANDOM_STATE, TARGET = 1200, "is_fraud"
DATA = sys.argv[sys.argv.index("--data") + 1]
TAG = "_" + sys.argv[sys.argv.index("--tag") + 1]

df = pd.read_csv(data_file(DATA))
predictors = [c for c in df.columns if c not in (TARGET, "trans_time")]
cat = [c for c in predictors if df[c].dtype == object]
t = df["trans_time"]
c1, c2 = t.quantile(0.64), t.quantile(0.80)
tr, va = df[t <= c1], df[(t > c1) & (t <= c2)]
print(f"{len(predictors)} predictors, cat_features {cat}; train {len(tr):,} / valid {len(va):,}")

s = time.time()
m = CatBoostClassifier(iterations=500, learning_rate=0.02, depth=12, eval_metric="AUC",
                       random_seed=RANDOM_STATE, bagging_temperature=0.2, od_type="Iter",
                       metric_period=50, od_wait=100)        # d7_sweep.py's untuned CatBoost
m.fit(tr[predictors], tr[TARGET].values, cat_features=cat or None, verbose=False)
p = m.predict_proba(va[predictors])[:, 1]
yv = va[TARGET].values
row = dict(method="No Sampling", ratio="No Sampling (1:173)", model="CatBoost",
           mcc=matthews_corrcoef(yv, (p > 0.5).astype(int)), aucpr=average_precision_score(yv, p),
           seconds=time.time() - s)
pd.DataFrame([row]).to_csv(OUT / f"d7_sweep_rows{TAG}.csv", index=False)
print(f"untuned CatBoost: validation AUC-PR {row['aucpr']:.4f} ({row['seconds']:.0f}s)")
