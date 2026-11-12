"""
Fit-scope audit: which learned preprocessing steps saw the validation and test periods, and does it
matter? (Revision_Plan_V06.md Stage 2D; revision_plan_12_oct.md §2D.)

category_cost_check.py settled 'Average item cost' (category_3): its cut-offs are a full-sample
statistic, and the training split alone gives the identical map. This does the same for the other
steps whose parameters come from the transaction data rather than from external reference tables.
None of them uses a label.

  1. Location grid (loc_cluster_issue, loc_cluster_transac; encoding cell 34). The 32 x 32 grid
     spans the min and max latitude and longitude of all rows, cardholder and merchant locations
     together. Recomputed from the training split's bounds -> how many rows change cell?
  2. Occupation income quartile (job_1; cell 27). The cut-offs are the quartiles of the matched
     incomes of all 494 job titles in the file, unweighted. Recomputed over the titles that occur
     in the training split -> how many titles, and how many transactions, change quartile?
  3. Raw-baseline ordinal codes (city_ordinal, job_ordinal; raw_baseline_features.py). The codes
     are alphabetical ranks over every level in the file. No statistic is estimated, but a level
     first seen after training gets a code the models never trained on -> how many such levels
     and rows?

Steps whose cut-offs come from external data are out of scope here and listed in
Data_Provenance_V01.md: county income quartiles, Gini, broadband and 5G quintiles, urban influence
(lookup), education and career level (fixed bins), cyber-knowledge quintiles (over all O*NET
occupations). The client features fit nothing: each is a function of the card's strictly earlier
transactions.

  python fit_scope_check.py   ->  output/fit_scope_check_V01.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
TRAIN_FRACTION = 0.64
GRID_SIZE = 32

raw = pd.read_csv(PATH / "fraudTrain.csv",
                  usecols=["lat", "long", "merch_lat", "merch_long", "city", "state", "job", "is_fraud"])
enc = pd.read_csv(PATH / "fraudTrain_encoded.csv",
                  usecols=["trans_time", "loc_cluster_issue", "loc_cluster_transac", "job_1"])
assert len(raw) == len(enc)
is_train = (enc["trans_time"] <= enc["trans_time"].quantile(TRAIN_FRACTION)).values
rows = []

# --- 1. location grid ---------------------------------------------------------------
def grid(lat, lon, b):
    """Encoding cell 34, vectorised: int() truncates toward zero, as the cell's int() does."""
    r = np.trunc((lat - b["lat0"]) / (b["lat1"] - b["lat0"]) * (GRID_SIZE - 1)).astype(int)
    c = np.trunc((lon - b["lon0"]) / (b["lon1"] - b["lon0"]) * (GRID_SIZE - 1)).astype(int)
    return r * GRID_SIZE + c


def bounds(d):
    return dict(lat0=d[["lat", "merch_lat"]].min().min(), lat1=d[["lat", "merch_lat"]].max().max(),
                lon0=d[["long", "merch_long"]].min().min(), lon1=d[["long", "merch_long"]].max().max())


b_all, b_train = bounds(raw), bounds(raw[is_train])
for col, (la, lo) in {"loc_cluster_issue": ("lat", "long"), "loc_cluster_transac": ("merch_lat", "merch_long")}.items():
    full = grid(raw[la].values, raw[lo].values, b_all)
    assert (full == enc[col].values).all(), f"cell 34 not reproduced for {col}"
    tr = grid(raw[la].values, raw[lo].values, b_train)
    changed = full != tr
    rows.append(dict(step=f"{col}: grid bounds from training split instead of all rows",
                     units_changed=int(changed.sum()), units_total=len(raw),
                     rows_changed_after_training=int((changed & ~is_train).sum()),
                     note=f"bounds all {b_all} vs train {b_train}"))

# --- 2. occupation income quartile -----------------------------------------------------
inc = pd.read_csv(PATH / "encoding_table/tmp_parts/job_and_income.csv")
jobs = pd.read_csv(PATH / "encoding_table/csv/job_and_encoding.csv")
assert len(inc) == len(jobs) == raw["job"].nunique() and list(jobs["job"]) == sorted(raw["job"].unique())
inc["job"] = jobs["job"].values
assert (inc["Estimated_income_quartile"].values == jobs["job_1"].values).all()
q_all = pd.qcut(inc["Estimated_income"], q=4, labels=False)
assert (q_all.values == jobs["job_1"].values).all(), "cell 27 not reproduced"
seen = set(raw.loc[is_train, "job"])
cuts = np.percentile(inc.loc[inc["job"].isin(seen), "Estimated_income"], [25, 50, 75])
# pd.qcut's bins are right-closed: a value equal to a cut-off falls in the lower quartile
q_train = np.searchsorted(cuts, inc["Estimated_income"].values, side="left")
job_changed = set(inc.loc[q_train != q_all.values, "job"])
rows.append(dict(step="job_1: income quartiles over the training split's job titles instead of all titles",
                 units_changed=len(job_changed), units_total=len(inc),
                 rows_changed_after_training=int((raw["job"].isin(job_changed) & ~is_train).sum()),
                 note=f"{len(seen)} of {len(inc)} titles occur in training; "
                      f"{int(raw['job'].isin(job_changed).sum())} transactions in all carry a changed title"))

# --- 3. raw-baseline ordinal codes ------------------------------------------------------
for c, key in (("city", ["state", "city"]), ("job", ["job"])):
    lv_train = set(map(tuple, raw.loc[is_train, key].values))
    lv = list(map(tuple, raw[key].values))
    unseen = np.array([v not in lv_train for v in lv])
    rows.append(dict(step=f"{c}_ordinal: levels first seen after the training split",
                     units_changed=len({v for v, u in zip(lv, unseen) if u}),
                     units_total=len(set(lv)), rows_changed_after_training=int(unseen.sum()),
                     note=f"{int(raw.loc[unseen, 'is_fraud'].sum())} of those rows are fraudulent"))

res = pd.DataFrame(rows)
res.to_csv(OUT / "fit_scope_check_V01.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 120)
print(res.to_string(index=False))
