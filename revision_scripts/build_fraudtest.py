"""
Features for the untouched confirmation period, Found_DataSet/fraudTest.csv
(Revision_Phase7_Plan_V01.md §9, step 1).

fraudTest.csv is Sparkov's later period (2020-06-21 to 2020-12-31). It is scored once, last, by
confirm_run.py. This script only builds features. It computes no statistic of the labels; is_fraud
is carried through to the output files untouched.

Every transformation is the one the models were trained with, applied with its fraudTrain
parameters frozen:
  E   the corrected encoding (encoding_V25): category codes, the county lookup (V25's own city
      cells re-run over fraudTrain's and fraudTest's cities -- fraudTrain's values must come out
      unchanged), the state grid, the job lookup, integer age, the 32 x 32 location grid with
      fraudTrain's bounds, geodesic distance (geopy, as the encoding)
  H   the seven client features over fraudTrain + fraudTest as one series, so a fraudTest card's
      history includes its fraudTrain transactions
  R   the raw baseline: fraudTrain's one-hot columns; city and job ordinal codes from fraudTrain's
      levels, unseen levels -> -1
plus the hybrid (R + E + H) and catnative (+ H) sets that confirm_run.py scores.

THE CHECK THAT MATTERS: the same builders, applied to fraudTrain.csv, must reproduce
fraudTrain_encoded.csv, fraudTrain_raw_baseline.csv and the client features exactly, before
anything is written for fraudTest.

New levels in fraudTest, handled explicitly:
  - 12 cities: resolved through the same ZIP-database -> county pipeline. One county needs a
    manual entry: Craig, AK lies in Prince of Wales-Hyder Census Area, which the 2024 Urban
    Influence file lists under a misspelling ("Price of Wales Hyder Census Area", FIPS 02198,
    UIC 9 -> rural). The entry is added to the scratch copy only; no fraudTrain city is affected.
  - 3 job titles, matched by hand (NEW_JOBS below), recorded in Data_Provenance_V01.md:
      'Engineer, water'               = existing 'Water engineer' (same title, words reordered)
      'Operational investment banker' = existing 'Investment banker, operational' (same)
      'Software engineer'             from OEWS "Software Developers and Software Quality Assurance
                                      Analysts and Testers" ($114,270 -> quartile 3) and O*NET
                                      "Software Developers" (Job Zone 4 -> 3); O*NET rates no
                                      education or knowledge for Software Developers, so those come
                                      from the nearest rated occupation, "Software Quality
                                      Assurance Analysts and Testers" (bachelor's -> 2; computers
                                      and electronics 5.47 -> quintile 4)

  python build_fraudtest.py   ->  ../confirm_data/fraudTest_{encoded,encoded_client,raw_baseline,
                                   raw_baseline_client,hybrid_client,catnative_client}.csv
"""
import ast
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from geopy.distance import geodesic

import client_features

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
DEST = PATH / "confirm_data"; DEST.mkdir(exist_ok=True)
NB = PATH / "MLProj01_fraudTrain_encoding_V25.ipynb"
TEST_FILE = PATH.parent.parent / "Found_DataSet" / "fraudTest.csv"
T, Y = "trans_time", "is_fraud"
GRID_SIZE = 32
NEW_JOBS = {"Engineer, water": "Water engineer",
            "Operational investment banker": "Investment banker, operational",
            "Software engineer": {"job_1": 3, "job_2": 2, "job_3": 3, "job_4": 4}}

cells = ["".join(c["source"]) for c in json.load(open(NB))["cells"]]


def literal(cell, name):
    body = re.search(rf"{name} = (\{{.*?\}})", cells[cell], re.S).group(1)
    return ast.literal_eval(re.sub(r"#[^\n]*", "", body))


CATEGORY_CODE = literal(13, "category_code")
STATE_CODE = literal(25, "state_code")
CAT = ["category_1", "category_2", "category_3", "category_4a", "category_4b", "category_4c",
       "category_interaction_1", "category_interaction_2"]
CITY = ["city_1", "city_2", "city_3", "city_4a", "city_4b"]


def city_lookup(cities):
    """V25's city cells (16-22) over the given (city, state) pairs, in a scratch copy."""
    tmp = Path(tempfile.mkdtemp(prefix="city_fraudtest_"))
    shutil.copytree(PATH / "encoding_table", tmp / "encoding_table",
                    ignore=shutil.ignore_patterns("_old", "*.xlsx"))
    ns = {"pd": pd, "np": np, "display": lambda *a, **k: None, "HTML": lambda x: x, "df_enc": cities}
    cwd = os.getcwd(); os.chdir(tmp)
    try:
        for i in range(16, 22):
            exec(compile(cells[i], f"V25_cell{i}", "exec"), ns)
        # the UIC file's misspelt entry for FIPS 02198 (see the docstring); cell 22 reads this table
        uic_path = Path("encoding_table/ref_data/city_city_type.csv")
        uic = pd.read_csv(uic_path)
        if not ((uic.State == "AK") & (uic.County_Name == "Prince of Wales-Hyder Census Area")).any():
            uic = pd.concat([uic, pd.DataFrame([{"State": "AK", "County_Name": "Prince of Wales-Hyder Census Area",
                                                 "Attribute": "UIC_2024", "Value": 9, "Encoded_urban_type": 0}])],
                            ignore_index=True)
            uic.to_csv(uic_path, index=False)
        exec(compile(cells[22], "V25_cell22", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns["df_city"]


def encode(raw, t0, bounds, df_city, jobs):
    """The 27 encoded predictors (+ trans_time, is_fraud), in fraudTrain_encoded.csv's order."""
    ts = pd.to_datetime(raw["trans_date_trans_time"])
    out = pd.DataFrame(index=raw.index)
    out[T] = np.floor((ts - t0).dt.total_seconds() / 3600)
    h = ts.dt.hour
    out["trans_time_grp"] = np.select([h < 8, h < 16], [0, 1], 2)
    digits = raw["category"].map(CATEGORY_CODE)
    assert digits.notna().all(), "unknown category"
    for i, c in enumerate(CAT):
        out[c] = digits.str[i].astype(int)
    out["amt"] = raw["amt"].values
    out["gender"] = (raw["gender"] == "M").astype(int)          # LabelEncoder: F 0, M 1
    m = raw[["state", "city"]].merge(df_city, how="left", left_on=["state", "city"],
                                     right_on=["State", "City"], validate="many_to_one")
    assert len(m) == len(raw) and m[CITY].notna().all().all(), "unresolved city"
    for c in CITY:
        out[c] = m[c].astype(int).values
    sc = raw["state"].map(STATE_CODE)
    assert sc.notna().all(), "unknown state"
    out["state_x"] = sc.str[0].astype(int); out["state_y"] = sc.str[1].astype(int)
    out["city_pop"] = raw["city_pop"].values
    jm = raw[["job"]].merge(jobs, on="job", how="left", validate="many_to_one")
    assert len(jm) == len(raw) and jm[["job_1", "job_2", "job_3", "job_4"]].notna().all().all(), "unmatched job"
    for c in ("job_1", "job_2", "job_3", "job_4"):
        out[c] = jm[c].astype(int).values
    # encoding cell 33: whole years, truncated
    days = (ts - pd.to_datetime(raw["dob"])).astype("int64")
    out["cardholder_age"] = (days / (10**6 * 60 * 60 * 24 * 365.25 * 1000)).astype("int64")
    # encoding cell 34: the 32 x 32 grid with fraudTrain's bounds, int() truncating toward zero
    def grid(lat, lon):
        r = np.trunc((lat - bounds["lat0"]) / (bounds["lat1"] - bounds["lat0"]) * (GRID_SIZE - 1)).astype(int)
        c = np.trunc((lon - bounds["lon0"]) / (bounds["lon1"] - bounds["lon0"]) * (GRID_SIZE - 1)).astype(int)
        return r * GRID_SIZE + c
    out["loc_cluster_issue"] = grid(raw["lat"].values, raw["long"].values)
    out["loc_cluster_transac"] = grid(raw["merch_lat"].values, raw["merch_long"].values)
    out["issue_transac_dist"] = [geodesic((a, b), (c, d)).kilometers for a, b, c, d in
                                 zip(raw["lat"], raw["long"], raw["merch_lat"], raw["merch_long"])]
    out[Y] = raw[Y].values
    return out


def raw_baseline(raw, t0, onehot_cols, levels):
    """raw_baseline_features.py's 77 predictors with fraudTrain's columns and codes."""
    ts = pd.to_datetime(raw["trans_date_trans_time"])
    out = pd.DataFrame(index=raw.index)
    out[T] = ((ts - t0).dt.total_seconds() // 3600).astype(int)
    for c in ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]:
        out[c] = raw[c].values
    out["gender"] = (raw["gender"].str.upper() == "M").astype(int)
    out["age"] = ((ts - pd.to_datetime(raw["dob"])).dt.days / 365.25).astype(float)
    out["hour_of_day"] = ts.dt.hour.astype(int)
    out["day_of_week"] = ts.dt.dayofweek.astype(int)
    for c in ("category", "state"):
        d = pd.get_dummies(raw[c], prefix=c, dtype=int)
        d.columns = [s.replace(" ", "_") for s in d.columns]
        assert set(d.columns) <= set(onehot_cols[c]), f"new {c} level"
        out = pd.concat([out, d.reindex(columns=onehot_cols[c], fill_value=0)], axis=1)
    for c in ("city", "job"):
        out[f"{c}_ordinal"] = raw[c].map(levels[c]).fillna(-1).astype(int)   # unseen -> -1
    out[Y] = raw[Y].values
    return out


def with_client(df, cf):
    out = df.copy()
    for c in client_features.FEATURES:
        out.insert(out.columns.get_loc(Y), c, cf[c].values)
    return out


# ------------------------------------------------------------------------------------------
train = pd.read_csv(PATH / "fraudTrain.csv")
test = pd.read_csv(TEST_FILE)
t0 = pd.to_datetime(train["trans_date_trans_time"]).min()
bounds = dict(lat0=train[["lat", "merch_lat"]].min().min(), lat1=train[["lat", "merch_lat"]].max().max(),
              lon0=train[["long", "merch_long"]].min().min(), lon1=train[["long", "merch_long"]].max().max())

# the county lookup over both periods' cities; fraudTrain's values must not move
pairs = pd.concat([train[["city", "state"]], test[["city", "state"]]]).drop_duplicates().reset_index(drop=True)
df_city = city_lookup(pairs)
committed = pd.read_csv(PATH / "encoding_table/csv/city_and_encoding.csv")
chk = committed.merge(df_city, on=["State", "City"], suffixes=("_c", "_n"), validate="one_to_one")
assert len(chk) == len(committed) and all((chk[f"{c}_c"] == chk[f"{c}_n"]).all() for c in CITY), \
    "fraudTrain cities' values changed"
print(f"county lookup: {len(df_city)} cities ({len(df_city) - len(committed)} new), fraudTrain's unchanged")

jobs = pd.read_csv(PATH / "encoding_table/csv/job_and_encoding.csv")
add = []
for title, src in NEW_JOBS.items():
    vals = jobs.loc[jobs.job == src, ["job_1", "job_2", "job_3", "job_4"]].iloc[0].to_dict() if isinstance(src, str) else src
    add.append({"job": title, **vals})
jobs = pd.concat([jobs, pd.DataFrame(add)], ignore_index=True)

# --- 1. the builders reproduce fraudTrain's files ------------------------------------------
enc_train = encode(train, t0, bounds, df_city, jobs)
want = pd.read_csv(PATH / "fraudTrain_encoded.csv")
assert list(enc_train.columns) == list(want.columns), (list(enc_train.columns), list(want.columns))
for c in want.columns:
    tol = 1e-9 if c == "issue_transac_dist" else 0
    assert np.abs(enc_train[c].values - want[c].values).max() <= tol, f"E: {c} not reproduced"
print("E builder reproduces fraudTrain_encoded.csv (27 predictors; distance to 1e-9 km)")

onehot_cols = {c: [k for k in pd.read_csv(PATH / "fraudTrain_raw_baseline.csv", nrows=1).columns if k.startswith(c + "_")]
               for c in ("category", "state")}
levels = {c: {v: i for i, v in enumerate(sorted(train[c].unique()))} for c in ("city", "job")}
r_train = raw_baseline(train, t0, onehot_cols, levels)
want = pd.read_csv(PATH / "fraudTrain_raw_baseline.csv")
assert list(r_train.columns) == list(want.columns)
for c in want.columns:
    assert np.allclose(r_train[c].values, want[c].values, rtol=0, atol=1e-12), f"R: {c} not reproduced"
print("R builder reproduces fraudTrain_raw_baseline.csv (77 predictors)")

both = pd.concat([train[client_features.NEEDED], test[client_features.NEEDED]], ignore_index=True)
cf = client_features.compute(both)
want = pd.read_csv(PATH / "fraudTrain_encoded_client.csv", usecols=client_features.FEATURES)
for c in client_features.FEATURES:
    assert np.abs(cf[c].values[:len(train)] - want[c].values).max() <= 1e-9, f"H: {c} changed for fraudTrain"
print("H over the joined series leaves fraudTrain's client features unchanged (causal)")
cf_test = cf.iloc[len(train):].reset_index(drop=True)

# --- 2. fraudTest ---------------------------------------------------------------------------
E = encode(test, t0, bounds, df_city, jobs)
R = raw_baseline(test, t0, onehot_cols, levels)
EH, RH = with_client(E, cf_test), with_client(R, cf_test)
E_pred = [c for c in E.columns if c not in (T, Y)]
R_pred = [c for c in R.columns if c not in (T, Y)]
hybrid = pd.concat([R[[T] + R_pred], E[[c for c in E_pred if c not in R_pred]], R[[Y]]], axis=1)
HC = with_client(hybrid, cf_test)
NUMERIC = [c for c in R_pred if not c.startswith(("category_", "state_")) and not c.endswith("_ordinal")]
cn = pd.concat([R[[T] + NUMERIC]], axis=1)
cn["category"] = test["category"].values; cn["state"] = test["state"].values
cn["city"] = (test["city"] + "|" + test["state"]).values; cn["job"] = test["job"].values
cn[Y] = test[Y].values
CNH = with_client(cn, cf_test)

ref = {"encoded": PATH / "fraudTrain_encoded.csv", "encoded_client": PATH / "fraudTrain_encoded_client.csv",
       "raw_baseline": PATH / "fraudTrain_raw_baseline.csv", "raw_baseline_client": PATH / "fraudTrain_raw_baseline_client.csv",
       "hybrid_client": PATH / "ablation_data/fraudTrain_hybrid_client.csv",
       "catnative_client": PATH / "ablation_data/fraudTrain_catnative_client.csv"}
for name, df in {"encoded": E, "encoded_client": EH, "raw_baseline": R, "raw_baseline_client": RH,
                 "hybrid_client": HC, "catnative_client": CNH}.items():
    cols = list(pd.read_csv(ref[name], nrows=0).columns)
    assert list(df.columns) == cols, f"{name}: columns differ from its fraudTrain counterpart"
    df.to_csv(DEST / f"fraudTest_{name}.csv", index=False)
    print(f"wrote fraudTest_{name}.csv {df.shape}")

out_of_grid = int(((E["loc_cluster_issue"] < 0) | (E["loc_cluster_issue"] >= GRID_SIZE**2) |
                   (E["loc_cluster_transac"] < 0) | (E["loc_cluster_transac"] >= GRID_SIZE**2)).sum())
print(f"fraudTest: {len(test):,} rows; unseen city codes {int((R['city_ordinal'] < 0).sum())} rows, "
      f"unseen job codes {int((R['job_ordinal'] < 0).sum())} rows; locations outside fraudTrain's grid {out_of_grid}")
