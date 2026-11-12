"""
Feature sets for the Phase 7 comparisons (Revision_Phase7_Plan_V01.md §4; revision_plan_12_oct.md §3A).

Built from the corrected fraudTrain_encoded.csv (E), fraudTrain_raw_baseline.csv (R) and the
seven client-history features, all row-aligned with fraudTrain.csv (checked below).

Hybrid configurations -- does the encoding add anything when the raw information is kept?
  hybrid          R + E                       (amt, city_pop, gender are shared: kept once)
  hybrid_client   R + E + H

Targeted ablations of E -- one feature family changed at a time, everything else as E:
  abl_cat_onehot       the 8 encoded category features replaced by one-hot `category` (14)
  abl_plus_onehot      E + one-hot `category` (the original field retained alongside)
  abl_hour             trans_time_grp (3 bands) replaced by hour_of_day (0-23)
  abl_plus_dow         E + day_of_week (information E does not carry at all)
  abl_no_interactions  E without the two interaction terms
  abl_no_external      E without the externally enriched features: city_1..city_4b (USDA, ACS,
                       ERS, FCC) and job_1..job_4 (BLS, O*NET). This is also the configuration
                       that excludes every source dated after the transaction window.
  abl_no_geo_occ       E without all 14 geographic and occupational features (adds state_x/y,
                       the two location clusters and issue_transac_dist to abl_no_external)
CatBoost with native categorical handling (sensitivity, Phase 7 plan §4.2):
  catnative        R with the one-hot and ordinal blocks replaced by the original strings
                   category, state, city ("city|state", since city names repeat across states)
                   and job, for CatBoost's cat_features
  catnative_client the same + H
Phase 8 (Revision_Phase8_Plan_V01.md, C1) -- the semantic attributes kept, the detail restored:
  eplus            E with trans_time_grp replaced by hour_of_day + day_of_week, plus one-hot
                   `category` (14): 42 predictors
  eplus_client     the same + H
Reduced pair -- amount, category and time only, raw against encoded:
  red_raw              amt + one-hot category + hour_of_day
  red_enc              amt + category_1..interaction_2 + trans_time_grp

Each file keeps trans_time (the split key, not a predictor) and is_fraud. Where a family is
replaced, the replacement takes the replaced columns' position, so the order of the other
predictors is unchanged.

  python ablation_features.py [--only name1,name2]   ->  ../ablation_data/fraudTrain_<name>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import client_features

PATH = Path(__file__).resolve().parent.parent
DEST = PATH / "ablation_data"; DEST.mkdir(exist_ok=True)
T, Y = "trans_time", "is_fraud"

E = pd.read_csv(PATH / "fraudTrain_encoded.csv")
R = pd.read_csv(PATH / "fraudTrain_raw_baseline.csv")
H = pd.read_csv(PATH / "fraudTrain_encoded_client.csv", usecols=client_features.FEATURES)[client_features.FEATURES]

# row alignment: all three come from fraudTrain.csv in its row order
assert len(E) == len(R) == len(H)
assert (E[T].values == R[T].values).all() and (E[Y].values == R[Y].values).all()
assert np.allclose(E["amt"].values, R["amt"].values)
assert (E["gender"].values == R["gender"].values).all(), "gender coded differently"
assert np.allclose(E["city_pop"].values, R["city_pop"].values)

CAT_ENC = ["category_1", "category_2", "category_3", "category_4a", "category_4b", "category_4c",
           "category_interaction_1", "category_interaction_2"]
CAT_ONEHOT = [c for c in R.columns if c.startswith("category_")]
assert len(CAT_ONEHOT) == 14 and not set(CAT_ONEHOT) & set(E.columns)
EXTERNAL = ["city_1", "city_2", "city_3", "city_4a", "city_4b", "job_1", "job_2", "job_3", "job_4"]
GEO_OCC = EXTERNAL + ["state_x", "state_y", "loc_cluster_issue", "loc_cluster_transac", "issue_transac_dist"]
assert len(GEO_OCC) == 14
E_PRED = [c for c in E.columns if c not in (T, Y)]
R_PRED = [c for c in R.columns if c not in (T, Y)]
assert len(E_PRED) == 27 and len(R_PRED) == 77


def replace(cols, old, new):
    """cols with the block `old` replaced by `new`, at the position of old's first member."""
    i = cols.index(old[0])
    rest = [c for c in cols if c not in old]
    return rest[:i] + new + rest[i:]


def frame(spec):
    """spec: list of (source frame, column) in predictor order."""
    out = pd.DataFrame({T: E[T].values})
    for src, c in spec:
        assert c not in out.columns, f"duplicate column {c}"
        out[c] = src[c].values
    out[Y] = E[Y].values
    return out


def from_E(cols):
    src = {**{c: E for c in E_PRED}, **{c: R for c in R_PRED}}
    return frame([(src[c], c) for c in cols])


sets = {
    "hybrid": frame([(R, c) for c in R_PRED] + [(E, c) for c in E_PRED if c not in R_PRED]),
    "abl_cat_onehot": from_E(replace(E_PRED, CAT_ENC, CAT_ONEHOT)),
    "abl_plus_onehot": from_E(E_PRED + CAT_ONEHOT),
    "abl_hour": from_E(replace(E_PRED, ["trans_time_grp"], ["hour_of_day"])),
    "abl_plus_dow": from_E(E_PRED + ["day_of_week"]),
    "abl_no_interactions": from_E([c for c in E_PRED if c not in ("category_interaction_1", "category_interaction_2")]),
    "abl_no_external": from_E([c for c in E_PRED if c not in EXTERNAL]),
    "abl_no_geo_occ": from_E([c for c in E_PRED if c not in GEO_OCC]),
    "red_raw": frame([(R, "amt")] + [(R, c) for c in CAT_ONEHOT] + [(R, "hour_of_day")]),
    "red_enc": frame([(E, "amt")] + [(E, c) for c in CAT_ENC] + [(E, "trans_time_grp")]),
    "eplus": from_E(replace(E_PRED, ["trans_time_grp"], ["hour_of_day", "day_of_week"]) + CAT_ONEHOT),
}
# catnative: R's numeric columns in R's order, the strings where R's encoded blocks were
src = pd.read_csv(PATH / "fraudTrain.csv", usecols=["category", "state", "city", "job"])
NUMERIC = [c for c in R_PRED if not c.startswith(("category_", "state_")) and not c.endswith("_ordinal")]
cn = pd.DataFrame({T: E[T].values})
for c in NUMERIC:
    cn[c] = R[c].values
cn["category"] = src["category"].values
cn["state"] = src["state"].values
cn["city"] = (src["city"] + "|" + src["state"]).values
cn["job"] = src["job"].values
cn[Y] = E[Y].values
sets["catnative"] = cn
cc = cn.copy()
for c in client_features.FEATURES:
    cc.insert(cc.columns.get_loc(Y), c, H[c].values)
sets["catnative_client"] = cc

hc = sets["hybrid"].copy()
for c in client_features.FEATURES:
    hc.insert(hc.columns.get_loc(Y), c, H[c].values)
sets["hybrid_client"] = hc

ec = sets["eplus"].copy()
for c in client_features.FEATURES:
    ec.insert(ec.columns.get_loc(Y), c, H[c].values)
sets["eplus_client"] = ec

ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None
rows = []
for name, df in sets.items():
    if ONLY is not None and name not in ONLY:
        continue
    pred = [c for c in df.columns if c not in (T, Y)]
    assert df[pred].isna().sum().sum() == 0 and len(df) == len(E)
    df.to_csv(DEST / f"fraudTrain_{name}.csv", index=False)
    added = [c for c in pred if c not in E_PRED]
    dropped = [c for c in E_PRED if c not in pred]
    rows.append({"set": name, "predictors": len(pred),
                 "relative_to_E_added": " ".join(added) if len(added) < 20 else f"{len(added)} columns",
                 "relative_to_E_dropped": " ".join(dropped)})
    print(f"{name:20s} {len(pred):3d} predictors")
_log = Path(__file__).resolve().parent / "output" / "ablation_feature_sets_V01.csv"
if ONLY is not None and _log.exists():           # partial run: update those rows only
    _old = pd.read_csv(_log)
    rows = _old[~_old["set"].isin([r["set"] for r in rows])].to_dict("records") + rows
pd.DataFrame(rows).to_csv(_log, index=False)
print(f"wrote {sum(ONLY is None or n in ONLY for n in sets)} files to {DEST}")
