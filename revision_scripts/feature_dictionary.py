"""
The authoritative feature dictionary (Revision_Plan_V06.md Stage 2C; revision_plan_12_oct.md §2C).

One row per predictor of the encoded set (E, 27), the client-history set (H, 7) and the raw
baseline (R, 77, with its one-hot blocks grouped). Each row runs from the original Sparkov field
through the construct to the transformation and resulting feature, with its source, fitting
period, handling of missing or unseen values, and whether it measures the cardholder or is a proxy.

The text is written here once; the script checks it against the data rather than trusting it:
  - the E, H and R feature lists must equal the data files' predictor columns exactly;
  - each feature's observed values (range, number of distinct values) are read from the data.
Provenance details are in Data_Provenance_V01.md; the category values in
category_encoding_audit_V01.csv (encoding_audit.py).

  python feature_dictionary.py   ->  output/feature_dictionary_V02.csv  (V01: before the construct renaming of CCF_MM_V04)
"""
from pathlib import Path

import pandas as pd

import client_features

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)

TRAIN = "training split (to 2019-12-10)"
FULL = "all rows, no labels (training split gives the identical map)"
EXT = "external reference data (not fitted on transactions)"
NONE = "nothing fitted"
CAT_UNSEEN = "every category occurs in training; a new category has no code (would fail loudly)"
COUNTY = ("city -> county via ZIP database; 5 cities and 8-9 counties per table filled by hand "
          "(Data_Provenance_V01.md §2); no defaults")
JOB = ("nearest O*NET/OEWS title by sentence-embedding cosine similarity, no threshold: every "
       "title gets a match")

# feature: (original field, construct, transformation, source, fitting period, missing/unseen, measured or proxy)
E = {
    "trans_time_grp": ("trans_date_trans_time", "routine activity: time of day",
                       "hour 0-7 -> 0 (bed time), 8-15 -> 1 (office hours), 16-23 -> 2 (evening)",
                       "transaction timestamp", NONE, "none possible", "measured (coarsened)"),
    "category_1": ("category", "purchase channel", "0 offline or mixed, 1 online only",
                   "hand-authored map, Categories_encoding_V04.xlsx", NONE, CAT_UNSEEN, "attribute of the category"),
    "category_2": ("category", "category necessity (coded; category-level, not the purchase)", "0 non-essential, 1 essential",
                   "hand-authored map", NONE, CAT_UNSEEN, "attribute of the category"),
    "category_3": ("category", "category typical amount (quartile of mean transaction amount; not item price)", "quartile (0-3) of the category's mean amount among the 14 categories",
                   "transaction amounts", FULL, CAT_UNSEEN, "attribute of the category"),
    "category_4a": ("category", "category purpose (coded): household/wellbeing", "1 if household/wellbeing", "hand-authored map", NONE, CAT_UNSEEN, "attribute of the category"),
    "category_4b": ("category", "category purpose (coded): entertainment/lifestyle", "1 if entertainment/lifestyle", "hand-authored map", NONE, CAT_UNSEEN, "attribute of the category"),
    "category_4c": ("category", "category purpose (coded): shopping/miscellaneous", "1 if shopping/miscellaneous", "hand-authored map", NONE, CAT_UNSEEN, "attribute of the category"),
    "category_interaction_1": ("category", "necessity x typical amount", "category_2 x category_3: 0 = non-essential OR essential in the lowest amount quartile; 1-3 = essential in quartile 2-4",
                               "derived", NONE, CAT_UNSEEN, "attribute of the category"),
    "category_interaction_2": ("category", "necessity x household/wellbeing", "category_2 x category_4a: 1 = essential household/wellbeing item",
                               "derived", NONE, CAT_UNSEEN, "attribute of the category"),
    "amt": ("amt", "transaction amount", "as is (USD)", "transaction", NONE, "none", "measured"),
    "gender": ("gender", "cardholder gender", "F 0, M 1", "transaction", NONE, "none", "measured (as simulated)"),
    "city_1": ("city, state", "county socioeconomic status", "quartile of county median household income",
               "USDA ERS, 2022 (retrospective)", EXT, COUNTY, "proxy: county of residence"),
    "city_2": ("city, state", "county income inequality", "quintile of county Gini index", "ACS 5-year 2016-20 B19083", EXT, COUNTY, "proxy: county of residence"),
    "city_3": ("city, state", "county type", "0 noncore (UIC 6, 8, 9), 1 micropolitan (UIC 2, 5, 7, and noncore code 3), 2 metropolitan (UIC 1, 4)", "USDA ERS UIC 2024 (retrospective)", EXT, COUNTY, "proxy: county of residence"),
    "city_4a": ("city, state", "digital connectivity: fixed", "quintile of mean high-speed fixed broadband coverage", "FCC BDC Dec 2022 (retrospective)", EXT, COUNTY, "proxy: county of residence"),
    "city_4b": ("city, state", "digital connectivity: mobile", "quintile of 5G area coverage", "FCC BDC Dec 2022 (retrospective)", EXT, COUNTY, "proxy: county of residence"),
    "state_x": ("state", "west-east position", "column (0-8) of a hand-built 9 x 10 state grid", "hand-authored map", NONE, "every state has a cell", "attribute of the state"),
    "state_y": ("state", "north-south position", "row (0-9) of the state grid", "hand-authored map", NONE, "every state has a cell", "attribute of the state"),
    "city_pop": ("city_pop", "city population", "as is", "transaction (simulated)", NONE, "none", "measured (as simulated)"),
    "job_1": ("job", "occupational income", "quartile of matched OEWS mean annual wage over the 494 titles", "BLS OEWS May 2020", FULL, JOB, "proxy: nearest occupation"),
    "job_2": ("job", "education requirement", "modal O*NET required-education category binned: below high school / high school+ / bachelor's+", "O*NET 2024 (retrospective)", EXT, JOB, "proxy: nearest occupation"),
    "job_3": ("job", "occupational preparation (O*NET Job Zone; not seniority)", "O*NET Job Zone - 1 (0-4)", "O*NET 2024 (retrospective)", EXT, JOB, "proxy: nearest occupation"),
    "job_4": ("job", "occupational computer knowledge (not personal cybersecurity knowledge)", "quintile of O*NET 'Computers and Electronics' knowledge level", "O*NET 2024 (retrospective)", EXT, JOB, "proxy: nearest occupation"),
    "cardholder_age": ("dob, trans_date_trans_time", "cardholder age", "whole years at the transaction", "transaction", NONE, "none", "measured (as simulated)"),
    "loc_cluster_issue": ("lat, long", "cardholder home location", "cell of a 32 x 32 grid over the data's lat/long range",
                          "transaction", "grid bounds from all rows (training bounds move 0.43% of cells)", "a location outside the bounds gets an out-of-grid id", "measured (coarsened)"),
    "loc_cluster_transac": ("merch_lat, merch_long", "merchant location", "cell of the same grid",
                            "transaction", "as above (0.13% of cells move)", "as above", "measured (coarsened)"),
    "issue_transac_dist": ("lat, long, merch_lat, merch_long", "distance from home to merchant", "geodesic km", "transaction", NONE, "none", "measured"),
}
H = {
    "card_amt_zscore": ("cc_num, amt", "amount unusual for this card", "z-score against the card's prior amounts; 0 until 5 priors; clipped +/-10"),
    "card_hours_since_prev": ("cc_num, unix_time", "transaction velocity", "hours since the card's previous transaction, capped at 720"),
    "card_txn_1h": ("cc_num, unix_time", "burst activity", "card's transactions in the preceding hour"),
    "card_txn_24h": ("cc_num, unix_time", "burst activity", "card's transactions in the preceding 24 hours"),
    "card_txn_7d": ("cc_num, unix_time", "recent activity", "card's transactions in the preceding 7 days"),
    "card_dist_from_usual_km": ("cc_num, merch_lat, merch_long, lat, long", "unusual location", "km from the centroid of the card's previous merchants (home for the first)"),
    "card_new_category": ("cc_num, category", "novel purchase type", "1 if the card's first purchase in this category"),
}
R = {
    "amt": ("amt", "transaction amount", "as is"),
    "city_pop": ("city_pop", "city population", "as is"),
    "lat": ("lat", "cardholder home latitude", "as is"),
    "long": ("long", "cardholder home longitude", "as is"),
    "merch_lat": ("merch_lat", "merchant latitude", "as is"),
    "merch_long": ("merch_long", "merchant longitude", "as is"),
    "gender": ("gender", "cardholder gender", "M 1, F 0"),
    "age": ("dob, trans_date_trans_time", "cardholder age", "years at the transaction, fractional"),
    "hour_of_day": ("trans_date_trans_time", "time of day", "0-23"),
    "day_of_week": ("trans_date_trans_time", "day of week", "0 Monday - 6 Sunday"),
    "category_* (14)": ("category", "transaction category", "one-hot"),
    "state_* (51)": ("state", "state", "one-hot"),
    "city_ordinal": ("city", "city identity", "alphabetical rank over all levels (no statistic fitted); 22 levels first seen after training"),
    "job_ordinal": ("job", "occupation identity", "alphabetical rank; 4 levels first seen after training"),
}


def observed(s):
    return f"{s.min():g} to {s.max():g}; {s.nunique()} distinct"


enc = pd.read_csv(PATH / "fraudTrain_encoded.csv")
cli = pd.read_csv(PATH / "fraudTrain_encoded_client.csv", usecols=client_features.FEATURES)
raw = pd.read_csv(PATH / "fraudTrain_raw_baseline.csv")
e_pred = [c for c in enc.columns if c not in ("trans_time", "is_fraud")]
r_pred = [c for c in raw.columns if c not in ("trans_time", "is_fraud")]
assert list(E) == e_pred, "dictionary E list differs from fraudTrain_encoded.csv"
assert list(H) == client_features.FEATURES
r_groups = {"category_* (14)": [c for c in r_pred if c.startswith("category_")],
            "state_* (51)": [c for c in r_pred if c.startswith("state_")]}
assert len(r_groups["category_* (14)"]) == 14 and len(r_groups["state_* (51)"]) == 51
r_single = [c for c in r_pred if c not in sum(r_groups.values(), [])]
assert set(r_single) | set(r_groups) == set(R) and len(r_pred) == 77

rows = []
for f, (orig, construct, trans, source, fit, unseen, kind) in E.items():
    rows.append(dict(set="E (encoded)", feature=f, original_field=orig, construct=construct,
                     transformation=trans, source=source, fitting_period=fit,
                     missing_or_unseen=unseen, measured_or_proxy=kind, observed=observed(enc[f])))
for f, (orig, construct, trans) in H.items():
    rows.append(dict(set="H (client history)", feature=f, original_field=orig, construct=construct,
                     transformation=trans, source="the card's strictly earlier transactions; no label",
                     fitting_period=NONE + " (computed online over the whole series)",
                     missing_or_unseen="cold-start defaults (Data_Provenance_V01.md §4)",
                     measured_or_proxy="measured (behavioural)", observed=observed(cli[f])))
for f, (orig, construct, trans) in R.items():
    cols = r_groups.get(f, [f])
    rows.append(dict(set="R (conventional)", feature=f, original_field=orig, construct=construct,
                     transformation=trans, source="transaction",
                     fitting_period=NONE, missing_or_unseen="unseen level: one-hot all zero / ordinal code -1 (fraudTest)",
                     measured_or_proxy="measured (as simulated)",
                     observed=observed(raw[cols[0]]) if len(cols) == 1 else f"{len(cols)} indicator columns"))
out = pd.DataFrame(rows)
out.to_csv(OUT / "feature_dictionary_V02.csv", index=False)   # V02: construct names of CCF_MM_V04 (labels_v04.py)
print(out.groupby("set").size().to_string())
print(f"E {len(E)} + H {len(H)} + R {len(r_pred)} predictors ({len(R)} rows); lists match the data files")
