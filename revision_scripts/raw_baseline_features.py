"""
The conventional baseline feature set, for the J1 encoding ablation.
(Revision_Phase6_Plan_V01.md §3.)

The paper claims theory-informed, interpretable feature construction as its contribution but
has never compared against not having it. This builds the baseline a competent practitioner
would produce from the same raw Sparkov columns the encoding is derived from, so that the
comparison measures THE ENCODING and not a difference in the columns available.

  numeric, as-is        amt, city_pop, lat, long, merch_lat, merch_long
  gender                0/1
  dob                   age in years at the transaction (the quantity cardholder_age holds)
  timestamp             hour of day, day of week. NO elapsed-time feature: under a
                        chronological split it is a row index (CLAUDE.md, Known pitfalls),
                        and giving the baseline one would hand it leakage the encoded set
                        does not have. trans_time is still written, as the split key only.
  category, state       one-hot (14 + 51 columns)
  city, job             ordinal (label) encoding -- an integer code per level
  excluded              cc_num, trans_num, first, last, street, unix_time, merchant, zip

`merchant` is excluded deliberately: merchant_probe.py shows Sparkov gives every category
exactly 50 merchants and spreads fraud among them at sampling noise (observed/expected sd
= 1.011), so it is a 50-way relabelling of `category` with no signal of its own. `zip` is
excluded as a second target encoding of the same geography as `city`.

WHY ORDINAL AND NOT TARGET ENCODING. An earlier version target-encoded city and job, out of
fold within the training split. It was sound on leakage -- the mapping never saw validation or
test -- but it crippled the baseline: with a 0.58% base rate and ~930 training rows per city,
an out-of-fold estimate carries about 5 frauds, so the column is almost pure noise presented
as a continuous feature with hundreds of split points, which a greedy splitter fits and does
not generalise. baseline_variants.py measures every conventional alternative on one fixed
LightGBM setting (validation AUC-PR): dropping city/job 0.9110, frequency encoding 0.9108,
one-hot 0.9104, ordinal 0.9086, and target encoding 0.4555 / 0.5347 / 0.6894 at smoothing
20 / 200 / 2000.

Ordinal is chosen from among the four that work:
  - it is within 0.002 of the best, so the baseline is not handicapped;
  - it keeps the column count at 77, where one-hot would be 1,463 -- random forest and k-NN
    have to run on this too;
  - it uses the same raw columns the encoding is built from, where dropping city/job would not;
  - it is precisely the treatment Section 3.3 argues against (ordinal bias on nominal data).
    A baseline the paper itself calls naive cannot be dismissed as a straw man.
The seven variants belong in the appendix so the choice is auditable.

Leakage: the codes come from the sorted level inventory and use NO labels at all, so unlike
target encoding there is nothing to fit on the training split and nothing that could leak.

Usage: python raw_baseline_features.py
Writes ../fraudTrain_raw_baseline.csv and ../fraudTrain_raw_baseline_client.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

import client_features

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
TRAIN_FRACTION = 0.64
TARGET = "is_fraud"
ONEHOT = ["category", "state"]
ORDINAL = ["city", "job"]

raw = pd.read_csv(PATH / "fraudTrain.csv")
print(f"{len(raw):,} rows, {raw[TARGET].sum():,} fraud")

out = pd.DataFrame(index=raw.index)

# --- the split key (not a predictor) ----------------------------------------------
ts = pd.to_datetime(raw["trans_date_trans_time"])
out["trans_time"] = ((ts - ts.min()).dt.total_seconds() // 3600).astype(int)

# --- numeric, as-is ---------------------------------------------------------------
for c in ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]:
    out[c] = raw[c].values

# --- gender, age, cyclical time ---------------------------------------------------
out["gender"] = (raw["gender"].str.upper() == "M").astype(int)
dob = pd.to_datetime(raw["dob"])
out["age"] = ((ts - dob).dt.days / 365.25).astype(float)
out["hour_of_day"] = ts.dt.hour.astype(int)
out["day_of_week"] = ts.dt.dayofweek.astype(int)

# --- one-hot ----------------------------------------------------------------------
for c in ONEHOT:
    d = pd.get_dummies(raw[c], prefix=c, dtype=int)
    d.columns = [s.replace(" ", "_") for s in d.columns]
    out = pd.concat([out, d], axis=1)
    print(f"one-hot {c}: {d.shape[1]} columns")

# --- ordinal (label) encoding: no labels used, so nothing can leak ----------------
cut_train = out["trans_time"].quantile(TRAIN_FRACTION)
is_train = (out["trans_time"] <= cut_train).values
y = raw[TARGET].values
print(f"training split: {is_train.sum():,} rows, fraud rate {y[is_train].mean():.5f}")

for c in ORDINAL:
    codes = pd.Categorical(raw[c], categories=sorted(raw[c].dropna().unique())).codes
    assert (codes >= 0).all(), f"unencodable {c} values"
    out[f"{c}_ordinal"] = codes.astype(int)
    print(f"ordinal-encode {c}: {raw[c].nunique()} levels -> codes 0..{codes.max()}")

out[TARGET] = y

# --- checks ----------------------------------------------------------------------
predictors = [c for c in out.columns if c not in ("trans_time", TARGET)]
assert TARGET not in predictors and "trans_time" not in predictors
assert out[predictors].isna().sum().sum() == 0, "NaNs in the predictors"
assert len(out) == len(raw)
for c in ORDINAL:
    v = out[f"{c}_ordinal"]
    assert v.min() == 0 and v.max() == raw[c].nunique() - 1 and v.dtype.kind == "i"
# row alignment with the encoded file, as the client-features notebook checks
enc = pd.read_csv(PATH / "fraudTrain_encoded.csv", usecols=["amt", "is_fraud", "trans_time"])
assert len(enc) == len(out)
assert np.allclose(enc["amt"].values, out["amt"].values)
assert (enc[TARGET].values == out[TARGET].values).all()
assert (enc["trans_time"].values == out["trans_time"].values).all()
print(f"\n{len(predictors)} predictors (+ trans_time as the split key, + {TARGET})")

out.to_csv(PATH / "fraudTrain_raw_baseline.csv", index=False)
print("wrote fraudTrain_raw_baseline.csv", out.shape)

# --- the same, plus the seven client-history features ------------------------------
cf = client_features.compute(raw)
withc = out.copy()
for c in client_features.FEATURES:
    withc.insert(withc.columns.get_loc(TARGET), c, cf[c].values)
withc.to_csv(PATH / "fraudTrain_raw_baseline_client.csv", index=False)
print("wrote fraudTrain_raw_baseline_client.csv", withc.shape)

print("\nrank correlation with is_fraud, strongest 12 (a sanity read, not a test):")
r = out[predictors].corrwith(out[TARGET], method="spearman").abs().sort_values(ascending=False)
print(r.head(12).round(4).to_string())
