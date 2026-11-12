"""
Does `merchant` carry fraud signal beyond `category`?
(J1, Revision_Phase6_Plan_V01.md §1 — matched vs +merchant as the ablation baseline.)

The encoding is derived from ten raw columns and `merchant` is not one of them, so a
baseline that target-encodes `merchant` would be answering a different question from the
one J1 asks. How much different depends on whether `merchant` is informative at all once
`category` is known, which is an empirical question about Sparkov, not a judgement call.

Three measurements, all on the chronological training split (the earliest 64%), scored on
the validation split (the next 16%) where scoring is involved:

  1. nesting -- how many categories each merchant appears in;
  2. within-category dispersion -- the sd of per-merchant fraud rate inside each category,
     against the sd expected if fraud were assigned uniformly within the category
     (sqrt(p(1-p)/n) per merchant). Observed ~= expected means `merchant` is noise given
     `category`;
  3. single-feature ranking -- AUC-PR on validation of three encodings used alone:
     the category fraud rate, the merchant fraud rate, and the merchant's deviation from
     its own category's rate (the part of `merchant` that `category` cannot express).
     All three mappings are fitted on the training split only.

Usage: python merchant_probe.py
Writes output/merchant_probe_V01.csv (per-category dispersion) and prints the verdicts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)

df = pd.read_csv(HERE.parent / "fraudTrain.csv",
                 usecols=["trans_date_trans_time", "merchant", "category", "is_fraud"])
df["t"] = pd.to_datetime(df["trans_date_trans_time"])
c1, c2 = df["t"].quantile(0.64), df["t"].quantile(0.80)
tr, va = df[df["t"] <= c1], df[(df["t"] > c1) & (df["t"] <= c2)]
print(f"train {len(tr):,} (fraud {tr.is_fraud.sum():,}) / valid {len(va):,} (fraud {va.is_fraud.sum():,})")
print(f"{df.merchant.nunique()} merchants, {df.category.nunique()} categories\n")

# 1 -- nesting
per_merchant = df.groupby("merchant")["category"].nunique()
print("1. categories per merchant:", per_merchant.value_counts().to_dict())
print(f"   merchants per category: {df.groupby('category')['merchant'].nunique().describe()[['min','50%','max']].to_dict()}\n")

# 2 -- within-category dispersion of merchant fraud rate, training split
rows = []
for cat, g in tr.groupby("category"):
    p = g.is_fraud.mean()
    m = g.groupby("merchant")["is_fraud"].agg(["mean", "size"])
    expected = np.sqrt(p * (1 - p) / m["size"]).mean()   # binomial sd under uniform risk
    rows.append(dict(category=cat, fraud_rate=p, n_merchants=len(m),
                     observed_sd=m["mean"].std(ddof=1), expected_sd=expected,
                     ratio=m["mean"].std(ddof=1) / expected if expected else np.nan))
disp = pd.DataFrame(rows).sort_values("fraud_rate", ascending=False)
disp.to_csv(OUT / "merchant_probe_V01.csv", index=False)
print("2. within-category dispersion of per-merchant fraud rate (training split)")
print(disp.round(5).to_string(index=False))
print(f"   mean observed/expected ratio: {disp.ratio.mean():.3f}  "
      "(1.0 = merchant is pure noise given category)\n")

# 3 -- single-feature ranking on validation, mappings fitted on train only
prior = tr.is_fraud.mean()
cat_rate = tr.groupby("category")["is_fraud"].mean()
mer_rate = tr.groupby("merchant")["is_fraud"].mean()
mer_cat = tr.groupby("merchant")["category"].first()
mer_dev = mer_rate - mer_cat.map(cat_rate)                  # merchant minus its category

feats = {"category fraud rate": va.category.map(cat_rate).fillna(prior),
         "merchant fraud rate": va.merchant.map(mer_rate).fillna(prior),
         "merchant deviation from its category": va.merchant.map(mer_dev).fillna(0.0)}
print("3. validation AUC-PR of each encoding used alone (fraud rate "
      f"{va.is_fraud.mean():.5f} = the no-skill floor)")
for name, x in feats.items():
    print(f"   {name:38s} {average_precision_score(va.is_fraud, x):.4f}")
