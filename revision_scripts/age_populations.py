"""
Cardholder age and fraud over three card populations, and Table A12 without its filter
(Revision_Plan_V08.md, B1 and B5; review of CCF_MM_V03, items 3 and 10).

Why. fraud_only_cards.py compares "all cards" with a column it calls "mixed only", but that column
selects every card except the 75 fraud-only ones: 908 cards, of which 221 carry no fraud at all.
The paper copied the label. This script reports the three populations separately:

  all cards                     983   the original descriptive association
  all except fraud-only cards   908   isolates removing the separate synthetic population
  mixed cards only              687   both genuine and fraudulent transactions; selected on the outcome

Age is the encoded `cardholder_age` (the model's feature, also used by Table A12), split at
"under 50" / "50 and over". The band rows use the same variable in ten-year bands. The gap is
the pooled fraud rate of 50-and-over minus that of under-50, in percentage points, with a
cardholder bootstrap (1,000 draws: cards resampled with replacement, all their transactions kept).

Table A12 as published excluded the 731 genuine transactions above $2,000 (no fraud is above
$2,000). It is recomputed here on every transaction.

Usage: python age_populations.py
Writes result_csv/age_populations_V01.csv, result_csv/age_time_table_V02.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE.parent, HERE.parent / "result_csv"
N_BOOT, SEED = 1000, 1200
BANDS = [0, 30, 40, 50, 60, 70, 200]
BAND_LABEL = ["under 30", "30-39", "40-49", "50-59", "60-69", "70 and over"]

raw = pd.read_csv(DATA / "fraudTrain.csv", usecols=["cc_num", "amt", "is_fraud"])
enc = pd.read_csv(DATA / "fraudTrain_encoded.csv", usecols=["cardholder_age", "trans_time_grp", "is_fraud"])
assert np.array_equal(raw.is_fraud.values, enc.is_fraud.values)       # row-aligned files
df = raw.assign(age=enc.cardholder_age.values, grp=enc.trans_time_grp.values)
df["over"] = df.age >= 50
df["band"] = pd.cut(df.age, BANDS, right=False, labels=BAND_LABEL)

rate = df.groupby("cc_num").is_fraud.mean()
kind = pd.Series(np.where(rate == 1, "fraud-only", np.where(rate == 0, "genuine-only", "mixed")), index=rate.index)
df["kind"] = df.cc_num.map(kind)
POPS = {"all cards": df,
        "all except fraud-only cards": df[df.kind != "fraud-only"],
        "mixed cards only": df[df.kind == "mixed"]}


def gap(over, fraud):
    return 100 * (fraud[over].mean() - fraud[~over].mean())


rng = np.random.default_rng(SEED)
rows = []
for name, d in POPS.items():
    # cardholder bootstrap: per-card sums, resampled (a card whose holder turns 50 within the data
    # contributes to both groups, as its transactions do)
    per = d.groupby(["cc_num", "over"]).is_fraud.agg(["sum", "count"]).unstack(fill_value=0)
    s_o, n_o = per[("sum", True)].values, per[("count", True)].values
    s_u, n_u = per[("sum", False)].values, per[("count", False)].values
    draws = []
    for _ in range(N_BOOT):
        c = rng.multinomial(len(per), np.full(len(per), 1 / len(per)))
        draws.append(100 * ((c @ s_o) / (c @ n_o) - (c @ s_u) / (c @ n_u)))
    for grp_name, m in [("50 and over", d.over), ("under 50", ~d.over)]:
        rows.append(dict(population=name, cards_in_population=d.cc_num.nunique(), group=grp_name,
                         cards=int(d.loc[m].cc_num.nunique()), transactions=int(m.sum()),
                         frauds=int(d.is_fraud[m].sum()), fraud_rate_pct=100 * d.is_fraud[m].mean()))
    for b in BAND_LABEL:
        m = d.band == b
        rows.append(dict(population=name, cards_in_population=d.cc_num.nunique(), group=f"band {b}",
                         cards=int(d.loc[m].cc_num.nunique()), transactions=int(m.sum()),
                         frauds=int(d.is_fraud[m].sum()), fraud_rate_pct=100 * d.is_fraud[m].mean()))
    rows.append(dict(population=name, cards_in_population=d.cc_num.nunique(),
                     group="gap: 50 and over minus under 50 (pp)", fraud_rate_pct=gap(d.over, d.is_fraud),
                     ci_lo=np.percentile(draws, 2.5), ci_hi=np.percentile(draws, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "age_populations_V01.csv", index=False)
print(out.round(4).to_string(index=False))
crossers = int((df.groupby("cc_num").over.nunique() > 1).sum())
print(f"\ncards whose holder crosses 50 within the data: {crossers} (their transactions fall on both sides)")

# Table A12 on every transaction
GRP = {0: "Bed time", 1: "Office hours", 2: "Evening"}
t = []
for g, lab in GRP.items():
    r = {"time group": lab}
    for over, olab in [(False, "under 50"), (True, "50 and over")]:
        for fr, flab in [(0, "genuine"), (1, "fraudulent")]:
            m = (df.over == over) & (df.is_fraud == fr)
            n = int((m & (df.grp == g)).sum())
            r[f"{olab} {flab} n"] = n
            r[f"{olab} {flab} %"] = 100 * n / int(m.sum())
    t.append(r)
a12 = pd.DataFrame(t)
a12.to_csv(RESULTS / "age_time_table_V02.csv", index=False)
print("\n" + a12.round(2).to_string(index=False))
print("genuine totals:", int(((~df.over) & (df.is_fraud == 0)).sum()), int((df.over & (df.is_fraud == 0)).sum()))
