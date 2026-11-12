"""Cross-check of the review of CCF_MM_V03 (2026-10-27), items 3 and 10 (Revision_Plan_V08.md, X1-X2).

1. Age gap over three card populations. fraud_only_cards.py labels its restricted population
   "mixed only" but selects df[~fraud_only] -- all 908 cards except the 75 fraud-only ones, which
   still include the 221 genuine-only cards. Its 0.150 pp is therefore the 908-card figure; the
   687 mixed cards give 0.190 pp. Reports the script's band-mean gap (age = 2020 - birth year,
   age 50 in the lower group), pooled gaps for both boundary conventions, and a cardholder
   bootstrap interval (300 draws) for the pooled >50 vs <=50 gap.
2. Table A12's denominator: genuine counts by encoded cardholder_age (<50 / >=50), with and
   without the 731 genuine transactions above $2,000.

Usage: python age_population_check.py fraudTrain.csv     (diagnostic; prints only)
"""
import numpy as np
import pandas as pd
import sys
df = pd.read_csv(sys.argv[1], usecols=["trans_date_trans_time","cc_num","dob","amt","category","is_fraud"])
df["age"] = 2020 - pd.to_datetime(df["dob"]).dt.year
r = df.groupby("cc_num")["is_fraud"].mean()
kind = pd.Series(np.where(r==1,"fraud_only",np.where(r==0,"genuine_only","mixed")), index=r.index)
df["kind"] = df.cc_num.map(kind)
print(kind.value_counts().to_dict())
BINS=[0,30,40,50,60,70,200]
def gaps(d):
    b = d.groupby(pd.cut(d.age,BINS),observed=False)["is_fraud"].mean().values
    band = np.nanmean(b[3:]) - np.nanmean(b[:3])       # script's definition (age 50 in 'under')
    pooled50 = d.loc[d.age>50,"is_fraud"].mean() - d.loc[d.age<=50,"is_fraud"].mean()
    pooledge50 = d.loc[d.age>=50,"is_fraud"].mean() - d.loc[d.age<50,"is_fraud"].mean()
    return band, pooled50, pooledge50
pops = {"all 983": df, "excl fraud-only (908)": df[df.kind!="fraud_only"], "mixed 687": df[df.kind=="mixed"]}
rng = np.random.default_rng(0)
for name,d in pops.items():
    cards = d.cc_num.unique(); g = gaps(d)
    over = d.age>50
    cc = d.groupby("cc_num").agg(age=("age","first"))
    print(f"\n{name}: cards over50={int((cc.age>50).sum())} <=50={int((cc.age<=50).sum())}; "
          f"tx over={int(over.sum())} under={int((~over).sum())}; fraud over={int(d.is_fraud[over].sum())} under={int(d.is_fraud[~over].sum())}")
    print(f"  gap band-mean {g[0]*100:.3f}pp | pooled >50 vs <=50 {g[1]*100:.3f}pp | pooled >=50 vs <50 {g[2]*100:.3f}pp")
    idx = {c: ix for c, ix in d.groupby("cc_num").indices.items()}
    arr = d.reset_index(drop=True)
    bs=[]
    for _ in range(300):
        pick = rng.choice(cards, len(cards))
        ii = np.concatenate([idx[c] for c in pick])
        bs.append(gaps(arr.iloc[ii])[1])
    print(f"  card-bootstrap 95% (pooled >50): [{np.percentile(bs,2.5)*100:.3f}, {np.percentile(bs,97.5)*100:.3f}]pp")
# Table A12 denominator
print("\ngenuine >$2000:", int(((df.amt>2000)&(df.is_fraud==0)).sum()), " fraud >$2000:", int(((df.amt>2000)&(df.is_fraud==1)).sum()))
h = pd.to_datetime(df.trans_date_trans_time).dt.hour
print("genuine total", int((df.is_fraud==0).sum()), "genuine <=2000", int(((df.is_fraud==0)&(df.amt<=2000)).sum()))
for lab,m in [("<50",df.age<50),(">=50",df.age>=50),("<=50",df.age<=50),(">50",df.age>50)]:
    print(lab, "genuine all", int(((df.is_fraud==0)&m).sum()), "genuine <=2000", int(((df.is_fraud==0)&m&(df.amt<=2000)).sum()), "fraud", int((df.is_fraud[m]).sum()))
# Table A12 as printed: encoded cardholder_age, <50 vs >=50, by trans_time_grp
from pathlib import Path
e = pd.read_csv(Path(sys.argv[1]).with_name("fraudTrain_encoded.csv"),
                usecols=["cardholder_age", "trans_time_grp", "is_fraud"])
e["amt"] = df["amt"].values                      # the two files are row-aligned
for lab, m in [("<50", e.cardholder_age < 50), (">=50", e.cardholder_age >= 50)]:
    g = e[m & (e.is_fraud == 0)]
    print(f"encoded age {lab}: genuine {len(g)}, genuine <=$2000 {int((g.amt <= 2000).sum())}, "
          f"fraud {int(e.is_fraud[m].sum())}; genuine <=$2000 by time group "
          f"{g[g.amt <= 2000].trans_time_grp.value_counts().sort_index().to_dict()}")
