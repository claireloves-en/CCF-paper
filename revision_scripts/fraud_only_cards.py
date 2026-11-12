"""
The fraud-only cardholders: a Sparkov artefact that affects what the paper can claim.
(J2 / J3.4, Revision_Phase6_Plan_V01.md §8. Found while counting new cardholders.)

Counting cards for the J3.4 new-cardholder check turned up something else: every
transaction on a card that does not appear in the training window is fraudulent. Following
that up:

  75 of the 983 cards (7.6%) carry NOTHING BUT fraud across the whole dataset -- 741
  transactions, 9.9% of all 7,506 frauds. Their 75 cardholders (first/last/dob/street)
  appear on no other card, so these are not compromised cards belonging to customers who
  also transact genuinely; they are a separate synthetic population that exists only as
  fraud. The realistic compromised-card case is the other 687 mixed cards, which carry the
  remaining 90.1% of frauds.

Three consequences, each measured below:

  1. J3.4 is degenerate, not infeasible. The test window contains 19 cards unseen in
     training and all 174 of their transactions are fraudulent, so a "new cardholder"
     subset has no negatives and AUC-PR is undefined on it. Section 5.3 should say that
     the dataset cannot test new-cardholder generalisation because every new card is a
     fraud-only card -- a sharper statement than "there are no new cards".
  2. The client-history features (Section 3.3.1) mean something different on these cards.
     On a fraud-only card, "deviation from the card's own behaviour" is deviation from
     fraudulent behaviour. The features remain causal and never touch is_fraud, but the
     part of their contribution that comes from these 741 transactions does not transfer
     to real data, where a compromised card has genuine history.
  3. The cardholder-age finding is partly an artefact. The 75 fraud-only cardholders are
     older than the rest (mean 59.6 vs 49.2; 77.3% over 50 vs 44.3%), so they inflate the
     association between age and fraud that Sections 4.3-4.4 report.

Usage: python fraud_only_cards.py
Writes output/fraud_only_cards_V01.csv (per-group summary) and
       output/fraud_only_age_V01.csv (fraud rate by age band, all cards vs mixed only).
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
AGE_BINS = [0, 30, 40, 50, 60, 70, 200]
REF_YEAR = 2020          # the dataset ends 2020-06-21; ages are approximate by design

df = pd.read_csv(HERE.parent / "fraudTrain.csv",
                 usecols=["trans_date_trans_time", "cc_num", "first", "last", "street",
                          "dob", "is_fraud"])
df["t"] = pd.to_datetime(df["trans_date_trans_time"])
df["age"] = REF_YEAR - pd.to_datetime(df["dob"]).dt.year
df["person"] = df["first"] + "|" + df["last"] + "|" + df["dob"] + "|" + df["street"]

rate = df.groupby("cc_num")["is_fraud"].mean()
fraud_only = set(rate[rate == 1].index)
df["fraud_only"] = df["cc_num"].isin(fraud_only)

# --- are these the same people as the ordinary cardholders? -------------------
p_fo = set(df.loc[df.fraud_only, "person"].unique())
p_ot = set(df.loc[~df.fraud_only, "person"].unique())
print(f"{len(fraud_only)} fraud-only cards of {df.cc_num.nunique()}; "
      f"{(rate == 0).sum()} cards with no fraud; {((rate > 0) & (rate < 1)).sum()} mixed")
print(f"persons: {len(p_fo)} on fraud-only cards, {len(p_ot)} on the rest, "
      f"{len(p_fo & p_ot)} shared  <- 0 means they are a separate population")

# --- how much of the fraud sits on them, overall and in the test window -------
c2 = df["t"].quantile(0.80)
test = df[df["t"] > c2]
rows = [dict(scope="whole dataset", rows=len(df), fraud=int(df.is_fraud.sum()),
             fraud_on_fraud_only=int(df.loc[df.fraud_only, "is_fraud"].sum())),
        dict(scope="test window", rows=len(test), fraud=int(test.is_fraud.sum()),
             fraud_on_fraud_only=int(test.loc[test.fraud_only, "is_fraud"].sum()))]
summary = pd.DataFrame(rows)
summary["share_of_fraud"] = (summary.fraud_on_fraud_only / summary.fraud).round(4)
summary.to_csv(OUT / "fraud_only_cards_V01.csv", index=False)
print("\n" + summary.to_string(index=False))

# --- the age skew, and what it does to the paper's age finding ----------------
card = df.groupby("cc_num").agg(age=("age", "first"), fraud_only=("fraud_only", "first"))
print("\ncardholder age by group")
print(card.groupby("fraud_only")["age"].agg(["count", "mean", "median"]).round(1).to_string())
print("share over 50:", card.groupby("fraud_only")["age"].apply(lambda a: round((a > 50).mean(), 3)).to_dict())

band = lambda d: d.groupby(pd.cut(d.age, AGE_BINS), observed=True)["is_fraud"].agg(["mean", "sum"])
all_b, mixed_b = band(df), band(df[~df.fraud_only])
age = pd.DataFrame({"rate_all_cards": all_b["mean"], "fraud_all_cards": all_b["sum"],
                    "rate_mixed_only": mixed_b["mean"], "fraud_mixed_only": mixed_b["sum"]})
age["rate_diff"] = (age.rate_all_cards - age.rate_mixed_only)
age.index.name = "age_band"
age.to_csv(OUT / "fraud_only_age_V01.csv")
print("\nfraud rate by cardholder age -- all cards vs mixed cards only")
print(age.round(5).to_string())

over, under = age.index[3:], age.index[:3]
lift_all = age.loc[over, "rate_all_cards"].mean() - age.loc[under, "rate_all_cards"].mean()
lift_mix = age.loc[over, "rate_mixed_only"].mean() - age.loc[under, "rate_mixed_only"].mean()
print(f"\nover-50 minus under-50 fraud rate: all cards {lift_all:+.5f}, "
      f"mixed cards only {lift_mix:+.5f} ({lift_mix / lift_all:.0%} of it survives)")
