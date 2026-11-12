"""
Can this dataset test generalisation to NEW cardholders? (J3.4, Revision_Phase6_Plan_V01.md §8)

The assessment notes that a chronological split tests future transactions from largely the
same cardholders, and tests neither new cardholders nor fraudster adaptation. Whether the
first of those can be tested here is a counting question: fraudTrain.csv has 983 cards over
18 months, and if essentially all of them transact in the first 64% then the test window
contains no new cardholders to score.

Counts, on the chronological 64/16/20 split:
  - cards in train / validation / test, and how many test cards are unseen in train;
  - transactions and frauds on unseen cards in the test window.

If there are enough frauds on unseen cards to score (>= 30, say), report AUC-PR on that
subset in Phase 6; if not, Section 5.3 says the dataset's fixed card population does not
permit the test. Either way the counts are what the text needs.

Usage: python new_cardholder_check.py
Writes output/new_cardholder_check_V01.csv
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
MIN_FRAUD_TO_SCORE = 30

df = pd.read_csv(HERE.parent / "fraudTrain.csv",
                 usecols=["trans_date_trans_time", "cc_num", "is_fraud"])
df["t"] = pd.to_datetime(df["trans_date_trans_time"])
c1, c2 = df["t"].quantile(0.64), df["t"].quantile(0.80)
parts = {"train": df[df["t"] <= c1],
         "valid": df[(df["t"] > c1) & (df["t"] <= c2)],
         "test": df[df["t"] > c2]}

seen = set(parts["train"]["cc_num"].unique())
rows = []
for name, p in parts.items():
    cards = set(p["cc_num"].unique())
    unseen = cards - seen
    m = p["cc_num"].isin(unseen)
    rows.append(dict(split=name, rows=len(p), fraud=int(p.is_fraud.sum()),
                     cards=len(cards), cards_unseen_in_train=len(unseen),
                     rows_on_unseen_cards=int(m.sum()),
                     fraud_on_unseen_cards=int(p.loc[m, "is_fraud"].sum())))
res = pd.DataFrame(rows)
res.to_csv(OUT / "new_cardholder_check_V01.csv", index=False)
print(res.to_string(index=False))

n_cards, n_fraud = res.loc[res.split == "test", ["cards_unseen_in_train", "fraud_on_unseen_cards"]].iloc[0]
print(f"\ntotal cards in the dataset: {df.cc_num.nunique()}")
print(f"test-window cards unseen in training: {n_cards}; frauds on them: {n_fraud}")
print("verdict:", f"scoreable (>= {MIN_FRAUD_TO_SCORE} frauds)" if n_fraud >= MIN_FRAUD_TO_SCORE
      else f"NOT scoreable (< {MIN_FRAUD_TO_SCORE} frauds) -- Section 5.3 reports the count instead")
