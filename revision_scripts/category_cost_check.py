"""
Is 'Average item cost' (category_3) fitted on data the models should not see?
(J3, Revision_Plan_V05.md — the journal assessment's preprocessing-leakage point.)

Section 3.3 says the average cost of an item "was coded by quartile of average item
cost across all transactions". The map itself is hand-authored, in
encoding_table/Categories_encoding_V04.xlsx, so the code gives no clue where the
cut-offs came from. This script reconstructs them:

  1. mean `amt` per `category` over all 1,296,675 rows of fraudTrain.csv, binned at
     the quartiles of those 14 means -> compared with the workbook's category_3;
  2. the same computed on the chronological training split only (trans_time <= its 64th
     percentile, the cut SPLIT_MODE = "temporal" uses) -> compared with (1).

(1) reproducing the workbook exactly would confirm the cut-offs are data-derived and
so were fitted on the test period as well; (2) agreeing with (1) would show the
leakage changes no feature value, which is what the paper has to be able to state.

Usage: python category_cost_check.py
Writes output/category_cost_check_V01.csv and prints the two verdicts.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
TRAIN_FRACTION = 0.64          # model_V25: chronological 64/16/20

# category_3 as curated in encoding_table/Categories_encoding_V04.xlsx, keyed by the
# raw `category` value rather than by the workbook's human-readable name.
WORKBOOK = {"home": 1, "kids_pets": 1, "entertainment": 2, "food_dining": 0,
            "health_fitness": 0, "personal_care": 0, "gas_transport": 2, "travel": 3,
            "grocery_net": 0, "grocery_pos": 3, "misc_net": 3, "misc_pos": 1,
            "shopping_net": 3, "shopping_pos": 2}


def quartile_bins(means):
    """Bin each category mean at the quartiles of the 14 means (0 = cheapest quartile)."""
    cuts = means.quantile([0.25, 0.50, 0.75]).values
    return means.map(lambda v: int(np.searchsorted(cuts, v, side="right")))


# Cut on trans_time (hours since the first transaction), the column the models split on,
# not on the raw timestamp: the two quantiles differ by 48 rows and the training split has
# to be exactly the models'. No hour straddles the boundary.
df = pd.read_csv(HERE.parent / "fraudTrain.csv", usecols=["category", "amt"])
df["trans_time"] = pd.read_csv(HERE.parent / "fraudTrain_encoded.csv",
                               usecols=["trans_time"])["trans_time"].values
cut = df["trans_time"].quantile(TRAIN_FRACTION)
train = df[df["trans_time"] <= cut]

mean_all, mean_train = df.groupby("category")["amt"].mean(), train.groupby("category")["amt"].mean()
res = pd.DataFrame({"mean_amt_all": mean_all.round(4),
                    "bin_all": quartile_bins(mean_all),
                    "mean_amt_train": mean_train.round(4),
                    "bin_train": quartile_bins(mean_train),
                    "category_3_workbook": pd.Series(WORKBOOK)}).sort_values("mean_amt_all")
res.index.name = "category"
res.to_csv(OUT / "category_cost_check_V01.csv")

print(res.to_string(), "\n")
print(f"training split: {len(train):,} of {len(df):,} rows, trans_time <= {cut:g}")
print("1. cut-offs are data-derived (bin_all == workbook):",
      bool((res["bin_all"] == res["category_3_workbook"]).all()))
print("2. training data alone gives the same bins (bin_train == bin_all):",
      bool((res["bin_train"] == res["bin_all"]).all()))
