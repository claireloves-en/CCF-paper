"""
Appendix Tables A5-A9 and A13 of the manuscript (from CCF_MM_V11), recomputed from fraudTrain.csv.

Until now these tables had no generating code: they were read off the EDA plots (Fig. A8) for the
submitted manuscript and carried forward as typed numbers. This script recomputes every count and
checks it against the published values, so the boundaries the tables use are on record:

  * age is the encoded `cardholder_age` (age at the transaction, the models' variable);
  * Table A7 splits at 25 or under / over 25; Tables A8-A9 at 50 or under / over 50
    (both checked against the strict "< 25" / "< 50" split, which does not reproduce them);
  * Table A6's middle age group is 25 to 50 inclusive;
  * amount edges are not handled uniformly by the published tables; the counts are reproduced only
    with $17 and $250 in the band above (one transaction sits on each) and $24.84 and $200 in the
    band below (74 gas-and-transport and 3 point-of-sale-grocery transactions sit on them). $600
    and $2,000 carry no transaction. LOWER marks the edges that belong to the band above.

Time band: `trans_time_grp` (0 bed time 00:00-07:59, 1 office hours 08:00-15:59, 2 evening).
Writes result_csv/appendix_tables_a5_a13_V01.csv (long format: table, row, column, value) and
exits non-zero if any published count is not reproduced.

Usage: python appendix_tables_a5_a13.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
from data_paths import data_file  # noqa: E402

raw = pd.read_csv(PROJ / "fraudTrain.csv", usecols=["category", "amt", "is_fraud"])
enc = pd.read_csv(data_file("fraudTrain_encoded.csv"), usecols=["cardholder_age", "trans_time_grp"])
assert len(raw) == len(enc) == 1_296_675
d = raw.assign(age=enc.cardholder_age.values, grp=enc.trans_time_grp.values)
LOWER = {17, 250}
BANDS = {0: "Bed time", 1: "Office hours", 2: "Evening"}

rows = []


def add(table, row, col, value):
    rows.append({"table": table, "row": row, "column": col, "value": value})


def amount_table(table, sub, cuts, labels):
    edges = [c - 1e-9 if c in LOWER else c for c in cuts]
    band = pd.cut(sub.amt, [-1] + edges + [float("inf")], labels=labels, right=True)
    for lab in labels:
        s = sub[band == lab]
        add(table, lab, "n", len(s)); add(table, lab, "fraud", int(s.is_fraud.sum()))
    add(table, "Total", "n", len(sub)); add(table, "Total", "fraud", int(sub.is_fraud.sum()))
    return band


def age_time_table(table, sub, cut):
    for side, mask in [(f"age <= {cut}", sub.age <= cut), (f"age > {cut}", sub.age > cut)]:
        for g, name in BANDS.items():
            for f, lab in [(0, "genuine"), (1, "fraud")]:
                add(table, name, f"{side} {lab}", int(((sub.grp == g) & (sub.is_fraud == f) & mask).sum()))


# A5 gas and transport; A6 point-of-sale grocery (with the age split); A7-A9 age x time band
amount_table("A5", d[d.category == "gas_transport"], [17, 24.84], ["5.93-17", "17-24.84", ">24.84"])
pos = d[d.category == "grocery_pos"]
band = amount_table("A6", pos, [200, 250], ["<=200", "200-250", ">250"])
for lab in ["<=200", "200-250", ">250"]:
    s = pos[band == lab]
    add("A6", lab, "age 25-50", int(s.age.between(25, 50).sum()))
add("A6", "Total", "age 25-50", int(pos.age.between(25, 50).sum()))
age_time_table("A7", d[d.category == "misc_net"], 25)
age_time_table("A8", d[d.category == "shopping_net"], 50)
age_time_table("A9", d[d.category == "shopping_pos"], 50)
amount_table("A13", d, [250, 600, 2000], ["<=250", "250-600", "600-2000", ">2000"])

out = pd.DataFrame(rows)
out.to_csv(PROJ / "result_csv" / "appendix_tables_a5_a13_V01.csv", index=False)

# published values (CCF_MM_V10, identical to the submitted manuscript)
PUB = {
    ("A5", "5.93-17", "n"): 483, ("A5", "5.93-17", "fraud"): 483, ("A5", "17-24.84", "n"): 254,
    ("A5", "17-24.84", "fraud"): 135, ("A5", ">24.84", "n"): 130922, ("A5", "Total", "n"): 131659,
    ("A5", "Total", "fraud"): 618,
    ("A6", "<=200", "n"): 112002, ("A6", "<=200", "fraud"): 0, ("A6", "200-250", "n"): 9712,
    ("A6", "200-250", "fraud"): 4, ("A6", ">250", "n"): 1924, ("A6", ">250", "fraud"): 1739,
    ("A6", "Total", "n"): 123638, ("A6", "Total", "fraud"): 1743, ("A6", "<=200", "age 25-50"): 63917,
    ("A6", "200-250", "age 25-50"): 9624, ("A6", ">250", "age 25-50"): 883, ("A6", "Total", "age 25-50"): 74424,
    ("A7", "Bed time", "age <= 25 genuine"): 942, ("A7", "Bed time", "age <= 25 fraud"): 58,
    ("A7", "Bed time", "age > 25 genuine"): 33147, ("A7", "Bed time", "age > 25 fraud"): 137,
    ("A7", "Office hours", "age <= 25 genuine"): 2702, ("A7", "Office hours", "age <= 25 fraud"): 12,
    ("A7", "Office hours", "age > 25 genuine"): 18122, ("A7", "Office hours", "age > 25 fraud"): 45,
    ("A7", "Evening", "age <= 25 genuine"): 4556, ("A7", "Evening", "age <= 25 fraud"): 45,
    ("A7", "Evening", "age > 25 genuine"): 2903, ("A7", "Evening", "age > 25 fraud"): 618,
    ("A8", "Bed time", "age <= 50 genuine"): 7853, ("A8", "Bed time", "age <= 50 fraud"): 186,
    ("A8", "Bed time", "age > 50 genuine"): 19629, ("A8", "Bed time", "age > 50 fraud"): 0,
    ("A8", "Office hours", "age <= 50 genuine"): 22012, ("A8", "Office hours", "age <= 50 fraud"): 49,
    ("A8", "Office hours", "age > 50 genuine"): 9786, ("A8", "Office hours", "age > 50 fraud"): 60,
    ("A8", "Evening", "age <= 50 genuine"): 36550, ("A8", "Evening", "age <= 50 fraud"): 711,
    ("A8", "Evening", "age > 50 genuine"): 0, ("A8", "Evening", "age > 50 fraud"): 707,
    ("A9", "Bed time", "age <= 50 genuine"): 6487, ("A9", "Bed time", "age <= 50 fraud"): 134,
    ("A9", "Bed time", "age > 50 genuine"): 23434, ("A9", "Bed time", "age > 50 fraud"): 0,
    ("A9", "Office hours", "age <= 50 genuine"): 27043, ("A9", "Office hours", "age <= 50 fraud"): 23,
    ("A9", "Office hours", "age > 50 genuine"): 11838, ("A9", "Office hours", "age > 50 fraud"): 41,
    ("A9", "Evening", "age <= 50 genuine"): 47027, ("A9", "Evening", "age <= 50 fraud"): 277,
    ("A9", "Evening", "age > 50 genuine"): 0, ("A9", "Evening", "age > 50 fraud"): 368,
    ("A13", "<=250", "n"): 1258216, ("A13", "<=250", "fraud"): 1896, ("A13", "250-600", "n"): 27386,
    ("A13", "250-600", "fraud"): 2056, ("A13", "600-2000", "n"): 10342, ("A13", "600-2000", "fraud"): 3554,
    ("A13", ">2000", "n"): 731, ("A13", ">2000", "fraud"): 0, ("A13", "Total", "n"): 1296675,
    ("A13", "Total", "fraud"): 7506,
}
got = out.set_index(["table", "row", "column"]).value
bad = [(k, v, got.get(k)) for k, v in PUB.items() if got.get(k) != v]
for k, v, g in bad:
    print("MISMATCH", k, "published", v, "computed", g)
print(f"{len(PUB) - len(bad)}/{len(PUB)} published values reproduced")
sys.exit(1 if bad else 0)
