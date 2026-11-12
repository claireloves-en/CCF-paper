"""
Audit of the 'Transaction category' encoding, and the corrected Appendix Table A1
(Revision_Plan_V06.md, Stage 2A; revision_plan_12_oct.md §2A).

Three sources have to agree, and this script checks that they do:
  - category_code in MLProj01_fraudTrain_encoding_V25.ipynb, cell 13 (what the data were built from);
  - encoding_table/Categories_encoding_V04.xlsx (the hand-curated design);
  - the category columns of fraudTrain_encoded.csv (what the models saw).
It then checks the two interaction terms against their definitions in the workbook
  Interaction 1 = category_2 (essential, 0/1) x category_3 (cost quartile, zero-based 0-3)
  Interaction 2 = category_2 (essential, 0/1) x category_4a (household/wellbeing, 0/1)
and writes each category's values with descriptions generated from those definitions, so the
table cannot drift from the numbers.

The point the review raised: Interaction 1 = 0 holds for every non-essential category AND for
essential categories in the lowest cost quartile (food and dining, health and fitness, personal
care, online grocery). Table A1 described all of these as "Non-essential items (regardless of
cost)", which is wrong for the essential ones. The component features keep the distinction
(category_2 says essential), so the representation is not defective; the description was.

  python encoding_audit.py   ->  output/category_encoding_audit_V01.csv
"""
import ast
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)

CAT = ["category_1", "category_2", "category_3", "category_4a", "category_4b", "category_4c",
       "category_interaction_1", "category_interaction_2"]
NAMES = {  # Sparkov label -> the name used in the paper
    "home": "home products", "kids_pets": "kids and pets", "entertainment": "entertainment",
    "food_dining": "food and dining", "health_fitness": "health and fitness",
    "personal_care": "personal care", "gas_transport": "gas and transport", "travel": "travel",
    "grocery_net": "online grocery", "grocery_pos": "point-of-sale grocery",
    "misc_net": "online miscellaneous products", "misc_pos": "point-of-sale miscellaneous products",
    "shopping_net": "online shopping products", "shopping_pos": "point-of-sale shopping products"}
CHANNEL = {0: "offline or mixed", 1: "online only"}
ESSENTIAL = {0: "non-essential", 1: "essential"}
QUARTILE = {0: "lowest quartile", 1: "second quartile", 2: "third quartile", 3: "highest quartile"}
INT1 = {0: "non-essential (any cost), or essential in the lowest cost quartile",
        1: "essential, second cost quartile", 2: "essential, third cost quartile",
        3: "essential, highest cost quartile"}
INT2 = {0: "not an essential household/wellbeing item", 1: "essential household/wellbeing item"}


def purpose(r):
    p = [n for c, n in (("category_4a", "household/wellbeing"), ("category_4b", "entertainment/lifestyle"),
                        ("category_4c", "shopping/miscellaneous")) if r[c]]
    assert len(p) <= 1, r
    return p[0] if p else "none of the three"


# --- 1. the code table the data were built from ------------------------------------
nb = json.load(open(PATH / "MLProj01_fraudTrain_encoding_V25.ipynb"))
src = "".join(nb["cells"][13]["source"])
code = ast.literal_eval(re.sub(r"#[^\n]*", "", re.search(r"category_code = (\{.*?\})", src, re.S).group(1)))
assert set(code) == set(NAMES) and all(len(v) == 8 and v.isdigit() for v in code.values())
tab = pd.DataFrame([{"category": k, "code": v, **{c: int(v[i]) for i, c in enumerate(CAT)}}
                    for k, v in code.items()])

# --- 2. the curated workbook (same row order as category_list in cell 13) ------------
wb = pd.read_excel(PATH / "encoding_table/Categories_encoding_V04.xlsx")
wb_cols = [c for c in wb.columns if c.startswith(("Category_", "Category_Int"))]
assert len(wb_cols) == 8, wb_cols
assert (wb[wb_cols].values == tab[CAT].values).all(), "notebook code table and workbook disagree"

# --- 3. the data ------------------------------------------------------------------
enc = pd.read_csv(PATH / "fraudTrain_encoded.csv", usecols=CAT)
cat = pd.read_csv(PATH / "fraudTrain.csv", usecols=["category"])["category"]
seen = enc.assign(category=cat.values).drop_duplicates()
assert len(seen) == 14, "each category must map to exactly one encoding in the data"
merged = seen.merge(tab, on="category", suffixes=("_data", "_code"))
for c in CAT:
    assert (merged[f"{c}_data"] == merged[f"{c}_code"]).all(), c

# --- 4. the interaction definitions --------------------------------------------------
assert (tab["category_interaction_1"] == tab["category_2"] * tab["category_3"]).all()
assert (tab["category_interaction_2"] == tab["category_2"] * tab["category_4a"]).all()

# --- 5. categories the encoding cannot tell apart -------------------------------------
groups = tab.groupby("code")["category"].apply(list)
same = {k: [x for x in g if x != k] for g in groups for k in g if len(g) > 1}

rows = []
for _, r in tab.iterrows():
    rows.append({
        "category": NAMES[r["category"]], "sparkov_label": r["category"], "code": r["code"],
        "channel": f"{r['category_1']} ({CHANNEL[r['category_1']]})",
        "essential": f"{r['category_2']} ({ESSENTIAL[r['category_2']]})",
        "average_item_cost": f"{r['category_3']} ({QUARTILE[r['category_3']]})",
        "purpose": purpose(r),
        "interaction_1": f"{r['category_interaction_1']} ({INT1[r['category_interaction_1']]})",
        "interaction_2": f"{r['category_interaction_2']} ({INT2[r['category_interaction_2']]})",
        "identical_encoding_to": ", ".join(NAMES[x] for x in same.get(r["category"], [])),
    })
out = pd.DataFrame(rows)
out.to_csv(OUT / "category_encoding_audit_V01.csv", index=False)
pd.set_option("display.width", 250)
print(out.drop(columns=["sparkov_label", "code"]).to_string(index=False))
print("\nnotebook code table, workbook and data agree; both interaction definitions hold")
ambiguous = out[(tab["category_interaction_1"] == 0) & (tab["category_2"] == 1)]["category"].tolist()
print("essential categories with Interaction 1 = 0 (lowest cost quartile):", ambiguous)
print("pairs the encoding cannot distinguish:", sorted({tuple(sorted([NAMES[k]] + [NAMES[x] for x in v])) for k, v in same.items()}))
