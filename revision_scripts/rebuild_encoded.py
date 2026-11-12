"""
Rebuild fraudTrain_encoded.csv and fraudTrain_encoded_client.csv with the two encoding bugs fixed
(Revision_Plan_V06.md, Stage 2A).

Bug A — city features on the wrong rows. The USDA income table kept the US and state totals
(FIPS xx000); with ' County' stripped they collided with same-named counties, so
city_and_encoding.csv held 935 rows for 928 cities. Cell 23 then merged (1,312,122 rows) and
inserted the result into df_enc by index, so every row after the first duplicate received the
city values of another transaction.
Bug B — personal_care coded '001010001' (9 digits), shifting every category field by one.
Also fixed: the 5G table's 'Parish' fix-up appended copies in every state, which entered the
5G quintiles.

The encoding notebook needs the geo/NLP stack, which is not installed, so this script runs the
notebook's own pandas-only city cells (16-22) rather than a re-implementation:
  1. V24's cells, in a scratch copy of encoding_table/, must reproduce the committed
     city_and_encoding.csv, and V24's merge + insert must reproduce the committed city columns —
     the bug, reproduced exactly;
  2. V25's cells, likewise, give the corrected lookup, which is written back to encoding_table/;
  3. the category columns come from V25's category_code (cell 13);
  4. every other column of the encoded files must be unchanged.
The old files move to ../_old/ with a _V24 suffix.

  python rebuild_encoded.py            # check, rebuild and write
  python rebuild_encoded.py --dry-run  # check only
"""
import ast
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
NB = {"V24": PATH / "MLProj01_fraudTrain_encoding_V24.ipynb",
      "V25": PATH / "MLProj01_fraudTrain_encoding_V25.ipynb"}
CITY_CELLS = range(16, 23)
CITY = ["city_1", "city_2", "city_3", "city_4a", "city_4b"]
CAT = ["category_1", "category_2", "category_3", "category_4a", "category_4b", "category_4c",
       "category_interaction_1", "category_interaction_2"]
DRY = "--dry-run" in sys.argv


def cells(version):
    nb = json.load(open(NB[version]))
    return ["".join(c["source"]) for c in nb["cells"]]


def run_city_cells(version, raw):
    """Run the notebook's city cells in a scratch copy of encoding_table/; return df_city and
    the scratch directory holding the tables they wrote."""
    tmp = Path(tempfile.mkdtemp(prefix=f"city_{version}_"))
    shutil.copytree(PATH / "encoding_table", tmp / "encoding_table",
                    ignore=shutil.ignore_patterns("_old", "*.xlsx"))
    ns = {"pd": pd, "np": np, "display": lambda *a, **k: None, "HTML": lambda x: x,
          "df_enc": raw[["city", "state"]].copy()}
    src = cells(version)
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        for i in CITY_CELLS:
            exec(compile(src[i], f"{version}_cell{i}", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns["df_city"], tmp / "encoding_table"


def category_code(version):
    src = cells(version)[13]
    body = re.search(r"category_code = (\{.*?\})", src, re.S).group(1)
    body = re.sub(r"#[^\n]*", "", body)
    return ast.literal_eval(body)


raw = pd.read_csv(PATH / "fraudTrain.csv", usecols=["city", "state", "category", "amt", "is_fraud"])
old = pd.read_csv(PATH / "fraudTrain_encoded.csv")
old_client = pd.read_csv(PATH / "fraudTrain_encoded_client.csv")
n = len(raw)
assert len(old) == n == len(old_client)
assert np.allclose(old["amt"].values, raw["amt"].values) and (old["is_fraud"].values == raw["is_fraud"].values).all()
shared = [c for c in old.columns]
assert list(old_client.columns[:len(shared) - 1]) == shared[:-1], "client file = encoded + 7 before is_fraud"
for c in shared:  # floats to 1e-12: issue_transac_dist differs by 3.6e-15 (CSV decimal round trip)
    assert np.allclose(old[c].values, old_client[c].values, rtol=0, atol=1e-12), f"encoded and client files disagree on {c}"

# --- 1. V24 reproduces the committed lookup and the bug ------------------------------
city24, dir24 = run_city_cells("V24", raw)
committed = pd.read_csv(PATH / "encoding_table/csv/city_and_encoding.csv")
pd.testing.assert_frame_equal(pd.read_csv(dir24 / "csv/city_and_encoding.csv"), committed)
merged24 = raw.merge(city24, how="left", left_on=["state", "city"], right_on=["State", "City"])
assert len(merged24) == 1_312_122, len(merged24)
assert (merged24[CITY].values[:n].astype(int) == old[CITY].values).all()
print(f"V24 city cells reproduce city_and_encoding.csv ({len(committed)} rows for "
      f"{committed[['State', 'City']].drop_duplicates().shape[0]} cities) and the committed "
      f"city columns (merge gave {len(merged24):,} rows for {n:,})")

# --- 2. V25: one row per city, keyed join --------------------------------------------
city25, dir25 = run_city_cells("V25", raw)
assert not city25.duplicated(["State", "City"]).any()
assert len(city25) == committed[["State", "City"]].drop_duplicates().shape[0]
merged25 = raw[["state", "city"]].merge(city25, how="left", left_on=["state", "city"],
                                        right_on=["State", "City"], validate="many_to_one")
assert len(merged25) == n and merged25[CITY].notna().all().all()
new_city = merged25[CITY].astype(int).reset_index(drop=True)

# how the lookup itself changed, city by city (the quartile cut-offs moved when the
# state totals left the income table, and the duplicated cities resolve to their county)
cmp = committed.drop_duplicates(["State", "City"], keep="first").merge(
    city25, on=["State", "City"], suffixes=("_v24", "_v25"), validate="one_to_one")
lookup_changes = {c: int((cmp[f"{c}_v24"] != cmp[f"{c}_v25"]).sum()) for c in CITY}
print("cities whose lookup value changed V24 -> V25 (first of any duplicate):", lookup_changes)

# --- 3. category columns from V25's code table -----------------------------------------
code24, code25 = category_code("V24"), category_code("V25")
assert all(len(v) == 8 for v in code25.values())
assert {k for k in code25 if code25[k] != code24[k]} == {"personal_care"}
digits = raw["category"].map(code25)
new_cat = pd.DataFrame({c: digits.str[i].astype(int) for i, c in enumerate(CAT)})
old_cat_by_code = pd.DataFrame({c: raw["category"].map(code24).str[i].astype(int) for i, c in enumerate(CAT)})
assert (old_cat_by_code.values == old[CAT].values).all(), "V24 code table must reproduce the committed category columns"
cat_rows = (new_cat.values != old[CAT].values).any(axis=1)
assert set(raw.loc[cat_rows, "category"]) == {"personal_care"}

# --- 4. assemble, everything else unchanged ---------------------------------------------
def patch(df):
    out = df.copy()
    for c in CITY:
        out[c] = new_city[c].values
    for c in CAT:
        out[c] = new_cat[c].values
    return out

new = patch(old)
new_client = patch(old_client)
untouched = [c for c in old.columns if c not in CITY + CAT]
for a, b in ((old, new), (old_client, new_client)):
    for c in [c for c in a.columns if c not in CITY + CAT]:
        assert np.array_equal(a[c].values, b[c].values)
    assert list(a.columns) == list(b.columns) and len(b) == n

rows = []
for c in CITY + CAT:
    rows.append({"column": c, "rows_changed": int((old[c].values != new[c].values).sum()),
                 "share_changed": float((old[c].values != new[c].values).mean())})
rows.append({"column": "any city column", "rows_changed": int((old[CITY].values != new[CITY].values).any(1).sum()),
             "share_changed": float((old[CITY].values != new[CITY].values).any(1).mean())})
rows.append({"column": "any category column", "rows_changed": int(cat_rows.sum()), "share_changed": float(cat_rows.mean())})
for c, k in lookup_changes.items():
    rows.append({"column": f"lookup {c}: cities changed", "rows_changed": k, "share_changed": k / len(city25)})
report = pd.DataFrame(rows)
print(report.to_string(index=False))
print(f"{len(untouched)} other columns unchanged in both files")

OUT.mkdir(exist_ok=True)
report.to_csv(OUT / "encoding_fix_V01.csv", index=False)
cat_table = pd.DataFrame([{"category": k, "code": v, **{c: int(v[i]) for i, c in enumerate(CAT)}}
                          for k, v in code25.items()])
cat_table.to_csv(OUT / "category_codes_V25.csv", index=False)

if DRY:
    print("\n--dry-run: nothing written")
    sys.exit(0)

# --- write: corrected lookup tables, then the data files ---------------------------------
for rel in ["csv/city_and_encoding.csv", "ref_data/city_county.csv", "ref_data/city_income.csv",
            "ref_data/city_gini_index.csv", "ref_data/city_city_type.csv",
            "ref_data/city_fixed_broadband.csv", "ref_data/city_mobile_broadband.csv"]:
    shutil.copyfile(dir25 / rel, PATH / "encoding_table" / rel)
old_dir = PATH / "_old"
old_dir.mkdir(exist_ok=True)
for name in ("fraudTrain_encoded", "fraudTrain_encoded_client"):
    dest = old_dir / f"{name}_V24.csv"
    if not dest.exists():
        shutil.move(PATH / f"{name}.csv", dest)
new.to_csv(PATH / "fraudTrain_encoded.csv", index=False)
new_client.to_csv(PATH / "fraudTrain_encoded_client.csv", index=False)

# read back: what the models will load
for name, want in (("fraudTrain_encoded", new), ("fraudTrain_encoded_client", new_client)):
    got = pd.read_csv(PATH / f"{name}.csv")
    assert list(got.columns) == list(want.columns)
    for c in want.columns:
        assert np.allclose(got[c].values, want[c].values, rtol=0, atol=1e-12), c
print("wrote corrected encoding tables, fraudTrain_encoded.csv and fraudTrain_encoded_client.csv; "
      "old files in _old/ with suffix _V24")
