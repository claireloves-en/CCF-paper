"""
Write one configuration's tuned hyperparameters into the TUNED table of the model
notebook, from result_csv/hyperparameter_tuning_V02_<config>.csv (built by
build_tuned_table.py). Replaces an existing entry for that configuration.

Usage: python fill_tuned.py notime          (or notime_client)
Phase 7 (model_V26, V03 tables; the table's name can differ from the notebook's CONFIG):
       python fill_tuned.py p7_notime --version V03 --table p7_notime
       python fill_tuned.py p7_notime_raw --version V03 --table notime_raw
"""
import ast, json, sys
from pathlib import Path
import pandas as pd

CONFIG = sys.argv[1]
HERE = Path(__file__).resolve().parent
import os
NB = Path(os.environ.get("TUNED_NB", HERE.parent / "MLProj01_fraudTrain_model_V26.ipynb"))
VERSION = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "V02"
TABLE = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else CONFIG
RES = Path(os.environ.get("TUNED_RES", HERE.parent / "result_csv"))
ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
         "CatBoost", "XGBoost", "LightGBM", "K-NN"]

t = pd.read_csv(RES / f"hyperparameter_tuning_{VERSION}_{TABLE}.csv").set_index("model")
entry = f'    "{CONFIG}": {{\n'
for m in ORDER:
    p = json.loads(t.loc[m, "chosen_params"])
    note = "" if t.loc[m, "kept"] == "tuned" else "   # untuned: search did not beat it"
    entry += f"        {m!r}: {p!r},{note}\n"
entry += "    },\n"

nb = json.load(open(NB))
idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "TUNED = {" in "".join(c["source"])]
assert len(idx) == 1, idx
src = "".join(nb["cells"][idx[0]]["source"])

start = src.find(f'    "{CONFIG}": {{')
if start != -1:                                   # replace an existing entry
    end = src.index("    },\n", start) + len("    },\n")
    src = src[:start] + entry + src[end:]
else:                                             # insert before the closing brace
    marker = '    # "notime" and "notime_client" are filled in once their searches finish\n'
    p7_marker = '    # "p7_*" entries are filled in by revision_scripts/fill_tuned.py --version V03 once their searches finish\n'
    if CONFIG.startswith("p7_") and p7_marker in src:
        src = src.replace(p7_marker, entry + p7_marker)
    elif marker in src:
        src = src.replace(marker, entry + marker)
    else:
        close = src.index("\n}\nassert CONFIG in TUNED")
        src = src[:close + 1] + entry + src[close + 1:]
if all(f'"{c}": {{' in src for c in ("notime", "notime_client")):
    src = src.replace('    # "notime" and "notime_client" are filled in once their searches finish\n', "")

ast.parse(src)
nb["cells"][idx[0]]["source"] = src.splitlines(keepends=True)
nb["cells"][idx[0]]["outputs"] = []; nb["cells"][idx[0]]["execution_count"] = None
json.dump(nb, open(NB, "w"), indent=1)

# prove the table loads and has all eight models for this configuration
ns = {"pd": pd}
exec(src.split("assert CONFIG in TUNED")[0], ns)
assert set(ns["TUNED"][CONFIG]) == set(ORDER), ns["TUNED"][CONFIG].keys()
print(f"TUNED[{CONFIG!r}] written to {NB.name}:")
for m in ORDER:
    print(f"  {m:20s} {t.loc[m, 'kept']:34s} {ns['TUNED'][CONFIG][m]}")
