"""
The four runs of record of MLProj01_fraudTrain_model_V26.ipynb (Revision_Phase7_Plan_V01.md §8).

For each of RAW_BASELINE x CLIENT_FEATURES, writes a copy of the notebook with the two switches set
in cell 0 to ../executed/model_V26_<TAG>.ipynb and executes it in place with nbconvert -- never with
--allow-errors (CLAUDE.md). DYLD_LIBRARY_PATH must be set by the caller.

  python run_model_v26.py [--only temporal_p7_notime,...]
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
NB = PATH / "MLProj01_fraudTrain_model_V26.ipynb"
RUNS = [(False, False), (False, True), (True, False), (True, True)]   # (RAW_BASELINE, CLIENT_FEATURES)
only = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None

for raw, client in RUNS:
    tag = "temporal_p7_notime" + ("_raw" if raw else "") + ("_client" if client else "")
    if only and tag not in only:
        continue
    nb = json.load(open(NB))
    src = "".join(nb["cells"][0]["source"])
    for name, val in (("RAW_BASELINE", raw), ("CLIENT_FEATURES", client)):
        line = f"{name} = False\n"
        assert src.count(line) == 1, name
        src = src.replace(line, f"{name} = {val}\n")
    nb["cells"][0]["source"] = src.splitlines(keepends=True)
    out = PATH / "executed" / f"model_V26_{tag}.ipynb"
    json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"=== executing {out.name}", flush=True)
    r = subprocess.run([sys.executable.replace("python", "jupyter"), "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1", str(out)])
    if r.returncode:
        raise SystemExit(f"{out.name} FAILED (exit {r.returncode}); its results must not be used")
    print(f"=== {out.name} done", flush=True)
