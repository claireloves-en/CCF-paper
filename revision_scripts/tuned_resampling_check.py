"""
Does resampling help the TUNED models? (Revision_Plan_V03.md, A2 / section 4.1)

The sampling sweep (d7_sweep.py) uses default hyperparameters. Without trans_time
it ranks no resampling fifth, though within 0.003 mean AUC-PR of the best setting,
and default-configured LightGBM gains 0.057 from ROS 10:90. The final models are
tuned without resampling, so this checks the question that decides the paper's
claim: with each model's tuned hyperparameters (the kept set in
hyperparameter_tuning_V02_<config>.csv), does resampling the training set improve
validation AUC-PR?

Settings: none, ROS 1:150, ROS 10:90 (the sweep's best on average), RUS 1:150.
Usage: python tuned_resampling_check.py [--client-features]
       python tuned_resampling_check.py --data fraudTrain_encoded.csv --tag p7_notime --version V03
         (Phase 7: any data file, predictors in file order; kept set from hyperparameter_tuning_V03_<tag>.csv)
Writes output/tuned_resampling<TAG>.csv
"""
import importlib.util, json, sys, time
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

HERE = Path(__file__).resolve().parent
client = "--client-features" in sys.argv
GENERIC = "--data" in sys.argv
VERSION = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else "V02"
if GENERIC:
    _data, _tag = sys.argv[sys.argv.index("--data") + 1], sys.argv[sys.argv.index("--tag") + 1]
    sys.argv = ["d8_tuning.py", "--data", _data, "--tag", _tag]
else:
    sys.argv = ["d8_tuning.py", "--drop-trans-time", *(["--client-features"] if client else [])]
spec = importlib.util.spec_from_file_location("d8", HERE / "d8_tuning.py")
d8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d8)

cfg = _tag if GENERIC else ("notime_client" if client else "notime")
kept = pd.read_csv(HERE.parent / "result_csv" / f"hyperparameter_tuning_{VERSION}_{cfg}.csv").set_index("model")
SETTINGS = {"none": None,
            "ROS 1:150": RandomOverSampler(sampling_strategy=0.0066667, random_state=d8.RANDOM_STATE),
            "ROS 10:90": RandomOverSampler(sampling_strategy=0.111111, random_state=d8.RANDOM_STATE),
            "RUS 1:150": RandomUnderSampler(sampling_strategy=0.0066667, random_state=d8.RANDOM_STATE)}

rows = []
for name, sampler in SETTINGS.items():
    tr = d8.TRAIN
    if sampler is not None:
        X, y = sampler.fit_resample(d8.TRAIN[d8.predictors], d8.TRAIN[d8.target])
        tr = pd.DataFrame(X, columns=d8.predictors); tr[d8.target] = y
    for model, _grid, fit in d8.MODELS:
        p = json.loads(kept.loc[model, "chosen_params"])   # already in fitter format
        s = time.time()
        ap = average_precision_score(d8.yv, fit(p, tr))
        rows.append(dict(setting=name, model=model, aucpr=ap, seconds=time.time() - s))
        d8.log(f"  {name:10s} {model:20s} AUC-PR {ap:.4f}")
out = HERE / "output" / f"tuned_resampling_{cfg}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
w = pd.DataFrame(rows).pivot(index="model", columns="setting", values="aucpr")[list(SETTINGS)]
w["best gain over none"] = w.drop(columns="none").max(axis=1) - w["none"]
print(w.round(4).to_string())
print(f"wrote {out}")
