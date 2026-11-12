"""
The single-use confirmation on fraudTest.csv (Revision_Phase7_Plan_V01.md §9, step 2).

fraudTest.csv (Sparkov's later period, 2020-06-21 to 2020-12-31) has never been scored. This
script scores it ONCE, with every choice frozen:
  - each configuration is refitted on the same training split with its kept (V03)
    hyperparameters, exactly as ablation_run.py fits it;
  - each model's threshold is set on the validation split at the 0.5% alert budget;
  - GUARD: the refitted models' predictions on the original test window must equal the saved
    ablation_run.py predictions to 1e-9 for every model of every configuration. Only then is
    fraudTest read and scored;
  - the outputs are written once. The script refuses to run if any of them exists, and records
    the git commit it ran from and the frozen plan's commit.

Run in two steps, so a guard failure never coincides with a scored fraudTest:
  python confirm_run.py --guard-only     # refit + guard, fraudTest untouched
  python confirm_run.py                  # refit + guard + the one scoring

Outputs (output/):
  confirm_predictions_<tag>.csv.gz   per-row fraudTest probabilities, `row` = row in fraudTest.csv
  confirm_metrics_V01.csv            AUC-PR; fixed threshold, fixed capacity per day and over the
                                     window: alerts, TP, FP, precision, recall
  confirm_run_V01.log.json           commits, timestamps, guard results, row counts
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ablation_run import CONFIGS, FITTERS, MODEL_ORDER, TARGET, budget_threshold, RESULTS
from data_paths import data_file
from operating_points import capacity_flags, workload

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
CONFIRM = PATH / "confirm_data"
TEST_FILE = PATH.parent.parent / "Found_DataSet" / "fraudTest.csv"
FROZEN_PLAN_COMMIT = "5351e35"
DEFAULT = "p7_notime_raw,p7_notime_raw_client,p7_notime,p7_notime_client,p7_hybrid_client,p7_catnative_client"
FRAUDTEST_FILE = {   # fraudTrain configuration file -> its fraudTest counterpart (build_fraudtest.py)
    "fraudTrain_encoded.csv": "fraudTest_encoded.csv",
    "fraudTrain_encoded_client.csv": "fraudTest_encoded_client.csv",
    "fraudTrain_raw_baseline.csv": "fraudTest_raw_baseline.csv",
    "fraudTrain_raw_baseline_client.csv": "fraudTest_raw_baseline_client.csv",
    "ablation_data/fraudTrain_hybrid_client.csv": "fraudTest_hybrid_client.csv",
    "ablation_data/fraudTrain_catnative_client.csv": "fraudTest_catnative_client.csv",
}


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def git(*args):
    return subprocess.run(["git", *args], cwd=PATH, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=DEFAULT)
    ap.add_argument("--guard-only", action="store_true")
    args = ap.parse_args()
    configs = args.configs.split(",")
    outputs = [OUT / "confirm_metrics_V01.csv", OUT / "confirm_run_V01.log.json"] + \
              [OUT / f"confirm_predictions_{CONFIGS[c][2]}.csv.gz" for c in configs]
    if not args.guard_only:
        existing = [p.name for p in outputs if p.exists()]
        if existing:
            raise SystemExit(f"REFUSED: fraudTest has already been scored ({existing}). It is single-use.")
        assert subprocess.run(["git", "merge-base", "--is-ancestor", FROZEN_PLAN_COMMIT, "HEAD"],
                              cwd=PATH).returncode == 0, "the frozen plan's commit is not in this history"

    record = dict(started=datetime.now().isoformat(timespec="seconds"), head=git("rev-parse", "--short", "HEAD"),
                  frozen_plan_commit=FROZEN_PLAN_COMMIT, dirty_scripts=git("status", "--porcelain", "--", "."),
                  guard_only=args.guard_only, configs={})
    fitted = {}
    for cfg in configs:
        data, tune, tag = CONFIGS[cfg]
        df = pd.read_csv(data_file(data))
        predictors = [c for c in df.columns if c not in (TARGET, "trans_time")]
        t = df["trans_time"]
        c1, c2 = t.quantile(0.64), t.quantile(0.80)
        train, valid, test = df[t <= c1], df[(t > c1) & (t <= c2)], df[t > c2]
        saved = pd.read_csv(OUT / f"test_predictions_{tag}.csv.gz")
        assert (saved["row"].values == test.index.values).all(), f"{cfg}: saved rows differ"
        tuned = pd.read_csv(RESULTS / tune).set_index("model")
        models = [m for m in MODEL_ORDER if m in saved.columns]
        fitted[cfg] = {}
        record["configs"][cfg] = {"tag": tag, "models": {}}
        for m in models:
            s = time.time()
            predict = FITTERS[m](json.loads(tuned.loc[m, "chosen_params"]), train, valid, predictors)
            p_test = predict(test)
            worst = float(np.abs(p_test - saved[m].values).max())
            # XGBoost predicts in float32 (resolution ~6e-8 near 1), and its multithreaded sums are
            # not bit-reproducible at that precision; 1e-9 is below what it can represent. Float64
            # outputs keep 1e-9. (Revision_Phase7_Results_V01.md §0, item 3.)
            tol = 1e-6 if np.asarray(p_test).dtype == np.float32 else 1e-9
            thr, _a, usable = budget_threshold(predict(valid))
            record["configs"][cfg]["models"][m] = dict(guard_max_abs_diff=worst, guard_tol=tol,
                                                       threshold=thr, usable=bool(usable))
            log(f"{cfg} {m}: guard max |diff| {worst:.2e} (tol {tol:.0e}) ({time.time() - s:.0f}s)")
            if worst > tol:
                raise SystemExit(f"GUARD FAILED for {cfg} / {m}: {worst:.3e}. fraudTest not read.")
            fitted[cfg][m] = (predict, thr, usable, predictors)
    log("GUARD PASSED for every model of every configuration")
    if args.guard_only:
        print(json.dumps(record, indent=1)); return

    # ---------------------------------------------- the one scoring of fraudTest
    raw = pd.read_csv(TEST_FILE, usecols=["trans_date_trans_time", "unix_time", "cc_num", TARGET])
    day = pd.to_datetime(raw["trans_date_trans_time"]).dt.date.values
    chrono = raw["unix_time"].values
    rows = []
    for cfg in configs:
        data, _tune, tag = CONFIGS[cfg]
        ft = pd.read_csv(CONFIRM / FRAUDTEST_FILE[data])
        assert len(ft) == len(raw) and (ft[TARGET].values == raw[TARGET].values).all()
        y = ft[TARGET].values
        preds = {}
        for m, (predict, thr, usable, predictors) in fitted[cfg].items():
            s = predict(ft).astype(float)
            preds[m] = s
            rules = [("fixed threshold (validation)", s >= thr if usable else None),
                     ("fixed capacity, 0.5% per day", capacity_flags(s, day, chrono)),
                     ("fixed capacity, 0.5% of window", capacity_flags(s, None, chrono))]
            for rule, flags in rules:
                w = workload(flags, y) if flags is not None else {}
                rows.append(dict(config=cfg, model=m, aucpr=average_precision_score(y, s), rule=rule, **w))
            log(f"{cfg} {m}: fraudTest AUC-PR {rows[-1]['aucpr']:.4f}")
        pd.DataFrame(preds).assign(**{TARGET: y, "row": np.arange(len(y))}).to_csv(
            OUT / f"confirm_predictions_{tag}.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(OUT / "confirm_metrics_V01.csv", index=False)

    # the population facts the plan asked for, recorded with the scoring
    train_cards = set(pd.read_csv(PATH / "fraudTrain.csv", usecols=["cc_num"])["cc_num"])
    new = ~raw["cc_num"].isin(train_cards)
    rate = raw[new].groupby("cc_num")[TARGET].mean()
    record.update(finished=datetime.now().isoformat(timespec="seconds"), fraudtest_rows=len(raw),
                  fraudtest_fraud=int(raw[TARGET].sum()), new_cards=int(new.groupby(raw["cc_num"]).first().sum()),
                  new_card_rows=int(new.sum()), new_cards_fraud_only=int((rate == 1).sum()),
                  new_card_fraud=int(raw.loc[new, TARGET].sum()))
    json.dump(record, open(OUT / "confirm_run_V01.log.json", "w"), indent=1)
    log(f"fraudTest scored once: {len(raw):,} rows, {int(raw[TARGET].sum())} fraudulent. Outputs written; "
        f"commit them now.")


if __name__ == "__main__":
    main()
