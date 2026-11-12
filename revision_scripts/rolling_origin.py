"""
Repeated temporal evaluation: is the ranking stable over time? (J4, Revision_Phase6_Plan_V01.md §7.)

The assessment asks for confidence intervals OR repeated temporal evaluations. uncertainty_check.py
supplies the first from the frozen test predictions; this supplies the second, which the bootstrap
cannot: the bootstrap resamples one evaluation window, so it says nothing about whether performance
holds in a different period.

DESIGN. The data is one contiguous ~18 months (2019-01-01 to 2020-06-21), which constrains this more
than compute does. Three EXPANDING-origin folds, all of them inside the first 80% of the series, so
the 80-100% test period is never touched and the one-time-test rule holds absolutely:

  fold 1   train <= 32%   threshold on 32-40%   evaluate 40-48%
  fold 2   train <= 48%   threshold on 48-56%   evaluate 56-64%
  fold 3   train <= 64%   threshold on 64-72%   evaluate 72-80%     (train = the published split)

Expanding rather than sliding: the training window grows, as it would in deployment, and a sliding
window would waste the early data. Each evaluation block is 8% of the series, about 104,000
transactions with 400-700 frauds.

SCOPE. Five models -- the four ensembles plus the decision tree -- which carry every claim the check
is needed for. Three configurations: 1 (raw baseline), 3 (encoded) and 4 (encoded + client), so that
"does the J1 margin survive fold variation?" is answered as well as "does each model's performance?".

HYPERPARAMETERS ARE HELD FIXED at each configuration's kept set rather than re-tuned per fold.
Re-tuning would multiply the cost by 40 and answer a different question. The consequence has to be
stated in the paper: those sets were selected on the 64-80% window, which lies AFTER folds 1 and 2,
so those folds carry a mild lookahead advantage. It applies to every model equally, so it does not
distort the ranking, which is what the check is for.

READ THE FOLDS DOWN, NOT ACROSS. The evaluation blocks have different fraud rates and AUC-PR scales
with the base rate, so a fold-to-fold change in one model's AUC-PR is not by itself instability.
The base rate is reported per fold; compare models within a fold.

Usage: python rolling_origin.py [--configs notime_raw,notime,notime_client] [--models "..."]
Writes output/rolling_origin_V01.csv
"""
import argparse, json, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score

from ablation_run import CONFIGS, FITTERS, TARGET, budget_threshold, ALERT_RATE
from data_paths import data_file as resolve

HERE = Path(__file__).resolve().parent
PATH, RESULTS = HERE.parent, HERE.parent / "result_csv"
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)

# (train end, threshold window end, evaluation window end) as quantiles of trans_time
FOLDS = [(0.32, 0.40, 0.48), (0.48, 0.56, 0.64), (0.64, 0.72, 0.80)]
DEFAULT_MODELS = ["Decision Tree", "Random Forest", "CatBoost", "XGBoost", "LightGBM"]


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="notime_raw,notime,notime_client")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out", default="rolling_origin_V01.csv",
                    help="Phase 7: rolling_origin_V02.csv with --configs p7_notime_raw,p7_notime,p7_notime_client")
    args = ap.parse_args()
    models = args.models.split(",")

    rows = []
    for cfg in args.configs.split(","):
        data_file, tune_file, tag = CONFIGS[cfg]
        df = pd.read_csv(resolve(data_file))
        predictors = [c for c in df.columns if c not in (TARGET, "trans_time")]
        tuned = pd.read_csv(RESULTS / tune_file).set_index("model")
        t = df["trans_time"]
        log(f"=== {cfg}: {len(predictors)} predictors ===")

        for fi, (q_tr, q_th, q_ev) in enumerate(FOLDS, start=1):
            c_tr, c_th, c_ev = t.quantile(q_tr), t.quantile(q_th), t.quantile(q_ev)
            train = df[t <= c_tr]
            thresh = df[(t > c_tr) & (t <= c_th)]
            evalw = df[(t > c_th) & (t <= c_ev)]
            y = evalw[TARGET].values
            log(f"  fold {fi}: train {len(train):,} / threshold {len(thresh):,} / "
                f"evaluate {len(evalw):,}; eval fraud {int(y.sum())} ({y.mean():.5f})")

            for name in models:
                params = json.loads(tuned.loc[name, "chosen_params"])
                s = time.time()
                predict = FITTERS[name](params, train, thresh, predictors)
                p_th, p_ev = predict(thresh), predict(evalw)
                thr, _a, usable = budget_threshold(p_th)
                row = dict(config=cfg, fold=fi, model=name,
                           train_rows=len(train), eval_rows=len(evalw),
                           eval_fraud=int(y.sum()), eval_fraud_rate=float(y.mean()),
                           aucpr=average_precision_score(y, p_ev),
                           auc=roc_auc_score(y, p_ev))
                if usable:
                    alerted = (p_ev >= thr).astype(int)
                    row.update(threshold=thr, alert_rate=float(alerted.mean()),
                               recall=recall_score(y, alerted),
                               precision=precision_score(y, alerted, zero_division=0))
                else:
                    row.update(threshold=thr, alert_rate=np.nan, recall=np.nan, precision=np.nan)
                rows.append(row)
                log(f"    {name}: AUC-PR {row['aucpr']:.4f} ({time.time() - s:.0f}s)")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / args.out, index=False)

    pd.set_option("display.width", 200)
    print("\nAUC-PR by fold (read DOWN a column; fraud rates differ between folds)\n")
    piv = res.pivot_table(index=["config", "model"], columns="fold", values="aucpr")
    print(piv.round(4).to_string())
    print("\neval-window fraud rate by fold")
    print(res.groupby("fold")["eval_fraud_rate"].first().round(5).to_string())

    print("\nper-model spread across folds (max - min AUC-PR)")
    print(res.groupby(["config", "model"])["aucpr"].agg(lambda a: a.max() - a.min()).round(4).to_string())

    pre = "p7_" if {"p7_notime", "p7_notime_raw"} <= set(res.config) else ""
    if {pre + "notime", pre + "notime_raw"} <= set(res.config):
        a = res[res.config == pre + "notime"].set_index(["model", "fold"])["aucpr"]
        b = res[res.config == pre + "notime_raw"].set_index(["model", "fold"])["aucpr"]
        d = (a - b).unstack()
        print("\nJ1 margin per fold: encoded minus raw baseline (negative = baseline ahead)\n")
        print(d.round(4).to_string())
        print(f"\nsign consistent across every fold and model: "
              f"{bool((d.values < 0).all() or (d.values > 0).all())}")
    print(f"\nwrote {OUT / args.out}")


if __name__ == "__main__":
    main()
