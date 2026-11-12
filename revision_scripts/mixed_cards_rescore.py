"""
Re-score the test set on the mixed cards only, to estimate what generalises.
(J2 / §8 of Revision_Phase6_Plan_V01.md. Authors' decision, 2026-10-07.)

75 of the 983 cards carry nothing but fraud, and their cardholders appear on no other card
(fraud_only_cards.py): a synthetic population that exists only as fraud, not compromised
cards belonging to customers who also transact genuinely. They hold 174 of the test window's
1,538 frauds. On such a card the client-history features describe deviation from FRAUDULENT
history, so whatever part of their contribution comes from those 174 transactions does not
transfer to real data.

This recomputes each model's test metrics over the test window less those 174 transactions,
from the FROZEN per-row probabilities ablation_run.py saved. Nothing is refitted, nothing is
re-tuned, and each model keeps the operating threshold it set on the validation set, so this
is a re-scoring of a single test evaluation rather than a second look at the test set.

Read the result with one caution. Dropping 174 all-fraud rows lowers the test fraud rate from
0.593% to 0.526%, and AUC-PR scales with the base rate, so part of any fall is mechanical.
Both rates are printed; compare models within a column, and read the configuration difference
(client features minus base) rather than the levels.

Usage: python mixed_cards_rescore.py [--configs notime,notime_client]
Writes output/mixed_cards_rescore_V01.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                             precision_score, f1_score, matthews_corrcoef)

from data_paths import data_file as resolve
from ablation_run import CONFIGS, MODEL_ORDER, TARGET

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="notime,notime_client")
    ap.add_argument("--out", default="mixed_cards_rescore_V01.csv",
                    help="Phase 7: mixed_cards_rescore_V02.csv with --configs p7_notime,p7_notime_client")
    args = ap.parse_args()
    configs = args.configs.split(",")

    raw = pd.read_csv(PATH / "fraudTrain.csv", usecols=["cc_num", "is_fraud"])
    rate = raw.groupby("cc_num")["is_fraud"].mean()
    fraud_only = set(rate[rate == 1].index)
    print(f"{len(fraud_only)} fraud-only cards of {raw.cc_num.nunique()}")

    rows = []
    for cfg in configs:
        data_file, _tune, tag = CONFIGS[cfg]
        pred_path = OUT / f"test_predictions_{tag}.csv.gz"
        bud_path = OUT / f"test_operating_point_V01_{tag}.csv"
        if not pred_path.exists():
            print(f"SKIP {cfg}: {pred_path.name} not found -- run ablation_run.py --config {cfg}")
            continue

        # rebuild the test mask exactly as ablation_run.py cuts it, to line the saved
        # predictions up with cc_num by row position
        t = pd.read_csv(resolve(data_file), usecols=["trans_time"])["trans_time"]
        test_mask = (t > t.quantile(0.80)).values
        preds = pd.read_csv(pred_path)
        assert len(preds) == test_mask.sum(), f"{len(preds)} predictions vs {test_mask.sum()} test rows"
        y = preds[TARGET].values
        assert (y == raw[TARGET].values[test_mask]).all(), "prediction rows are not aligned"

        cc = raw["cc_num"].values[test_mask]
        mixed = ~np.isin(cc, list(fraud_only))
        thr = pd.read_csv(bud_path).set_index("metric\\model").loc["Operating threshold (from validation)"]
        print(f"\n{cfg}: test {len(y):,} rows / {int(y.sum())} fraud ({y.mean():.5f}) -> "
              f"mixed {int(mixed.sum()):,} / {int(y[mixed].sum())} fraud ({y[mixed].mean():.5f}); "
              f"dropped {int((~mixed).sum())} rows, {int(y[~mixed].sum())} of them fraud")

        for m in MODEL_ORDER:
            if m not in preds.columns:
                continue
            p = preds[m].values
            r = dict(config=cfg, model=m,
                     aucpr_all=average_precision_score(y, p),
                     aucpr_mixed=average_precision_score(y[mixed], p[mixed]),
                     auc_all=roc_auc_score(y, p),
                     auc_mixed=roc_auc_score(y[mixed], p[mixed]))
            th = float(thr[m])
            for label, sel in [("all", slice(None)), ("mixed", mixed)]:
                a = (p[sel] >= th).astype(int)
                yy = y[sel]
                r[f"alert_rate_{label}"] = float(a.mean())
                r[f"recall_{label}"] = recall_score(yy, a)
                r[f"precision_{label}"] = precision_score(yy, a, zero_division=0)
                r[f"f1_{label}"] = f1_score(yy, a)
                r[f"mcc_{label}"] = matthews_corrcoef(yy, a)
            rows.append(r)

    if not rows:
        raise SystemExit("nothing to score")
    res = pd.DataFrame(rows)
    res["aucpr_diff"] = res.aucpr_mixed - res.aucpr_all
    res.to_csv(OUT / args.out, index=False)
    print("\n" + res[["config", "model", "aucpr_all", "aucpr_mixed", "aucpr_diff",
                      "recall_mixed", "precision_mixed"]].round(4).to_string(index=False))

    # the question the decision was taken for: how much of the client-feature gain survives?
    pre = "p7_" if {"p7_notime", "p7_notime_client"} <= set(res.config) else ""
    if {pre + "notime", pre + "notime_client"} <= set(res.config):
        a = res[res.config == pre + "notime"].set_index("model")
        b = res[res.config == pre + "notime_client"].set_index("model")
        gain = pd.DataFrame({"gain_all_cards": b.aucpr_all - a.aucpr_all,
                             "gain_mixed_only": b.aucpr_mixed - a.aucpr_mixed})
        gain["share_surviving"] = (gain.gain_mixed_only / gain.gain_all_cards).round(3)
        print("\nclient-feature gain in test AUC-PR, all cards vs mixed cards only")
        print(gain.round(4).to_string())
    print(f"\nwrote {OUT / args.out}")


if __name__ == "__main__":
    main()
