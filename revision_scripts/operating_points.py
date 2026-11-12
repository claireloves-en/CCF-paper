"""
Fixed-threshold and fixed-capacity operating points, with realised workloads
(Revision_Phase7_Plan_V01.md §5; revision_plan_12_oct.md §3C).

The paper's operating point is a threshold set on the validation set to flag 0.5% of validation
transactions and then held fixed. On test it flags a different share (0.500-0.566% in Phase 6), so
two configurations compared at "the 0.5% operating point" may be reviewing different amounts of
work. This reports both evaluations from the saved test predictions, refitting nothing:

  fixed threshold   the validation threshold, unchanged: alerts, TP, FP, realised alert rate,
                    precision, recall
  fixed capacity    within each calendar day, the top round_half_up(0.005 x n_day) transactions by
                    score; ties at the cut-off broken by chronological order (earlier first). No
                    label is used to set the cut.
  capacity, window  the same over the whole evaluation window at once (secondary)

capacity_flags() is shared with uncertainty_check.py, which puts intervals on these numbers.

  python operating_points.py [--configs p7_notime_raw,p7_notime_raw_client,p7_notime,p7_notime_client]
  -> output/operating_points_V01.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ablation_run import CONFIGS, MODEL_ORDER, TARGET
from data_paths import data_file

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
RATE = 0.005


def round_half_up(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(int)


def capacity_flags(scores, group, chrono):
    """Flag the top round_half_up(RATE x n_g) rows by score within each group, ties broken by
    chronological order (earlier first). group: per-row group key (e.g. calendar day) or None for
    one group; chrono: per-row sortable time (unix_time, then row number as a final tie-break)."""
    n = len(scores)
    group = np.zeros(n, dtype=int) if group is None else pd.factorize(group)[0]
    # sort by group, then score descending, then time ascending, then row
    order = np.lexsort((np.arange(n), chrono, -scores, group))
    g_sorted = group[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    sizes = np.diff(np.r_[starts, n])
    k = round_half_up(RATE * sizes)
    rank_in_group = np.arange(n) - np.repeat(starts, sizes)
    flags = np.zeros(n, dtype=bool)
    flags[order] = rank_in_group < np.repeat(k, sizes)
    return flags


def test_rows(cfg, preds):
    """Row numbers in fraudTrain.csv of the saved test predictions."""
    if "row" in preds.columns:
        return preds["row"].values
    t = pd.read_csv(data_file(CONFIGS[cfg][0]), usecols=["trans_time"])["trans_time"]
    rows = np.flatnonzero((t > t.quantile(0.80)).values)   # Phase 6 files: rebuild the split
    assert len(rows) == len(preds)
    return rows


def load(cfg):
    tag = CONFIGS[cfg][2]
    preds = pd.read_csv(OUT / f"test_predictions_{tag}.csv.gz")
    thr = pd.read_csv(OUT / f"test_operating_point_V01_{tag}.csv").set_index("metric\\model")
    return preds, thr.loc["Operating threshold (from validation)"]


def workload(flags, y):
    tp, fp = int((flags & (y == 1)).sum()), int((flags & (y == 0)).sum())
    return dict(alerts=tp + fp, tp=tp, fp=fp, alert_rate=(tp + fp) / len(y),
                precision=tp / (tp + fp) if tp + fp else np.nan, recall=tp / int((y == 1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="p7_notime_raw,p7_notime_raw_client,p7_notime,p7_notime_client")
    args = ap.parse_args()
    raw = pd.read_csv(PATH / "fraudTrain.csv", usecols=["trans_date_trans_time", "unix_time", "is_fraud"])
    day = pd.to_datetime(raw["trans_date_trans_time"]).dt.date.values

    out = []
    for cfg in args.configs.split(","):
        try:
            preds, thr = load(cfg)
        except FileNotFoundError:
            print(f"SKIP {cfg}: predictions not found"); continue
        rows = test_rows(cfg, preds)
        y = preds[TARGET].values
        assert (y == raw[TARGET].values[rows]).all(), f"{cfg}: rows not aligned"
        chrono = raw["unix_time"].values[rows]
        d = day[rows]
        for m in [m for m in MODEL_ORDER if m in preds.columns]:
            s = preds[m].values.astype(float)
            for rule, flags in (("fixed threshold (validation)", s >= float(thr[m])),
                                ("fixed capacity, 0.5% per day", capacity_flags(s, d, chrono)),
                                ("fixed capacity, 0.5% of window", capacity_flags(s, None, chrono))):
                out.append(dict(config=cfg, model=m, rule=rule, **workload(flags, y)))
        print(f"{cfg}: {len(y):,} test rows over {len(set(d))} days")
    res = pd.DataFrame(out)
    res.to_csv(OUT / "operating_points_V01.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
