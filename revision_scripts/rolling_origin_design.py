"""
The design of the repeated (rolling-origin) evaluation, for the paper (Revision_Plan_V08.md, A8; review of
CCF_MM_V03, item 6b). rolling_origin.py fixes the folds as quantiles of trans_time; this turns them into
dates and counts, and records which of the published windows each evaluation block lies in.

Models: random forest, CatBoost, XGBoost, LightGBM and the decision tree. Configurations: R, E, E+H.
Each model is refitted on the fold's expanding training window; hyperparameters are NOT re-tuned -- they are
the kept sets selected on the published validation window (64-80% of the series), which lies after folds 1
and 2 and contains fold 3's threshold and evaluation blocks.

Usage: python rolling_origin_design.py  -> result_csv/rolling_origin_design_V01.csv
"""
from pathlib import Path

import pandas as pd

from rolling_origin import FOLDS

PATH = Path(__file__).resolve().parent.parent
e = pd.read_csv(PATH / "fraudTrain_encoded.csv", usecols=["trans_time", "is_fraud"])
ts = pd.to_datetime(pd.read_csv(PATH / "fraudTrain.csv", usecols=["trans_date_trans_time"]).trans_date_trans_time)
t = e.trans_time
c64, c80 = t.quantile(0.64), t.quantile(0.80)


def where(lo, hi):
    if hi <= c64: return "published training window"
    if lo >= c64 and hi <= c80: return "published validation window"
    return "spans windows"


rows = []
for fi, (q_tr, q_th, q_ev) in enumerate(FOLDS, start=1):
    c_tr, c_th, c_ev = t.quantile(q_tr), t.quantile(q_th), t.quantile(q_ev)
    for part, m, lo, hi in [("train", t <= c_tr, t.min(), c_tr), ("threshold", (t > c_tr) & (t <= c_th), c_tr, c_th),
                            ("evaluation", (t > c_th) & (t <= c_ev), c_th, c_ev)]:
        rows.append(dict(fold=fi, block=part, first=ts[m].min(), last=ts[m].max(), transactions=int(m.sum()),
                         frauds=int(e.is_fraud[m].sum()), fraud_rate_pct=100 * e.is_fraud[m].mean(),
                         lies_in=where(lo, hi) if part != "train" else "published training window"))
out = pd.DataFrame(rows)
out.to_csv(PATH / "result_csv" / "rolling_origin_design_V01.csv", index=False)
print(out.to_string(index=False))
