"""
Location grid from training-split bounds: does the downstream result change?
(Revision_Plan_V07.md item R3; the review of CCF_MM_V02, 2026-10-26, §3C.)

fit_scope_check.py showed that the 32 x 32 location grid (encoding cell 34) spans the lat/long range
of all rows, and that training-only bounds move 0.43% of cardholder cells and 0.13% of merchant cells.
That is a feature-level difference. This refits the two encoded configurations with the training-only
grid and compares the test results with the runs of record.

  python grid_bounds_check.py --build        # ablation_data/fraudTrain_grid_train{,_client}.csv
  python ablation_run.py --config p7_grid_train
  python ablation_run.py --config p7_grid_train_client
  python grid_bounds_check.py --compare      # -> output/grid_bounds_check_V01.csv

The refits keep E's and E+H's kept hyperparameters (the grid is not re-tuned), so a difference here is
the effect of the bounds alone. Differences are paired, from 1,000 stratified bootstrap resamples of the
test period with seed 1200, as for Table 3.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

TRAIN_FRACTION = 0.64
GRID_SIZE = 32


def grid(lat, lon, b):
    """Encoding cell 34, vectorised (as fit_scope_check.py; build() asserts it reproduces the files)."""
    r = np.trunc((lat - b["lat0"]) / (b["lat1"] - b["lat0"]) * (GRID_SIZE - 1)).astype(int)
    c = np.trunc((lon - b["lon0"]) / (b["lon1"] - b["lon0"]) * (GRID_SIZE - 1)).astype(int)
    return r * GRID_SIZE + c


def bounds(d):
    return dict(lat0=d[["lat", "merch_lat"]].min().min(), lat1=d[["lat", "merch_lat"]].max().max(),
                lon0=d[["long", "merch_long"]].min().min(), lon1=d[["long", "merch_long"]].max().max())


HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
ABL = PATH / "ablation_data"
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
               "CatBoost", "XGBoost", "LightGBM", "K-NN"]
PAIRS = {"p7_notime": "p7_grid_train", "p7_notime_client": "p7_grid_train_client"}
RAW = {"p7_notime": "p7_notime_raw", "p7_notime_client": "p7_notime_raw_client"}
N_BOOT, SEED = 1000, 1200


def build():
    raw = pd.read_csv(PATH / "fraudTrain.csv", usecols=["lat", "long", "merch_lat", "merch_long"])
    for src, dst in [("fraudTrain_encoded.csv", "fraudTrain_grid_train.csv"),
                     ("fraudTrain_encoded_client.csv", "fraudTrain_grid_train_client.csv")]:
        enc = pd.read_csv(PATH / src)
        assert len(enc) == len(raw)
        is_train = (enc["trans_time"] <= enc["trans_time"].quantile(TRAIN_FRACTION)).values
        b_all, b_train = bounds(raw), bounds(raw[is_train])
        for col, (la, lo) in {"loc_cluster_issue": ("lat", "long"),
                              "loc_cluster_transac": ("merch_lat", "merch_long")}.items():
            assert (grid(raw[la].values, raw[lo].values, b_all) == enc[col].values).all()
            new = grid(raw[la].values, raw[lo].values, b_train)
            print(f"{dst} {col}: {int((new != enc[col].values).sum()):,} rows change cell")
            enc[col] = new                       # same position, same column order
        enc.to_csv(ABL / dst, index=False)
        print("wrote", ABL / dst)


def strat_boot_idx(y, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def compare():
    rows = []
    for base, alt in PAIRS.items():
        a = pd.read_csv(OUT / f"test_predictions_temporal_{base}.csv.gz")
        b = pd.read_csv(OUT / f"test_predictions_temporal_{alt}.csv.gz")
        r = pd.read_csv(OUT / f"test_predictions_temporal_{RAW[base]}.csv.gz")
        assert (a["row"].values == b["row"].values).all() and (a["row"].values == r["row"].values).all()
        y = a["is_fraud"].values
        opa = pd.read_csv(OUT / f"test_operating_point_V01_temporal_{base}.csv").set_index("metric\\model")
        opb = pd.read_csv(OUT / f"test_operating_point_V01_temporal_{alt}.csv").set_index("metric\\model")
        rng = np.random.default_rng(SEED)
        idx = [strat_boot_idx(y, rng) for _ in range(N_BOOT)]
        for m in MODEL_ORDER:
            pa, pb, pr = a[m].values, b[m].values, r[m].values
            d = [average_precision_score(y[i], pb[i]) - average_precision_score(y[i], pa[i]) for i in idx]
            ap_a, ap_b, ap_r = (average_precision_score(y, p) for p in (pa, pb, pr))
            rows.append(dict(config=base, model=m, aucpr_record=ap_a, aucpr_grid_train=ap_b,
                             diff=ap_b - ap_a, diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5),
                             gap_E_minus_R_record=ap_a - ap_r, gap_E_minus_R_grid_train=ap_b - ap_r,
                             max_abs_prob_change=float(np.abs(pb - pa).max()),
                             recall_budget_record=opa.loc["Recall (budget)", m],
                             recall_budget_grid_train=opb.loc["Recall (budget)", m],
                             precision_budget_record=opa.loc["Precision (budget)", m],
                             precision_budget_grid_train=opb.loc["Precision (budget)", m]))
            print(f"{base:18s} {m:20s} {ap_a:.4f} -> {ap_b:.4f}  diff {ap_b - ap_a:+.4f} "
                  f"[{rows[-1]['diff_lo']:+.4f}, {rows[-1]['diff_hi']:+.4f}]")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "grid_bounds_check_V01.csv", index=False)
    print("wrote", OUT / "grid_bounds_check_V01.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    build() if args.build else compare()
