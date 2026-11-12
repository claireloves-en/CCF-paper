"""
Matched compression: does the encoding's theory-chosen coarsening lose less than an arbitrary coarsening of
the same granularity? (Revision_Phase8_Plan_V01.md, C2; frozen at commit 10ceada.)

Configuration E (fraudTrain_encoded.csv), random forest and CatBoost at E's kept hyperparameters
(hyperparameter_tuning_V03_p7_notime.csv), fitted exactly as ablation_run.py fits them.

  draw 0        E itself -- must reproduce ablation_run.py's saved test predictions to 1e-9 first
  time i        trans_time_grp (E: hours 0-7, 8-15, 16-23 -> 0, 1, 2) replaced by three random
                contiguous bands on the 24-hour circle: three distinct cut hours, every band >= 2
                hours, partition != E's; coded 0, 1, 2 clockwise from the band containing hour 0.
                rng = default_rng(1200 + i), i = 1..20
  category i    the 14 rows of the category code table (8 attributes, interactions included) assigned
                to the 14 categories by a random non-identity permutation: same attributes, same 11
                distinct profiles, consistent interactions; only the theory's assignment is lost.
                rng = default_rng(2200 + i), i = 1..20

Per draw: validation and test AUC-PR of each model and of the two-model mean. fraudTest is not read.

Usage: python matched_compression.py [--draws 20]
Writes result_csv/matched_compression_V01.csv (per draw) and prints E's percentile rank per arm.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ablation_run import CONFIGS, OUT, RESULTS, TARGET, fit_cat, fit_forest, log, resolve

N_DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 20
MODELS = {"Random Forest": fit_forest, "CatBoost": fit_cat}
CAT_COLS = ["category_1", "category_2", "category_3", "category_4a", "category_4b", "category_4c",
            "category_interaction_1", "category_interaction_2"]
E_BANDS = (0, 8, 16)


def bands_of(cuts):
    """hour -> band code for sorted cut hours (each cut starts a band), coded clockwise from hour 0's band."""
    cuts = sorted(cuts)
    code = np.empty(24, dtype=int)
    for h in range(24):
        # the band containing h starts at the largest cut <= h, or wraps to the last cut
        start = max([c for c in cuts if c <= h], default=cuts[-1])
        code[h] = cuts.index(start)
    return (code - code[0]) % 3


def draw_cuts(rng):
    while True:
        cuts = sorted(rng.choice(24, 3, replace=False).tolist())
        lens = [cuts[1] - cuts[0], cuts[2] - cuts[1], 24 - cuts[2] + cuts[0]]
        if min(lens) >= 2 and tuple(cuts) != E_BANDS:
            return cuts


def main():
    data_file, tune_file, tag = CONFIGS["p7_notime"]
    df = pd.read_csv(resolve(data_file))
    src = pd.read_csv(resolve("fraudTrain.csv"), usecols=["trans_date_trans_time", "category"])
    hour = pd.to_datetime(src.trans_date_trans_time).dt.hour.values
    assert (bands_of(list(E_BANDS))[hour] == df.trans_time_grp.values).all(), "E's bands not as documented"
    cat = src.category.values
    table = df[CAT_COLS].assign(category=cat).drop_duplicates()
    assert len(table) == 14 and table.category.is_unique, "category -> code is not a function"
    table = table.set_index("category")[CAT_COLS]
    cats = list(table.index)

    pred = [c for c in df.columns if c not in (TARGET, "trans_time")]
    t = df["trans_time"]
    c1, c2 = t.quantile(0.64), t.quantile(0.80)
    i_tr, i_va, i_te = (t <= c1).values, ((t > c1) & (t <= c2)).values, (t > c2).values
    tuned = pd.read_csv(RESULTS / tune_file).set_index("model")
    saved = pd.read_csv(OUT / f"test_predictions_{tag}.csv.gz")

    def evaluate(frame, arm, i, desc):
        tr, va, te = frame[i_tr], frame[i_va], frame[i_te]
        row = dict(arm=arm, draw=i, variant=desc)
        for m, fitter in MODELS.items():
            predict = fitter(json.loads(tuned.loc[m, "chosen_params"]), tr, va, pred)
            p_va, p_te = predict(va), predict(te)
            if arm == "E":
                worst = float(np.abs(p_te - saved[m].values).max())
                assert worst <= 1e-9, f"draw 0 does not reproduce the saved {m} predictions ({worst:.2e})"
            row[f"val_{m}"] = average_precision_score(va[TARGET].values, p_va)
            row[f"test_{m}"] = average_precision_score(te[TARGET].values, p_te)
        for s in ("val", "test"):
            row[f"{s}_mean"] = np.mean([row[f"{s}_{m}"] for m in MODELS])
        return row

    rows = []
    s = time.time()
    rows.append(evaluate(df, "E", 0, "E: 0-7 / 8-15 / 16-23; theory assignment"))
    log(f"E: test mean {rows[-1]['test_mean']:.4f} (guard passed, {time.time() - s:.0f}s)")
    for i in range(1, N_DRAWS + 1):
        cuts = draw_cuts(np.random.default_rng(1200 + i))
        d = df.copy()
        d["trans_time_grp"] = bands_of(cuts)[hour]
        rows.append(evaluate(d, "time", i, "bands starting at " + "/".join(f"{c}h" for c in cuts)))
        log(f"time {i}: {rows[-1]['variant']}: test mean {rows[-1]['test_mean']:.4f}")
    for i in range(1, N_DRAWS + 1):
        rng = np.random.default_rng(2200 + i)
        while True:
            perm = rng.permutation(len(cats))
            if (perm != np.arange(len(cats))).any():
                break
        mapping = {cats[k]: cats[perm[k]] for k in range(len(cats))}    # category k takes perm[k]'s codes
        d = df.copy()
        d[CAT_COLS] = table.loc[[mapping[c] for c in cat]].values
        moved = sum(cats[k] != cats[perm[k]] for k in range(len(cats)))
        rows.append(evaluate(d, "category", i, f"{moved} of 14 categories reassigned"))
        log(f"category {i}: {rows[-1]['variant']}: test mean {rows[-1]['test_mean']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "matched_compression_V01.csv", index=False)
    e = out.iloc[0]
    for arm in ("time", "category"):
        a = out[out.arm == arm]
        for col in ("test_mean", "test_Random Forest", "test_CatBoost", "val_mean"):
            rank = (np.r_[a[col].values, e[col]] <= e[col]).sum() / (len(a) + 1)
            print(f"{arm:8s} {col:20s} E {e[col]:.4f}  draws median {a[col].median():.4f} "
                  f"[{a[col].min():.4f}, {a[col].max():.4f}]  E's percentile rank {rank:.2f}")


if __name__ == "__main__":
    main()
