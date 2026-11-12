"""
k-NN with standardised features: a sensitivity analysis for the run of record (Revision_Plan_V08.md, B2).

The review of CCF_MM_V03 (item 7) noted that no scaling is stated for k-NN. None was applied:
ablation_run.py fits KNeighborsClassifier on the raw columns, so its Euclidean distance is
dominated by the widest-ranged ones (city population, amount, coordinates). The paper's
explanation of k-NN's gain from the encoding ("twenty-seven dense ordinal features give a more
meaningful distance") cannot be judged without knowing what the scaled model does.

This refits k-NN on R, R+H, E and E+H with a StandardScaler fitted on the training split, keeps
distance weighting and Euclidean distance (the kept settings), and lets the validation set choose
k from {5, 11, 25, 50, 100} -- the unscaled search chose k = 50 for all four, but on a different
geometry. One neighbour query at k = 100 serves every smaller k. The test period is scored with
the chosen k; fraudTest is not touched.

Reported beside the run of record (saved predictions, output/test_predictions_temporal_p7_*):
test AUC-PR per configuration, and the encoding difference E - R and (E+H) - (R+H) with a paired
stratified bootstrap (1,000 draws, uncertainty_check.py's estimator and seed).

Usage: python knn_scaled.py
Writes result_csv/knn_scaled_V01.csv
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ablation_run import CONFIGS, OUT, RESULTS, TARGET, resolve
from uncertainty_check import APScorer, boot_p, ci, holm

KS = [5, 11, 25, 50, 100]
N_BOOT, SEED = 1000, 1200
CFGS = ["p7_notime_raw", "p7_notime_raw_client", "p7_notime", "p7_notime_client"]
LABEL = {"p7_notime_raw": "R", "p7_notime_raw_client": "R+H", "p7_notime": "E", "p7_notime_client": "E+H"}


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def knn_proba(dist, idx, y_train, k):
    """Distance-weighted fraud probability from the first k neighbours (sklearn's 'distance' rule)."""
    d, yi = dist[:, :k], y_train[idx[:, :k]]
    with np.errstate(divide="ignore"):
        w = 1.0 / d
    exact = np.isinf(w)
    rows = exact.any(axis=1)                      # an exact match takes all the weight, as in sklearn
    w[rows] = exact[rows].astype(float)
    return (w * yi).sum(axis=1) / w.sum(axis=1)


def main():
    rows, test_scores, y_test, rec = [], {}, None, {}
    for cfg in CFGS:
        s = time.time()
        df = pd.read_csv(resolve(CONFIGS[cfg][0]))
        pred = [c for c in df.columns if c not in (TARGET, "trans_time")]
        t = df["trans_time"]
        c1, c2 = t.quantile(0.64), t.quantile(0.80)
        tr, va, te = df[t <= c1], df[(t > c1) & (t <= c2)], df[t > c2]
        sc = StandardScaler().fit(tr[pred])
        nn = NearestNeighbors(n_neighbors=max(KS), n_jobs=-1).fit(sc.transform(tr[pred]))
        ytr = tr[TARGET].values
        dv, iv = nn.kneighbors(sc.transform(va[pred]))
        val = {k: average_precision_score(va[TARGET].values, knn_proba(dv, iv, ytr, k)) for k in KS}
        k_best = max(val, key=val.get)
        dt, it = nn.kneighbors(sc.transform(te[pred]), n_neighbors=k_best)
        p = knn_proba(dt, it, ytr, k_best)
        y = te[TARGET].values
        y_test = y if y_test is None else y_test
        assert np.array_equal(y, y_test)
        saved = pd.read_csv(OUT / f"test_predictions_{CONFIGS[cfg][2]}.csv.gz")
        assert np.array_equal(saved[TARGET].values, y)
        test_scores[cfg], rec[cfg] = p, saved["K-NN"].values
        rows.append(dict(config=LABEL[cfg], predictors=len(pred), k_chosen=k_best,
                         **{f"val_aucpr_k{k}": v for k, v in val.items()},
                         test_aucpr_scaled=average_precision_score(y, p),
                         test_aucpr_of_record=average_precision_score(y, rec[cfg])))
        log(f"{LABEL[cfg]}: k={k_best}, val {val[k_best]:.4f}, test scaled {rows[-1]['test_aucpr_scaled']:.4f} "
            f"vs of record {rows[-1]['test_aucpr_of_record']:.4f} ({time.time() - s:.0f}s)")

    # paired stratified bootstrap of the encoding differences, scaled and of record
    pos, neg = np.flatnonzero(y_test == 1), np.flatnonzero(y_test == 0)
    rng = np.random.default_rng(SEED)
    sc_s = {c: APScorer(v, y_test) for c, v in test_scores.items()}
    sc_r = {c: APScorer(v, y_test) for c, v in rec.items()}
    pairs = [("E - R", "p7_notime", "p7_notime_raw"), ("(E+H) - (R+H)", "p7_notime_client", "p7_notime_raw_client")]
    draws = {(lab, kind): [] for lab, _, _ in pairs for kind in ("scaled", "of_record")}
    for _ in range(N_BOOT):
        w = np.zeros(len(y_test))
        w[pos] = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)))
        w[neg] = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)))
        for lab, a, b in pairs:
            draws[(lab, "scaled")].append(sc_s[a](w) - sc_s[b](w))
            draws[(lab, "of_record")].append(sc_r[a](w) - sc_r[b](w))
    diff_rows = []
    for (lab, kind), d in draws.items():
        a, b = [x for x in pairs if x[0] == lab][0][1:]
        src = test_scores if kind == "scaled" else rec
        lo, hi = ci(d)
        diff_rows.append(dict(config=f"difference {lab}", kind=kind,
                              diff=average_precision_score(y_test, src[a]) - average_precision_score(y_test, src[b]),
                              lo=lo, hi=hi, p=boot_p(d)))
    for kind in ("scaled", "of_record"):
        idx = [i for i, r in enumerate(diff_rows) if r["kind"] == kind]
        for i, q in zip(idx, holm([diff_rows[i]["p"] for i in idx])):
            diff_rows[i]["p_holm"] = q
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(diff_rows)], ignore_index=True)
    out.to_csv(RESULTS / "knn_scaled_V01.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
