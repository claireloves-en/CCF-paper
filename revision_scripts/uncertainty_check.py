"""
Paired bootstrap intervals for the test-set metrics. (J4, Revision_Phase6_Plan_V01.md §7.)

The paper calls four ensembles "practically tied" -- they sit within 0.010 test AUC-PR of each
other. The assessment asks for confidence intervals or repeated evaluation to support claims at
that resolution. This puts intervals on each model's metrics AND on the differences that matter:
model minus model within a configuration, and client-features minus base within a model.

Method: stratified bootstrap of the test set, N_BOOT resamples, resampling fraudulent and
genuine transactions separately so every resample holds the observed class counts. The SAME
resample is applied to every model and configuration, which is what gives the differences
intervals -- an unpaired bootstrap would leave the shared test-set variation in them.

This does NOT re-touch the test set. It resamples the FROZEN per-row probabilities that
ablation_run.py saved, each model keeping the threshold it set on the validation set, so the
one-time test evaluation stands as it is.

A resample is applied as multinomial weights rather than row indices, and AUC-PR is computed
from per-distinct-score group sums, so no sort is repeated inside the loop. The weighted
estimator is checked against sklearn at unit weights before the loop runs.

Usage: python uncertainty_check.py [--configs notime,notime_client] [--boot 1000]
Writes output/test_bootstrap_V01.csv          per model and configuration, with 95% intervals
       output/test_bootstrap_diffs_V01.csv    differences, with 95% intervals

Phase 7 (Revision_Phase7_Plan_V01.md §§6-7) adds --scheme, which selects the resampling unit and
writes output/test_bootstrap{,_diffs}_V02_<scheme>.csv:
  stratified  as above (the primary scheme)
  card        the cards present in the test window, resampled with replacement; each card's
              transactions stay together (within-cardholder dependence); not class-stratified
  timeblock   circular moving-block bootstrap over the window's calendar days, 7-day blocks
              (short-range temporal dependence)
It also scores the fixed-capacity operating point (operating_points.capacity_flags, 0.5% per day)
and, for the p7 configurations, the two primary hypotheses -- ensemble-mean encoded minus raw,
without and with client history -- with two-sided bootstrap p-values, Holm-adjusted over the two.
  python uncertainty_check.py --scheme card --configs p7_notime_raw,p7_notime_raw_client,p7_notime,p7_notime_client
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ablation_run import CONFIGS, MODEL_ORDER, TARGET
from operating_points import capacity_flags, test_rows

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
RANDOM_STATE = 1200
ENSEMBLES = ["Random Forest", "CatBoost", "XGBoost", "LightGBM"]
# (label, configuration a, configuration b) -> interval on a's AUC-PR minus b's, per model.
# The first two are the J1 ablation's headline differences; the last two the client features'.
CONFIG_PAIRS = [
    ("3 - 1  encoding, without client history", "notime", "notime_raw"),
    ("4 - 2  encoding, with client history", "notime_client", "notime_raw_client"),
    ("2 - 1  client history, on raw features", "notime_raw_client", "notime_raw"),
    ("4 - 3  client history, on encoded features", "notime_client", "notime"),
]
# Phase 7: the same four on the corrected data, then the secondary families of the frozen plan §6
P7_PAIRS = [
    ("P1  3 - 1  encoding, without client history", "p7_notime", "p7_notime_raw"),
    ("P2  4 - 2  encoding, with client history", "p7_notime_client", "p7_notime_raw_client"),
    ("2 - 1  client history, on raw features", "p7_notime_raw_client", "p7_notime_raw"),
    ("4 - 3  client history, on encoded features", "p7_notime_client", "p7_notime"),
    ("hybrid - raw", "p7_hybrid", "p7_notime_raw"),
    ("hybrid + client - raw + client", "p7_hybrid_client", "p7_notime_raw_client"),
    ("catnative - raw", "p7_catnative", "p7_notime_raw"),
    ("catnative + client - raw + client", "p7_catnative_client", "p7_notime_raw_client"),
    ("reduced: encoded - raw", "p7_red_enc", "p7_red_raw"),
] + [(f"ablation {a} - E", f"p7_{a}", "p7_notime") for a in (
    "abl_cat_onehot", "abl_plus_onehot", "abl_hour", "abl_plus_dow", "abl_no_interactions",
    "abl_no_external", "abl_no_geo_occ")]
PRIMARY = P7_PAIRS[:2]


def holm(p):
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def boot_p(d):
    """Two-sided bootstrap p-value for a difference: twice the smaller tail share at zero."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if d.size else np.nan


def at_flags(flags, y, w):
    """Weighted recall and precision of a fixed set of alerts."""
    tp = float(np.sum(w[flags & (y == 1)]))
    fp = float(np.sum(w[flags & (y == 0)]))
    pos = float(np.sum(w[y == 1]))
    return (tp / pos if pos else np.nan, tp / (tp + fp) if tp + fp else 0.0)


class APScorer:
    """Weighted average precision for one model, with the sort done once."""

    def __init__(self, scores, y):
        order = np.argsort(-scores, kind="mergesort")
        self.order = order
        s = scores[order]
        # one group per distinct score: AP steps at distinct thresholds, so ties must be
        # pooled exactly as sklearn pools them
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        self.y = y[order].astype(float)

    def __call__(self, w):
        wo = w[self.order]
        wy = wo * self.y
        tp = np.add.reduceat(wy, self.starts).cumsum()
        n = np.add.reduceat(wo, self.starts).cumsum()
        total_pos = tp[-1]
        if total_pos <= 0:
            return np.nan
        # Groups that a resample leaves with zero weight hold no transactions and add nothing.
        # (Before 2026-10-12 this was tp / n, which is 0/0 = NaN whenever the top-scored group
        # had zero weight -- about a third of the resamples for models whose top score is unique,
        # such as CatBoost and XGBoost. ci() dropped those draws silently, so the Phase 6
        # intervals rest on the remaining two thirds. Revision_Phase7_Results_V01.md.)
        precision = np.divide(tp, n, out=np.zeros_like(tp), where=n > 0)
        recall = tp / total_pos
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def at_threshold(scores, y, thr, w):
    """Weighted recall, precision and alert rate at a fixed threshold."""
    a = scores >= thr
    tp = float(np.sum(w[a & (y == 1)]))
    fp = float(np.sum(w[a & (y == 0)]))
    pos = float(np.sum(w[y == 1]))
    tot = float(np.sum(w))
    return (tp / pos if pos else np.nan,
            tp / (tp + fp) if tp + fp else 0.0,
            (tp + fp) / tot if tot else np.nan)


def ci(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a.size else (np.nan, np.nan)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--configs", default="notime_raw,notime_raw_client,notime,notime_client")
    ap_.add_argument("--boot", type=int, default=1000)
    ap_.add_argument("--scheme", choices=["stratified", "card", "timeblock"], default=None,
                     help="Phase 7: resampling unit; omit for the Phase 6 behaviour and file names")
    ap_.add_argument("--block-days", type=int, default=7)
    ap_.add_argument("--period", choices=["test", "confirm"], default="test",
                     help="confirm: the fraudTest predictions of confirm_run.py (plan §9.4)")
    args = ap_.parse_args()
    phase7 = args.scheme is not None
    scheme = args.scheme or "stratified"
    suffix = f"V02_{scheme}" if phase7 else "V01"
    confirm = args.period == "confirm"
    if confirm:
        suffix = f"V03_confirm_{scheme}"

    loaded = {}
    for cfg in args.configs.split(","):
        _d, _t, tag = CONFIGS[cfg]
        pred_path = OUT / (f"confirm_predictions_{tag}.csv.gz" if confirm else f"test_predictions_{tag}.csv.gz")
        if not pred_path.exists():
            print(f"SKIP {cfg}: {pred_path.name} not found")
            continue
        preds = pd.read_csv(pred_path)
        thr = pd.read_csv(OUT / f"test_operating_point_V01_{tag}.csv").set_index("metric\\model")
        loaded[cfg] = (preds, thr.loc["Operating threshold (from validation)"])
    if not loaded:
        raise SystemExit("no predictions found -- run ablation_run.py first")

    y = next(iter(loaded.values()))[0][TARGET].values
    for cfg, (preds, _t) in loaded.items():
        assert (preds[TARGET].values == y).all(), f"{cfg} has a different test label vector"
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n = len(y)

    # the resampling units: cards and calendar days of the test rows
    first = next(iter(loaded))
    rows = loaded[first][0]["row"].values if confirm else test_rows(first, loaded[first][0])
    for cfg, (preds, _t) in loaded.items():
        assert ((preds["row"].values if confirm else test_rows(cfg, preds)) == rows).all(), f"{cfg}: rows differ"
    meta_file = (Path(__file__).resolve().parent.parent.parent.parent / "Found_DataSet" / "fraudTest.csv" if confirm
                 else Path(__file__).resolve().parent.parent / "fraudTrain.csv")
    meta = pd.read_csv(meta_file,
                       usecols=["cc_num", "trans_date_trans_time", "unix_time"])
    card = pd.factorize(meta["cc_num"].values[rows])[0]
    day_key = pd.to_datetime(meta["trans_date_trans_time"].values[rows]).date
    day = pd.factorize(pd.Series(day_key).astype(str).values, sort=True)[0]
    chrono = meta["unix_time"].values[rows]
    n_card, n_day = card.max() + 1, day.max() + 1
    print(f"scheme {scheme}: {n_card} cards, {n_day} days in the test window")
    print(f"test set {n:,} rows, {len(pos)} fraud ({y.mean():.5f}); "
          f"{len(loaded)} configurations, {args.boot} resamples")

    # build the scorers and check the weighted estimator against sklearn at unit weights
    scorers, models = {}, {}
    ones = np.ones(n)
    for cfg, (preds, thr) in loaded.items():
        for m in MODEL_ORDER:
            if m not in preds.columns:
                continue
            s = preds[m].values.astype(float)
            sc = APScorer(s, y)
            ref = average_precision_score(y, s)
            got = sc(ones)
            assert abs(got - ref) < 1e-12, f"{cfg}/{m}: weighted AP {got} vs sklearn {ref}"
            scorers[(cfg, m)] = sc
            models[(cfg, m)] = (s, float(thr[m]), capacity_flags(s, day, chrono))
    print(f"weighted AUC-PR matches sklearn for all {len(scorers)} model-configuration pairs "
          f"(max |diff| < 1e-12)")

    keys = list(scorers)
    point = {k: dict(aucpr=scorers[k](ones)) for k in keys}
    for k in keys:
        s, t, cap = models[k]
        r, p, a = at_threshold(s, y, t, ones)
        point[k].update(recall=r, precision=p, alert_rate=a)
        if phase7:
            rc, pc = at_flags(cap, y, ones)
            point[k].update(cap_recall=rc, cap_precision=pc)

    METRICS = ("aucpr", "recall", "precision") + (("cap_recall", "cap_precision") if phase7 else ())
    draws = {k: {m: [] for m in METRICS} for k in keys}
    rng = np.random.default_rng(RANDOM_STATE)

    def weights():
        if scheme == "stratified":
            w = np.zeros(n)
            w[pos] = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)))
            w[neg] = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)))
            return w
        if scheme == "card":
            counts = rng.multinomial(n_card, np.full(n_card, 1 / n_card))
            return counts[card].astype(float)
        # timeblock: circular blocks of block_days consecutive days until the window is covered
        L = args.block_days
        mult = np.zeros(n_day)
        for start in rng.integers(0, n_day, size=int(np.ceil(n_day / L))):
            mult[(start + np.arange(L)) % n_day] += 1
        return mult[day]

    for b in range(args.boot):
        w = weights()
        for k in keys:                       # the same resample for every model: paired
            s, t, cap = models[k]
            draws[k]["aucpr"].append(scorers[k](w))
            r, p, _a = at_threshold(s, y, t, w)
            draws[k]["recall"].append(r)
            draws[k]["precision"].append(p)
            if phase7:
                rc, pc = at_flags(cap, y, w)
                draws[k]["cap_recall"].append(rc)
                draws[k]["cap_precision"].append(pc)
        if (b + 1) % 200 == 0:
            print(f"  {b + 1}/{args.boot}", flush=True)

    rows = []
    for (cfg, m) in keys:
        r = dict(config=cfg, model=m, **{f"{k}": v for k, v in point[(cfg, m)].items()})
        for met in METRICS:
            lo, hi = ci(draws[(cfg, m)][met])
            r[f"{met}_lo"], r[f"{met}_hi"] = lo, hi
            r[f"{met}_width"] = hi - lo
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"test_bootstrap_{suffix}.csv", index=False)

    # differences, from the paired draws
    diffs = []
    for cfg in loaded:
        present = [m for m in ENSEMBLES if (cfg, m) in draws]
        for i, a in enumerate(present):
            for bm in present[i + 1:]:
                d = np.array(draws[(cfg, a)]["aucpr"]) - np.array(draws[(cfg, bm)]["aucpr"])
                lo, hi = ci(d)
                diffs.append(dict(kind="model vs model", config=cfg, a=a, b=bm,
                                  diff=point[(cfg, a)]["aucpr"] - point[(cfg, bm)]["aucpr"],
                                  lo=lo, hi=hi, separated=bool(lo > 0 or hi < 0)))
    # the J1 ablation differences, and the client-feature differences, per model
    pairs = P7_PAIRS if phase7 else CONFIG_PAIRS
    for label, ca, cb in pairs:
        for m in MODEL_ORDER:
            if (ca, m) in draws and (cb, m) in draws:
                d = np.array(draws[(ca, m)]["aucpr"]) - np.array(draws[(cb, m)]["aucpr"])
                lo, hi = ci(d)
                diffs.append(dict(kind=label, config=f"{ca} - {cb}", a=m, b="",
                                  diff=point[(ca, m)]["aucpr"] - point[(cb, m)]["aucpr"],
                                  lo=lo, hi=hi, separated=bool(lo > 0 or hi < 0),
                                  p=boot_p(d) if phase7 else np.nan))
    if phase7:
        # Holm within each secondary family (frozen plan §6), then the two primary hypotheses
        for label, _a, _b in pairs:
            idx = [i for i, r in enumerate(diffs) if r["kind"] == label and r["b"] == ""]
            if idx:
                for i, q in zip(idx, holm([diffs[i]["p"] for i in idx])):
                    diffs[i]["p_holm"] = q
        prim = []
        for label, ca, cb in PRIMARY:
            if all((c, m) in draws for c in (ca, cb) for m in ENSEMBLES):
                da = np.mean([draws[(ca, m)]["aucpr"] for m in ENSEMBLES], axis=0)
                db = np.mean([draws[(cb, m)]["aucpr"] for m in ENSEMBLES], axis=0)
                d = da - db
                lo, hi = ci(d)
                pt = (np.mean([point[(ca, m)]["aucpr"] for m in ENSEMBLES])
                      - np.mean([point[(cb, m)]["aucpr"] for m in ENSEMBLES]))
                prim.append(dict(kind="PRIMARY " + label, config=f"{ca} - {cb}", a="ensemble mean (RF, CatBoost, XGBoost, LightGBM)",
                                 b="", diff=pt, lo=lo, hi=hi, separated=bool(lo > 0 or hi < 0), p=boot_p(d)))
        for r, q in zip(prim, holm([r["p"] for r in prim]) if prim else []):
            r["p_holm"] = q
        diffs = prim + diffs
    dd = pd.DataFrame(diffs)
    dd.to_csv(OUT / f"test_bootstrap_diffs_{suffix}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nAUC-PR with 95% intervals")
    print(res[["config", "model", "aucpr", "aucpr_lo", "aucpr_hi", "aucpr_width"]].round(4).to_string(index=False))
    print("\nensemble pairs: is the AUC-PR difference separated from zero?")
    print(dd[dd.kind == "model vs model"][["config", "a", "b", "diff", "lo", "hi", "separated"]].round(4).to_string(index=False))
    if phase7:
        print("\nprimary hypotheses (Holm over the two)")
        print(dd[dd.kind.str.startswith("PRIMARY")][["kind", "diff", "lo", "hi", "p", "p_holm"]].round(4).to_string(index=False))
    for label, _a, _b in pairs:
        sub = dd[dd.kind == label]
        if len(sub):
            print(f"\n{label}")
            print(sub[["a", "diff", "lo", "hi", "separated"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
