"""
Paired bootstrap for Phase 8 C1 (Revision_Phase8_Plan_V01.md, frozen at 10ceada): E+ against R and E, with and
without client history, for random forest and CatBoost and their mean.

Fixed test predictions (ablation_run.py's output/test_predictions_temporal_<config>.csv.gz), 1,000 resamples
applied in common to every model and configuration, seed 1200; uncertainty_check.py's weighted AP estimator.
Schemes: stratified (primary) and card (whole cardholders resampled).

Families (Holm within each):
  C1-primary    E+ - R and (E+ + H) - (R + H), two-model mean
  C1-secondary  E+ - E and (E+ + H) - (E + H), two-model mean; and all four contrasts per model

Usage: python phase8_compare.py
Writes result_csv/phase8_bootstrap_diffs_V01_{stratified,card}.csv and phase8_test_aucpr_V01.csv
"""
import numpy as np
import pandas as pd

from ablation_run import CONFIGS, OUT, RESULTS, TARGET
from uncertainty_check import APScorer, boot_p, ci, holm

MODELS = ["Random Forest", "CatBoost"]
CFG = {"R": "p7_notime_raw", "R+H": "p7_notime_raw_client", "E": "p7_notime", "E+H": "p7_notime_client",
       "E+": "p7_eplus", "E+ +H": "p7_eplus_client"}
CONTRASTS = [("E+ - R", "E+", "R"), ("(E+ +H) - (R+H)", "E+ +H", "R+H"),
             ("E+ - E", "E+", "E"), ("(E+ +H) - (E+H)", "E+ +H", "E+H")]
N_BOOT, SEED = 1000, 1200

pred = {}
y = rows = None
for k, cfg in CFG.items():
    d = pd.read_csv(OUT / f"test_predictions_{CONFIGS[cfg][2]}.csv.gz")
    if y is None:
        y, rows = d[TARGET].values, d["row"].values
    assert np.array_equal(d[TARGET].values, y) and np.array_equal(d["row"].values, rows)
    for m in MODELS:
        pred[(k, m)] = d[m].values
card = pd.read_csv(RESULTS.parent / "fraudTrain.csv", usecols=["cc_num"]).cc_num.values[rows]
_, card_idx = np.unique(card, return_inverse=True)
n_card = card_idx.max() + 1

point = {key: APScorer(v, y)(np.ones(len(y))) for key, v in pred.items()}
pd.DataFrame([dict(config=k, model=m, test_aucpr=v) for (k, m), v in point.items()]).to_csv(
    RESULTS / "phase8_test_aucpr_V01.csv", index=False)
scorers = {key: APScorer(v, y) for key, v in pred.items()}
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

for scheme in ("stratified", "card"):
    rng = np.random.default_rng(SEED)
    draws = {key: [] for key in pred}
    for _ in range(N_BOOT):
        if scheme == "stratified":
            w = np.zeros(len(y))
            w[pos] = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)))
            w[neg] = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)))
        else:
            w = rng.multinomial(n_card, np.full(n_card, 1 / n_card))[card_idx].astype(float)
        for key, s in scorers.items():
            draws[key].append(s(w))
    draws = {k: np.array(v) for k, v in draws.items()}
    out = []
    for lab, a, b in CONTRASTS:
        fam = "C1-primary" if "R" in b else "C1-secondary"
        d = np.mean([draws[(a, m)] for m in MODELS], axis=0) - np.mean([draws[(b, m)] for m in MODELS], axis=0)
        lo, hi = ci(d)
        out.append(dict(family=fam, contrast=lab, model="mean (RF, CatBoost)",
                        diff=np.mean([point[(a, m)] for m in MODELS]) - np.mean([point[(b, m)] for m in MODELS]),
                        lo=lo, hi=hi, p=boot_p(d)))
        for m in MODELS:
            d = draws[(a, m)] - draws[(b, m)]
            lo, hi = ci(d)
            out.append(dict(family="C1-secondary", contrast=lab, model=m, diff=point[(a, m)] - point[(b, m)],
                            lo=lo, hi=hi, p=boot_p(d)))
    out = pd.DataFrame(out)
    for fam in out.family.unique():
        i = out.family == fam
        out.loc[i, "p_holm"] = holm(out.loc[i, "p"].values)
    out.to_csv(RESULTS / f"phase8_bootstrap_diffs_V01_{scheme}.csv", index=False)
    print(f"\n{scheme}\n" + out.round(4).to_string(index=False))
