"""
Permutation importance with source fields as the primary unit, and the E+ category split
(Revision_Plan_V08.md, B3; Revision_Phase8_Plan_V01.md, C1b; review of CCF_MM_V03, item 4).

Why. Shuffling one derived column on its own can produce records the encoding cannot generate -- e.g.
Interaction 1 = 3 with necessity = 0 -- so a per-feature drop partly measures the model on impossible
inputs. Shuffling all of a source field's columns with one permutation keeps them mutually consistent
(each row receives another real row's complete set). It still breaks dependence BETWEEN source fields
(category and amount, say), so it is reliance under a marginal perturbation of the field, not a unique or
causal share.

For each configuration, random forest and CatBoost are refitted with the kept hyperparameters (guard:
test predictions equal ablation_run.py's saved ones to 1e-9) and scored on the full validation set:

  group     every source field's columns shuffled jointly, 10 repeats (mean, SD)
  feature   each column alone, 10 repeats -- reported as marginal reliance only (appendix); skipped for
            R's one-hot columns, which mean nothing alone
  E+ only   the category field split into its two blocks -- the 8 semantic attributes and the 14
            indicators -- each shuffled as a block; and tree-path SHAP on the balanced validation sample
            (1,000 + 1,000, seed 1200) summed per block, as a share of the category field's mean |SHAP|

  python explain_grouped.py --configs R,E        (E+ needs the Phase 8 tuning first)
  python explain_grouped.py --combine            -> result_csv/explain_grouped_V01.csv
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score

from ablation_run import CONFIGS, OUT, RESULTS, TARGET, RANDOM_STATE, resolve
from explain_compare import fit, group_of, proba, shap_values

REPEATS, N_BAL = 10, 1000
CFG = {"R": "p7_notime_raw", "E": "p7_notime", "E+": "p7_eplus", "R+H": "p7_notime_raw_client",
       "E+H": "p7_notime_client", "E+ +H": "p7_eplus_client"}
MODELS = ["Random Forest", "CatBoost"]
SEMANTIC = ["category_1", "category_2", "category_3", "category_4a", "category_4b", "category_4c",
            "category_interaction_1", "category_interaction_2"]


def run(key):
    cfg = CFG[key]
    df = pd.read_csv(resolve(CONFIGS[cfg][0]))
    pred = [c for c in df.columns if c not in (TARGET, "trans_time")]
    t = df["trans_time"]
    c1, c2 = t.quantile(0.64), t.quantile(0.80)
    train, valid, test = df[t <= c1], df[(t > c1) & (t <= c2)], df[t > c2]
    tuned = pd.read_csv(RESULTS / CONFIGS[cfg][1]).set_index("model")
    saved = pd.read_csv(OUT / f"test_predictions_{CONFIGS[cfg][2]}.csv.gz")
    X, yv = valid[pred], valid[TARGET].values
    is_raw = "raw" in cfg
    indicators = [c for c in pred if c.startswith("category_") and c not in SEMANTIC]
    units = [(g, [c for c in pred if group_of(c) == g], "group") for g in sorted({group_of(c) for c in pred})]
    if key.startswith("E+"):
        units += [("category: semantic attributes", [c for c in SEMANTIC if c in pred], "category block"),
                  ("category: indicators", indicators, "category block")]
    units += [(c, [c], "feature") for c in pred
              if not (is_raw and group_of(c) in ("category", "state")) and not (key.startswith("E+") and c in indicators)]
    rng_bal = np.random.default_rng(RANDOM_STATE)
    pos, neg = valid.index.values[yv == 1], valid.index.values[yv == 0]
    balanced = np.r_[rng_bal.choice(pos, N_BAL, replace=False), rng_bal.choice(neg, N_BAL, replace=False)]
    rows = []
    for m in MODELS:
        s = time.time()
        mod = fit(m, json.loads(tuned.loc[m, "chosen_params"]), train, pred)
        worst = float(np.abs(proba(mod, test[pred]) - saved[m].values).max())
        assert worst <= 1e-9, f"{cfg} {m}: refit does not reproduce the reported model ({worst:.2e})"
        base = average_precision_score(yv, proba(mod, X))
        prng = np.random.default_rng(RANDOM_STATE)
        for name, cols, kind in units:
            drops = []
            for _ in range(REPEATS):
                Xp = X.copy()
                Xp[cols] = X[cols].values[prng.permutation(len(X))]
                drops.append(base - average_precision_score(yv, proba(mod, Xp)))
            rows.append(dict(config=key, model=m, kind=kind, unit=name, n_columns=len(cols), base_aucpr=base,
                             aucpr_drop=float(np.mean(drops)), aucpr_drop_sd=float(np.std(drops, ddof=1)),
                             aucpr_drop_min=float(np.min(drops)), aucpr_drop_max=float(np.max(drops))))
        if key.startswith("E+"):
            v = shap_values(shap.TreeExplainer(mod), df.loc[balanced, pred], m)
            cat_total = np.abs(v[:, [pred.index(c) for c in pred if group_of(c) == "category"]].sum(axis=1)).mean()
            for name, cols in (("category: semantic attributes", SEMANTIC), ("category: indicators", indicators)):
                blk = np.abs(v[:, [pred.index(c) for c in cols]].sum(axis=1)).mean()
                rows.append(dict(config=key, model=m, kind="SHAP category block", unit=name, n_columns=len(cols),
                                 mean_abs_shap=float(blk), share_of_category=float(blk / cat_total),
                                 units="probability" if m == "Random Forest" else "log-odds"))
        print(f"{key} {m}: base {base:.4f}, guard {worst:.1e}, {time.time() - s:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"explain_grouped_{cfg}.csv", index=False)


if __name__ == "__main__":
    if "--combine" in sys.argv:
        parts = [OUT / f"explain_grouped_{CFG[k]}.csv" for k in CFG]
        out = pd.concat([pd.read_csv(p) for p in parts if p.exists()], ignore_index=True)
        out.to_csv(RESULTS / "explain_grouped_V01.csv", index=False)
        g = out[out.kind == "group"].pivot_table(index="unit", columns=["config", "model"], values="aucpr_drop")
        print(g.round(4).to_string())
    else:
        for k in sys.argv[sys.argv.index("--configs") + 1].split(","):
            run(k)
