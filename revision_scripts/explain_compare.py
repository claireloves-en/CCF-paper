"""
Raw-model against encoded-model explanations, on the same transactions (Revision_Phase7_Plan_V01.md
§10, 4A-4C; revision_plan_12_oct.md §4).

The paper's interpretability argument compared readable encoded explanations with raw identifiers.
The fair comparison is with a raw-feature model explained in the same format, with readable labels:
"Category = grocery (in store)", "Hour of day", amounts in USD. This fits random forest and CatBoost
on configurations 1 (R, p7_notime_raw) and 3 (E, p7_notime) with their kept V03 hyperparameters.
A guard checks the fits reproduce ablation_run.py's saved test predictions, so these are the models
the paper reports. It then explains them on identical validation transactions:

  samples (validation rows, identical for R and E; seed 1200)
    balanced    1,000 fraudulent + 1,000 genuine: the case-control comparison of the paper's figures
    natural     20,000 at random, natural prevalence (~0.56% fraud): what the importances mean for
                the transaction stream
    examples    12 by a fixed rule: the 3 highest- and 3 lowest-scored frauds and genuines of the
                balanced sample under E's CatBoost
  SHAP
    TreeExplainer, tree-path-dependent (the paper's setting): random forest in probability units,
    CatBoost in log-odds
    interventional sensitivity: 1,000 natural-prevalence TRAINING rows as background, on the first
    500 balanced rows
    grouped by source variable (category, time, amount, city, ...), so R and E are compared on the
    same footing
  permutation importance
    drop in validation AUC-PR when a feature (E) or a source group (R and E) is shuffled, 3 repeats:
    one consistently defined measure across models, replacing Table 5's average of unlike quantities

Outputs (output/, copied to result_csv/ as results):
  shap_summary_V01.csv      config, model, sample, explainer, feature, label, group, mean |SHAP|
  shap_grouped_V01.csv      the same by source group
  permutation_importance_V01.csv
  explain_examples_V01.csv  12 transactions x 2 configs x 2 models: score and top-5 contributions

  python explain_compare.py [--quick]   (--quick: 200 balanced, 2,000 natural, 1 repeat -- a code test)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from ablation_run import CONFIGS, RESULTS, TARGET, RANDOM_STATE, NO_JOBS
from data_paths import data_file

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
QUICK = "--quick" in sys.argv
N_BAL, N_NAT, REPEATS, N_INTERV, N_BACK = (100, 2000, 1, 100, 200) if QUICK else (1000, 20000, 3, 500, 1000)
CONFIGS_USED = {"R": "p7_notime_raw", "E": "p7_notime"}
MODELS = ["Random Forest", "CatBoost"]


def group_of(c):
    """The source variable a predictor comes from -- the same groups for R and E."""
    if c.startswith("card_"): return "client history"
    if c.startswith("category_"): return "category"
    if c in ("trans_time_grp", "hour_of_day", "day_of_week"): return "time"
    if c == "amt": return "amount"
    if c == "gender": return "gender"
    if c in ("cardholder_age", "age"): return "age"
    if c.startswith("city_") and c != "city_pop" or c == "city_ordinal": return "city"
    if c.startswith("state_"): return "state"
    if c == "city_pop": return "city population"
    if c.startswith("job_"): return "occupation"
    if c in ("lat", "long", "merch_lat", "merch_long", "loc_cluster_issue", "loc_cluster_transac", "issue_transac_dist"):
        return "location"
    raise KeyError(c)


def labels(columns, raw):
    """Readable labels from model_V26's cell 5 (one dictionary for the notebook and this script)."""
    cell = "".join(json.load(open(PATH / "MLProj01_fraudTrain_model_V26.ipynb"))["cells"][5]["source"])
    ns = {"RAW_BASELINE": raw, "df_enc": pd.DataFrame(columns=columns)}
    exec(cell, ns)
    return ns["column_names_dict"]


def fit(model, params, train, predictors):
    """As ablation_run.py's fit_forest / fit_cat, returning the model object."""
    if model == "Random Forest":
        p = {k: (None if k == "max_depth" and v == 0 else v) for k, v in params.items()}
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=NO_JOBS, verbose=False, **p).fit(
            train[predictors], train[TARGET].values)
    m = CatBoostClassifier(**params, eval_metric="AUC", random_seed=RANDOM_STATE, od_type="Iter", od_wait=100, verbose=False)
    return m.fit(train[predictors], train[TARGET].values, verbose=False)


def proba(m, X):
    return m.predict_proba(X)[:, 1]


def shap_values(explainer, X, model):
    v = explainer.shap_values(X)
    v = np.asarray(v)
    if v.ndim == 3:                      # sklearn trees: (rows, features, classes) -> fraud class
        v = v[:, :, 1]
    return v


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    data, splits = {}, {}
    for key, cfg in CONFIGS_USED.items():
        df = pd.read_csv(data_file(CONFIGS[cfg][0]))
        t = df["trans_time"]; c1, c2 = t.quantile(0.64), t.quantile(0.80)
        data[key] = df
        splits[key] = (df[t <= c1], df[(t > c1) & (t <= c2)], df[t > c2])
    valid_idx = splits["E"][1].index.values
    assert (splits["R"][1].index.values == valid_idx).all(), "R and E validation rows differ"
    y_valid = splits["E"][1][TARGET].values
    pos, neg = valid_idx[y_valid == 1], valid_idx[y_valid == 0]
    balanced = np.r_[rng.choice(pos, N_BAL, replace=False), rng.choice(neg, N_BAL, replace=False)]
    natural = rng.choice(valid_idx, N_NAT, replace=False)
    background = rng.choice(splits["E"][0].index.values, N_BACK, replace=False)
    print(f"balanced {len(balanced)}, natural {len(natural)} ({data['E'].loc[natural, TARGET].mean():.4%} fraud), "
          f"background {len(background)} training rows")

    models, preds = {}, {}
    for key, cfg in CONFIGS_USED.items():
        train, valid, test = splits[key]
        predictors = [c for c in data[key].columns if c not in (TARGET, "trans_time")]
        tuned = pd.read_csv(RESULTS / CONFIGS[cfg][1]).set_index("model")
        saved = pd.read_csv(OUT / f"test_predictions_{CONFIGS[cfg][2]}.csv.gz")
        for m in MODELS:
            s = time.time()
            mod = fit(m, json.loads(tuned.loc[m, "chosen_params"]), train, predictors)
            worst = float(np.abs(proba(mod, test[predictors]) - saved[m].values).max())
            assert worst <= 1e-9, f"{cfg} {m}: refit does not reproduce the reported model ({worst:.2e})"
            models[(key, m)] = (mod, predictors)
            preds[(key, m)] = proba(mod, data[key].loc[balanced, predictors])
            print(f"{key} {m}: fitted, guard {worst:.1e} ({time.time() - s:.0f}s)")

    # the 12 examples, by rule, under E's CatBoost
    sc = pd.Series(preds[("E", "CatBoost")], index=balanced)
    yb = data["E"].loc[balanced, TARGET]
    ex = []
    for cls in (1, 0):
        s = sc[yb == cls].sort_values()
        ex += list(s.index[-3:][::-1]) + list(s.index[:3])
    examples = np.array(ex)

    rows, grows, ex_rows, perm_rows = [], [], [], []

    def grouped(v, predictors, **keys):
        """Exact grouped SHAP: per row, the sum of a group's values; then mean |.|."""
        for gname in sorted({group_of(c) for c in predictors}):
            cols = [j for j, c in enumerate(predictors) if group_of(c) == gname]
            gv = v[:, cols].sum(axis=1)
            grows.append(dict(**keys, group=gname, n_features=len(cols),
                              mean_abs_shap=float(np.abs(gv).mean()), mean_shap=float(gv.mean())))
    for (key, m), (mod, predictors) in models.items():
        lab = labels(list(data[key].columns), raw=(key == "R"))
        units = "probability" if m == "Random Forest" else "log-odds"
        pd_exp = shap.TreeExplainer(mod)
        for sample_name, idx in (("balanced case-control", balanced), ("natural prevalence", natural)):
            s = time.time()
            v = shap_values(pd_exp, data[key].loc[idx, predictors], m)
            for j, c in enumerate(predictors):
                rows.append(dict(config=key, model=m, sample=sample_name, explainer="tree-path-dependent",
                                 units=units, feature=c, label=lab[c], group=group_of(c),
                                 mean_abs_shap=float(np.abs(v[:, j]).mean()), mean_shap=float(v[:, j].mean())))
            grouped(v, predictors, config=key, model=m, sample=sample_name, explainer="tree-path-dependent", units=units)
            if sample_name == "balanced case-control":
                pos_of = {r: i for i, r in enumerate(idx)}
                for r in examples:
                    contrib = sorted(zip(predictors, v[pos_of[r]]), key=lambda kv: -abs(kv[1]))[:5]
                    ex_rows.append(dict(row=int(r), is_fraud=int(data[key].loc[r, TARGET]), config=key, model=m,
                                        score=float(proba(mod, data[key].loc[[r], predictors])[0]), units=units,
                                        **{f"top{k + 1}": f"{lab[c]} = {data[key].loc[r, c]:g}: {val:+.4f}"
                                           for k, (c, val) in enumerate(contrib)}))
            print(f"{key} {m} SHAP {sample_name}: {time.time() - s:.0f}s")
        # interventional sensitivity
        s = time.time()
        masker = shap.maskers.Independent(data[key].loc[background, predictors], max_samples=N_BACK)
        iv_exp = shap.TreeExplainer(mod, data=masker, feature_perturbation="interventional")
        v = shap_values(iv_exp, data[key].loc[balanced[:N_INTERV], predictors], m)
        for j, c in enumerate(predictors):
            rows.append(dict(config=key, model=m, sample=f"balanced, first {N_INTERV}",
                             explainer=f"interventional ({N_BACK} training rows)", units=units, feature=c,
                             label=lab[c], group=group_of(c), mean_abs_shap=float(np.abs(v[:, j]).mean()),
                             mean_shap=float(v[:, j].mean())))
        grouped(v, predictors, config=key, model=m, sample=f"balanced, first {N_INTERV}",
                explainer=f"interventional ({N_BACK} training rows)", units=units)
        print(f"{key} {m} SHAP interventional: {time.time() - s:.0f}s")

        # permutation importance on the full validation set: per feature, and per source group
        valid = splits[key][1]
        X, yv = valid[predictors].copy(), valid[TARGET].values
        base = average_precision_score(yv, proba(mod, X))
        units_perm = [(c, [c], "feature") for c in predictors] + \
                     [(g, [c for c in predictors if group_of(c) == g], "group")
                      for g in sorted({group_of(c) for c in predictors})]
        prng = np.random.default_rng(RANDOM_STATE)
        s = time.time()
        for name, cols, kind in units_perm:
            if kind == "feature" and key == "R" and group_of(name) in ("category", "state"):
                continue                     # one-hot columns: only meaningful as a group
            drops = []
            for _ in range(REPEATS):
                Xp = X.copy()
                perm = prng.permutation(len(Xp))
                Xp[cols] = X[cols].values[perm]          # shuffle the group's columns together
                drops.append(base - average_precision_score(yv, proba(mod, Xp)))
            perm_rows.append(dict(config=key, model=m, kind=kind, unit=name,
                                  label=lab.get(name, name), aucpr_drop=float(np.mean(drops)),
                                  aucpr_drop_sd=float(np.std(drops)), base_aucpr=base))
        print(f"{key} {m} permutation: {time.time() - s:.0f}s")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "shap_summary_V01.csv", index=False)
    pd.DataFrame(grows).to_csv(OUT / "shap_grouped_V01.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(OUT / "permutation_importance_V01.csv", index=False)
    pd.DataFrame(ex_rows).to_csv(OUT / "explain_examples_V01.csv", index=False)
    pd.set_option("display.width", 220)
    p = pd.DataFrame(perm_rows)
    print(p[p.kind == "group"].pivot_table(index="unit", columns=["config", "model"], values="aucpr_drop").round(4).to_string())


if __name__ == "__main__":
    main()
