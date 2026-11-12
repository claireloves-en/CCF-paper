"""
LIME on the source variables, with derived features regenerated (Revision_Phase7_Plan_V01.md §10, 4B;
revision_plan_12_oct.md §4B).

The paper's LIME perturbs each encoded feature independently. `category_1..interaction_2` are 8
functions of one field, so independent perturbation produces category profiles that exist in no
category, and LIME then shares the field's effect among them in ways that mean little. The paper
already concedes this. Here LIME perturbs the ORIGINAL fields, and every model feature is
regenerated from them through the feature dictionary, so every perturbed transaction is a valid one:

  source variable                 -> E (encoded) features                    -> R (raw) features
  category                        -> category_1..interaction_2               -> one-hot category
  city (with its state, home location, population)
                                  -> city_1..4b, state_x/y, city_pop,        -> lat, long, city_pop,
                                     loc_cluster_issue, issue_transac_dist      one-hot state, city_ordinal
  job                             -> job_1..4                                -> job_ordinal
  hour of day                     -> trans_time_grp                          -> hour_of_day
  day of week                     -> (not in E)                              -> day_of_week
  amount, gender                  -> amt, gender                             -> amt, gender
  age                             -> cardholder_age (whole years)            -> age
  merchant location               -> loc_cluster_transac, issue_transac_dist -> merch_lat, merch_long

A perturbed city carries the home location and population of that city's first training
transaction; the explained transaction keeps its own. Both R and E are therefore explained in the
same eleven source variables, so their explanations can be compared directly.

Reported, per configuration (R, E) and model (random forest, CatBoost):
  - mean LIME weight per source variable over the first 500 of explain_compare.py's balanced sample;
  - local fidelity: LIME's own R^2 of the local surrogate (`score`);
  - stability over 5 seeds on the first 100: Spearman rho of the source-variable ranks between
    seed pairs, and the overlap of the top 5.
CHECK FIRST: the mapping, applied to unperturbed source rows, must reproduce E's and R's feature
rows exactly (distance to 1e-9 km).

  python lime_grouped.py [--quick]  ->  output/lime_grouped_V01.csv, lime_grouped_stability_V01.csv
"""
import ast
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

import explain_compare as ec
from ablation_run import CONFIGS, RESULTS, TARGET, RANDOM_STATE
from data_paths import data_file

HERE = Path(__file__).resolve().parent
PATH = HERE.parent
OUT = HERE / "output"
QUICK = "--quick" in sys.argv
N_EXPLAIN, N_STAB, N_SAMPLES, SEEDS = (20, 10, 500, 3) if QUICK else (500, 100, 2000, 5)
SOURCES = ["category", "city", "job", "hour", "day_of_week", "amount", "gender", "age", "merch_lat", "merch_long"]
CATEGORICAL = ["category", "city", "job", "hour", "day_of_week", "gender"]
GRID = 32
WGS84 = Geodesic.WGS84

cells = ["".join(c["source"]) for c in json.load(open(PATH / "MLProj01_fraudTrain_encoding_V25.ipynb"))["cells"]]
lit = lambda i, n: ast.literal_eval(re.sub(r"#[^\n]*", "", re.search(rf"{n} = (\{{.*?\}})", cells[i], re.S).group(1)))
CATEGORY_CODE, STATE_CODE = lit(13, "category_code"), lit(25, "state_code")
CAT_E = ["category_1", "category_2", "category_3", "category_4a", "category_4b", "category_4c",
         "category_interaction_1", "category_interaction_2"]

# ---- the source table and lookups (fraudTrain.csv) ------------------------------------------
raw = pd.read_csv(PATH / "fraudTrain.csv")
ts = pd.to_datetime(raw["trans_date_trans_time"])
age_days = (ts - pd.to_datetime(raw["dob"])).astype("int64") / (10**9 * 86400)   # exact days
src = pd.DataFrame({
    "category": raw["category"], "city": raw["city"] + "|" + raw["state"], "job": raw["job"],
    "hour": ts.dt.hour, "day_of_week": ts.dt.dayofweek, "amount": raw["amt"],
    "gender": (raw["gender"] == "M").astype(int), "age": age_days / 365.25,
    "merch_lat": raw["merch_lat"], "merch_long": raw["merch_long"]})
home = raw[["lat", "long", "city_pop"]]                       # the explained row's own, carried
city_rep = (pd.concat([src["city"], home], axis=1).drop_duplicates("city").set_index("city"))   # first occurrence
bounds = dict(lat0=raw[["lat", "merch_lat"]].min().min(), lat1=raw[["lat", "merch_lat"]].max().max(),
              lon0=raw[["long", "merch_long"]].min().min(), lon1=raw[["long", "merch_long"]].max().max())
cityenc = pd.read_csv(PATH / "encoding_table/csv/city_and_encoding.csv")
cityenc["city"] = cityenc["City"] + "|" + cityenc["State"]
cityenc = cityenc.set_index("city")[["city_1", "city_2", "city_3", "city_4a", "city_4b"]]
jobenc = pd.read_csv(PATH / "encoding_table/csv/job_and_encoding.csv").set_index("job")
rawcols = list(pd.read_csv(PATH / "fraudTrain_raw_baseline.csv", nrows=0).columns)
city_levels = {v: i for i, v in enumerate(sorted(raw["city"].unique()))}
job_levels = {v: i for i, v in enumerate(sorted(raw["job"].unique()))}
# LIME works on numbers: categorical sources are coded 0..k-1
codes = {c: sorted(src[c].unique()) for c in CATEGORICAL}


def to_numeric(S):
    X = S.copy()
    for c in CATEGORICAL:
        X[c] = X[c].map({v: i for i, v in enumerate(codes[c])})
    return X[SOURCES].astype(float)


def from_numeric(X):
    S = pd.DataFrame(X, columns=SOURCES)
    for c in CATEGORICAL:
        S[c] = np.asarray(codes[c], dtype=object)[np.clip(np.rint(S[c]).astype(int), 0, len(codes[c]) - 1)]
    return S


def dist_km(a, b, c, d):
    return np.array([WGS84.Inverse(*q)["s12"] / 1000 for q in zip(a, b, c, d)])


def grid(lat, lon):
    r = np.trunc((lat - bounds["lat0"]) / (bounds["lat1"] - bounds["lat0"]) * (GRID - 1)).astype(int)
    c = np.trunc((lon - bounds["lon0"]) / (bounds["lon1"] - bounds["lon0"]) * (GRID - 1)).astype(int)
    return r * GRID + c


def homes(S, own_city, own_home):
    """Home lat/long/population: the row's own while its city is unchanged, else the city's."""
    h = city_rep.loc[S["city"].values].reset_index(drop=True)
    keep = (S["city"].values == own_city)
    for c in ("lat", "long", "city_pop"):
        h.loc[keep, c] = own_home[c]
    return h


def to_E(S, h):
    out = pd.DataFrame(index=range(len(S)))
    hr = S["hour"].astype(int).values
    out["trans_time_grp"] = np.select([hr < 8, hr < 16], [0, 1], 2)
    d = S["category"].map(CATEGORY_CODE)
    for i, c in enumerate(CAT_E):
        out[c] = d.str[i].astype(int).values
    out["amt"] = S["amount"].values
    out["gender"] = S["gender"].astype(int).values
    ce = cityenc.loc[S["city"].values].reset_index(drop=True)
    for c in ce.columns:
        out[c] = ce[c].astype(int).values
    st = S["city"].str.split("|").str[1].map(STATE_CODE)
    out["state_x"], out["state_y"] = st.str[0].astype(int).values, st.str[1].astype(int).values
    out["city_pop"] = h["city_pop"].values
    je = jobenc.loc[S["job"].values].reset_index(drop=True)
    for c in ("job_1", "job_2", "job_3", "job_4"):
        out[c] = je[c].astype(int).values
    out["cardholder_age"] = np.trunc(S["age"].values.astype(float)).astype(int)
    out["loc_cluster_issue"] = grid(h["lat"].values, h["long"].values)
    out["loc_cluster_transac"] = grid(S["merch_lat"].values, S["merch_long"].values)
    out["issue_transac_dist"] = dist_km(h["lat"].values, h["long"].values, S["merch_lat"].values, S["merch_long"].values)
    return out


def to_R(S, h):
    out = pd.DataFrame(0, index=range(len(S)), columns=[c for c in rawcols if c not in ("trans_time", TARGET)])
    out["amt"] = S["amount"].values
    out["city_pop"], out["lat"], out["long"] = h["city_pop"].values, h["lat"].values, h["long"].values
    out["merch_lat"], out["merch_long"] = S["merch_lat"].values, S["merch_long"].values
    out["gender"] = S["gender"].astype(int).values
    out["age"] = np.floor(S["age"].values.astype(float) * 365.25 + 1e-9) / 365.25
    out["hour_of_day"], out["day_of_week"] = S["hour"].astype(int).values, S["day_of_week"].astype(int).values
    for i, c in enumerate(S["category"].values):
        out.iat[i, out.columns.get_loc(f"category_{c}")] = 1
    for i, c in enumerate(S["city"].str.split("|").str[1].values):
        out.iat[i, out.columns.get_loc(f"state_{c}")] = 1
    out["city_ordinal"] = S["city"].str.split("|").str[0].map(city_levels).values
    out["job_ordinal"] = S["job"].map(job_levels).values
    return out


def main():
    E = pd.read_csv(data_file("fraudTrain_encoded.csv"))
    R = pd.read_csv(PATH / "fraudTrain_raw_baseline.csv")
    t = E["trans_time"]; c1, c2 = t.quantile(0.64), t.quantile(0.80)
    train_idx, valid_idx = E.index[t <= c1].values, E.index[(t > c1) & (t <= c2)].values

    # ---- CHECK: the mapping reproduces the real rows -----------------------------------
    chk = np.random.default_rng(0).choice(valid_idx, 2000, replace=False)
    S = src.loc[chk].reset_index(drop=True)
    h = home.loc[chk].reset_index(drop=True)
    e_pred = [c for c in E.columns if c not in ("trans_time", TARGET)]
    r_pred = [c for c in R.columns if c not in ("trans_time", TARGET)]
    ge, gr = to_E(S, h)[e_pred], to_R(S, h)[r_pred]
    for c in e_pred:
        tol = 1e-9 if c == "issue_transac_dist" else 0
        assert np.abs(ge[c].values - E.loc[chk, c].values).max() <= tol, f"E {c}"
    for c in r_pred:
        assert np.abs(gr[c].values - R.loc[chk, c].values).max() <= 1e-9, f"R {c}"
    print("source -> feature mapping reproduces 2,000 validation rows of E and R exactly")

    # ---- the explained rows: explain_compare.py's balanced sample, same seed, same draws ----
    rng = np.random.default_rng(RANDOM_STATE)
    yv = E.loc[valid_idx, TARGET].values
    pos, neg = valid_idx[yv == 1], valid_idx[yv == 0]
    balanced = np.r_[rng.choice(pos, ec.N_BAL, replace=False), rng.choice(neg, ec.N_BAL, replace=False)]
    explain = balanced[:N_EXPLAIN]
    lime_train = to_numeric(src.loc[np.random.default_rng(RANDOM_STATE).choice(train_idx, 20000, replace=False)])

    rows, stab = [], []
    for key, cfg in ec.CONFIGS_USED.items():
        df = E if key == "E" else R
        predictors = e_pred if key == "E" else r_pred
        tuned = pd.read_csv(RESULTS / CONFIGS[cfg][1]).set_index("model")
        train = df.loc[train_idx]
        for m in ec.MODELS:
            mod = ec.fit(m, json.loads(tuned.loc[m, "chosen_params"]), train, predictors)
            mapper = to_E if key == "E" else to_R
            s0 = time.time()
            per_seed = {}
            for seed in range(SEEDS):
                n_rows = N_EXPLAIN if seed == 0 else N_STAB
                lime = LimeTabularExplainer(lime_train.values, feature_names=SOURCES, class_names=["genuine", "fraud"],
                                            categorical_features=[SOURCES.index(c) for c in CATEGORICAL],
                                            discretize_continuous=True, random_state=RANDOM_STATE + seed)
                weights = []
                for r in explain[:n_rows]:
                    own_city, own_home = src.at[r, "city"], home.loc[r]

                    def predict(X, own_city=own_city, own_home=own_home):
                        S = from_numeric(X)
                        return mod.predict_proba(mapper(S, homes(S, own_city, own_home))[predictors])

                    exp = lime.explain_instance(to_numeric(src.loc[[r]]).values[0], predict,
                                                num_features=len(SOURCES), num_samples=N_SAMPLES)
                    w = dict(exp.as_map()[1])
                    weights.append([w.get(i, 0.0) for i in range(len(SOURCES))])
                    if seed == 0:
                        rows.append(dict(config=key, model=m, row=int(r), is_fraud=int(E.at[r, TARGET]),
                                         fidelity_r2=float(exp.score), **{s: w.get(i, 0.0) for i, s in enumerate(SOURCES)}))
                per_seed[seed] = np.array(weights)
            # stability over seeds, on the first N_STAB rows
            rhos, tops = [], []
            for a, b in combinations(range(SEEDS), 2):
                for i in range(N_STAB):
                    wa, wb = np.abs(per_seed[a][i]), np.abs(per_seed[b][i])
                    rhos.append(spearmanr(wa, wb).statistic)
                    tops.append(len(set(np.argsort(-wa)[:5]) & set(np.argsort(-wb)[:5])) / 5)
            stab.append(dict(config=key, model=m, seeds=SEEDS, rows=N_STAB, spearman_rho_mean=float(np.nanmean(rhos)),
                             spearman_rho_p10=float(np.nanpercentile(rhos, 10)), top5_overlap_mean=float(np.mean(tops))))
            print(f"{key} {m}: LIME done ({time.time() - s0:.0f}s); fidelity R2 median "
                  f"{np.median([r['fidelity_r2'] for r in rows if r['config'] == key and r['model'] == m]):.3f}; "
                  f"seed stability rho {stab[-1]['spearman_rho_mean']:.3f}, top-5 overlap {stab[-1]['top5_overlap_mean']:.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "lime_grouped_V01.csv", index=False)
    pd.DataFrame(stab).to_csv(OUT / "lime_grouped_stability_V01.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.groupby(["config", "model"])[SOURCES].apply(lambda d: d.abs().mean()).round(4).T.to_string())


if __name__ == "__main__":
    main()
