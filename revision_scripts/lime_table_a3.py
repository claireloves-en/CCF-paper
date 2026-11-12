"""
Appendix Table A3: average LIME contribution of each feature condition in the four
interpreted models, with how many of the explained transactions fall in each
condition, by class.

Usage: python lime_table_a3.py [TAG]      TAG defaults to temporal_notime
Reads  result_csv/LIME_{model}_{TAG}.csv (written by model_V25)
Writes result_csv/LIME_TableA3_V01_{TAG}.csv and .xlsx

Why a script and not model_V25 cells 217-218. Cell 218 counts transactions in bins
that are hardcoded from the submitted run (amount 9.66 / 47.62 / 83.27, ...). LIME
bins each feature at its training-set quartiles, and under the chronological split
those are different (amount 21.x / 47.40 / 83.06), so the notebook's counts do not
line up with the coefficients they sit beside. Here the bins come from LIME's own
discretiser, built exactly as the notebook builds its explainer, so every count
refers to the same condition as the coefficient in its row.

Which transactions. Cell 60 draws 1,000 genuine and 1,000 fraudulent validation
transactions and shuffles them (df_random); the LIME cells explain the first 500
of those. The counts here are over those 500, i.e. the transactions the averages
are taken over. The reconstruction of df_random is checked against the summary
statistics cell 219 printed in the executed notebook.
"""
import random, sys, warnings
from pathlib import Path
from data_paths import data_file  # encoded files: current, or archived via CCF_ENCODED_VERSION
import numpy as np
import pandas as pd
import lime.lime_tabular

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PATH = HERE.parent
RES = PATH / "result_csv"
RANDOM_STATE, N_EXPLAINED = 1200, 500
MODELS = ["Decision Tree", "Random Forest", "CatBoost", "XGBoost"]

# ---- the notebook's data preparation (model_V25 cells 3-14) ---------------------
NAMES = {
    'trans_time': 'Time difference from the first transaction', 'trans_time_grp': 'Transaction time group',
    'category_1': 'Offline vs. online', 'category_2': 'Non-essential vs. essential',
    'category_3': 'Average item cost', 'category_4a': 'Purpose of purchase: household/wellbeing',
    'category_4b': 'Purpose of purchase: entertainment/lifestyle',
    'category_4c': 'Purpose of purchase: shopping-related',
    'category_interaction_1': 'Interaction 1: Non-essential vs. essential*Average item cost',
    'category_interaction_2': 'Interaction 2: Non-essential vs. essential*Purpose of purchase: household or wellbeing',
    'amt': 'Amount spent (USD)', 'gender': 'Gender', 'city_1': 'City socioeconomic status',
    'city_2': 'City social inequality level', 'city_3': 'City Rural-Urban',
    'city_4a': 'City high-speed fixed broadband coverage', 'city_4b': 'City high-speed mobile broadband',
    'state_x': 'State West vs East', 'state_y': 'State North vs South', 'city_pop': 'Population of the city',
    'job_1': 'Occupation income level', 'job_2': 'Occupation education requirement',
    'job_3': 'Occupation career level', 'job_4': 'Occupation cyber knowledge',
    'cardholder_age': 'Cardholder age', 'loc_cluster_issue': 'Location Cluster Card Issue',
    'loc_cluster_transac': 'Location Cluster Transaction',
    'issue_transac_dist': 'Distance between where card was issued and transaction made',
    'card_amt_zscore': 'Card amount vs own history (z-score)',
    'card_hours_since_prev': 'Card hours since previous transaction',
    'card_txn_1h': 'Card transactions in past 1h', 'card_txn_24h': 'Card transactions in past 24h',
    'card_txn_7d': 'Card transactions in past 7d',
    'card_dist_from_usual_km': 'Card distance from usual location (km)',
    'card_new_category': 'Card first purchase in category',
}
ENCODED = ['trans_time_grp', 'category_1', 'category_2', 'category_3', 'category_4a', 'category_4b',
           'category_4c', 'category_interaction_1', 'category_interaction_2', 'amt', 'gender',
           'city_1', 'city_2', 'city_3', 'city_4a', 'city_4b', 'state_x', 'state_y', 'city_pop',
           'job_1', 'job_2', 'job_3', 'job_4', 'cardholder_age', 'loc_cluster_issue',
           'loc_cluster_transac', 'issue_transac_dist']          # trans_time dropped
CLIENT_COLS = ['card_amt_zscore', 'card_hours_since_prev', 'card_txn_1h', 'card_txn_24h',
               'card_txn_7d', 'card_dist_from_usual_km', 'card_new_category']
BASE_ORDER = ['Amount spent (USD)', 'Offline vs. online', 'Non-essential vs. essential', 'Average item cost',
    'Purpose of purchase: household/wellbeing', 'Purpose of purchase: entertainment/lifestyle',
    'Purpose of purchase: shopping-related',
    'Interaction 1: Non-essential vs. essential*Average item cost',
    'Interaction 2: Non-essential vs. essential*Purpose of purchase: household or wellbeing',
    'Cardholder age', 'Transaction time group', 'Population of the city', 'Gender',
    'Location Cluster Card Issue', 'Location Cluster Transaction',
    'Distance between where card was issued and transaction made',
    'State West vs East', 'State North vs South', 'Occupation income level',
    'Occupation education requirement', 'Occupation career level', 'Occupation cyber knowledge',
    'City socioeconomic status', 'City social inequality level', 'City Rural-Urban',
    'City high-speed fixed broadband coverage', 'City high-speed mobile broadband']


def prepare(tag):
    """The notebook's data preparation (model_V25 cells 3-14) and df_random (cell 60).

    Returns train_df, valid_df, df_random, predictors (notebook order) and ORDER
    (the paper's presentation order, model_V25 cell 217, then the client features).
    """
    client = tag.endswith("_client")
    predictors = [NAMES[c] for c in ENCODED + (CLIENT_COLS if client else [])]
    order = BASE_ORDER + ([NAMES[c] for c in CLIENT_COLS] if client else [])
    assert sorted(order) == sorted(predictors)

    df = pd.read_csv(data_file("fraudTrain_encoded_client.csv" if client else "fraudTrain_encoded.csv"))
    df = df.rename(columns=NAMES)
    t = df['Time difference from the first transaction']
    c1, c2 = t.quantile(0.64), t.quantile(0.80)
    train_df, valid_df = df[t <= c1], df[(t > c1) & (t <= c2)]

    nonfraud = list(valid_df.loc[valid_df.is_fraud == 0].index)
    random.seed(RANDOM_STATE); r0 = np.array(random.sample(nonfraud, 1000))
    fraud = list(valid_df.loc[valid_df.is_fraud == 1].index)
    random.seed(RANDOM_STATE); r1 = np.array(random.sample(fraud, 1000))
    idx = np.concatenate((r0, r1)); random.shuffle(idx)
    df_random = valid_df.loc[idx, :]
    # cell 219 in executed/model_V25_temporal_notime.ipynb printed these for df_random
    if tag == "temporal_notime":
        a = df_random['Amount spent (USD)']
        got = (a.min(), a.quantile(.25), a.quantile(.5), a.quantile(.75), a.max())
        assert np.allclose(got, (1.02, 21.67, 96.425, 600.1325, 1294.83), atol=0.006), got
        assert df_random['Cardholder age'].quantile(.5) == 45.0
    return train_df, valid_df, df_random, predictors, order


def main(TAG):
    train_df, valid_df, df_random, predictors, ORDER = prepare(TAG)
    explained = df_random.iloc[:N_EXPLAINED]

    # ---- LIME's discretiser, built as in the LIME cells -----------------------------
    explainer = lime.lime_tabular.LimeTabularExplainer(
        train_df[predictors].values, random_state=RANDOM_STATE, feature_names=predictors,
        class_names=['class_0', 'class_1'], discretize_continuous=True)
    disc = explainer.discretizer
    bins = disc.discretize(explained[predictors].values)

    coef = {}
    for m in MODELS:
        s = pd.read_csv(RES / f"LIME_{m}_{TAG}.csv", index_col=0).iloc[:, 0]
        coef[m] = s

    rows = []
    for feat in ORDER:
        j = predictors.index(feat)
        labels = disc.names[j] if j in disc.names else None
        if labels is None:            # LIME leaves a feature undiscretised only if it is categorical
            raise RuntimeError(f"{feat} was not discretised")
        for b, label in enumerate(labels):
            inbin = bins[:, j] == b
            g = int((inbin & (explained.is_fraud.values == 0)).sum())
            f = int((inbin & (explained.is_fraud.values == 1)).sum())
            vals = [coef[m].get(label, np.nan) for m in MODELS]
            rows.append(dict(feature=feat, condition=label, genuine=g, fraud=f,
                             **dict(zip(MODELS, vals))))
    out = pd.DataFrame(rows)
    ng, nf = int((explained.is_fraud == 0).sum()), int((explained.is_fraud == 1).sum())
    out["genuine (%)"] = [f"{g} ({100*g/ng:.1f}%)" for g in out.genuine]
    out["fraud (%)"] = [f"{f} ({100*f/nf:.1f}%)" for f in out.fraud]
    out["mean"] = out[MODELS].mean(axis=1)
    out["positive in"] = (out[MODELS] > 0).sum(axis=1).astype(str) + " of 4"

    # every LIME coefficient must have found its row, and every row its coefficients
    for m in MODELS:
        missing = set(coef[m].index) - set(out.condition)
        assert not missing, (m, missing)
    # LIME names a bin above the upper quartile even when the quartile is the maximum
    # (every binary and small ordinal feature); nothing falls in it, so no model has a
    # coefficient for it. Drop those rows; any other gap is an error.
    empty = (out.genuine + out.fraud) == 0
    assert out.loc[empty, MODELS].isna().all().all() and not out.loc[~empty, MODELS].isna().any().any()
    out = out[~empty].reset_index(drop=True)
    assert (out.groupby("feature").genuine.sum() == ng).all() and (out.groupby("feature").fraud.sum() == nf).all()

    cols = ["feature", "condition", "genuine (%)", "fraud (%)", *MODELS, "mean", "positive in"]
    out[cols].to_csv(RES / f"LIME_TableA3_V01_{TAG}.csv", index=False)
    # submission copy: one block per feature, readable headers, a notes sheet
    x = out[cols].copy()
    x.loc[x.feature.duplicated(), "feature"] = ""
    x.columns = ["Feature", "Condition (LIME interval)", "Genuine, n (%)", "Fraudulent, n (%)",
                 "Decision tree", "Random forest", "CatBoost", "XGBoost", "Mean of four models",
                 "Positive in"]
    notes = [
        "Table A3. Average LIME contribution of each feature condition in the four interpreted models.",
        f"LIME explained {N_EXPLAINED} validation transactions ({ng} genuine, {nf} fraudulent), the first "
        f"{N_EXPLAINED} of a shuffled balanced sample of 1,000 genuine and 1,000 fraudulent transactions, "
        "with 2,000 perturbation samples each.",
        "Intervals are bounded by the feature's quartiles in the training set. Each coefficient is the "
        "model's LIME weight for that interval, averaged over the explained transactions that fall in it; "
        "positive values push the prediction towards fraud.",
        "n (%) is the number of explained transactions of each class in the interval, and its share of that class.",
        "Intervals above a feature's maximum, which LIME defines but no transaction occupies, are omitted.",
        f"Source: result_csv/LIME_*_{TAG}.csv (model_V25); built by revision_scripts/lime_table_a3.py.",
    ]
    with pd.ExcelWriter(RES / f"LIME_TableA3_V01_{TAG}.xlsx", engine="openpyxl") as w:
        x.to_excel(w, index=False, sheet_name="Table A3")
        pd.DataFrame({"Notes": notes}).to_excel(w, index=False, sheet_name="Notes")
        ws = w.sheets["Table A3"]
        for col, width in zip("ABCDEFGHIJ", [44, 62, 15, 17, 13, 14, 11, 11, 13, 11]):
            ws.column_dimensions[col].width = width
        for row in ws.iter_rows(min_row=2, min_col=5, max_col=9):
            for cell in row:
                cell.number_format = "0.0000"
        ws.freeze_panes = "C2"
        w.sheets["Notes"].column_dimensions["A"].width = 120
    print(f"explained: {ng} genuine, {nf} fraudulent of {N_EXPLAINED}")
    print(out[cols].head(12).to_string())
    print("wrote", RES / f"LIME_TableA3_V01_{TAG}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "temporal_notime")
