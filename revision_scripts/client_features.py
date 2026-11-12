"""
The seven client-level behavioural features (D10), as a reusable module.
(Revision_Phase6_Plan_V01.md §3 — configurations 2 and 4 must use the SAME features.)

The computation was written in cells 4-6 of MLProj01_fraudTrain_clientfeatures_V01.ipynb,
which writes fraudTrain_encoded_client.csv. The raw-baseline configurations need the same
seven features over a different predictor set, so the code lives here and the notebook
imports it rather than the two drifting apart.

Every feature is causal: it uses only the card's transactions strictly EARLIER than the
current one, and none of them uses is_fraud. The notebook's brute-force causality check
(cell 6) still applies and must still pass.

  card_amt_zscore          how unusual this amount is for this card, from the card's prior
                           amounts; 0 until MIN_HISTORY priors; clipped to +/-10
  card_hours_since_prev    hours since the card's previous transaction, capped at HOURS_CAP
  card_txn_1h/24h/7d       the card's transactions in the preceding 1h / 24h / 7d, excluding
                           this one
  card_dist_from_usual_km  km from this merchant to the centroid of the card's previous
                           merchants; the cardholder's home for a first transaction
  card_new_category        1 if this is the card's first purchase in this category

This module is verified rather than trusted: run it as a script to check that compute()
reproduces the seven columns of the committed fraudTrain_encoded_client.csv exactly.

  python client_features.py
"""
import numpy as np
import pandas as pd

MIN_HISTORY = 5          # prior transactions needed before a z-score is trusted
HOURS_CAP = 720          # 30 days
EARTH_KM = 6371.0088

FEATURES = ["card_amt_zscore", "card_hours_since_prev", "card_txn_1h", "card_txn_24h",
            "card_txn_7d", "card_dist_from_usual_km", "card_new_category"]

# the columns compute() needs from fraudTrain.csv
NEEDED = ["cc_num", "unix_time", "amt", "category", "merch_lat", "merch_long", "lat", "long"]


def _trailing_counts(times, window_s):
    """For each position, how many earlier positions fall in (t - window, t]."""
    j = np.searchsorted(times, times - window_s, side="right")
    return np.maximum(np.arange(len(times)) - j, 0)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return EARTH_KM * 2 * np.arcsin(np.sqrt(a))


def compute(raw):
    """The seven features for `raw`, returned in raw's own row order.

    raw must carry the NEEDED columns of fraudTrain.csv. Nothing is read from is_fraud.
    """
    missing = [c for c in NEEDED if c not in raw.columns]
    if missing:
        raise ValueError(f"compute() needs {missing} in the input")

    # Order each card's history in time. The raw file is not globally time-ordered;
    # ties on unix_time are broken by original row order.
    df = raw[NEEDED].copy()
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["cc_num", "unix_time", "_row"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("cc_num", sort=False)

    # --- amount relative to the card's own past ---------------------------------
    k = g.cumcount()                                        # number of prior txns
    s1 = g["amt"].cumsum() - df["amt"]                      # sum of prior amounts
    s2 = (df["amt"] ** 2).groupby(df["cc_num"]).cumsum() - df["amt"] ** 2
    mean_prev = s1 / k.replace(0, np.nan)
    var_prev = (s2 - k * mean_prev ** 2) / (k - 1).replace(0, np.nan)
    std_prev = np.sqrt(var_prev.clip(lower=0))
    z = (df["amt"] - mean_prev) / std_prev.replace(0, np.nan)
    df["card_amt_zscore"] = z.where(k >= MIN_HISTORY, 0.0).fillna(0.0).clip(-10, 10)

    # --- time since the card's previous transaction ------------------------------
    df["card_hours_since_prev"] = (g["unix_time"].diff() / 3600.0).fillna(HOURS_CAP).clip(upper=HOURS_CAP)

    # --- trailing counts ---------------------------------------------------------
    for col, w in [("card_txn_1h", 3600), ("card_txn_24h", 86400), ("card_txn_7d", 7 * 86400)]:
        out = np.empty(len(df), dtype=np.int32)
        for idx in g.indices.values():
            out[idx] = _trailing_counts(df["unix_time"].values[idx], w)
        df[col] = out

    # --- distance from the card's usual merchant location ------------------------
    c_lat = (g["merch_lat"].cumsum() - df["merch_lat"]) / k.replace(0, np.nan)
    c_lon = (g["merch_long"].cumsum() - df["merch_long"]) / k.replace(0, np.nan)
    c_lat, c_lon = c_lat.fillna(df["lat"]), c_lon.fillna(df["long"])   # first txn: home
    df["card_dist_from_usual_km"] = _haversine_km(c_lat, c_lon, df["merch_lat"], df["merch_long"])

    # --- first purchase in this category -----------------------------------------
    df["card_new_category"] = (df.groupby(["cc_num", "category"], sort=False).cumcount() == 0).astype(int)

    # back to the caller's row order
    df = df.sort_values("_row").reset_index(drop=True)
    assert (df["_row"].values == np.arange(len(df))).all()
    return df[FEATURES]


if __name__ == "__main__":
    from pathlib import Path
    PATH = Path(__file__).resolve().parent.parent
    raw = pd.read_csv(PATH / "fraudTrain.csv", usecols=NEEDED)
    got = compute(raw)
    want = pd.read_csv(PATH / "fraudTrain_encoded_client.csv", usecols=FEATURES)[FEATURES]
    assert len(got) == len(want), f"{len(got)} rows vs {len(want)}"
    # The counts and the flag are integers and must match exactly. The three float
    # columns are compared to 1e-9: fraudTrain_encoded_client.csv stores them as decimal
    # text, so a round trip through it cannot be bit-exact. Observed worst case 1.8e-15.
    EXACT = ["card_txn_1h", "card_txn_24h", "card_txn_7d", "card_new_category"]
    bad, worst = [], 0.0
    for c in FEATURES:
        d = np.abs(got[c].values - want[c].values)
        worst = max(worst, float(d.max()))
        tol = 0 if c in EXACT else 1e-9
        if d.max() > tol:
            bad.append((c, int((d > tol).sum()), float(d.max()), tol))
    if bad:
        for c, n, d, tol in bad:
            print(f"MISMATCH {c}: {n} rows beyond {tol}, max |diff| = {d}")
        raise SystemExit(1)
    print(f"compute() reproduces all {len(FEATURES)} columns of fraudTrain_encoded_client.csv "
          f"over {len(got):,} rows; integer columns exact, worst float |diff| = {worst:.2e}")
