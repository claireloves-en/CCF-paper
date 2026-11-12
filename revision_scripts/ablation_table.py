"""
The J1 ablation table: what does the theory-informed encoding contribute?
(Revision_Phase6_Plan_V01.md §6.)

Four configurations under one chronological split, one tuning budget and one test evaluation:

  1  raw            conventional handling of the raw Sparkov columns the encoding is built from
  2  raw + client   the same, plus the seven client-history features
  3  encoded        the paper's 27 theory-informed features
  4  encoded+client the paper's second configuration

and the four differences that answer the assessment's question:

  3 - 1   the encoding's contribution without behavioural history
  4 - 2   the encoding's contribution once behavioural history is available
  2 - 1   the client features' contribution on raw features
  4 - 3   the client features' contribution on encoded features

Reads the per-configuration tables ablation_run.py writes (test AUC-PR, AUC, and recall and
precision at each model's validation-set operating point) and reshapes them. Computes nothing
new, so it cannot disagree with the runs of record.

Usage: python ablation_table.py [--phase7]
Writes output/ablation_V01.csv        one row per model and configuration
       output/ablation_diffs_V01.csv  the four differences, per model
--phase7 reads the p7_* runs (corrected encoding; Revision_Phase7_Plan_V01.md) and writes
ablation_V02.csv / ablation_diffs_V02.csv, plus ablation_targeted_V01.csv: the random forest and
CatBoost diagnostics (hybrid, one-change ablations of E, reduced pair, catnative) against their
reference configuration, with the information each arm changes.
"""
import sys
from pathlib import Path

import pandas as pd

from ablation_run import CONFIGS, MODEL_ORDER

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
ORDER = ["notime_raw", "notime_raw_client", "notime", "notime_client"]
LABEL = {"notime_raw": "1 raw", "notime_raw_client": "2 raw + client",
         "notime": "3 encoded", "notime_client": "4 encoded + client"}
DIFFS = [("3 - 1  encoding, without client history", "notime", "notime_raw"),
         ("4 - 2  encoding, with client history", "notime_client", "notime_raw_client"),
         ("2 - 1  client history, on raw features", "notime_raw_client", "notime_raw"),
         ("4 - 3  client history, on encoded features", "notime_client", "notime")]
PHASE7 = "--phase7" in sys.argv
V = "V02" if PHASE7 else "V01"
if PHASE7:
    ORDER = ["p7_" + c for c in ORDER]
    LABEL = {"p7_" + k: v for k, v in LABEL.items()}
    DIFFS = [(l, "p7_" + a, "p7_" + b) for l, a, b in DIFFS]
# the diagnostics: (configuration, reference, what the arm changes relative to the reference)
TARGETED = [
    ("p7_hybrid", "p7_notime_raw", "R + E: the encoded features added to the raw ones"),
    ("p7_hybrid_client", "p7_notime_raw_client", "R + E + H against R + H"),
    ("p7_abl_cat_onehot", "p7_notime", "category: 8 encoded features -> one-hot (14)"),
    ("p7_abl_plus_onehot", "p7_notime", "category: one-hot added alongside the encoding"),
    ("p7_abl_hour", "p7_notime", "time: 3 bands -> hour of day (24)"),
    ("p7_abl_plus_dow", "p7_notime", "time: day of week added (absent from E)"),
    ("p7_abl_no_interactions", "p7_notime", "the 2 interaction terms removed"),
    ("p7_abl_no_external", "p7_notime", "external enrichment removed (city_1..4b, job_1..4): also every post-2020 source"),
    ("p7_abl_no_geo_occ", "p7_notime", "all 14 geographic and occupational features removed"),
    ("p7_red_enc", "p7_red_raw", "reduced pair: amt + category + time, encoded against raw"),
    ("p7_catnative", "p7_notime_raw", "CatBoost native categorical handling of the raw strings"),
    ("p7_catnative_client", "p7_notime_raw_client", "the same, with H"),
]

rows = []
for cfg in ORDER:
    _d, _t, tag = CONFIGS[cfg]
    met_p, bud_p = (OUT / f"test_metric_result_V01_{tag}.csv",
                    OUT / f"test_operating_point_V01_{tag}.csv")
    if not met_p.exists():
        print(f"SKIP {cfg}: run `ablation_run.py --config {cfg}` first")
        continue
    met = pd.read_csv(met_p).set_index("metric\\model")
    bud = pd.read_csv(bud_p).set_index("metric\\model")
    for m in MODEL_ORDER:
        if m not in met.columns:
            continue
        rows.append(dict(configuration=LABEL[cfg], config=cfg, model=m,
                         test_aucpr=met.loc["AUC-PR", m], test_auc=met.loc["AUC", m],
                         alert_rate=bud.loc["Alert rate on test", m],
                         recall_budget=bud.loc["Recall (budget)", m],
                         precision_budget=bud.loc["Precision (budget)", m]))

if not rows:
    raise SystemExit("no configuration results found")
res = pd.DataFrame(rows)
res.to_csv(OUT / f"ablation_{V}.csv", index=False)

piv = res.pivot(index="model", columns="configuration", values="test_aucpr").reindex(MODEL_ORDER)
piv = piv[[LABEL[c] for c in ORDER if LABEL[c] in piv.columns]]
print("Test AUC-PR by model and configuration\n")
print(piv.round(4).to_string())

have = set(res.config)
drows = []
for label, a, b in DIFFS:
    if not {a, b} <= have:
        continue
    da = res[res.config == a].set_index("model")["test_aucpr"]
    db = res[res.config == b].set_index("model")["test_aucpr"]
    for m in MODEL_ORDER:
        if m in da.index and m in db.index:
            drows.append(dict(difference=label, model=m, value=da[m] - db[m]))
if drows:
    dd = pd.DataFrame(drows)
    dd.to_csv(OUT / f"ablation_diffs_{V}.csv", index=False)
    print("\nDifferences in test AUC-PR\n")
    print(dd.pivot(index="model", columns="difference", values="value")
            .reindex(MODEL_ORDER).round(4).to_string())
    print("\nmean over the four ensembles")
    ens = ["Random Forest", "CatBoost", "XGBoost", "LightGBM"]
    print(dd[dd.model.isin(ens)].groupby("difference")["value"].mean().round(4).to_string())
print(f"\nwrote {OUT / f'ablation_{V}.csv'}"
      + (f" and {OUT / f'ablation_diffs_{V}.csv'}" if drows else ""))

if PHASE7:
    def aucpr(cfg):
        p = OUT / f"test_metric_result_V01_{CONFIGS[cfg][2]}.csv"
        return pd.read_csv(p).set_index("metric\\model").loc["AUC-PR"] if p.exists() else None
    trows = []
    for cfg, ref, what in TARGETED:
        a, b = aucpr(cfg), aucpr(ref)
        if a is None or b is None:
            print(f"SKIP {cfg}: run it and {ref} first"); continue
        for m in ("Random Forest", "CatBoost"):
            if m in a.index and m in b.index:
                trows.append(dict(config=cfg, reference=ref, change=what, model=m,
                                  test_aucpr=a[m], reference_aucpr=b[m], difference=a[m] - b[m]))
    if trows:
        t = pd.DataFrame(trows)
        t.to_csv(OUT / "ablation_targeted_V01.csv", index=False)
        print("\nTargeted comparisons (random forest, CatBoost)\n")
        print(t.pivot_table(index=["config", "change"], columns="model", values="difference", sort=False).round(4).to_string())
