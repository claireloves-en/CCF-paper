"""
Build the revised Results tables from model_V25's exports.

Inputs (result_csv/), for each configuration tag T:
    test_metric_result_V01_<T>.csv      threshold-0.5 and threshold-free metrics, test set
    test_operating_point_V01_<T>.csv    metrics at the 0.5% alert budget, test set
    models_metric_result_V06_<T>.csv    the same on the validation set (for the optimism gap)
Outputs (result_csv/):
    results_table_performance_<T>.csv   Table 3: all metrics, test set, best per row flagged
    results_table_client_effect.csv     base vs base + client features, per model
    results_summary.md                  the same, readable, with the best value in bold

Usage: python results_tables.py [base_tag] [client_tag]
       defaults: temporal_notime temporal_notime_client
"""
import sys
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "result_csv"
BASE = sys.argv[1] if len(sys.argv) > 1 else "temporal_notime"
CLIENT = sys.argv[2] if len(sys.argv) > 2 else "temporal_notime_client"
ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "AdaBoost",
         "CatBoost", "XGBoost", "LightGBM", "K-NN"]
# rows where "best" is meaningful (higher is better); the threshold and the alert
# rate are properties of the operating point, not scores
SCORE_ROWS = ["Recall", "Precision", "Specificity", "F1 Score", "AUC", "AUC-PR", "MCC",
              "Balanced Classification Rate", "Recall (budget)", "Precision (budget)",
              "F1 (budget)", "MCC (budget)"]


def load(prefix, tag):
    f = RES / f"{prefix}_{tag}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.set_index(d.columns[0])[ORDER].astype(float)


def performance(tag, split="test"):
    m = load("test_metric_result_V01" if split == "test" else "models_metric_result_V06", tag)
    o = load("test_operating_point_V01" if split == "test" else "operating_point_V01", tag)
    if m is None:
        return None
    return pd.concat([m, o]) if o is not None else m


def md_table(df, digits=4):
    cols = list(df.columns)
    out = ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r, row in df.iterrows():
        best = row.max() if r in SCORE_ROWS else None
        cells = []
        for v in row:
            s = "—" if pd.isna(v) else f"{v:.{digits}f}"
            cells.append(f"**{s}**" if best is not None and not pd.isna(v) and v == best else s)
        out.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(out)


lines = []
for tag in (BASE, CLIENT):
    t = performance(tag, "test")
    if t is None:
        lines.append(f"## {tag}: test results not available yet\n")
        continue
    flagged = t.copy()
    flagged["best"] = [t.loc[r].idxmax() if r in SCORE_ROWS else "" for r in t.index]
    flagged.to_csv(RES / f"results_table_performance_{tag}.csv")
    lines += [f"## Test-set performance — {tag}", "", md_table(t), ""]
    v = performance(tag, "valid")
    if v is not None:
        gap = (v.loc["AUC-PR"] - t.loc["AUC-PR"]).round(4)
        lines += ["Validation minus test AUC-PR (positive = validation optimistic; AUC-PR also scales with the fraud rate, 0.56% valid vs 0.59% test): "
                  + ", ".join(f"{m} {g:+.4f}" for m, g in gap.items()), ""]

b, c = performance(BASE, "test"), performance(CLIENT, "test")
if b is not None and c is not None:
    rows = []
    for m in ORDER:
        rows.append({"model": m,
                     "AUC-PR base": b.loc["AUC-PR", m], "AUC-PR + client": c.loc["AUC-PR", m],
                     "AUC-PR gain": c.loc["AUC-PR", m] - b.loc["AUC-PR", m],
                     "AUC base": b.loc["AUC", m], "AUC + client": c.loc["AUC", m],
                     "Recall@budget base": b.loc["Recall (budget)", m] if "Recall (budget)" in b.index else float("nan"),
                     "Recall@budget + client": c.loc["Recall (budget)", m] if "Recall (budget)" in c.index else float("nan"),
                     "Precision@budget base": b.loc["Precision (budget)", m] if "Precision (budget)" in b.index else float("nan"),
                     "Precision@budget + client": c.loc["Precision (budget)", m] if "Precision (budget)" in c.index else float("nan")})
    ce = pd.DataFrame(rows).set_index("model")
    ce.to_csv(RES / "results_table_client_effect.csv")
    lines += ["## Effect of the client-level features (test set)", "",
              "| Model | AUC-PR base | + client | gain | Recall@0.5% base | + client | Precision@0.5% base | + client |",
              "|---|---|---|---|---|---|---|---|"]
    for m, r in ce.iterrows():
        lines.append(f"| {m} | {r['AUC-PR base']:.4f} | {r['AUC-PR + client']:.4f} | {r['AUC-PR gain']:+.4f} | "
                     f"{r['Recall@budget base']:.4f} | {r['Recall@budget + client']:.4f} | "
                     f"{r['Precision@budget base']:.4f} | {r['Precision@budget + client']:.4f} |")
else:
    lines.append("## Client-feature comparison: needs both test result sets")

(RES / "results_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
