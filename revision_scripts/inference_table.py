"""
The inference record behind every interval and test the paper reports (Revision_Plan_V08.md, B4 and A7;
review of CCF_MM_V03, item 6).

Reads the paired-bootstrap difference files written by uncertainty_check.py (test period, three schemes)
and by confirm_run.py (confirmation period, two schemes), and writes:

  result_csv/inference_table_V01.csv          every contrast: period, scheme, family, number of tests in
                                              the family, contrast, model, difference, 95% percentile
                                              interval, bootstrap p, Holm-adjusted p
  result_csv/inference_families_V01.csv       per period x scheme x family: tests, and how many are
                                              separated from zero after Holm at 0.05 (Appendix Table A15)

Method, as implemented (uncertainty_check.py): B = 1,000 resamples of fixed test predictions, applied in
common to every model and configuration. Intervals are ordinary 2.5-97.5% percentile intervals, not
adjusted for multiplicity. p = min(1, 2 * min(share of draws <= 0, share >= 0)); a share of zero is
reported as p < 0.001 (= 1/B), never as 0. Holm's correction is applied within each family and, for the
two primary hypotheses, over those two. The 'model vs model' rows carry no p-value: they were specified
as descriptive (frozen plan section 6).

Usage: python inference_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "result_csv"
B = 1000
SOURCES = [("test", "stratified", "test_bootstrap_diffs_V02_stratified.csv"),
           ("test", "card", "test_bootstrap_diffs_V02_card.csv"),
           ("test", "time block (7 days)", "test_bootstrap_diffs_V02_timeblock.csv"),
           ("confirmation", "stratified", "test_bootstrap_diffs_V03_confirm_stratified.csv"),
           ("confirmation", "card", "test_bootstrap_diffs_V03_confirm_card.csv")]


def fmt_p(p):
    if pd.isna(p):
        return ""
    return "<0.001" if p < 1 / B else f"{p:.3f}"


rows = []
for period, scheme, fname in SOURCES:
    d = pd.read_csv(RESULTS / fname)
    d["family"] = np.where(d.kind.str.startswith("PRIMARY"), "primary (ensemble mean, Holm over 2)", d.kind)
    n = d.groupby("family").kind.transform("size")
    for (_, r), k in zip(d.iterrows(), n):
        model = r.a if pd.isna(r.b) or r.b == "" else f"{r.a} vs {r.b}"
        rows.append(dict(period=period, scheme=scheme, family=r.family, tests_in_family=int(k),
                         contrast=r.config, model=model, diff=r["diff"], lo=r.lo, hi=r.hi,
                         p=fmt_p(r.p), p_holm=fmt_p(r.get("p_holm")),
                         separated_after_holm=(bool(r.p_holm < 0.05) if pd.notna(r.get("p_holm"))
                                               else bool(r.separated))))
full = pd.DataFrame(rows)
full.to_csv(RESULTS / "inference_table_V01.csv", index=False)

fam = (full.groupby(["period", "scheme", "family"], sort=False)
       .agg(tests=("model", "size"), separated=("separated_after_holm", "sum")).reset_index())
fam.to_csv(RESULTS / "inference_families_V01.csv", index=False)
pd.set_option("display.width", 200)
print(fam.to_string(index=False))
print("\nprimary:")
print(full[full.family.str.startswith("primary")][["period", "scheme", "contrast", "diff", "lo", "hi", "p", "p_holm"]]
      .round(4).to_string(index=False))
