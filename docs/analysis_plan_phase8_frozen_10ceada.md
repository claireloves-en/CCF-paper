# Revision Phase 8 — analysis plan (frozen before any run)
Round plan: `Revision_Plan_V08.md`, Stage C · Review: assessment of `CCF_MM_V03` (2026-10-27), items 1 and 4
Date: 2026-10-27 · Frozen at the commit that adds this file; results go to `Revision_Phase8_Results_V01.md`

The review asks two things Phase 7 cannot answer:
- whether semantic descriptors can be **kept** without the information lost to coarsening;
- whether the encoding's theory-chosen coarsening is **any better than arbitrary coarsening** of the same
  granularity.

This plan fixes the configurations, models, procedure, contrasts and expectations before any of them is run.
Deviations are logged in the results file's §0.

**Data and constraints.**
- Corrected encoded files (V25), the chronological 64/16/20 split, `trans_time` not a predictor.
- No resampling.
- The test period (20%) has been scored many times; these are post hoc analyses and are labelled as such.
- **`Found_DataSet/fraudTest.csv` is not read.**

## C1 — keeping the semantic attributes and the detail (E⁺)

**Configurations** (built by `ablation_features.py`, sets `eplus` and `eplus_client`):
- **E⁺** = E with `trans_time_grp` replaced by `hour_of_day` and `day_of_week`, plus the 14 one-hot `category`
  indicators. Everything else is E's: the 8 semantic category attributes, the county, occupation and location
  features, amount, age, gender and city population. 27 − 1 + 2 + 14 = 42 predictors.
- **E⁺ + H**: the same plus the seven client-history features.

**Models:** random forest and CatBoost, the two interpreted ensembles, as for the hybrid.

**Tuning:** identical to the hybrid (`run_phase7_tuning.sh`'s hybrid block):
1. untuned baseline (`d7_sweep.py --only-no-sampling`);
2. 40-draw random search (`d8_tuning.py`);
3. best-3 refit on the full training set (`refit_top_k.py`);
4. keep-or-untuned table (`build_tuned_table.py --version V03`).

Then `ablation_run.py --config p7_eplus[_client]` fits and scores the test period.

**Contrasts.** Test AUC-PR, per model and for the two-model mean, with paired bootstrap of fixed predictions
(1,000 draws, seed 1200). Stratified is primary; card is the sensitivity scheme.

| Family | Contrasts |
|---|---|
| C1-primary | E⁺ − R; (E⁺+H) − (R+H) (two-model mean; Holm over the two) |
| C1-secondary | E⁺ − E; (E⁺+H) − (E+H); per-model versions of all four (Holm within family) |

**Expectation, recorded before the run:** E⁺ ≈ R, within ±0.005, because the hybrid R+E is indistinguishable
from R and E⁺ carries the same time and category information as R. E⁺ − E should be close to E's deficit
(about +0.03 to +0.04).

**C1b — explanations of E⁺** (`explain_grouped.py`, random forest and CatBoost on E⁺; R and E alongside for
reference):
- **Grouped permutation importance** by source field: the field's columns shuffled jointly, 10 repeats, mean and
  SD, validation AUC-PR.
- **Within the category field**, the semantic attributes (8 columns) and the indicators (14 columns) are each
  permuted as a block, and the drops are compared.
- **Tree-path-dependent SHAP** on the balanced validation sample (1,000 + 1,000, seed 1200). It is summed over
  the category field, then split into the semantic and indicator sub-blocks (share of the field's mean
  |SHAP|).

The question is descriptive: when both are available, does the model's reliance on category go through the
semantic attributes or through the indicators? No hypothesis test.

## C2 — matched compression: theory-chosen versus arbitrary coarsening

Script `matched_compression.py`. Configuration E, random forest and CatBoost at **E's kept hyperparameters**
(`hyperparameter_tuning_V03_p7_notime.csv`), fitted as in `ablation_run.py`. Draw 0 is E itself and must
reproduce the saved test predictions to 1e-9 before any draw is fitted.

**C2-time, 20 draws.**
- `trans_time_grp` (E: hours 0–7, 8–15, 16–23, coded 0, 1, 2) is replaced by three random contiguous bands on
  the 24-hour circle.
- Three distinct cut hours are drawn uniformly, subject to every band being at least 2 hours long and the
  partition differing from E's.
- Bands are coded 0, 1, 2 in clockwise order from the band containing hour 0.

**C2-category, 20 draws.**
- The 14 rows of the category code table (8 attributes each, interactions included) are assigned to the 14
  categories by a uniformly random permutation other than the identity.
- Every draw therefore keeps the same 8 attributes, the same 11 distinct profiles with the same merge
  structure, and internally consistent interaction terms. Only the theory's assignment of codes to categories
  is lost.

**Seeds:** draw *i* uses `numpy.random.default_rng(1200 + i)` for C2-time and `default_rng(2200 + i)` for
C2-category.

**Measure.** Test AUC-PR of each model and of the two-model mean (validation AUC-PR reported alongside). For each
arm: E's value, the draws' median and range, and E's percentile rank among the 20 draws plus E itself (rank / 21).

**Interpretation, fixed before the run:**
- E in the top quarter of its arm's distribution → the theory-chosen coarsening loses less than arbitrary
  coarsening of the same granularity.
- E within the middle half → the cost is that of compression, and theory confers no measurable advantage in
  what is retained.
- E in the bottom quarter → the theory-chosen coarsening is worse than arbitrary.

No p-value: with 20 draws the percentile is descriptive.

## Outputs

- `result_csv/hyperparameter_tuning_V03_p7_eplus{,_client}.csv`
- `result_csv/test_metric_result_V01_temporal_p7_eplus{,_client}.csv` (+ operating points)
- `result_csv/phase8_bootstrap_diffs_V01_{stratified,card}.csv`
- `result_csv/explain_grouped_V01.csv`
- `result_csv/matched_compression_V01.csv`
