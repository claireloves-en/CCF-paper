# Revision Phase 7 — frozen analysis plan
Paper: *Protecting Consumers from Credit Card Fraud* · Date: 2026-10-12
Round plan: `Revision_Plan_V06.md` · Review: `revision_plan_12_oct.md` · Evidence: `Revision_Phase7_Results_V01.md`

**This plan is frozen by the git commit that adds it.** Every analysis below is specified before it
is run on the corrected data. A change after the freeze is allowed only as a recorded deviation: the
results file names it, gives the reason, and reports the frozen version as well where it can be run.

---

## 1. Why there is a Phase 7

The correctness audit that `revision_plan_12_oct.md` §2 asked for found two bugs in
`MLProj01_fraudTrain_encoding_V24.ipynb`. Both are baked into every result built on the encoded
files: configurations 3 and 4 of the J1 ablation, Tables 2–5 and 8, Figs 3–7 and 12, SHAP, LIME,
and the bootstrap, rolling-origin and mixed-card results for those configurations.

- **The city features were on the wrong rows.** The USDA income table kept its US and state totals.
  With " County" stripped, these collided with same-named counties (Arkansas, Hawaii, New York,
  Utah, DC, …), so the city lookup held 935 rows for 928 cities. Cell 23 merged it into the
  transactions (1,312,122 rows for 1,296,675) and inserted the result back by index. Every row
  after row 29 therefore received the five city features of a different transaction:
  **1,290,523 rows (99.5%)**. No label was involved, so there is no leakage — but `city_1..city_4b`
  were noise by construction.
- **`personal_care` was coded `'001010001'`** — nine digits, so every field shifted by one. Its
  90,758 transactions (7.0%) were encoded as non-essential, second cost quartile, entertainment;
  the design (and the curated workbook) says essential, lowest quartile, household/wellbeing.
- **Two smaller defects in the same cells.** The state totals were counted when the income-quartile
  cut-offs were set, which moves 18 cities' `city_1`. The 5G table's "Parish" patch appended copies
  in every state (e.g. "Jefferson Parish, AL") that entered the 5G quintiles, which moves 6 cities'
  `city_4b`.

Fixed in `MLProj01_fraudTrain_encoding_V25.ipynb`. `revision_scripts/rebuild_encoded.py` applies the
fix. It runs the notebook's own city cells — V24's, which reproduce the committed files and the bug
exactly, then V25's — and asserts that the other 16 columns are unchanged. The old files are kept as
`_old/fraudTrain_encoded{,_client}_V24.csv`, and `CCF_ENCODED_VERSION=V24` makes the scripts read
them (`revision_scripts/data_paths.py`), so every committed result remains reproducible.
`result_csv/encoding_fix_V01.csv` gives the counts.

**A procedural asymmetry found at the same time.** The encoded configurations' hyperparameters were
selected with `refit_top_k.py` (the best 3 search candidates refitted on the full training set, best
kept). The raw-baseline configurations were not — only the search's single winner was refitted.
Phase 6's J1 comparison therefore did not give the two sides the same selection procedure, and the
difference favoured the encoding. Phase 7 applies the top-3 refit to all four configurations.

## 2. Evaluation history — what has already seen the test window

The test window (2020-03-06 08:00 to 2020-06-21 12:13; 259,284 transactions, 1,538 fraudulent) has
been scored many times:

| When | Analysis | Status |
|---|---|---|
| submission | shuffled split; test scored | superseded (leakage) |
| round 1 (model_V24/V25) | chronological split, encoded configurations, one test evaluation each | **original**, but on the buggy encoding |
| Phase 6 | raw baselines, bootstrap, mixed-card re-scoring | post-hoc |
| Phase 7 (this plan) | every configuration below | post-hoc |

**Consequences for the text.** The paper no longer says the test set was "used once". It says that
model choices were made on validation, that the test window was then evaluated for the original
configurations, and that the revision's analyses re-use that window and are post hoc. **No Phase 7
choice is made on test results.** Every selection — hyperparameters, thresholds, model families,
ablation arms, bootstrap schemes, the capacity rule — is fixed here or on validation.

**Untouched data — `Found_DataSet/fraudTest.csv`.** This is Sparkov's later period: 2020-06-21 12:14
to 2020-12-31, 555,719 transactions, 2,145 fraudulent (0.386%), 924 cards of which 16 are new. It has
never been read by any model or analysis in this project, apart from the row, date-range and
unseen-level counts taken to write this plan. **It is scored once, last** (§9). It is
within-simulator out-of-time confirmation. It is not real-world validation and the paper does not
call it that.

## 3. Data and split

- **E**: corrected `fraudTrain_encoded.csv` (27 predictors). **E+H**: corrected
  `fraudTrain_encoded_client.csv` (34). **R**: `fraudTrain_raw_baseline.csv` (77; unchanged,
  never used the buggy file). **R+H**: `fraudTrain_raw_baseline_client.csv` (84).
- Hybrid and ablation sets are built by `revision_scripts/ablation_features.py` into
  `ablation_data/` (§4).
- Split unchanged: chronological 64/16/20 on `trans_time` quantiles. Train ≤ 2019-12-10 18:00
  (829,920 rows / 4,797 fraud), validation to 2020-03-06 07:58 (207,471 / 1,171), test as above.
- No resampling. Predictors are taken in data-file column order everywhere in Phase 7, so no
  configuration inherits `d8_tuning.py`'s hardcoded order.

## 4. Configurations

### 4.1 Main benchmark — eight models, as before

| # | Config | Features | Selection |
|---|---|---|---|
| 1 | `p7_notime_raw` | R | Phase 6 search (unchanged trials) + **top-3 refit** → `hyperparameter_tuning_V03_notime_raw.csv` |
| 2 | `p7_notime_raw_client` | R+H | same → `…V03_notime_raw_client.csv` |
| 3 | `p7_notime` | E (corrected) | **new** 40-draw search + top-3 refit → `…V03_p7_notime.csv` |
| 4 | `p7_notime_client` | E+H (corrected) | same → `…V03_p7_notime_client.csv` |

The selection procedure is identical for all four and is the Phase 2 one:
- 40 random draws per model (`d8_tuning.py` grids, seed 1200), each fitted on a stratified 25%
  training subsample and scored on validation AUC-PR;
- the best 3 refitted on the full training set (`refit_top_k.py`), best kept;
- the kept set is used only if it beats the model's untuned baseline on the same data
  (`d7_sweep.py --only-no-sampling`; `build_tuned_table.py`), otherwise the untuned set is used.

AdaBoost and the other models follow the rule as it falls.

### 4.2 Diagnostic configurations — random forest and CatBoost only

Two model families, to keep the workload proportionate (review §3): **random forest** (the leading
model on E in Phase 6) and **CatBoost** (leading on R, R+H and E+H).

| Config | Features | Hyperparameters | Question |
|---|---|---|---|
| `p7_hybrid` | R + E (101) | own search, as §4.1, these 2 models | does E add anything when R is kept? |
| `p7_hybrid_client` | R + E + H (108) | own search | the same, with history |
| `p7_abl_cat_onehot` | E, 8 category features → one-hot `category` (14) | E's kept set | the category encoding's cost |
| `p7_abl_plus_onehot` | E + one-hot `category` | E's | is anything lost by discarding the original field? |
| `p7_abl_hour` | E, `trans_time_grp` (3 bands) → `hour_of_day` (24) | E's | the time encoding's cost |
| `p7_abl_plus_dow` | E + `day_of_week` | E's | information E never carried |
| `p7_abl_no_interactions` | E − 2 interaction terms | E's | do the interactions help? |
| `p7_abl_no_external` | E − `city_1..4b`, `job_1..4` | E's | the externally enriched features; **also the configuration that excludes every source dated after the transaction window** |
| `p7_abl_no_geo_occ` | E − all 14 geographic/occupational | E's | formalises Phase 6 §8C with tuned models |
| `p7_red_raw` / `p7_red_enc` | amt + category + time, raw / encoded | E's | the reduced pair already quoted in §4.7 |
| `p7_catnative` / `p7_catnative_client` | R with `category`, `state`, `city`, `job` as strings, CatBoost `cat_features` | own CatBoost search, as §4.1 | does the result depend on the conventional preprocessing? Sensitivity, reported apart from the benchmark |

Holding the ablation arms at E's hyperparameters isolates the feature change. It disadvantages an
arm whose best hyperparameters differ, so the arms measure the effect of the change *at E's
setting*, and the text says so.

**Information differences the text must state** (the review's comparison check):
- R carries hour of day and day of week; E carries 3 time bands and no day of week.
- R has raw lat/long and merchant lat/long; E has a 32×32 grid cell for each and the geodesic
  distance.
- R has age as a float; E has it as an integer.
- R has `state` one-hot; E has a 2-number grid.
- R has `city`/`job` as ordinal codes; E has their external attributes.
- `merchant` and `zip` are in neither.
- E maps the 14 categories onto 11 distinct codes. Three pairs are indistinguishable: home /
  kids-pets, health-fitness / personal care, online misc / online shopping
  (`category_encoding_audit_V01.csv`).

## 5. Metrics and operating points

- **Primary metric:** test AUC-PR from predicted probabilities.
- **Fixed threshold.** The threshold that flags 0.5% of *validation* transactions is applied
  unchanged to test. Report the threshold, realised test alert rate, alert count, TP, FP, precision
  and recall. (`ablation_run.py` now writes the counts.)
- **Fixed capacity.** Within each calendar day of the evaluation window, flag the top
  `round_half_up(0.005 × n_day)` transactions by score. Ties at the cut-off are broken by
  chronological row order (earlier first). No label is used to set the cut. The secondary variant
  takes the top `round_half_up(0.005 × n)` over the whole window. Report alert count, TP, FP,
  precision and recall. (`revision_scripts/operating_points.py`, from saved predictions; no refits.)
- The threshold-0.5 metrics stay in the appendix for continuity.

## 6. Comparisons and multiplicity

**Primary — two hypotheses, Holm over the two.** Ensemble = mean of random forest, CatBoost, XGBoost
and LightGBM test AUC-PR.
- **P1:** ensemble(3) − ensemble(1): encoded minus raw, without history.
- **P2:** ensemble(4) − ensemble(2): encoded minus raw, with history.

Each gets a paired stratified-bootstrap 95% interval and a two-sided bootstrap p-value
(2 × the smaller tail share of the difference's draws at 0). Holm-adjusted p-values are reported.

**Secondary — exploratory, Holm within each family, reported as such:**
- per-model 3−1 and 4−2 (8 + 8);
- the client-history gains 2−1 and 4−3 (8 + 8);
- ensemble pairs within a configuration (6 per configuration);
- hybrid minus R and hybrid_client minus R+H (2 models each);
- each ablation arm minus E (2 models × 9 arms);
- catnative minus R and catnative_client minus R+H (CatBoost).

## 7. Uncertainty

`uncertainty_check.py --scheme {stratified,card,timeblock}`, 1,000 resamples each, seed 1200. The
same resample is applied to every model and configuration (paired).

- **stratified** (primary, as Phase 6): fraudulent and genuine rows resampled separately.
- **card** (sensitivity): resample the 927 cards present in the test window with replacement. Each
  card's transactions stay together, which preserves within-cardholder dependence. Not
  class-stratified.
- **timeblock** (sensitivity): circular moving-block bootstrap over the window's 108 calendar days,
  in 7-day blocks. This preserves short-range temporal dependence.

The text states that the card and block schemes address different dependence structures, and that
all three are conditional on the fitted models. They resample frozen predictions, so they carry no
training or tuning uncertainty. The rolling-origin check (§8) speaks to that.

## 8. Re-runs of Phase 6 analyses on the corrected data

All on the `p7_*` configurations with the V03 hyperparameters:
- `ablation_table.py` and `fig_ablation.py`;
- `mixed_cards_rescore.py` (configurations 3 and 4);
- `rolling_origin.py` (3 folds inside the first 80%, 5 models, configurations 1, 3 and 4 as in
  Phase 6);
- `baseline_variants.py` (re-measures the "14 noise features" result);
- the full `d7_sweep.py` resampling sweep on corrected E, then `tuned_resampling_check.py`
  (Table 2, Fig 3).

The model notebook of record becomes `MLProj01_fraudTrain_model_V26.ipynb`. It adds a `RAW_BASELINE`
switch and the V03 `TUNED` entries, with four runs of record (configurations 1–4) that produce
Tables 3–5, Figs 4–7, SHAP and LIME. `ablation_run.py --check` must reproduce its test tables to
1e-9.

## 9. Confirmation on fraudTest — once, last

1. `build_fraudtest.py` builds E, E+H, R, R+H, hybrid and catnative features for fraudTest:
   - the corrected encoding;
   - its 12 new cities through the same county pipeline;
   - its 3 new jobs matched by hand to O*NET/OEWS titles, documented in the provenance appendix;
   - R's ordinal codes taken from fraudTrain, with unseen levels coded −1;
   - client history computed over the concatenated fraudTrain + fraudTest series;
   - whether the 16 new cards are fraud-only, reported.
2. `confirm_run.py` refits each frozen configuration on the training split with its kept
   hyperparameters and sets thresholds on validation.
   - **Guard:** its test-window predictions must equal the saved ones to 1e-9 before fraudTest is
     read.
   - It then scores fraudTest once, writes the outputs and refuses to run again if they exist. It
     logs the commit hash of this plan.
3. Configurations: 1–4 (8 models), `p7_hybrid_client` (2 models), `p7_catnative_client` (CatBoost).
4. Report:
   - AUC-PR;
   - both operating points (fixed capacity per day of fraudTest);
   - P1 and P2 with the stratified and card intervals.

   fraudTest's base rate (0.386%) is lower than the test window's (0.593%), so AUC-PR levels are
   compared within a period, never across.
5. Runs only after every other Phase 7 result is written to `Revision_Phase7_Results_V01.md` and
   committed.

## 10. Interpretation (Stage 4)

On random forest and CatBoost, configurations 1 and 3, the same validation cases:
- **4A — paired explanations.** Raw-model SHAP with readable labels ("Category = grocery (in
  store)", "Hour of day", amounts in $) beside the encoded model's, for the same 1,000 + 1,000
  balanced case-control sample and 12 example transactions chosen by a fixed rule: the 3 highest-
  and 3 lowest-scored frauds and genuines under configuration 3.
- **4B — dependent features.**
  - Grouped LIME: perturb `category`, `city` and `job` as source variables and regenerate their
    derived features from the dictionary through a `predict_fn` wrapper. Report local fidelity
    (LIME's R²) and stability over 5 seeds (Spearman ρ of feature ranks; top-5 overlap).
  - SHAP: TreeExplainer's tree-path-dependent default, plus an interventional run with a
    1,000-row natural-prevalence training background as sensitivity, and SHAP grouped by source
    variable.
- **4C — what the summaries represent.**
  - The balanced figures are labelled case-control.
  - Add a natural-prevalence summary: mean |SHAP| over a random 20,000-row validation sample.
  - Table 5 drops the cross-model "average". It shows native importance per model, plus validation
    permutation importance in AUC-PR as one consistently defined cross-model measure.
- **Claim level:** "semantically explicit / theory-informed"; investigator understanding not tested
  (authors' decision, 2026-10-12).

## 11. What the results will be allowed to say

Set out before the corrected numbers exist.

| Outcome of P1 and P2 | What the paper says |
|---|---|
| both negative, Holm-separated | as Phase 6: the encoding costs accuracy; it buys explicit semantics, and the cost is stated |
| either within its interval | no measurable cost on that comparison; interpretability is not traded against accuracy there |
| positive | the encoding adds predictive information on that comparison; say which, and check whether the hybrid agrees |

The hybrid decides the second question independently. If R+E+H beats R+H, the encoding adds
structure as additional features, whatever replacement costs. If it does not, the encoding's value
is semantic only. The ablation arms attribute any gap to families. The text attributes a gap to a
family only where that family's arm moves AUC-PR by more than its interval.

## 12. Compute

| Step | Cost |
|---|---|
| corrected-E untuned baselines + 2 searches + top-3 refits | ~6–8 h |
| raw top-3 refits | ~1 h |
| hybrid searches (2 models × 2) | ~2–3 h |
| catnative searches (CatBoost × 2) | ~2 h |
| `ablation_run` fits (all configurations) | ~2 h |
| full sweep | ~3.5 h |
| rolling origin | ~1 h |

Everything runs in the background with `DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib`.
