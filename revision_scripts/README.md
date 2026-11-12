# Revision scripts

Standalone scripts behind the analyses in the revision (see `Revision_Plan_V05.md`).
Run from this directory with `/opt/anaconda3/bin/python <script>`. They read
`../fraudTrain_encoded.csv` and write to `output/` (git-ignored); the copies kept as
results live in `../result_csv/`.

| Script | Produced | Runtime |
|---|---|---|
| `phase1_reproduce.py` | `models_metric_result_V05_replica.csv` (default); `models_metric_result_V04_repro_diff.csv` (`--unseeded`) | ~10 min |
| `phase2_splits.py` | `split_comparison_V01.csv` | ~10 min per config |
| `d7_sweep.py` | `class_ratio_{rus,ros,smote}{,_aucpr}_V06.csv`, `best_sampling_per_model_V01.csv` | ~205 min |
| `reshape_sweep.py [TAG]` | reshapes `output/d7_sweep_rows{TAG}.csv` into `class_ratio_*_V06{TAG}.csv`, `best_sampling_per_model_V01{TAG}.csv`, `sampling_mean_by_setting_V01{TAG}.csv` | seconds |
| `tuned_resampling_check.py [--client-features]` | does resampling help the *tuned* models? `output/tuned_resampling_notime{,_client}.csv` | ~1 h |
| `d8_tuning.py` | `hyperparameter_tuning_V01.csv`, `hyperparameter_trials_V01.csv`. **The only record of the search spaces** | ~110 min |
| `lime_table_a3.py [TAG]` | Appendix Table A3: `LIME_TableA3_V01_{TAG}.csv/.xlsx`, with bins from LIME's own discretiser. Also provides `prepare()`, the notebook's data split and SHAP/LIME sample | ~1 min |
| `fig1_flow.py` | paper Fig 1, the analysis flow as revised: `figures/fig1_flow.png` | seconds |
| `fig3_sampling.py` | paper Fig 3 from the trans_time-free sweep: `figures/fig3_sampling_notime.png` | seconds |
| `fig5_7_interpretation.py` | paper Figs 5-7: importance (from Table 5's CSV), SHAP beeswarms (from the executed notebook), CatBoost SHAP dependence (refit, checked against the notebook's AUC-PR) | ~5 min |
| `export_for_github.py` | re-exports `../for_GitHub/` (notebooks, HTML, these scripts, Readme) from the current notebooks, sanitising machine paths | ~2 min |
| `highlight_changes.py BASE REVISED OUT [--force ids]` | highlights in `OUT` every change from `BASE` (word-level, matched by paragraph id); used to make `Paper/main_V03.docx` from `main_V01.docx` and `main_V03_clean.docx` with `--force 5B2CB169` (Fig 1's caption, since its image changed); `Paper/CCF_MM_V02.docx` from `CCF_MM_7_10.docx` and `CCF_MM_V02_clean.docx` with `--force 5C5F8B51,6A3D9618,0F4EFA07,64866403,53BA6091,0311DC61` (the captions of Figs 1, 5, 6, 7, 8 and 9) | seconds |
| `client_features.py` | the seven D10 client features as a module; run it to verify `compute()` reproduces `fraudTrain_encoded_client.csv` | ~1 min |
| `raw_baseline_features.py` | J1 §3: the conventional baseline feature set (77 predictors). `../fraudTrain_raw_baseline{,_client}.csv` | ~3 min |
| `ablation_run.py --config C [--check]` | J1 §5: fit 8 models on one configuration and score the test set once. `--check` gates it against the committed tables. `output/test_{metric_result,operating_point,predictions}_*` | ~1.5 h |
| `ablation_table.py` | J1 §6: reshapes the four configurations' test tables into the ablation table and the four differences. `ablation_V01.csv`, `ablation_diffs_V01.csv` | seconds |
| `fig_ablation.py` | paper Fig 12: test AUC-PR by model and configuration, and the encoding's contribution with intervals. `figures/fig_ablation.png` | seconds |
| `baseline_variants.py` | J1 §8: how to handle `city`/`job` in the baseline, and which encoded features carry signal. `baseline_variants_V01.csv` | ~4 min |
| `rolling_origin.py` | J4 §7: repeated temporal evaluation — 3 expanding folds inside the first 80%, so the test period is never touched. `rolling_origin_V01.csv` | ~1 h |
| `uncertainty_check.py [--boot N]` | J4 §7: paired stratified bootstrap of the frozen test predictions; intervals on each metric, on model-vs-model differences and on the four J1 ablation differences. `test_bootstrap{,_diffs}_V01.csv` | ~3 min |
| `mixed_cards_rescore.py` | J2 §8: re-score the frozen test predictions on the mixed cards only, and how much of the client-feature gain survives. `mixed_cards_rescore_V01.csv` | ~1 min |
| `new_cardholder_check.py` | J3.4 §8: can the dataset test new-cardholder generalisation? `new_cardholder_check_V01.csv` | ~1 min |
| `fraud_only_cards.py` | J2 §8: the 75 fraud-only cards and what they do to the age finding. `fraud_only_cards_V01.csv`, `fraud_only_age_V01.csv` | ~1 min |
| `merchant_probe.py` | J1 §1: does `merchant` carry fraud signal beyond `category`? `merchant_probe_V01.csv` | ~1 min |
| `category_cost_check.py` | J3.2: is `category_3` ('Average item cost') fitted on data the models should not see? `category_cost_check_V01.csv` | seconds |
| `verify_restored.py` | — checks the four scripts against the committed files. Run with `CCF_ENCODED_VERSION=V24` since 2026-10-12 (the committed results are from the pre-fix encoded files) | — |

### Phase 7 (`Revision_Phase7_Plan_V01.md`, frozen at `5351e35`)

The encoded files were rebuilt on 2026-10-12 with two encoding bugs fixed. Every script that reads
them goes through `data_paths.py`: unset, the corrected files; `CCF_ENCODED_VERSION=V24`, the
archived pre-fix files in `../_old/`, which every result committed before then was computed from.

| Script | Produced | Runtime |
|---|---|---|
| `rebuild_encoded.py [--dry-run]` | the corrected `fraudTrain_encoded{,_client}.csv` and city lookup, from encoding_V25's own city cells; reproduces the V24 bug first. `encoding_fix_V01.csv`, `category_codes_V25.csv` | ~3 min |
| `encoding_audit.py` | corrected Table A1: `category_encoding_audit_V01.csv` (code table = workbook = data; interaction definitions) | ~1 min |
| `fit_scope_check.py` | which learned steps saw the test period, and whether it matters: `fit_scope_check_V01.csv` | ~1 min |
| `grid_bounds_check.py --build` / `--compare` | the location grid refitted on training-split bounds (configs `p7_grid_train[_client]` in `ablation_run.py`) against the runs of record: `grid_bounds_check_V01.csv` | ~1.5 h (the two refits) |
| `package_predictions.py` | bundles the saved test and confirmation predictions with a README and SHA256SUMS for a data archive: `output/CCF_predictions_V01.zip` | ~1 min |
| `build_ccf_mm_v02.py`, `build_ccf_mm_v03.py` | the manuscript builds: `Paper/CCF_MM_V02_clean.docx` from `CCF_MM_V01_clean.docx`, then `CCF_MM_V03_clean.docx` from V02 (title, abstract, keywords, the grid check, the §6 wording). Highlighted copies with `highlight_changes.py` as above | seconds |
| `feature_dictionary.py` | the feature dictionary: `feature_dictionary_V01.csv` | ~1 min |
| `data_paths.py` | (module) current or archived encoded files | — |
| `ablation_features.py [--only a,b]` | hybrid, targeted-ablation, reduced and catnative feature sets in `../ablation_data/` (git-ignored) | ~10 min |
| `run_phase7_tuning.sh` | searches for corrected E, E+H, hybrid ×2; top-3 refit for R, R+H → `hyperparameter_tuning_V03_*.csv` | ~8–10 h |
| `catnative_baseline.py`, `run_phase7_catnative.sh` | CatBoost with native categorical handling: untuned baseline, search, table | ~2 h |
| `run_phase7_runs.sh` | every Phase 7 fit and analysis after the searches, the model_V26 runs of record, the sweep | ~12 h |
| `operating_points.py` | fixed threshold vs fixed capacity (0.5% per day; of the window), realised workloads: `operating_points_V01.csv` | ~1 min |
| `uncertainty_check.py --scheme {stratified,card,timeblock}` | Phase 7 bootstrap: `test_bootstrap{,_diffs}_V02_<scheme>.csv`, with the Holm-adjusted primary hypotheses | ~5 min each |
| `explain_compare.py [--quick]` | raw vs encoded explanations on the same transactions; grouped and natural-prevalence SHAP; permutation importance | ~1–2 h |
| `lime_grouped.py [--quick]` | LIME on source variables with derived features regenerated; fidelity and seed stability | ~1 h |
| `run_model_v26.py` | the four model_V26 runs of record into `../executed/` | hours |
| `build_fraudtest.py` | fraudTest features in `../confirm_data/` (git-ignored); the builders must first reproduce fraudTrain's files exactly | ~15 min |
| `fig_ablation.py --phase7`, `fig_targeted.py`, `pr_figure_paired.py --phase7`, `fig5_7_interpretation.py --phase7` | the corrected-data figures: `figures/fig_ablation_V02.png`, `fig_targeted_V01.png`, `pr_curves_test_V02.png`, `fig5/6/7_*_V02.png` | ~5 min |
| `run_phase7_tail.sh` | the script-against-notebook guards, the resampling sweep and the tuned-resampling check (re-run after the 0.5-threshold fix) | ~5 h |
| `confirm_run.py [--guard-only]` | **the single-use scoring of fraudTest**, last; refuses to run twice | ~1 h |

### Second revision (`Revision_Plan_V08.md`; Phase 8 plan `Revision_Phase8_Plan_V01.md`, frozen at `10ceada`)

| Script | Produces | Time |
|---|---|---|
| `age_population_check.py fraudTrain.csv` | diagnostic of the review's age point: `fraud_only_cards.py`'s "mixed only" column is the 908 cards without fraud-only ones | ~1 min |
| `age_populations.py` | age gap over 983 / 908 / 687 cards with cardholder bootstrap; Table A12 on every transaction: `age_populations_V01.csv`, `age_time_table_V02.csv` | ~2 min |
| `knn_scaled.py` | k-NN on standardised features, k chosen on validation, beside the unscaled run of record: `knn_scaled_V01.csv` | ~20 min |
| `inference_table.py` | every bootstrap contrast with p and Holm p, and the per-family summary: `inference_table_V01.csv`, `inference_families_V01.csv` | seconds |
| `rolling_origin_design.py` | fold dates and counts, and which published window each block lies in: `rolling_origin_design_V01.csv` | ~1 min |
| `explain_grouped.py --configs R,E[,E+]`, `--combine` | permutation by source field (primary), per feature (marginal), 10 repeats; E+'s category split (semantic vs indicators, permutation and SHAP): `explain_grouped_V01.csv` | ~1 h per config |
| `run_phase8.sh` | C1: E+ feature sets, tuning (RF, CatBoost), fit + test → `hyperparameter_tuning_V03_p7_eplus*.csv`, test tables | ~4 h |
| `phase8_compare.py` | C1 paired bootstrap (stratified, card): `phase8_bootstrap_diffs_V01_*.csv`, `phase8_test_aucpr_V01.csv` | ~5 min |
| `matched_compression.py` | C2: E against 20 random time bandings and 20 random category-code assignments: `matched_compression_V01.csv` | ~3–6 h |
| `labels_v04.py` | (module) the renamed construct labels, used by the V04 tables and figures | — |
| `fig5_7_interpretation.py --v04 --skip-fig7` | Figs 5–6 with renamed labels; Fig 6 recomputed from SHAP (guard: validation AUC-PR equals the notebook's) → `fig5/6_*_V03.png` | ~15 min |

These scripts first ran from a temporary session directory that was later wiped. They
were restored on 2026-09-14 and verified: on the partial re-runs listed in
`verify_restored.py`, every checked value matches the committed results exactly
(max |diff| = 0). That includes all 80 hyperparameter-search trials for logistic
regression and the decision tree, so the search's random stream is reproduced as well.

The restored `d8_tuning.py` also writes each trial's hyperparameters, as a JSON column.
The original did not, so the committed `hyperparameter_trials_V01.csv` has scores but
no parameters.

### Submission (CCF_MM_V11, 2026-11-12)

| Script | Produces | Time |
|---|---|---|
| `fig_ablation.py --v11` | Fig 2 with the standardised k-NN of Table 4 (`knn_scaled_V01.csv`) and the manuscript's R/E/H labels: `figures/fig_ablation_V03.png` | seconds |
| `appendix_tables_a5_a13.py` | Tables A5–A9 and A13, which had no generating code; checks all 65 published counts and records the age and amount boundaries they use: `appendix_tables_a5_a13_V01.csv` | ~1 min |
| `ccf_mm_v11_figures.py IN OUT` | puts the 300-dpi source PNGs of Figs 1, 2 and 6 back into the manuscript (V10 carried downscaled or redrawn copies), checked by checksum | seconds |
| `build_submission_package.py` | the code package for GitHub + Zenodo: `../submission_package/CCF_code_package/` and `.zip` (git-ignored) | ~1 min |
