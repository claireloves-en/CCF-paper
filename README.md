# When theory meets prediction: feature design and interpretive limits in a synthetic credit card fraud benchmark

Code, reference data and result tables for the paper above (Journal of Big Data, revised
submission). Licence: MIT for the code; the reference data keep their own terms (see `LICENSE`).

## Data

The transactions are the public *Credit Card Transactions Fraud Detection Dataset* (Kaggle),
generated with the Sparkov Data Generation tool. They are **not included**. Download
`fraudTrain.csv` and `fraudTest.csv` and place `fraudTrain.csv` in this directory (the root of the
package). `fraudTest.csv` (the later confirmation period) is read by `build_fraudtest.py` and
`confirm_run.py` from `Found_DataSet/` two levels above this directory (the original project
layout: `<project>/Found_DataSet/` beside `<project>/Python_proj_code/MLProj01_US/`); it was scored
exactly once, on 14 October 2026, and `revision_scripts/confirm_run.py` refuses to run again if its
outputs exist.

All derived data files (`fraudTrain_encoded.csv`, `fraudTrain_encoded_client.csv`,
`fraudTrain_raw_baseline*.csv`, `ablation_data/`, `confirm_data/`) are rebuilt by the code below.

## Layout

| Path | Contents |
|---|---|
| `MLProj01_fraudTrain_EDA_V10.ipynb` | exploratory analysis (Appendix 5 figures) |
| `MLProj01_fraudTrain_encoding_V25.ipynb` | the theory-informed encoding (E): writes `fraudTrain_encoded.csv` |
| `MLProj01_fraudTrain_clientfeatures_V01.ipynb` | the seven cardholder-history features (H), with a causality check: writes `fraudTrain_encoded_client.csv` |
| `MLProj01_fraudTrain_findratio_V10.ipynb` | the resampling sweep (code; the reported sweep was run by `revision_scripts/d7_sweep.py`, the same code) |
| `MLProj01_fraudTrain_model_V26.ipynb` | the model notebook: 8 models, operating points, SHAP, LIME |
| `executed/model_V26_*.ipynb` | the four executed runs of record, with outputs: E (`p7_notime`), E + H (`_client`), R (`_raw`), R + H (`_raw_client`) |
| `revision_scripts/` | every other analysis in the paper (see its `README.md` and the map below) |
| `encoding_table/` | the hand-authored category, city and occupation codings, the cached lookup tables, and the reference data (`ref_data/original/`) |
| `result_csv/` | the result tables behind every table and figure, and `figures/` |
| `docs/` | the analysis plans as frozen before the analyses (git commits `5351e35`, `10ceada`), the data and pipeline provenance, and Additional file 1 (feature dictionary) |

## Running

Set up the environment in `environment.txt`. On macOS, start every process with
`DYLD_LIBRARY_PATH` pointing at Homebrew's OpenMP (`/usr/local/opt/libomp/lib` or
`/opt/homebrew/opt/libomp/lib`): LightGBM and XGBoost/scikit-learn otherwise load two OpenMP
runtimes, which can crash without a traceback.

1. `MLProj01_fraudTrain_encoding_V25.ipynb`, or `python revision_scripts/rebuild_encoded.py`,
   which rebuilds the encoded file from the cached lookup tables without the NLP stack.
2. `MLProj01_fraudTrain_clientfeatures_V01.ipynb` (or `revision_scripts/client_features.py`).
3. `python revision_scripts/raw_baseline_features.py` (the conventional representation R, 77
   predictors) and `python revision_scripts/ablation_features.py` (hybrid, one-change, reduced,
   native-categorical and E⁺ feature sets).
4. Hyperparameter search: `revision_scripts/run_phase7_tuning.sh`, `run_phase7_catnative.sh`,
   `run_phase8.sh` (E⁺). Selected values: `result_csv/hyperparameter_tuning_V03_*.csv`;
   every trial: `hyperparameter_trials_V03_*.csv`.
5. Fits and analyses: `revision_scripts/run_phase7_runs.sh`, `run_phase7_tail.sh`, then the
   scripts in the map below. `revision_scripts/run_model_v26.py` re-executes the four model
   notebooks; `ablation_run.py --check` reproduces their test tables to 1e-16.
6. `revision_scripts/confirm_run.py` — the single scoring of the later period (already done;
   see `result_csv/confirm_run_V01.log`).

Scripts run from `revision_scripts/`, read `../fraudTrain_encoded*.csv` and write to
`revision_scripts/output/`; the copies kept as results are in `result_csv/`. Total compute for
everything is roughly 30–40 hours on a laptop. `revision_scripts/verify_restored.py` checks
results computed before the encoding fix of 12 October 2026 and needs the pre-fix files, which are
not distributed.

## Where each table and figure comes from

| Manuscript | Script | Result file |
|---|---|---|
| Table 4, Fig. 2: test AUC-PR, E − R differences | `ablation_run.py`, `ablation_table.py`, `uncertainty_check.py --scheme stratified/card/timeblock`, `fig_ablation.py --v11` | `ablation_V02.csv`, `test_bootstrap_V02_*.csv`, `test_bootstrap_diffs_V02_*.csv`, `fig_ablation_V03.png` |
| Table 4 / Fig. 2, k-NN row (features standardised on training, k chosen on validation) | `knn_scaled.py` | `knn_scaled_V01.csv` |
| Table 5: E⁺ | `run_phase8.sh`, `phase8_compare.py`, `explain_grouped.py` | `phase8_test_aucpr_V01.csv`, `phase8_bootstrap_diffs_V01_*.csv`, `explain_grouped_V01.csv` |
| Fig. 3: hybrid and one-change configurations; native-categorical CatBoost | `ablation_run.py`, `fig_targeted.py` | `ablation_targeted_V01.csv`, `fig_targeted_V01.png` |
| §4.2 and Table A20: random time bands and category codes | `matched_compression.py` | `matched_compression_V01.csv` |
| §4.3: history gains without fraud-only cards | `mixed_cards_rescore.py` | `mixed_cards_rescore_V02.csv` |
| Fig. 4: precision–recall curves | `pr_figure_paired.py --phase7` | `pr_curves_test_temporal_p7_*.csv`, `pr_curves_test_V02.png` |
| Table 6: operating rules | `operating_points.py` | `operating_points_V01.csv` |
| Table 7, Fig. 5: native importance | executed E notebook, `fig5_7_interpretation.py --v04` | `feature_rank_table_V06_temporal_p7_notime.csv`, `fig5_importance_notime_V03.png` |
| Table 8, Table A19: permutation importance | `explain_grouped.py` | `explain_grouped_V01.csv` |
| Fig. 6, Fig. 7, natural-prevalence and interventional SHAP, §4.6 example | `fig5_7_interpretation.py`, `explain_compare.py` | `fig6_shap_beeswarm_notime_V03.png`, `fig7_shap_dependence_catboost_notime_V02.png`, `shap_grouped_V01.csv`, `shap_summary_V01.csv`, `explain_examples_V01.csv` |
| Table A4: LIME fidelity and stability | `lime_grouped.py` | `lime_grouped_V01.csv`, `lime_grouped_stability_V01.csv` |
| §4.7, Table A12: age populations | `age_populations.py`, `fraud_only_cards.py` | `age_populations_V01.csv`, `fraud_only_cards_V01.csv` |
| Table A14: age × time band | `age_populations.py` | `age_time_table_V02.csv` |
| Tables A5–A9, A13 | `appendix_tables_a5_a13.py` (checks every published count) | `appendix_tables_a5_a13_V01.csv` |
| Fig. 8, Table 9: resampling | `d7_sweep.py --drop-trans-time`, `reshape_sweep.py`, `tuned_resampling_check.py`, `fig3_sampling.py` | `sampling_mean_by_setting_V01_p7_notime.csv`, `best_sampling_per_model_V01_p7_notime.csv`, `tuned_resampling_p7_notime*.csv`, `fig3_sampling_notime_V02.png` |
| Table 10: later confirmation period | `build_fraudtest.py`, `confirm_run.py` | `confirm_metrics_V01.csv`, `test_bootstrap*_V03_confirm_*.csv` |
| Tables A10, A11: hyperparameters | `d8_tuning.py` (search spaces), `refit_top_k.py`, `catnative_baseline.py` | `hyperparameter_tuning_V03_*.csv` |
| Table A16: encoded models at two thresholds | executed E notebook | `test_metric_result_V01_temporal_p7_notime.csv`, `test_operating_point_V01_temporal_p7_notime.csv` |
| Table A17: comparison families | `inference_table.py` | `inference_table_V01.csv`, `inference_families_V01.csv` |
| Table A18: three earlier periods | `rolling_origin.py`, `rolling_origin_design.py` | `rolling_origin_V02.csv`, `rolling_origin_design_V01.csv` |
| Table A1: category coding | `encoding_audit.py` | `category_encoding_audit_V01.csv` |
| §3.1: merchants; fraud-only cards | `merchant_probe.py`, `fraud_only_cards.py`, `new_cardholder_check.py` | `merchant_probe_V01.csv`, `fraud_only_cards_V01.csv`, `new_cardholder_check_V01.csv` |
| §3.2.2, Appendix 6: handling of city and occupation in R | `baseline_variants.py` | `baseline_variants_V02.csv` |
| Appendix 6: grid bounds, fit scope, encoding fix | `grid_bounds_check.py`, `fit_scope_check.py`, `category_cost_check.py`, `rebuild_encoded.py` | `grid_bounds_check_V01.csv`, `fit_scope_check_V01.csv`, `category_cost_check_V01.csv`, `encoding_fix_V01.csv` |
| Fig. 1 | `fig1_flow.py` | `fig1_flow_V02.png` |
| Appendix 5 figures (A1–A8) | `MLProj01_fraudTrain_EDA_V10.ipynb` | — |

Two notes on reading the outputs:

- **k-NN.** The paper's principal k-NN comparison (Table 4, Fig. 2) standardises features on the
  training period (`knn_scaled.py`). The executed model notebooks, Fig. 4, the resampling
  analysis (Fig. 8, Table 9), Table 10 and Table A16 use the original unstandardised k-NN
  (k = 50, distance-weighted), reported in the paper as a sensitivity run.
- **Configuration names.** In file names, `p7_notime` is E, `p7_notime_client` E + H,
  `p7_notime_raw` R, `p7_notime_raw_client` R + H, `p7_hybrid[_client]` R + E [+ H],
  `p7_eplus[_client]` E⁺ [+ H], `p7_catnative[_client]` native-categorical CatBoost,
  `p7_abl_*` the one-change configurations and `p7_red_*` the reduced pair.

## Citation

Please cite the paper (and this archive's DOI).
