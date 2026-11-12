#!/bin/zsh
# Phase 7 runs after hyperparameter selection (Revision_Phase7_Plan_V01.md §§4-8): every fit and
# test evaluation on the corrected data, the analyses built on them, and the four notebook runs
# of record. fraudTest is NOT touched here (confirm_run.py, last). DYLD_LIBRARY_PATH set by caller.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/bin/python
stamp() { echo "=== $(date +%H:%M:%S) $* ===" }
RC="Random Forest,CatBoost"

stamp "catnative searches";  ./run_phase7_catnative.sh

stamp "TUNED entries in model_V26"
$PY fill_tuned.py p7_notime --version V03 --table p7_notime
$PY fill_tuned.py p7_notime_client --version V03 --table p7_notime_client
$PY fill_tuned.py p7_notime_raw --version V03 --table notime_raw
$PY fill_tuned.py p7_notime_raw_client --version V03 --table notime_raw_client

for c in p7_notime_raw p7_notime_raw_client p7_notime p7_notime_client; do
  stamp "fit + test $c";  $PY ablation_run.py --config $c
done
for c in p7_hybrid p7_hybrid_client p7_abl_cat_onehot p7_abl_plus_onehot p7_abl_hour p7_abl_plus_dow \
         p7_abl_no_interactions p7_abl_no_external p7_abl_no_geo_occ p7_red_raw p7_red_enc; do
  stamp "fit + test $c";  $PY ablation_run.py --config $c --models "$RC"
done
for c in p7_catnative p7_catnative_client; do
  stamp "fit + test $c";  $PY ablation_run.py --config $c --models CatBoost
done

ALL=p7_notime_raw,p7_notime_raw_client,p7_notime,p7_notime_client,p7_hybrid,p7_hybrid_client,p7_abl_cat_onehot,p7_abl_plus_onehot,p7_abl_hour,p7_abl_plus_dow,p7_abl_no_interactions,p7_abl_no_external,p7_abl_no_geo_occ,p7_red_raw,p7_red_enc,p7_catnative,p7_catnative_client
stamp "tables";            $PY ablation_table.py --phase7
stamp "operating points";  $PY operating_points.py --configs $ALL
for s in stratified card timeblock; do
  stamp "bootstrap $s";    $PY uncertainty_check.py --scheme $s --configs $ALL
done
stamp "mixed cards";       $PY mixed_cards_rescore.py --configs p7_notime,p7_notime_client --out mixed_cards_rescore_V02.csv
stamp "rolling origin";    $PY rolling_origin.py --configs p7_notime_raw,p7_notime,p7_notime_client --out rolling_origin_V02.csv
stamp "baseline variants"; $PY baseline_variants.py --out baseline_variants_V02.csv

stamp "interpretation: paired SHAP, permutation"; $PY explain_compare.py
stamp "interpretation: grouped LIME";              $PY lime_grouped.py

stamp "model_V26 runs of record"; $PY run_model_v26.py
for c in p7_notime_raw p7_notime_raw_client p7_notime p7_notime_client; do
  stamp "guard $c against the notebook";  $PY ablation_run.py --config $c --check
done

stamp "resampling sweep, corrected E"; $PY d7_sweep.py --data fraudTrain_encoded.csv --tag p7_notime
$PY reshape_sweep.py _p7_notime
stamp "tuned resampling check"
$PY tuned_resampling_check.py --data fraudTrain_encoded.csv --tag p7_notime --version V03
$PY tuned_resampling_check.py --data fraudTrain_encoded_client.csv --tag p7_notime_client --version V03
stamp done
