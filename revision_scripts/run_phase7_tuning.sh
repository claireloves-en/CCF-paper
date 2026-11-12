#!/bin/zsh
# Phase 7 hyperparameter selection (Revision_Phase7_Plan_V01.md §4). Run only after the plan's
# freeze commit. Sequential, so the searches do not contend for CPU.
# DYLD_LIBRARY_PATH must be set by the caller -- two OpenMP runtimes (see CLAUDE.md).
#
# For every configuration the same procedure: untuned baselines (d7_sweep --only-no-sampling),
# 40-draw search (d8_tuning), best-3 refit on the full training set (refit_top_k), keep-or-untuned
# table (build_tuned_table --version V03).
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/bin/python
stamp() { echo "=== $(date +%H:%M:%S) $* ===" }

# 3, 4: the corrected encoded configurations -- a fresh search
for pair in "fraudTrain_encoded.csv p7_notime" "fraudTrain_encoded_client.csv p7_notime_client"; do
  data=${pair% *}; tag=${pair#* }
  stamp "d7 baselines $tag";  $PY d7_sweep.py --only-no-sampling --data $data --tag $tag
  stamp "d8 search $tag";     $PY d8_tuning.py --data $data --tag $tag
  stamp "top-3 refit $tag";   $PY refit_top_k.py --data $data --tag $tag
  stamp "table $tag";         $PY build_tuned_table.py _$tag --version V03
done

# 1, 2: the raw baselines -- Phase 6's search trials, now with the same top-3 refit
for flags in "" "--client-features"; do
  tag="_notime_raw${flags:+_client}"
  stamp "top-3 refit $tag";   $PY refit_top_k.py --drop-trans-time --raw-baseline ${=flags}
  stamp "table $tag";         $PY build_tuned_table.py $tag --version V03
done

# hybrid: random forest and CatBoost only
M="Random Forest,CatBoost"
for tag in p7_hybrid p7_hybrid_client; do
  data=ablation_data/fraudTrain_${tag#p7_}.csv
  stamp "d7 baselines $tag";  $PY d7_sweep.py --only-no-sampling --data $data --tag $tag
  stamp "d8 search $tag";     $PY d8_tuning.py --data $data --tag $tag --models "$M"
  stamp "top-3 refit $tag";   $PY refit_top_k.py --data $data --tag $tag --models "$M"
  stamp "table $tag";         $PY build_tuned_table.py _$tag --version V03 --models "$M"
done
stamp done
