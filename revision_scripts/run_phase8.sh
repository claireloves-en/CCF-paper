#!/bin/zsh
# Phase 8 C1 (Revision_Phase8_Plan_V01.md, frozen at 10ceada): E+ feature sets, the hybrid's tuning
# procedure for random forest and CatBoost, then fit + test. DYLD_LIBRARY_PATH set by the caller.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/bin/python
stamp() { echo "=== $(date +%H:%M:%S) $* ===" }
M="Random Forest,CatBoost"
stamp "feature sets";          $PY ablation_features.py --only eplus,eplus_client
for tag in p7_eplus p7_eplus_client; do
  data=ablation_data/fraudTrain_${tag#p7_}.csv
  stamp "d7 baselines $tag";  $PY d7_sweep.py --only-no-sampling --data $data --tag $tag
  stamp "d8 search $tag";     $PY d8_tuning.py --data $data --tag $tag --models "$M"
  stamp "top-3 refit $tag";   $PY refit_top_k.py --data $data --tag $tag --models "$M"
  stamp "table $tag";         $PY build_tuned_table.py _$tag --version V03 --models "$M"
  stamp "fit + test $tag";    $PY ablation_run.py --config $tag --models "$M"
done
stamp done
