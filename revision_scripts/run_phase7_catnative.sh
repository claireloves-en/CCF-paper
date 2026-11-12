#!/bin/zsh
# Phase 7 §4.2: CatBoost with native categorical handling -- untuned baseline, 40-draw search,
# best-3 refit, keep-or-untuned table. After run_phase7_tuning.sh; DYLD_LIBRARY_PATH set by caller.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/bin/python
stamp() { echo "=== $(date +%H:%M:%S) $* ===" }
for tag in p7_catnative p7_catnative_client; do
  data=ablation_data/fraudTrain_${tag#p7_}.csv
  stamp "untuned baseline $tag"; $PY catnative_baseline.py --data $data --tag $tag
  stamp "d8 search $tag";        $PY d8_tuning.py --data $data --tag $tag --models CatBoost
  stamp "top-3 refit $tag";      $PY refit_top_k.py --data $data --tag $tag --models CatBoost
  stamp "table $tag";            $PY build_tuned_table.py _$tag --version V03 --models CatBoost
done
stamp done
