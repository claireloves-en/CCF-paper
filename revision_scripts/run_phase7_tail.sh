#!/bin/zsh
# The tail of run_phase7_runs.sh after the 0.5-threshold fix in ablation_run.py (2026-10-13): the four
# script-vs-notebook guards, the resampling sweep and the tuned-resampling check.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/bin/python
stamp() { echo "=== $(date +%H:%M:%S) $* ===" }
for c in p7_notime_raw p7_notime_raw_client p7_notime p7_notime_client; do
  stamp "guard $c against the notebook";  $PY ablation_run.py --config $c --check
done
stamp "resampling sweep, corrected E"; $PY d7_sweep.py --data fraudTrain_encoded.csv --tag p7_notime
$PY reshape_sweep.py _p7_notime
stamp "tuned resampling check"
$PY tuned_resampling_check.py --data fraudTrain_encoded.csv --tag p7_notime --version V03
$PY tuned_resampling_check.py --data fraudTrain_encoded_client.csv --tag p7_notime_client --version V03
stamp done
