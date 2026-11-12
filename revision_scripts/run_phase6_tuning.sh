#!/bin/zsh
# J1 step 5 (Revision_Phase6_Plan_V01.md): tune the two baseline configurations.
# For each: the untuned baselines first (d7_sweep --only-no-sampling, the comparison
# build_tuned_table.py applies its keep-or-discard rule against), then the 40-draw search,
# then the kept-hyperparameter table. Sequential, so the two do not contend for CPU.
# DYLD_LIBRARY_PATH must be set by the caller -- two OpenMP runtimes (see CLAUDE.md).
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/bin/python
for flags in "" "--client-features"; do
  tag="_notime_raw${flags:+_client}"
  echo "=== $(date +%H:%M:%S) d7_sweep baselines $tag ==="
  $PY d7_sweep.py --only-no-sampling --drop-trans-time --raw-baseline ${=flags}
  echo "=== $(date +%H:%M:%S) d8_tuning search $tag ==="
  $PY d8_tuning.py --drop-trans-time --raw-baseline ${=flags}
  echo "=== $(date +%H:%M:%S) build_tuned_table $tag ==="
  $PY build_tuned_table.py "$tag"
done
echo "=== $(date +%H:%M:%S) done ==="
