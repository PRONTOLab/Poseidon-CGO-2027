#!/bin/bash
# Full GENGA pipeline (Fig. 3, Fig. 4, unified); the ablations reuse the profiles of the frontier steps.
set -e
source "$(dirname "$0")/scripts/common.sh"
rm -f overhead.tsv

bash scripts/run_force.sh
bash scripts/run_force_mfoff.sh
if [ "$GENGA_HERBIE_USABLE" = 1 ]; then
  bash scripts/run_kepler.sh
  bash scripts/run_kepler_herbieoff.sh
  bash scripts/run_kepler_timeseries.sh
  bash scripts/run_unified.sh
else
  herbie_skip_note "the drift frontier, the drift time series and the unified solve"
  bash scripts/run_kepler_herbieoff.sh
fi
"$PY" genga_figures.py

echo "======== run_all.sh complete. Figures in ../../figures/ ========"
