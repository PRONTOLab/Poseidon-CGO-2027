#!/usr/bin/env bash
# Fig. 4: energy drift over the step count -> figures/genga_drift_timeseries.pdf.
# Reuses the drift solves of run_fig3.sh and runs them when they are missing.
set -e
source "$(dirname "$0")/scripts/common.sh"
[ "$GENGA_HERBIE_USABLE" = 1 ] || { herbie_skip_note "the drift time series"; exit 0; }
[ -s cache_kep_herbieoff/budgets.txt ] || bash scripts/run_kepler_herbieoff.sh
[ -d fpprofile_kep_rolled ] || bash scripts/run_kepler.sh
bash scripts/run_kepler_timeseries.sh
"$PY" genga_figures.py
