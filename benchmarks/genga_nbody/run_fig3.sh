#!/usr/bin/env bash
# Fig. 3: the force-kernel and drift-kernel frontiers -> figures/genga_force_drift.pdf.
set -e
source "$(dirname "$0")/scripts/common.sh"
bash scripts/run_force.sh
bash scripts/run_force_mfoff.sh
if [ "$GENGA_HERBIE_USABLE" = 1 ]; then bash scripts/run_kepler.sh; else herbie_skip_note "the drift frontier"; fi
bash scripts/run_kepler_herbieoff.sh
"$PY" genga_figures.py
