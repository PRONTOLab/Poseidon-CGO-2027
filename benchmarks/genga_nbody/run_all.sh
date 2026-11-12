#!/bin/bash
# Full paper pipeline for eval §5.1 (GENGA). Produces both figures and every
# reported number. Ordering matters: the ablation arms and the timeseries reuse
# the profiles / baselines produced by the force and kepler steps, and the
# timeseries takes its naive point from the Herbie-off frontier.
#   force            -> force_full_g0.tsv       (expansion / FP32 frontier)
#   force-mfoff      -> force_mfoff.tsv         (expansion-off ablation)
#   kepler           -> kepler_full_g0.tsv      (Herbie / drift frontier)
#   kepler-herbieoff -> kepler_herbieoff_g0.tsv (Herbie-off ablation)
#   timeseries       -> kepler_timeseries.tsv   (silent-failure trajectory)
#   unified          -> unified.tsv             (force-then-drift single solve)
# GENGA_HERBIE and PROFILES: see scripts/common.sh.
#   figures          -> ../../figures/genga_force_drift.pdf, genga_drift_timeseries.pdf
set -e
source "$(dirname "$0")/scripts/common.sh"
# fresh compilation-overhead log for this run
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
