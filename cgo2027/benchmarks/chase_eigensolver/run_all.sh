#!/bin/bash
# Full paper pipeline for eval §5.3 (chase_eigensolver).
#   run_ladder.sh  -> scvh_probe_on.tsv, scvh_probe_off.tsv, ablation_knee.tsv,
#                     chase_money_data.tsv, traces/eigresid_*.txt (the
#                     sensitivity-guided ladder + the blind ablation at the knee,
#                     avg-of-5; rows rchfsi-* of ../../EXPECTED_RESULTS.md)
#   plot_chase.py  -> ../../figures/rchfsi_residuals.pdf (residual-spectrum figure)
set -e
source "$(dirname "$0")/scripts/common.sh"
# fresh compilation-overhead log for this run
rm -f overhead.tsv

bash scripts/run_ladder.sh
"$PY" plot_chase.py

echo "======== run_all.sh complete. Figure in ../../figures/rchfsi_residuals.pdf ========"
