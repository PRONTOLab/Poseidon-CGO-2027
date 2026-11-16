#!/bin/bash
# Full R-ChFSI pipeline (Fig. 6).
set -e
source "$(dirname "$0")/scripts/common.sh"
rm -f overhead.tsv

bash scripts/run_strategies.sh
"$PY" plot_chase.py

echo "======== run_all.sh complete. Figure in ../../figures/rchfsi_residuals.pdf ========"
