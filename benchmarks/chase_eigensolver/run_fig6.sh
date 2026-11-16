#!/usr/bin/env bash
# Fig. 6: R-ChFSI residual convergence per strategy -> figures/rchfsi_residuals.pdf.
set -e
source "$(dirname "$0")/scripts/common.sh"
bash scripts/run_strategies.sh
"$PY" plot_chase.py
