#!/usr/bin/env bash
# Fig. 5: ozp convergence and frontier -> figures/ozp_convergence.pdf.
set -e
source "$(dirname "$0")/scripts/common.sh"
bash scripts/gen_data.sh
bash scripts/run_pareto.sh
bash scripts/run_dispatch.sh
bash scripts/run_direct_tc_lib.sh
bash scripts/run_autopick.sh
"$PY" plot_ozp.py
