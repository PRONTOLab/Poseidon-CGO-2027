#!/bin/bash
# Full ozp pipeline (Fig. 5 and the surrogate sweep).

set -e
source "$(dirname "$0")/scripts/common.sh"
rm -f overhead.tsv

bash scripts/gen_data.sh
bash scripts/run_pareto.sh
bash scripts/run_dispatch.sh
bash scripts/run_direct_tc_lib.sh
bash scripts/run_autopick.sh
bash scripts/run_surrogate.sh
"$PY" plot_ozp.py

echo "======== run_all.sh complete. Figures in ../../figures/ ========"
