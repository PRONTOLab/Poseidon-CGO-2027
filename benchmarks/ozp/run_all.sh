#!/bin/bash
# Full paper pipeline for eval §5.2 (ozp). Produces both figures and every
# reported number. Generates the synthetic inputs first, then:
#   pareto     -> pareto.tsv, traj_*.tsv        (in-kernel direct TC + FP16TCEC ladder)
#   dispatch   -> dispatch_rungs.tsv, traj_OZII_nm*.tsv (the frontier: direct
#                 library dispatches and Ozaki-II rungs)
#   direct-tc-lib -> direct_tc_lib.tsv, traj_LIB_*.tsv  (cuBLAS reference points)
#   autopick   -> autopick.tsv: the tau picks (EXPECTED_RESULTS.md rows ozp-tau*)
#   surrogate  -> surrogate.tsv                 (SCVH N-sweep, up to N=8192)
#   figures    -> ../../figures/ozp_convergence.pdf, ozp_surrogate.pdf

set -e
source "$(dirname "$0")/scripts/common.sh"
# fresh compilation-overhead log for this run
rm -f overhead.tsv

bash scripts/gen_data.sh
bash scripts/run_pareto.sh
bash scripts/run_dispatch_rungs.sh
bash scripts/run_direct_tc_lib.sh
bash scripts/run_autopick.sh
bash scripts/run_surrogate.sh
"$PY" plot_ozp.py
"$PY" plot_surrogate.py

echo "======== run_all.sh complete. Figures in ../../figures/ ========"
