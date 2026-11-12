#!/bin/bash
# Kick-the-tires: verify the Poseidon toolchain + in-tree Ozaki-II runtime run
# end-to-end on this GPU in a few minutes. Generates the two smallest inputs,
# profiles @512, and runs the tau=1e-7 call-site auto-pick at N=2048. The
# expected pick, RMSD and speedup are row ozp-tau1e-7 of ../../EXPECTED_RESULTS.md
# (the run prints whether the pick matches). Not a paper measurement — a smoke test.
set -e
source "$(dirname "$0")/scripts/common.sh"
# fresh compilation-overhead log for this run
rm -f overhead.tsv

echo "======== generate inputs (N=512 profile, N=2048 deploy) ========"
bash scripts/gen_data.sh 512 2048

echo "======== tau=1e-7 auto-pick (row ozp-tau1e-7 of EXPECTED_RESULTS.md) ========"
bash scripts/run_autopick.sh 1e-7

echo "======== quick.sh OK ========"
