#!/bin/bash
# Quick run: N = 512 profile, then the tau = 1e-7 decision at N = 2048.
set -e
source "$(dirname "$0")/scripts/common.sh"
rm -f overhead.tsv

echo "======== generate inputs (N=512 profile, N=2048 deploy) ========"
bash scripts/gen_data.sh 512 2048

echo "======== tau=1e-7 auto-pick (row ozp-tau1e-7 of expected/picks.tsv) ========"
bash scripts/run_autopick.sh 1e-7

echo "======== quick.sh OK ========"
