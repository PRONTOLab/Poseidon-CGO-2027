#!/bin/bash
# Quick run: N = 512 profile, one joint solve (the rchfsi-bf16-nm12 row, else tau = 1e-9), one run at N = 2048.
set -e
source "$(dirname "$0")/scripts/common.sh"

echo "======== 1. profile-gen @ N=512 NEV=32 (with each site's Kappa) ========"
chase_make profile PROF_DEFS="$PROF_DEFS"
rm -f cache/budgets.txt
echo "  (expect hemm Kappa ~1e4, the smallest, the washed-out filter; resid/gram/"
echo "   rrhv/rrproj ~1e11-1e13, kept in FP64)"

if [ "$ON_SHIPPED_MODEL" = 1 ]; then
  B="$(expected_share rchfsi-bf16-nm12)"
  echo "======== 2. solve at share $B, deploy N=2048 NEV=64 (solves read ./$PROFSRC) ========"
  chase_make solve BENCH_DEFS="$DEP_DEFS" BUDGET="$B" PROFDIR="$PROFSRC"
else
  echo "======== 2. solve at tau = 1e-9, deploy N=2048 NEV=64 (solves read ./$PROFSRC) ========"
  chase_make solve BENCH_DEFS="$DEP_DEFS" BUDGET=0 PROFDIR="$PROFSRC" EXTRA_SOLVE=-poseidon-tau=1e-9
fi
echo "  applied: $(solve_schemes)"
expected_report rchfsi-bf16-nm12 solve.log

echo "======== 3. verify (expect: locked 64/64, eigenvalue L2 FP64-class) ========"
NUM_RUNS=1 run_avg5 "quick: chase_opt.exe" ./chase_opt.exe
printf '%s\n' "$MEASURE_OUT"
echo "  solve_ms=$AVG_MS  eigenvalue L2=$EIG_L2  $LOCKED"

echo "======== quick.sh OK ========"
