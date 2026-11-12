#!/bin/bash
# Kick-the-tires (~10 min): exercise the full chase toolchain once. Profile at the
# N=512 surrogate (the run measures each site's condition number Kappa: the
# filter site, hemm, must come out with the SMALLEST one = washed out), then a
# single joint DP solve at the knee row of ../../EXPECTED_RESULTS.md
# (rchfsi-knee) deployed at N=2048, and verify it converges 64/64 at
# FP64-class accuracy. On a cost model measured on this device the knee is
# that model's own: the cheapest frontier point is not it, so the smoke test
# solves at budget 0 instead. Not a paper measurement — a smoke test.
set -e
source "$(dirname "$0")/scripts/common.sh"

echo "======== 1. profile-gen @ N=512 NEV=32 (with each site's Kappa) ========"
chase_make profile PROF_DEFS="$PROF_DEFS"
rm -f cache/budgets.txt
echo "  (expect hemm Kappa ~1e4, the smallest — the washed-out filter; resid/gram/"
echo "   rrhv/rrproj ~1e11-1e13, kept in FP64)"

if [ "$ON_SHIPPED_MODEL" = 1 ]; then B="$(expected_share rchfsi-knee)"; else B=0; fi
echo "======== 2. solve at share $B, deploy N=2048 NEV=64 (solves read ./$PROFSRC) ========"
chase_make solve BENCH_DEFS="$DEP_DEFS" BUDGET="$B" PROFDIR="$PROFSRC"
echo "  applied: $(solve_schemes)"
[ "$ON_SHIPPED_MODEL" = 1 ] && expected_report rchfsi-knee solve.log

echo "======== 3. verify (expect: locked 64/64, eigenvalue L2 FP64-class) ========"
chase_make verify

echo "======== quick.sh OK ========"
