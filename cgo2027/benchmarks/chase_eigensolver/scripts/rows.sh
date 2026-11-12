#!/bin/bash
# rows.sh <row id> <log> — compile the configuration of one R-ChFSI row of
# EXPECTED_RESULTS.md exactly as the pipeline does (`make solve` at the row's
# frontier share; the sensitivity-blind row from a copy of the profile without
# its Kappa lines, as run_ladder.sh makes it), writing the compiler's output to
# <log>. No binary is run, so no GPU is needed: the compile-only pick check
# (../../scripts/check_picks.sh) calls this for every row.
set -o pipefail
source "$(dirname "$0")/common.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"
PD="$PROFSRC"
if [ "$ROW" = rchfsi-blind ]; then
  rm -rf fpprofile_blind; cp -r "$PROFSRC" fpprofile_blind; sed -i '/^Kappa/d' fpprofile_blind/*.fpprofile
  PD=./fpprofile_blind
fi
B="$(expected_share "$ROW")" || exit 1
make solve BENCH_DEFS="$DEP_DEFS" BUDGET="$B" PROFDIR="$PD" > /dev/null 2>&1
cp solve.log "$LOG"
test -s chase_opt.exe
