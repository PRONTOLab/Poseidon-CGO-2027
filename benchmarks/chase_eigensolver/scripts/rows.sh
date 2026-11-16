#!/bin/bash
# rows.sh <row id> <log> [extra flags...]: compile one R-ChFSI decision row of expected/picks*.tsv
# as the pipeline does (rchfsi-blind from a profile without Kappa lines), without running it.
set -o pipefail
source "$(dirname "$0")/common.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"; shift 2
PD="$PROFSRC"
if [ "$ROW" = rchfsi-blind ]; then
  rm -rf fpprofile_blind; cp -r "$PROFSRC" fpprofile_blind; sed -i '/^Kappa/d' fpprofile_blind/*.fpprofile
  PD=./fpprofile_blind
fi
DEV="$(row_device "$ROW")" || exit 2
B="$(expected_share "$ROW")" || exit 1
X=""; [ $# -gt 0 ] && X="$(printf '%q ' "$@")"
make solve BENCH_DEFS="$DEP_DEFS" BUDGET="$B" PROFDIR="$PD" EXTRA_SOLVE="$X" > /dev/null 2>&1
cp solve.log "$LOG"
test -s chase_opt.exe
