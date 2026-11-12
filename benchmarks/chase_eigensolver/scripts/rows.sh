#!/bin/bash
# rows.sh <row id> <log> [extra compiler flags...]: compile the configuration
# of one R-ChFSI decision row (expected/picks*.tsv) exactly as the pipeline
# does (`make solve` at the row's frontier share; the sensitivity-blind row
# from a copy of the profile without its Kappa lines, as run_ladder.sh makes
# it), writing the compiler's output to <log>. No binary is run, so no GPU is
# needed: the compile-only pick check (../../scripts/check_picks.sh) and the
# explain helper (../../scripts/explain.sh) call this. COST_MODEL and GPU_ARCH
# are the caller's.
set -o pipefail
source "$(dirname "$0")/common.sh"
source "$ART/scripts/caches.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"; shift 2
PD="$PROFSRC"
if [ "$ROW" = rchfsi-blind ]; then
  rm -rf fpprofile_blind; cp -r "$PROFSRC" fpprofile_blind; sed -i '/^Kappa/d' fpprofile_blind/*.fpprofile
  PD=./fpprofile_blind
fi
DEV="$(row_device "$ROW")" || exit 2
seed_cache "$DEV" "$BENCH_DIR" cache
B="$(expected_share "$ROW")" || exit 1
X=""; [ $# -gt 0 ] && X="$(printf '%q ' "$@")"
make solve BENCH_DEFS="$DEP_DEFS" BUDGET="$B" PROFDIR="$PD" EXTRA_SOLVE="$X" > /dev/null 2>&1
cp solve.log "$LOG"
test -s chase_opt.exe
