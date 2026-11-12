#!/usr/bin/env bash
# pt.sh — phase timer for the compilation-overhead measurement.
#
#   pt.sh <phase> <detail> -- <cmd...>
#
# Runs <cmd...>, appends "<phase>\t<detail>\t<wall-seconds>" to
# $OVERHEAD_TSV (default ./overhead.tsv, i.e. the benchmark directory), and
# exits with the command's status. Phases used by the benchmarks:
#   baseline_compile  plain (non-Poseidon) compile, the overhead reference
#   profile_compile   --poseidon-profile-generate instrumented compile
#   profile_run       execution of the instrumented binary
#   solve_compile     --poseidon-profile-use optimization compile (detail = budget or
#                     tag; the first solve per cache is cold: DP table build +
#                     accuracy sampling; later budgets reuse the cached table)
# stdout/stderr of <cmd...> pass through untouched, so existing redirections
# and log greps keep working.
set -uo pipefail
PHASE="${1:?phase}" DETAIL="${2:?detail}"
[ "${3:-}" = "--" ] || { echo "pt.sh: expected '--' before command" >&2; exit 2; }
shift 3
OUT="${OVERHEAD_TSV:-./overhead.tsv}"
t0=$(date +%s.%N)
"$@"
rc=$?
t1=$(date +%s.%N)
awk -v p="$PHASE" -v d="$DETAIL" -v a="$t0" -v b="$t1" \
  'BEGIN{printf "%s\t%s\t%.2f\n", p, d, b-a}' >> "$OUT"
exit $rc
