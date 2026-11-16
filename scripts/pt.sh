#!/usr/bin/env bash
# pt.sh <phase> <detail> -- <cmd...>: run <cmd>, append "<phase> <detail> <seconds>" to
# $OVERHEAD_TSV (default ./overhead.tsv), exit with its status. PT_PROBE_SPLIT=1 records
# the probe's profile run and condition-number runs as separate rows.
set -uo pipefail
PHASE="${1:?phase}" DETAIL="${2:?detail}"
[ "${3:-}" = "--" ] || { echo "pt.sh: expected '--' before command" >&2; exit 2; }
shift 3
OUT="${OVERHEAD_TSV:-./overhead.tsv}"
t0=$(date +%s.%N)
if [ "${PT_PROBE_SPLIT:-0}" = 1 ]; then
  ERR="$(mktemp)"
  trap 'rm -f "$ERR"' EXIT
  { "$@" 2>&1 1>&3 3>&- | tee "$ERR" >&2 3>&-; rc=${PIPESTATUS[0]}; } 3>&1
  RUN="$(sed -n 's/^\[probe\] profile run: \([0-9.]*\) s$/\1/p' "$ERR" | tail -1)"
  MEASURED="$(sed -n 's/^\[probe\] condition-number measurement: [0-9]* runs, \([0-9.]*\) s$/\1/p' "$ERR" | tail -1)"
  if [ -n "$RUN" ] && [ -n "$MEASURED" ]; then
    printf '%s\t%s\t%s\n' "$PHASE" "$DETAIL" "$RUN" >> "$OUT"
    printf 'condition_runs\t%s\t%s\n' "$DETAIL" "$MEASURED" >> "$OUT"
    exit $rc
  fi
else
  "$@"
  rc=$?
fi
t1=$(date +%s.%N)
awk -v p="$PHASE" -v d="$DETAIL" -v a="$t0" -v b="$t1" \
  'BEGIN{printf "%s\t%s\t%.2f\n", p, d, b-a}' >> "$OUT"
exit $rc
