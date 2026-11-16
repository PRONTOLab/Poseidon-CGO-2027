#!/bin/bash
# rows.sh <row id> <log> [extra flags...]: compile one ozp decision row of expected/picks*.tsv
# as the pipeline does, without running it. ROWS_SOLVE_FRESH=1 drops the DP table first.
set -o pipefail
source "$(dirname "$0")/common.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"; shift 2
EXTRA=("$@")
mkdir -p tmp_rows cache
rm -f cache/*.ozdispatch
DEV="$(row_device "$ROW")" || exit 2
[ "${ROWS_SOLVE_FRESH:-0}" = 1 ] && rm -f cache/table.json
solve() { "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
            -poseidon-print "$@" ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "tmp_rows/$ROW.exe"; }
case "$(expected_field "$ROW" select)" in
  tau=*) T="$(expected_tau "$ROW")" || exit 1; solve -poseidon-tau="$T" "${EXTRA[@]}" > "$LOG" 2>&1 ;;
  share=*) B="$(expected_share "$ROW")" || exit 1; solve -poseidon-cost-budget="$B" "${EXTRA[@]}" > "$LOG" 2>&1 ;;
  *) echo "rows.sh: no ozp row '$ROW'" >&2; exit 2 ;;
esac
