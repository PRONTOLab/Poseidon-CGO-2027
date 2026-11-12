#!/bin/bash
# rows.sh <row id> <log> [extra compiler flags...]: compile the configuration
# of one ozp decision row (expected/picks*.tsv) exactly as the pipeline
# scripts do (common.sh's flag groups, deploy size APPLY_N), writing the
# compile's output to <log>. No binary is run, so no GPU is needed: the
# compile-only pick check (../../scripts/check_picks.sh) and the explain helper
# (../../scripts/explain.sh) call this. COST_MODEL and GPU_ARCH are the
# caller's. The content-addressed accuracy memo in cache/ is kept across rows
# and devices (the decision does not depend on it); the one-shot dispatch
# descriptors are not.
# ROWS_SOLVE_FRESH=1 (the explain helper) removes the DP table of the row's
# cache directory before the row's compile: a table read back from the cache
# skips pricing, and the explanation would lack the candidates' cost and error.
set -o pipefail
source "$(dirname "$0")/common.sh"
source "$ART/scripts/caches.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"; shift 2
EXTRA=("$@")
mkdir -p tmp_rows cache
rm -f cache/*.ozdispatch
DEV="$(row_device "$ROW")" || exit 2
seed_cache "$DEV" "$BENCH_DIR" cache
[ "${ROWS_SOLVE_FRESH:-0}" = 1 ] && rm -f cache/table.json
solve() { "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
            -poseidon-print "$@" ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "tmp_rows/$ROW.exe"; }
case "$(expected_field "$ROW" select)" in
  tau=*) T="$(expected_tau "$ROW")" || exit 1; solve -poseidon-tau="$T" "${EXTRA[@]}" > "$LOG" 2>&1 ;;
  share=*) B="$(expected_share "$ROW")" || exit 1; solve -poseidon-cost-budget="$B" "${EXTRA[@]}" > "$LOG" 2>&1 ;;
  *) echo "rows.sh: no ozp row '$ROW'" >&2; exit 2 ;;
esac
