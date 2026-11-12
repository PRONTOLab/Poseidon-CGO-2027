#!/bin/bash
# rows.sh <row id> <log> [extra compiler flags...]: compile the configuration
# of one GENGA decision row (expected/picks*.tsv) exactly as the pipeline
# scripts do (the same flag functions of common.sh, the same cache
# directories), writing the compile's output to <log>. No binary is run, so no
# GPU is needed: the compile-only pick check (../../scripts/check_picks.sh) and
# the explain helper (../../scripts/explain.sh) call this.
#
# The kernel follows from the row id (force, drift-naive, drift, unified), the
# selection from the row's `select`. COST_MODEL, GPU_ARCH and HERBIE_ANSWERS
# are the caller's (check_picks.sh sets them per device). A row of another
# device than the RTX 5090 solves in its own cache directory (<dir>_<device>):
# a DP table is kept per function and cache directory.
# ROWS_SOLVE_FRESH=1 (the explain helper) removes the DP table of the row's
# cache directory before the row's compile: a table read back from the cache
# skips pricing, and the explanation would lack the candidates' cost and error.
set -o pipefail
source "$(dirname "$0")/common.sh"
source "$ART/scripts/caches.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"; shift 2
EXTRA=("$@")
mkdir -p tmp_rows
DEV="$(row_device "$ROW")" || exit 2
SFX=""; [ "$DEV" = rtx5090 ] || SFX="_$DEV"
head_share() { tr ',' '\n' < "$1/budgets.txt" | sort -g | head -1; }

compile() {   # <name> <cuda flags array name> <src> <selection flag> <flags...>
  local name="$1" cuda="$2[@]" src="$3" sel="$4"; shift 4
  "$POSEIDON_CLANGXX" "${!cuda}" "$@" "$sel" "$src" -o "tmp_rows/$name.exe"
}
# The budget-0 solve that writes <cache dir>/budgets.txt, the frontier's shares.
enumerate_at0() { # <cache dir> <cuda flags array name> <src> <flags...>
  local cache="$1" cuda="$2" src="$3"; shift 3
  rm -f "$cache/budgets.txt" "$cache/table.json"; mkdir -p "$cache"
  seed_cache "$DEV" "$BENCH_DIR" "$cache"
  compile "$cache-b0" "$cuda" "$src" -poseidon-cost-budget=0 "$@" > "tmp_rows/$cache-b0.log" 2>&1 ||
    { echo "rows.sh: budget-0 solve failed (tmp_rows/$cache-b0.log)" >&2; return 1; }
}

# Per kernel: cache directory, flags, CUDA flags, source, and whether the
# pipeline enumerates the frontier (a budget-0 solve) before the row's compile.
case "$ROW" in
  *force*)
    CACHE="cache$SFX"; mapfile -t FP < <(force_flags -poseidon-cache="$CACHE")
    CU=COMMON_CUDA; SRC=genga_nbody.cu; ENUM=1 ;;
  *drift-naive*)
    CACHE=cache_kep_herbieoff
    [ -z "$SFX" ] || { echo "rows.sh: the naive drift row is solved for the RTX 5090 only" >&2; exit 2; }
    mapfile -t FP < <(kepler_herbieoff_flags)
    CU=KEPLER_CUDA; SRC=genga_kepler.cu; ENUM=1 ;;
  *drift*)
    CACHE="cache_kep_rolled$SFX"; mapfile -t FP < <(kepler_flags "$CACHE")
    CU=KEPLER_CUDA; SRC=genga_kepler.cu; ENUM=0 ;;
  *unified*)
    CACHE="cache_unified$SFX"; mapfile -t FP < <(unified_flags "$CACHE")
    CU=KEPLER_CUDA; SRC=genga_unified.cu; ENUM=0 ;;
  *) echo "rows.sh: no GENGA row '$ROW'" >&2; exit 2 ;;
esac
FP+=("${EXTRA[@]}")

SEL="$(expected_field "$ROW" select)" || exit 2
if [ "$ENUM" = 1 ] || [ "$SEL" = head ]; then
  enumerate_at0 "$CACHE" "$CU" "$SRC" "${FP[@]}" || exit 1
else
  seed_cache "$DEV" "$BENCH_DIR" "$CACHE"
fi
case "$SEL" in
  head)    S="-poseidon-cost-budget=$(head_share "$CACHE")" ;;
  zero)    S=-poseidon-cost-budget=0 ;;
  share=*) S="-poseidon-cost-budget=$(expected_share "$ROW")" || exit 2 ;;
  tau=*)   S="-poseidon-tau=$(expected_tau "$ROW")" || exit 2 ;;
  *) echo "rows.sh: row '$ROW' has an unknown select '$SEL'" >&2; exit 2 ;;
esac
[ "${ROWS_SOLVE_FRESH:-0}" = 1 ] && rm -f "$CACHE/table.json"
compile "$ROW" "$CU" "$SRC" "$S" "${FP[@]}" > "$LOG" 2>&1
