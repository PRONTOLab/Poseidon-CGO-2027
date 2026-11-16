#!/bin/bash
# rows.sh <row id> <log> [extra flags...]: compile one GENGA decision row of expected/picks*.tsv
# as the pipeline does, without running it. ROWS_SOLVE_FRESH=1 drops the row's DP table first.
set -o pipefail
source "$(dirname "$0")/common.sh"
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
enumerate_at0() { # <cache dir> <cuda flags array name> <src> <flags...>
  local cache="$1" cuda="$2" src="$3"; shift 3
  rm -f "$cache/budgets.txt" "$cache/table.json"; mkdir -p "$cache"
  compile "$cache-b0" "$cuda" "$src" -poseidon-cost-budget=0 "$@" > "tmp_rows/$cache-b0.log" 2>&1 ||
    { echo "rows.sh: budget-0 solve failed (tmp_rows/$cache-b0.log)" >&2; return 1; }
}

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
