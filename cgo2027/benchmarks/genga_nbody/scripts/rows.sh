#!/bin/bash
# rows.sh <row id> <log> — compile the configuration of one GENGA row of
# EXPECTED_RESULTS.md exactly as the pipeline scripts do (the same flag
# functions of common.sh, the same cache directories), writing the compile's
# output to <log>. No binary is run, so no GPU is needed: the compile-only pick
# check (../../scripts/check_picks.sh) calls this for every row.
set -o pipefail
source "$(dirname "$0")/common.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"
mkdir -p tmp_rows
head_share() { tr ',' '\n' < "$1/budgets.txt" | sort -g | head -1; }

compile() {   # <name> <cuda flags array name> <src> <budget> <flags...>
  local name="$1" cuda="$2[@]" src="$3" b="$4"; shift 4
  "$POSEIDON_CLANGXX" "${!cuda}" "$@" -poseidon-cost-budget="$b" "$src" -o "tmp_rows/$name.exe"
}
# The budget-0 solve that writes <cache dir>/budgets.txt, the frontier's shares.
enumerate_at0() { # <cache dir> <cuda flags array name> <src> <flags...>
  local cache="$1" cuda="$2" src="$3"; shift 3
  rm -f "$cache/budgets.txt" "$cache/table.json"; mkdir -p "$cache"
  compile "$cache-b0" "$cuda" "$src" 0 "$@" > "tmp_rows/$cache-b0.log" 2>&1 ||
    { echo "rows.sh: budget-0 solve failed (tmp_rows/$cache-b0.log)" >&2; return 1; }
}

case "$ROW" in
  genga-force-fp32|genga-force-expansion)
    mapfile -t FP < <(force_flags -poseidon-cache=cache)
    enumerate_at0 cache COMMON_CUDA genga_nbody.cu "${FP[@]}" || exit 1
    if [ "$ROW" = genga-force-fp32 ]; then B="$(head_share cache)"; else B="$(expected_share "$ROW")" || exit 1; fi
    compile "$ROW" COMMON_CUDA genga_nbody.cu "$B" "${FP[@]}" > "$LOG" 2>&1 ;;
  genga-drift-naive)
    mapfile -t FP < <(kepler_herbieoff_flags)
    enumerate_at0 cache_kep_herbieoff KEPLER_CUDA genga_kepler.cu "${FP[@]}" || exit 1
    B="$(head_share cache_kep_herbieoff)"
    compile "$ROW" KEPLER_CUDA genga_kepler.cu "$B" "${FP[@]}" > "$LOG" 2>&1 ;;
  genga-drift-algebraic)
    mapfile -t FP < <(kepler_flags cache_kep_rolled)
    B="$(expected_share "$ROW")" || exit 1
    compile "$ROW" KEPLER_CUDA genga_kepler.cu "$B" "${FP[@]}" > "$LOG" 2>&1 ;;
  genga-unified)
    mapfile -t FP < <(unified_flags cache_unified)
    B="$(expected_share "$ROW")" || exit 1
    compile "$ROW" KEPLER_CUDA genga_unified.cu "$B" "${FP[@]}" > "$LOG" 2>&1 ;;
  *) echo "rows.sh: no GENGA row '$ROW'" >&2; exit 2 ;;
esac
