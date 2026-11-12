#!/usr/bin/env bash
# check_picks.sh [row id ...] — compile-only check of every decision in
# EXPECTED_RESULTS.md, without a GPU.
#
# For each row of expected/picks.tsv (or the rows named), the benchmark's own
# scripts/rows.sh compiles that row's configuration the way the pipeline does,
# from the shipped profiles (PROFILES=shipped), the shipped RTX 5090 cost model
# and, for GENGA, the shipped Herbie answers; then the configuration the
# compiler applied (its `[poseidon] pin:` lines) is compared with the row's.
# Nothing is run on a GPU and nothing needs one: GPU_ARCH is the shipped model's.
# Output: picks_check/<row>.log per compile and picks_check/picks.tsv
# (row, select, result, applied configuration); exit 1 if a row differs.
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ART"
export POSEIDON_ARTIFACT_NO_GPU=1 PROFILES=shipped GENGA_HERBIE=replay
unset GPU_ID
export COST_MODEL="${COST_MODEL:-$ART/cost_models/cm_sm_120_RTX5090.csv}"
source "$ART/env.sh" || exit 1
source "$ART/scripts/expected.sh"

OUT="$ART/picks_check"; mkdir -p "$OUT"
ROWS=("$@")
[ ${#ROWS[@]} -gt 0 ] || mapfile -t ROWS < <(awk -F'\t' 'NR > 1 {print $1}' expected/picks.tsv)
printf 'row\tselect\tresult\tapplied\n' > "$OUT/picks.tsv"
echo "compiler: $POSEIDON_CLANGXX"
echo "plugin:   $(bash scripts/detect.sh plugin) (md5 $(md5sum < "$(bash scripts/detect.sh plugin)" | cut -c1-8))"
echo "model:    ${COST_MODEL#$ART/}"
FAIL=0
for R in "${ROWS[@]}"; do
  B="$(expected_field "$R" bench)" || { FAIL=1; continue; }
  T0=$(date +%s)
  if ( cd "benchmarks/$B" && bash scripts/rows.sh "$R" "$OUT/$R.log" ) > "$OUT/$R.out" 2>&1; then
    want="$(expected_field "$R" pin | tr ',' '\n' | sed '/^$/d' | sort -u)"
    got="$(pins_of_log "$OUT/$R.log")"
    if [ "$want" = "$got" ]; then res=SAME; else res=DIFFERENT; FAIL=1; fi
  else
    res=COMPILE-FAILED; got=""; FAIL=1
  fi
  printf '%s\t%s\t%s\t%s\n' "$R" "$(expected_field "$R" select)" "$res" "$(echo "$got" | paste -sd',' -)" >> "$OUT/picks.tsv"
  printf '%-30s %-15s %4ss\n' "$R" "$res" "$(( $(date +%s) - T0 ))"
done
echo "wrote picks_check/picks.tsv"
exit $FAIL
