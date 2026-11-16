#!/bin/bash
# Sourced by the chase_eigensolver scripts: configuration and launch helpers; flags live in the Makefile.
# PROFILES=fresh (default) | shipped (solve from profiles/surrogate512/).
set -o pipefail

ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
. "$ART/scripts/bcshim.sh"   # num_add / num_div: exact `bc -l` replacement
source "$ART/env.sh"
source "$ART/scripts/expected.sh"

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BENCH_DIR"

PROFILES="${PROFILES:-fresh}"
case "$PROFILES" in
  fresh) PROFSRC=fpprofile ;;
  shipped) PROFSRC=profiles/surrogate512 ;;
  *) echo "PROFILES=$PROFILES: use fresh or shipped" >&2; exit 2 ;;
esac
ON_SHIPPED_MODEL=0
cmp -s "$COST_MODEL" "$(bash "$ART/scripts/detect.sh" shipped_cost_model)" && ON_SHIPPED_MODEL=1

PROF_DEFS="${PROF_DEFS:--DCHASE_N=512 -DCHASE_NEV=32}"
DEP_DEFS="${DEP_DEFS:--DCHASE_N=2048 -DCHASE_NEV=64}"
NUM_RUNS="${NUM_RUNS:-5}"

# Frontier shares to measure; empty = a sample of this model's frontier plus the expected rows.
GUIDED_BUDGETS="${GUIDED_BUDGETS:-}"
# Shares of the sensitivity-blind strategies; empty = GUIDED_BUDGETS.
BLIND_BUDGETS="${BLIND_BUDGETS:-}"

echo "[config] POSEIDON_ROOT=$POSEIDON_ROOT GPU_ID=$GPU_ID  runs=$NUM_RUNS" >&2

chase_make() { make GPU_ID="$GPU_ID" "$@"; }

# Exit 7 is a diverged solve, recorded as such.
source "$ART/scripts/measure.sh"

# run_avg5 <launch> <exe> [trace file]: AVG_MS over NUM_RUNS runs, EIG_L2 and LOCKED of the
# last; the first run's output goes to the trace file.
run_avg5() {
  local _cl="$1" _ce="$2" _ct="${3:-}" _cr _cms _cl2 _clk _csum=0
  for _cr in $(seq 1 "$NUM_RUNS"); do
    MEASURE_ALLOW=7 measure "$_cl, run $_cr of $NUM_RUNS" _cms=time:solve_ms _cl2=acc:"eigval L2 vs planted" \
      -- timeout 600 env CUDA_VISIBLE_DEVICES="$GPU_ID" "$_ce"
    [ "$_cr" = 1 ] && [ -n "$_ct" ] && printf '%s\n' "$MEASURE_OUT" > "$_ct"
    if [ "$MEASURE_STATUS" = 7 ]; then
      echo "[strategies] $_cl, run $_cr: the solve diverged (exit 7); recorded as diverged"
      AVG_MS=diverged EIG_L2=diverged LOCKED=diverged
      return 0
    fi
    _clk=$(printf '%s\n' "$MEASURE_OUT" | grep -oE "locked [0-9]+/[0-9]+" | tail -1)
    [ -n "$_clk" ] || measure_fail "$_cl, run $_cr of $NUM_RUNS" "it printed no 'locked <n>/<m>'"
    _csum=$(num_add "$_csum" "$_cms")
  done
  AVG_MS=$(num_div "$_csum" "$NUM_RUNS") EIG_L2="$_cl2" LOCKED="$_clk"
}

# The applied configuration as <site>=<scheme>;... from solve.log's pin lines.
solve_schemes() {
  local sch
  sch=$(pins_of_log "${1:-solve.log}" | sed -E 's/^_Z[0-9]+([a-z]+)_opt[^:]*:M0=/\1=/' | paste -sd';' -)
  [ -z "$sch" ] && sch="all-FP64"
  echo "$sch"
}
