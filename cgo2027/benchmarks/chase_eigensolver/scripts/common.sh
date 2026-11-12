#!/bin/bash
# Shared configuration for the chase_eigensolver pipeline scripts. Config
# (compiler, CUDA, GPU_ID, python, in-tree Ozaki RT) comes from the artifact-wide
# env.sh, which auto-detects the toolchain (scripts/detect.sh) and is overridable
# from the environment or local.env; defaults are no longer duplicated here. The
# scripts drive the Makefile targets (profile/probe/solve/baseline), so the
# compiler flags live in ONE place (the Makefile) and `make` and
# `bash scripts/<x>.sh` agree.
set -o pipefail

ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
. "$ART/scripts/bcshim.sh"   # num_add / num_div: exact `bc -l` replacement
source "$ART/env.sh"
source "$ART/scripts/expected.sh"

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BENCH_DIR"

# PROFILES=fresh (default) | shipped: the solves read this run's N=512 profile
# (fpprofile/) or the shipped one (profiles/surrogate512/, with which the
# expected-results table's decisions were made). The sensitivity-blind arm
# reads a copy of the same profile with its Kappa lines removed.
PROFILES="${PROFILES:-fresh}"
case "$PROFILES" in
  fresh) PROFSRC=fpprofile ;;
  shipped) PROFSRC=profiles/surrogate512 ;;
  *) echo "PROFILES=$PROFILES: use fresh or shipped" >&2; exit 2 ;;
esac
ON_SHIPPED_MODEL=0
cmp -s "$COST_MODEL" "$(bash "$ART/scripts/detect.sh" shipped_cost_model)" && ON_SHIPPED_MODEL=1

PROF_DEFS="${PROF_DEFS:--DCHASE_N=512 -DCHASE_NEV=32}"   # SCVH surrogate profile
DEP_DEFS="${DEP_DEFS:--DCHASE_N=2048 -DCHASE_NEV=64}"    # deploy (paper figure)
NUM_RUNS="${NUM_RUNS:-5}"                                # avg-of-5 wall-clock

# Ladder budgets (shares): leave UNSET (the default) and run_ladder.sh sweeps an
# evenly spaced sample of THIS model's frontier, plus, on the shipped RTX 5090
# model, the frontier points of the rows of EXPECTED_RESULTS.md
# (expected/picks.tsv). Set explicitly only to sweep chosen shares.
LADDER_BUDGETS="${LADDER_BUDGETS:-}"
# Sensitivity-blind arm's budget list; empty means "the same list as the guided
# arm", which is what the shipped pipeline does. Set it to a subset when the
# guided arm sweeps every DP breakpoint of the device and the ablation is only
# reported at the knee (run_ladder.sh step 4b solves the knee in whichever arm
# missed it), so the ablation does not double a full sweep's node time.
LADDER_OFF_BUDGETS="${LADDER_OFF_BUDGETS:-}"

echo "[config] POSEIDON_ROOT=$POSEIDON_ROOT GPU_ID=$GPU_ID  runs=$NUM_RUNS" >&2

# make wrapper that always passes GPU_ID through.
chase_make() { make GPU_ID="$GPU_ID" "$@"; }

# Run the current chase_opt.exe NUM_RUNS times; echo "avg_ms<TAB>eigL2<TAB>locked",
# and stash run #1's full stdout at $1 (for the EIGRESID line).
run_avg5() {
  local trace="$1" sum=0 n=0 l2="FAIL" lk="?"
  for r in $(seq 1 "$NUM_RUNS"); do
    local o; o=$(timeout 600 env CUDA_VISIBLE_DEVICES="$GPU_ID" ./chase_opt.exe 2>&1)
    [ "$r" = 1 ] && [ -n "$trace" ] && echo "$o" > "$trace"
    local m; m=$(echo "$o" | grep -oE "solve_ms = [0-9.]+" | grep -oE "[0-9.]+$")
    [ -n "$m" ] && { sum=$(num_add "$sum" "$m"); n=$((n+1)); }
    l2=$(echo "$o" | grep "L2 vs planted" | grep -oE "= [0-9.e+-]+" | head -1 | tr -d '= ')
    lk=$(echo "$o" | grep -oE "locked [0-9]+/[0-9]+" | tail -1)
  done
  local avg; avg=$([ "$n" -gt 0 ] && num_div "$sum" "$n" || echo "?")
  printf "%s\t%s\t%s\n" "$avg" "${l2:-FAIL}" "${lk:-?}"
}

# The configuration the solver applied, from solve.log's pin lines: one
# <site>=<scheme> per rewritten site, sorted (hemm = the filter H*Z, resid,
# gram, rrhv and rrproj = the residual, Gram and two Rayleigh-Ritz products).
solve_schemes() {
  local sch
  sch=$(pins_of_log "${1:-solve.log}" | sed -E 's/^_Z[0-9]+([a-z]+)_opt[^:]*:M0=/\1=/' | paste -sd';' -)
  [ -z "$sch" ] && sch="all-FP64"
  echo "$sch"
}
