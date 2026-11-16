#!/bin/bash
# Sourced by the ozp scripts: configuration, compiler flag groups, launch helpers.
# PROFILES=fresh (default) | shipped (solve from profiles/surrogate512/).

set -o pipefail

ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ART/env.sh"
source "$ART/scripts/expected.sh"

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROF_N="${PROF_N:-512}"
APPLY_N="${APPLY_N:-2048}"
NUM_RUNS="${NUM_RUNS:-5}"
PROFILES="${PROFILES:-fresh}"
case "$PROFILES" in
  fresh) PROF=fpprofile ;;
  shipped) PROF=profiles/surrogate512 ;;
  *) echo "PROFILES=$PROFILES: use fresh or shipped" >&2; exit 2 ;;
esac

cd "$BENCH_DIR"
TMPD="$BENCH_DIR/tmp"; mkdir -p "$TMPD"

# The baselines are built by plain clang++, hence the explicit Poseidon include directory.
COMMON_CUDA=( -x cuda --cuda-gpu-arch="$GPU_ARCH" --cuda-path="$CUDA" -fcuda-rdc
  -O3 -Wno-deprecated-declarations -I"$POSEIDON_INC"
  -L"$CUDA/lib64" -lcudart -lcublas -lstdc++ -lm )

POSE=( -DPOSEIDON )

FP=( -poseidon-profile-use=./"$PROF" -poseidon-enable-herbie=0
  -poseidon-enable-pt=0 -poseidon-cache=cache
  -poseidon-cost-model="$COST_MODEL"
  -poseidon-loose-coverage -poseidon-raise-wmma )

DISPATCH=( -poseidon-ozaki-host-dispatch )
DISPATCH_RT=()

pick_of_log() { pins_of_log "$1" | sed 's/^.*:M0=//' | head -1; }

source "$ART/scripts/measure.sh"

# run_one <launch> <exe>: runs on $HMTX $KMTX; RUN_ROW = L2, RMSD, Energy, time_ms (tab-separated).
run_one() {
  local l2 rm en t
  measure "$1" l2=acc:"L2 relative error" rm=acc:RMSD en=acc:"Energy abs error" t=time:"Elapsed time" \
    -- env CUDA_VISIBLE_DEVICES="$GPU_ID" "$2" "$HMTX" "$KMTX"
  RUN_ROW="$(printf '%s\t%s\t%s\t%s' "$l2" "$rm" "$en" "$t")"
}

# end_if_build_failed <tsv> <column>: exits non-zero if a row records a failed compile.
end_if_build_failed() {
  local bad
  bad=$(awk -F'\t' -v c="$2" 'NR > 1 && ($c == "BUILD_FAIL" || $c == "COMPILE-FAILED") {print $1}' "$1" | paste -sd' ' -)
  [ -z "$bad" ] && return 0
  echo "$1: the compile failed for $bad (rows kept); the step ends with an error." >&2
  exit 1
}

# trace_one <launch> <exe> <tsv>: per-iteration residuals of one run.
trace_one() {
  measure "$1" -- env CUDA_VISIBLE_DEVICES="$GPU_ID" "$2" "$HMTX" "$KMTX" --trace
  printf '%s\n' "$MEASURE_OUT" | awk '/^iter/{print $2"\t"$4}' > "$3"
  [ -s "$3" ] || measure_fail "$1" "it printed no iteration"
}

echo "[config] POSEIDON_ROOT=$POSEIDON_ROOT GPU_ID=$GPU_ID arch=$GPU_ARCH" >&2
echo "[config] COST_MODEL=${COST_MODEL#$ART/}  NUM_RUNS=$NUM_RUNS  PROFILES=$PROFILES" >&2
