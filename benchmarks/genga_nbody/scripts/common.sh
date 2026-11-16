#!/bin/bash
# Sourced by the genga_nbody scripts: configuration, compiler flag groups, launch helpers.
# GENGA_HERBIE=replay (default; answers from $HERBIE_ANSWERS) | run (search with Herbie).
# PROFILES=fresh (default) | shipped (solve from profiles/<kernel>/).

set -o pipefail

ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ART/env.sh"
source "$ART/scripts/expected.sh"

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_RUNS="${NUM_RUNS:-5}"
GENGA_HERBIE="${GENGA_HERBIE:-auto}"
HERBIE_ANSWERS_GIVEN=0; [ -n "${HERBIE_ANSWERS:-}" ] && HERBIE_ANSWERS_GIVEN=1
HERBIE_ANSWERS="${HERBIE_ANSWERS:-$ART/devices/rtx5090/herbie_answers}"
PROFILES="${PROFILES:-fresh}"
case "$GENGA_HERBIE" in auto|replay|run) ;; *) echo "GENGA_HERBIE=$GENGA_HERBIE: use auto, replay or run" >&2; exit 2 ;; esac
case "$PROFILES" in fresh|shipped) ;; *) echo "PROFILES=$PROFILES: use fresh or shipped" >&2; exit 2 ;; esac

# The shipped Herbie answers are keyed by the Herbie platform of the shipped cost model;
# a measured model gets its platform generated beside it.
ON_SHIPPED_MODEL=0
cmp -s "$COST_MODEL" "$(bash "$ART/scripts/detect.sh" shipped_cost_model)" && ON_SHIPPED_MODEL=1
if [ ! -s "${COST_MODEL%.csv}.herbie.rkt" ]; then
  "$(bash "$ART/scripts/detect.sh" poseidon_calibrate)" --only herbie-platform --out "$COST_MODEL" >&2 ||
    { echo "common.sh: could not generate the Herbie platform of $COST_MODEL" >&2; return 1 2>/dev/null || exit 1; }
fi

# auto: replay the shipped Herbie answers on a shipped cost model; a model measured on this
# GPU has no shipped answers, so its Herbie questions are searched (when Herbie is available).
if [ "$GENGA_HERBIE" = auto ]; then
  GENGA_HERBIE=replay
  if [ "$ON_SHIPPED_MODEL" = 0 ] && [ "$HERBIE_ANSWERS_GIVEN" = 0 ] && [ -x "${HERBIE:-}" ]; then
    GENGA_HERBIE=run
  fi
fi
GENGA_HERBIE_USABLE=1
if [ "$GENGA_HERBIE" = replay ] && [ "$ON_SHIPPED_MODEL" = 0 ] && [ "$HERBIE_ANSWERS_GIVEN" = 0 ]; then
  GENGA_HERBIE_USABLE=0
fi
herbie_skip_note() {   # <what is skipped>
  echo "[skip] $1: SKIPPED, not failed. The shipped Herbie answers belong to the shipped cost"
  echo "[skip]   models, and this run prices with a model measured on this GPU, whose Herbie"
  echo "[skip]   questions have no shipped answer, and no Herbie was found to search them (HERBIE=<a"
  echo "[skip]   Herbie with --num-taylor-terms>; scripts/build_toolchain.sh --with-herbie builds one), or"
  echo "[skip]   HERBIE_ANSWERS=<pack> replays a pack made for this model. The GENGA decisions of"
  echo "[skip]   the RTX 5090, B300 and GH200 are checked compile-only: ./run_artifact.sh --picks --device all"
}

cd "$BENCH_DIR"

COMMON_CUDA=( -x cuda --cuda-gpu-arch="$GPU_ARCH" --cuda-path="$CUDA"
  -fcuda-rdc -O3 -L"$CUDA/lib64" -lstdc++ -lm )

# A rolled Newton loop keeps the drift kernel's subgraphs small enough for Herbie.
KEPLER_CUDA=( "${COMMON_CUDA[@]}" -fno-unroll-loops -DKEPLER_ROLLED )

# Herbie caps bound a search to minutes and ~13 GB; the drift kernel's series needs 4 Taylor terms.
GENGA_TAYLOR_TERMS="${GENGA_TAYLOR_TERMS-2,4}"
HERBIE_CAPS=( -poseidon-herbie-num-threads=4 -poseidon-herbie-num-pts=256
  -poseidon-herbie-num-enodes=2000 -poseidon-herbie-num-iters=3
  -poseidon-herbie-timeout=300
  ${GENGA_TAYLOR_TERMS:+-poseidon-herbie-num-taylor-terms=$GENGA_TAYLOR_TERMS} )
SPLIT=( -poseidon-min-uses-split=2 -poseidon-min-ops-split=10 )

SOLVE=( -poseidon-enable-pt -poseidon-num-samples=1024 -poseidon-loose-coverage
  -poseidon-cost-model="$COST_MODEL" )

# Early pruning is off: with 2- and 4-term answers combined it drops frontier points.
herbie_flags() {
  local k="$1" cache="$2"
  if [ "$GENGA_HERBIE" = replay ]; then
    # A nonexistent Herbie binary: a question without a shipped answer stops the compile.
    printf '%s\n' -poseidon-enable-herbie=1 "${SPLIT[@]}" "${HERBIE_CAPS[@]}" \
      -poseidon-early-prune=false -poseidon-cache="$cache" \
      -poseidon-import-herbie-answers="$HERBIE_ANSWERS/$k" \
      -poseidon-herbie-binary="$HERBIE_ANSWERS/no-herbie-in-replay-mode"
  else
    printf '%s\n' -poseidon-enable-herbie=1 "${SPLIT[@]}" "${HERBIE_CAPS[@]}" \
      -poseidon-early-prune=false -poseidon-cache="$cache"
    [ -z "${HERBIE:-}" ] || printf '%s\n' -poseidon-herbie-binary="$HERBIE"
  fi
}

# solve_profile <kernel> <fresh dir>; replayed Herbie answers need the profile they were asked from.
solve_profile() {
  local k="$1" fresh="$2"
  if [ "$PROFILES" = shipped ] || { [ "$k" != force ] && [ "$GENGA_HERBIE" = replay ]; }; then
    echo "profiles/$k"
  else
    echo "$fresh"
  fi
}

compare_profile() {
  "$PY" - "$1" "profiles/$2" <<'PY'
import glob, os, sys
def ranges(d):
    f = glob.glob(os.path.join(d, "*.fpprofile"))
    if len(f) != 1: return None
    keep = ("MinRes", "MaxRes", "Exec", "Operand", "RedTrip", "LaunchCount")
    return [l.strip() for l in open(f[0]) if l.strip().startswith(keep)]
a, b = ranges(sys.argv[1]), ranges(sys.argv[2])
if a is None or b is None:
    print("[profile] cannot compare %s with %s" % tuple(sys.argv[1:])); sys.exit(0)
diff = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
print("[profile] %s vs %s: %s" % (sys.argv[1], sys.argv[2],
      "same ranges and counts" if diff == 0 else "%d range/count lines differ" % diff))
PY
}

source "$ART/scripts/measure.sh"

# genga_run <launch> [<var>=<time|acc>:<label>...] -- <timeout s> <exe> [args...]; sets GENGA_MS.
genga_run() {
  local _gl="$1" _gs=() _gt
  shift
  while [ $# -gt 0 ] && [ "$1" != -- ]; do _gs+=("$1"); shift; done
  shift; _gt="$1"; shift
  measure "$_gl" GENGA_MS=time:"Elapsed time" "${_gs[@]}" \
    -- env CUDA_VISIBLE_DEVICES="$GPU_ID" timeout "$_gt" "$@"
}

# mean_time <launch> <timeout s> <exe> [args...]; sets MEAN_MS over NUM_RUNS launches.
mean_time() {
  local _gl="$1" _gi _gts=()
  shift
  for _gi in $(seq "$NUM_RUNS"); do
    genga_run "$_gl, run $_gi of $NUM_RUNS" -- "$@"
    _gts+=("$GENGA_MS")
  done
  MEAN_MS="$("$PY" -c "import sys; v = [float(x) for x in sys.argv[1:]]; print(f'{sum(v)/len(v):.6f}')" "${_gts[@]}")"
}

need_exe() {
  [ -s "$1" ] && return 0
  echo "[frontier] the compile at share $2 produced no $1; the step stops. Last lines of $3:" >&2
  tail -n 15 "$3" 2>/dev/null | sed 's/^/    /' >&2
  exit 1
}

herbie_summary() { grep -a -E '\[poseidon\] Herbie answers:' "$1" | sed 's/ replayed from .*/ replayed/' || true; }

# Solve flags, one per line, shared by the pipeline and rows.sh (`mapfile -t FP < <(...)`).
force_flags() {   # [extra flags...]
  printf '%s\n' -poseidon-profile-use=./"$(solve_profile force fpprofile)" \
    -poseidon-enable-herbie=0 "${SOLVE[@]}" "$@"
}
kepler_flags() {  # <cache dir> [extra flags...]
  local c="$1"; shift
  printf '%s\n' -poseidon-profile-use=./"$(solve_profile kepler fpprofile_kep_rolled)"
  herbie_flags kepler "$c"
  printf '%s\n' "${SOLVE[@]}" "$@"
}
kepler_herbieoff_flags() {  # [extra flags...]
  printf '%s\n' -poseidon-profile-use=./"$(solve_profile kepler fpprofile_kep_rolled)" \
    -poseidon-enable-herbie=0 "${SPLIT[@]}" -poseidon-cache=cache_kep_herbieoff "${SOLVE[@]}" "$@"
}
unified_flags() { # <cache dir> [extra flags...]
  local c="$1"; shift
  printf '%s\n' -poseidon-profile-use=./"$(solve_profile unified fpprofile_unified)"
  herbie_flags unified "$c"
  printf '%s\n' "${SOLVE[@]}" "$@"
}

echo "[config] POSEIDON_ROOT=$POSEIDON_ROOT GPU_ID=$GPU_ID arch=$GPU_ARCH" >&2
echo "[config] COST_MODEL=${COST_MODEL#$ART/}  NUM_RUNS=$NUM_RUNS  GENGA_HERBIE=$GENGA_HERBIE  GENGA_TAYLOR_TERMS=$GENGA_TAYLOR_TERMS  PROFILES=$PROFILES" >&2
