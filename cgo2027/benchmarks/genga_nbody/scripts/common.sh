#!/bin/bash
# Shared configuration for the genga_nbody pipeline scripts. Config (compiler,
# CUDA, GPU_ID, python, COST_MODEL) comes from the artifact-wide env.sh, which
# auto-detects the toolchain (scripts/detect.sh) and is overridable from the
# environment or local.env, so `make` and a direct `bash scripts/<x>.sh` see the
# same values.
#
# Knobs (environment):
#   GENGA_HERBIE=replay (default) | run
#     replay: the drift and unified solves take Herbie's answers from
#             $HERBIE_ANSWERS/<kernel>/ (-poseidon-import-herbie-answers) and
#             never start Herbie; a question with no answer there stops the
#             compile. HERBIE_ANSWERS defaults to herbie_answers/, the answers
#             searched for the shipped RTX 5090 model (an answer is keyed by
#             the device family of the cost model and by its Herbie platform,
#             so they replay only with that model); for another device, point
#             HERBIE_ANSWERS at an answer pack made for that device, or search.
#             They solve from the profiles the answers were searched
#             for (profiles/<kernel>/), because a Herbie question carries the
#             profiled ranges.
#     run:    Herbie searches every subgraph itself, from this run's profile,
#             caching into cache_<kernel>/ (HERBIE=<binary>, else the one the
#             compiler was configured with); about 10 minutes and up to 13 GB
#             of memory per solve. Each search runs once per Taylor-term cap in
#             GENGA_TAYLOR_TERMS (default 2,4), which needs a Herbie that
#             accepts --num-taylor-terms; GENGA_TAYLOR_TERMS= (empty) searches
#             once with Herbie's own cap (2 terms), which every Herbie accepts
#             but which does not find the drift kernel's 4-term series.
#   PROFILES=fresh (default) | shipped
#     shipped: every solve reads profiles/<kernel>/ instead of this run's
#     profile (decisions then match the expected-results table exactly).

set -o pipefail

ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ART/env.sh"
source "$ART/scripts/expected.sh"

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_RUNS="${NUM_RUNS:-5}"    # avg-of-5 wall-clock (artifact convention)
GENGA_HERBIE="${GENGA_HERBIE:-replay}"
HERBIE_ANSWERS_GIVEN=0; [ -n "${HERBIE_ANSWERS:-}" ] && HERBIE_ANSWERS_GIVEN=1
HERBIE_ANSWERS="${HERBIE_ANSWERS:-$BENCH_DIR/herbie_answers}"
PROFILES="${PROFILES:-fresh}"
case "$GENGA_HERBIE" in replay|run) ;; *) echo "GENGA_HERBIE=$GENGA_HERBIE: use replay or run" >&2; exit 2 ;; esac
case "$PROFILES" in fresh|shipped) ;; *) echo "PROFILES=$PROFILES: use fresh or shipped" >&2; exit 2 ;; esac

# GENGA solves on the device's cost model uniformly scaled x1e6. A uniform
# scale leaves every solver decision unchanged; the scaled copy is kept because
# the shipped Herbie answers were searched against the Herbie platform
# generated from it, and the platform is part of every Herbie question. The
# dimensionless *_rel rows (matrix-product ratios, unused here) are copied
# unscaled. For the shipped RTX 5090 model the scaled copy and its platform
# ship in cost_models/x1e6/; for a model measured on this device they are
# derived here.
SHIPPED_CM="$(bash "$ART/scripts/detect.sh" shipped_cost_model)"
ON_SHIPPED_MODEL=0
if cmp -s "$COST_MODEL" "$SHIPPED_CM" ||
   cmp -s "$COST_MODEL" "$ART/cost_models/x1e6/cm_native_x1e6.csv"; then
  ON_SHIPPED_MODEL=1
  GENGA_CM="$ART/cost_models/x1e6/cm_native_x1e6.csv"
elif head -1 "$COST_MODEL" | grep -q '^# poseidon-derived-from:'; then
  GENGA_CM="$COST_MODEL"   # already a scaled copy (a caller handed it back)
else
  GENGA_CM="$BENCH_DIR/measured/cm_native_x1e6.csv"
  bash "$ART/scripts/genga_x1e6.sh" "$COST_MODEL" "$GENGA_CM" ||
    { return 1 2>/dev/null || exit 1; }
fi
COST_MODEL="$GENGA_CM"

# Replay needs answers for this model's Herbie platform: the default ones are
# the RTX 5090 model's (devices/<device>/herbie_answers/ hold the B300's and
# GH200's saved models', which check_picks.sh passes), so a model measured on
# this GPU has none unless HERBIE_ANSWERS names a pack made for it. The drift
# and unified solves are then skipped, not failed: on such a model they need
# Herbie (GENGA_HERBIE=run).
GENGA_HERBIE_USABLE=1
if [ "$GENGA_HERBIE" = replay ] && [ "$ON_SHIPPED_MODEL" = 0 ] && [ "$HERBIE_ANSWERS_GIVEN" = 0 ]; then
  GENGA_HERBIE_USABLE=0
fi
herbie_skip_note() {   # <what is skipped>
  echo "[skip] $1: SKIPPED, not failed. The shipped Herbie answers belong to the shipped cost"
  echo "[skip]   models, and this run prices with a model measured on this GPU, whose Herbie"
  echo "[skip]   questions have no shipped answer. GENGA_HERBIE=run searches them (a Herbie with"
  echo "[skip]   --num-taylor-terms; scripts/build_toolchain.sh --with-herbie builds one), or"
  echo "[skip]   HERBIE_ANSWERS=<pack> replays a pack made for this model. The GENGA decisions of"
  echo "[skip]   the RTX 5090, B300 and GH200 are checked compile-only: ./run_artifact.sh --picks --device all"
}

cd "$BENCH_DIR"

# Common compiler flags shared by force / kepler / unified compiles.
# poseidon-clang++ supplies the two pass plugins, Poseidon's include directory,
# -ffp-contract=on (CUDA's default 'fast' fuses the error-free transforms an
# expansion is built on, collapsing it to float) and the profiler runtime.
#  -fcuda-rdc : the annotated kernels are compiled as relocatable device code.
COMMON_CUDA=( -x cuda --cuda-gpu-arch="$GPU_ARCH" --cuda-path="$CUDA"
  -fcuda-rdc -O3 -L"$CUDA/lib64" -lstdc++ -lm )

# Extra flags for the Kepler-drift solve: the Newton loop must stay ROLLED
# (-fno-unroll-loops -DKEPLER_ROLLED) so dEj is a loop-carried PHI leaf and the
# f-and-g cancellation subgraphs stay small enough for Herbie to rewrite.
KEPLER_CUDA=( "${COMMON_CUDA[@]}" -fno-unroll-loops -DKEPLER_ROLLED )

# Bounded-Herbie caps: keep a search in minutes and under ~13 GB of RAM.
# Chain-split isolates element-extraction so the 1-cos / sin-x expressions
# become tiny and Herbie-tractable. Every search runs once with Herbie's
# default Taylor cap (2 terms) and once with 4, and the candidates are the
# union: the drift kernel's Newton-step series needs 4 terms.
GENGA_TAYLOR_TERMS="${GENGA_TAYLOR_TERMS-2,4}"
HERBIE_CAPS=( -poseidon-herbie-num-threads=4 -poseidon-herbie-num-pts=256
  -poseidon-herbie-num-enodes=2000 -poseidon-herbie-num-iters=3
  -poseidon-herbie-timeout=300
  ${GENGA_TAYLOR_TERMS:+-poseidon-herbie-num-taylor-terms=$GENGA_TAYLOR_TERMS} )
SPLIT=( -poseidon-min-uses-split=2 -poseidon-min-ops-split=10 )

# Flags of every GENGA solve. The drift and unified solves are exact
# (-poseidon-early-prune=false): with the union of 2- and 4-term answers the
# pruned DP can drop a frontier point a shorter series reaches.
SOLVE=( -poseidon-enable-pt -poseidon-num-samples=1024 -poseidon-loose-coverage
  -poseidon-cost-model="$COST_MODEL" )

# herbie_flags <kernel> <cache dir>: the Herbie half of a drift or unified solve.
herbie_flags() {
  local k="$1" cache="$2"
  if [ "$GENGA_HERBIE" = replay ]; then
    # No Herbie process: the binary named here does not exist, so a question
    # without a shipped answer stops the compile instead of searching.
    printf '%s\n' -poseidon-enable-herbie=1 "${SPLIT[@]}" "${HERBIE_CAPS[@]}" \
      -poseidon-early-prune=false -poseidon-cache="$cache" \
      -poseidon-import-herbie-answers="$HERBIE_ANSWERS/$k" \
      -poseidon-herbie-binary="$BENCH_DIR/herbie_answers/no-herbie-in-replay-mode"
  else
    printf '%s\n' -poseidon-enable-herbie=1 "${SPLIT[@]}" "${HERBIE_CAPS[@]}" \
      -poseidon-early-prune=false -poseidon-cache="$cache"
    [ -z "${HERBIE:-}" ] || printf '%s\n' -poseidon-herbie-binary="$HERBIE"
  fi
}

# solve_profile <kernel> <fresh dir>: the profile directory a solve reads.
solve_profile() {
  local k="$1" fresh="$2"
  if [ "$PROFILES" = shipped ] || { [ "$k" != force ] && [ "$GENGA_HERBIE" = replay ]; }; then
    echo "profiles/$k"
  else
    echo "$fresh"
  fi
}

# compare_profile <fresh dir> <kernel>: whether this run's profile has the
# ranges of the shipped one, i.e. whether its Herbie questions are the shipped
# ones (sums of products may differ in the last digits: atomics).
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

# Herbie use of a solve log: searches run, answers read from the cache, replays.
herbie_summary() { grep -a -E '\[poseidon\] Herbie answers:' "$1" | sed 's/ replayed from .*/ replayed/' || true; }

# The flag set of each kind of solve, one flag per line (read them with
# `mapfile -t FP < <(kepler_flags ...)`). The pipeline scripts and the
# compile-only pick check (../../scripts/check_picks.sh) take them from here, so
# both compile with identical command lines.
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
