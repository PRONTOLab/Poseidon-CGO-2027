#!/usr/bin/env bash
# make_herbie_answers.sh: search again, compile-only and without a GPU, the Herbie
# answers a device's GENGA rows replay, and write them as an answer pack.
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ART"
usage() {
  cat <<'EOF'
Usage: scripts/make_herbie_answers.sh [--device rtx5090|b300|gh200] [--out <dir>]
  --device D       search the GENGA rows of device D (default: rtx5090)
  --out <dir>      the answer pack written (default: herbie_answers_<device>/)
  HERBIE=<binary>  a Herbie that accepts --num-taylor-terms (default: the compiler's)
EOF
}
DEV=rtx5090 OUTDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEV="${2:?--device needs a device}"; shift ;;
    --out) OUTDIR="${2:?--out needs a directory}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "make_herbie_answers.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
export POSEIDON_ARTIFACT_NO_GPU=1 PROFILES=shipped GENGA_HERBIE=run CUDA_VISIBLE_DEVICES=
unset GPU_ID
export COST_MODEL="${COST_MODEL:-$ART/devices/rtx5090/cost_models/cm_sm_120_RTX5090.csv}"
source "$ART/env.sh" || exit 1
source "$ART/scripts/expected.sh"
device_field "$DEV" gpu_arch > /dev/null || exit 2
OUTDIR="${OUTDIR:-$ART/herbie_answers_$DEV}"

BUILD="$(bash scripts/detect.sh build_dir)" || exit 1
export HERBIE="${HERBIE:-$(sed -n 's/^POSEIDON_HERBIE_BINARY:FILEPATH=//p' "$BUILD/CMakeCache.txt")}"
if [ ! -x "$HERBIE" ] || ! "$HERBIE" improve --help 2>&1 | grep -q -- '--num-taylor-terms'; then
  echo "make_herbie_answers.sh: no Herbie with --num-taylor-terms at '${HERBIE:-<unset>}'" >&2
  echo "make_herbie_answers.sh: (HERBIE=<binary>, or scripts/build_toolchain.sh --with-herbie)" >&2
  exit 1
fi

RC=0
for R in $(device_rows "$DEV"); do
  [ "$(expected_field "$R" bench)" = genga_nbody ] || continue
  case "$R" in
    *drift-naive*|*force*) continue ;;
    *drift*) K=kepler C=cache_kep_rolled ;;
    *unified*) K=unified C=cache_unified ;;
    *) continue ;;
  esac
  [ "$DEV" = rtx5090 ] || C="${C}_$DEV"
  rm -f "benchmarks/genga_nbody/$C"/cachedHerbie* "benchmarks/genga_nbody/$C/table.json"
  mkdir -p "$OUTDIR"
  echo "== $R: searching ($K, $(basename "$HERBIE"))"
  T0=$(date +%s)
  if ( row_env "$R" && unset HERBIE_ANSWERS &&
       cd benchmarks/genga_nbody && bash scripts/rows.sh "$R" "$OUTDIR/$R.log" ) > "$OUTDIR/$R.out" 2>&1; then
    rm -rf "${OUTDIR:?}/$K"; mkdir -p "$OUTDIR/$K"
    cp "benchmarks/genga_nbody/$C"/cachedHerbieOutput_* "benchmarks/genga_nbody/$C"/cachedHerbieQuestion_* "$OUTDIR/$K/"
    got="$(pins_of_log "$OUTDIR/$R.log")"
    want="$(expected_field "$R" pin | tr ',' '\n' | sed '/^$/d' | sort -u)"
    echo "   $(grep -a -m1 'Herbie answers:' "$OUTDIR/$R.log" | sed 's/^\[poseidon\] //'); $(( $(date +%s) - T0 ))s;" \
         "decision with these answers: $([ "$got" = "$want" ] && echo SAME || echo DIFFERENT)"
  else
    echo "   the compile failed: $OUTDIR/$R.log"; RC=1
  fi
done
[ -d "$OUTDIR" ] && (cd "$OUTDIR" && find . -mindepth 2 -type f | sed 's|^\./||' | sort | xargs -r md5sum > MD5SUMS)
echo "wrote $OUTDIR"
exit $RC
