#!/usr/bin/env bash
# bootstrap_cost_model.sh — make sure THIS GPU has a cost model before any
# experiment runs.
#
#   bash scripts/bootstrap_cost_model.sh               # shipped model on sm_120, else measure
#   bash scripts/bootstrap_cost_model.sh --calibrate   # measure this GPU even on sm_120
#
# The solver's decisions come from a per-device cost model: running with
# another device's CSV reproduces that device's picks. Three cases:
#
#   1. COST_MODEL is already resolved (the shipped RTX 5090 model on an sm_120
#      GPU, a model this script measured earlier, recorded in
#      cost_models/target.env, or COST_MODEL set by the caller): the model is
#      checked against this device with `poseidon-calibrate --check` and never
#      rewritten.
#   2. --calibrate, or a device with no model: poseidon-calibrate measures every
#      family the benchmarks price (per-op throughputs, tensor-core tiles, the
#      three host-dispatch families, the in-kernel raise classes and the native
#      DGEMM rows at R-ChFSI's two rectangular product shapes) into
#      cost_models/measured/, writes the Herbie platform beside it, and
#      cost_models/target.env records it for every later build. Delete
#      target.env to return to the shipped model.
#
# Two inputs belong to the artifact rather than to the compiler, and are passed
# here: the ozp surrogate profile the in-kernel arm enumerates its candidates
# from (the shipped one: the profile decides which candidates exist, never what
# they measure), and R-ChFSI's rectangular product shapes at its deploy size.
# The error-corrected (TCEC) dispatch row is measured against the backend the
# compiler links: cuMpSGEMM when the compiler was configured with
# -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON, its built-in cuBLAS backend otherwise.
set -eo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSEIDON_ARTIFACT_NO_MODEL=1 source "$ART/env.sh"

MODE=auto
case "${1:-}" in
  --calibrate) MODE=calibrate ;;
  "") ;;
  *) echo "usage: $0 [--calibrate]" >&2; exit 2 ;;
esac

GPU_NAME="$(bash "$ART/scripts/detect.sh" gpu_name)"
SHIPPED="$(bash "$ART/scripts/detect.sh" shipped_cost_model)"
echo "[bootstrap] device: $GPU_NAME ($GPU_ARCH, id $GPU_ID)"

if [ "$MODE" = auto ] && [ -n "$COST_MODEL" ]; then
  if cmp -s "$COST_MODEL" "$SHIPPED"; then
    echo "[bootstrap] cost model: the shipped RTX 5090 model ($(basename "$SHIPPED"))"
    [ "$GPU_NAME" = RTX5090 ] || {
      echo "[bootstrap] NOTE: this is a $GPU_NAME, an sm_120 card like the RTX 5090. The"
      echo "[bootstrap] solver makes the RTX 5090's decisions; the timings are this card's."
      echo "[bootstrap] ./run_artifact.sh --calibrate measures this card instead."; }
  else
    echo "[bootstrap] cost model: $COST_MODEL"
  fi
  "$POSEIDON_CALIBRATE" --gpu "$GPU_ID" --check --out "$COST_MODEL"
  exit 0
fi

# ---- measure this device -------------------------------------------------
BUILD="$(bash "$ART/scripts/detect.sh" build_dir)"
CUMPS_ARG=()
if grep -q '^POSEIDON_TCEC_USE_CUMPSGEMM:BOOL=ON' "$BUILD/CMakeCache.txt"; then
  CUMPS_ROOT="$(sed -n 's/^POSEIDON_CUMPSGEMM_ROOT:PATH=//p' "$BUILD/CMakeCache.txt")"
  CUMPS_ARG=( --cumpsgemm "$CUMPS_ROOT" )
  echo "[bootstrap] the compiler links cuMpSGEMM ($CUMPS_ROOT); the TCEC row is measured against it"
else
  echo "[bootstrap] the compiler links its built-in TCEC backend; the TCEC row is measured against it"
fi
INK_PROF="$ART/benchmarks/ozp/profiles/surrogate512"
# R-ChFSI's Gram and Rayleigh-Ritz products at its deploy size (N = 2048,
# NEV = 64): the compiler prices a native DGEMM call of compile-time extents
# from the row measured at exactly those extents.
RECT_SHAPES="${RECT_SHAPES:-112x112x2048,2048x112x2048}"

OUTDIR="$ART/cost_models/measured"
mkdir -p "$OUTDIR"
T0=$(date +%s)
"$POSEIDON_CALIBRATE" --gpu "$GPU_ID" --out "$OUTDIR" --force \
    --profile "$INK_PROF" --ozaki-rect "$RECT_SHAPES" "${CUMPS_ARG[@]}"
CSV="$(ls -t "$OUTDIR"/cm_"$GPU_ARCH"_*.csv 2>/dev/null | head -1)"
[ -s "$CSV" ] || { echo "[bootstrap] poseidon-calibrate wrote no cm_${GPU_ARCH}_*.csv in $OUTDIR" >&2; exit 1; }
echo "calibration_s	$(( $(date +%s) - T0 ))" >> "$ART/cost_models/bootstrap_timing.txt"
"$POSEIDON_CALIBRATE" --gpu "$GPU_ID" --check --out "$CSV"

cat > "$ART/cost_models/target.env" <<ENV
# Written by scripts/bootstrap_cost_model.sh for $GPU_NAME ($GPU_ARCH).
# Delete this file to return to the shipped RTX 5090 model (sm_120 only).
# Guard-style: the caller environment and local.env still override.
COST_MODEL="\${COST_MODEL:-$CSV}"
GPU_ARCH="\${GPU_ARCH:-$GPU_ARCH}"
ENV
echo "[bootstrap] wrote cost_models/target.env (COST_MODEL=${CSV#$ART/})"
