#!/usr/bin/env bash
# bootstrap_cost_model.sh [--calibrate [--fresh-profile]]: check this GPU's cost model,
# or measure one into cost_models/measured/ and record it in cost_models/target.env.
set -eo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSEIDON_ARTIFACT_NO_MODEL=1 source "$ART/env.sh"

MODE=auto FRESH_PROFILE=0
for _a in "$@"; do
  case "$_a" in
    --calibrate) MODE=calibrate ;;
    --fresh-profile) FRESH_PROFILE=1 ;;
    *) echo "usage: $0 [--calibrate [--fresh-profile]]" >&2; exit 2 ;;
  esac
done

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

INK_PROF="$ART/benchmarks/ozp/profiles/surrogate512"
if [ "$FRESH_PROFILE" = 1 ]; then
  echo "[bootstrap] profiling ozp on this GPU for the in-kernel candidates (scripts/profile_only.sh)"
  POSEIDON_ARTIFACT_NO_MODEL=1 bash "$ART/benchmarks/ozp/scripts/profile_only.sh"
  INK_PROF="$ART/benchmarks/ozp/fpprofile"
fi
# R-ChFSI's deploy-size products: a native DGEMM call is priced from the row of its exact extents.
RECT_SHAPES="${RECT_SHAPES:-112x112x2048,2048x112x2048}"

OUTDIR="$ART/cost_models/measured"
mkdir -p "$OUTDIR"
T0=$(date +%s)
"$POSEIDON_CALIBRATE" --gpu "$GPU_ID" --out "$OUTDIR" --force \
    --profile "$INK_PROF" --ozaki-rect "$RECT_SHAPES"
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
