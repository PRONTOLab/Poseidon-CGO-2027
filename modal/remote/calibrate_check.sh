#!/usr/bin/env bash
# calibrate_check.sh <out dir>: measure this GPU's cost model and compare it with the shipped B300
# model; informational, nothing uses the fresh model.
set -uo pipefail
OUT="${1:?usage: calibrate_check.sh <out dir>}"
TC="${TC:-/toolchain}"; BUNDLE="${BUNDLE:-/bundle}"; WORK="${WORK:-/work}"
PCAL="$TC/poseidon-build/Poseidon/bin/poseidon-calibrate"
mkdir -p "$OUT/fresh" "$WORK"
rm -rf "$WORK/calib" && mkdir -p "$WORK/calib" && tar -C "$WORK/calib" -xzf "$BUNDLE/artifact.tar.gz"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/device.txt"
T0=$(date +%s)
"$PCAL" --gpu 0 --out "$OUT/fresh" --force \
  --profile "$WORK/calib/artifact/benchmarks/ozp/profiles/surrogate512" \
  --ozaki-rect 112x112x2048,2048x112x2048 2>&1 | tee "$OUT/calibrate.log"
rc=${PIPESTATUS[0]}
echo "calibration_s $(( $(date +%s) - T0 )) exit $rc" | tee -a "$OUT/calibrate.log"
CSV="$(ls -t "$OUT"/fresh/cm_*.csv 2>/dev/null | head -1)"
[ -s "$CSV" ] || { echo "calibrate_check: no fresh model written"; exit 1; }
for saved in "$WORK"/calib/artifact/devices/b300/cost_models/*.csv; do
  python3 -I "$(dirname "$0")/cm_compare.py" "$CSV" "$saved" | tee "$OUT/compare_$(basename "$saved" .csv).txt"
done
