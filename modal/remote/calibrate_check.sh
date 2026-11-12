#!/usr/bin/env bash
# calibrate_check.sh <out dir>: measure this GPU's cost model into a scratch
# file and compare it with the saved B300 models of the replay pack. Nothing
# uses the fresh model; the report says whether the saved one still holds.
# The measurement is the artifact's bootstrap command (the same arms, the
# shipped ozp profile for the in-kernel arm, R-ChFSI's rectangular shapes).
set -uo pipefail
OUT="${1:?usage: calibrate_check.sh <out dir>}"
TC="${TC:-/toolchain}"; BUNDLE="${BUNDLE:-/bundle}"; WORK="${WORK:-/work}"
PACK="${PACK:-$WORK/replay_pack}"
if [ ! -f "$PACK/picks_b300.tsv" ] && [ -f "$BUNDLE/replay_pack.tar.gz" ]; then
  rm -rf "$PACK" && mkdir -p "$(dirname "$PACK")" && tar -C "$(dirname "$PACK")" -xzf "$BUNDLE/replay_pack.tar.gz"
fi
PCAL="$TC/poseidon-build/Poseidon/bin/poseidon-calibrate"
mkdir -p "$OUT/fresh" "$WORK"
rm -rf "$WORK/calib" && mkdir -p "$WORK/calib" && tar -C "$WORK/calib" -xzf "$BUNDLE/artifact.tar.gz"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/device.txt"
# The TCEC row is measured against the backend the compiler links, as the
# artifact's bootstrap does.
CUMPS_ARG=()
if grep -q '^POSEIDON_TCEC_USE_CUMPSGEMM:BOOL=ON' "$TC/poseidon-build/CMakeCache.txt"; then
  CUMPS_ARG=(--cumpsgemm "$(sed -n 's/^POSEIDON_CUMPSGEMM_ROOT:PATH=//p' "$TC/poseidon-build/CMakeCache.txt")")
fi
T0=$(date +%s)
"$PCAL" --gpu 0 --out "$OUT/fresh" --force "${CUMPS_ARG[@]}" \
  --profile "$WORK/calib/artifact/benchmarks/ozp/profiles/surrogate512" \
  --ozaki-rect 112x112x2048,2048x112x2048 2>&1 | tee "$OUT/calibrate.log"
rc=${PIPESTATUS[0]}
echo "calibration_s $(( $(date +%s) - T0 )) exit $rc" | tee -a "$OUT/calibrate.log"
CSV="$(ls -t "$OUT"/fresh/cm_*.csv 2>/dev/null | head -1)"
[ -s "$CSV" ] || { echo "calibrate_check: no fresh model written"; exit 1; }
for saved in "$PACK"/cost_models/*.csv; do
  python3 -I "$(dirname "$0")/cm_compare.py" "$CSV" "$saved" | tee "$OUT/compare_$(basename "$saved" .csv).txt"
done
