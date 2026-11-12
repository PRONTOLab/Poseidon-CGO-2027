#!/usr/bin/env bash
# check_picks.sh [--device rtx5090|b300|gh200|all] [--no-shipped-cache] [row id ...]
#
# Compile-only check of the compiler's decisions, without a GPU: for each
# decision row of the device (expected/picks.tsv for the RTX 5090,
# expected/picks_<device>.tsv for the others; the device of a named row is the
# file it is in), the benchmark's own scripts/rows.sh compiles the row's
# configuration the way the pipeline does, for that device's architecture,
# from the shipped profiles (PROFILES=shipped), the device's saved cost model
# (devices/devices.tsv, or the row's cost_model column) and, for GENGA, the
# device's Herbie answers; then the configuration the compiler applied (its
# `[poseidon] pin:` lines) is compared with the row's.
#
# Unless --no-shipped-cache (or POSEIDON_SHIPPED_CACHE=0), each compile starts
# from the device's shipped solve caches (scripts/caches.sh), which only save
# time: a compile whose inputs differ from theirs says so and solves itself.
# The `cache` column says what the caches did.
#
# Output: one line per row (SAME, DIFFERENT, COMPILE-FAILED), picks_check/<row>.log
# per compile and picks_check/picks.tsv (device, row, select, result, seconds,
# cache, applied configuration); exit 1 if a row is not SAME.
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ART"

usage() { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; }
DEVICES="" ROWS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICES="${2:?--device needs a device name}"; shift 2 ;;
    --device=*) DEVICES="${1#*=}"; shift ;;
    --no-shipped-cache) export POSEIDON_SHIPPED_CACHE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "check_picks.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    *) ROWS+=("$1"); shift ;;
  esac
done

export POSEIDON_ARTIFACT_NO_GPU=1 PROFILES=shipped GENGA_HERBIE=replay CUDA_VISIBLE_DEVICES=
unset GPU_ID
export COST_MODEL="${COST_MODEL:-$ART/cost_models/cm_sm_120_RTX5090.csv}"
source "$ART/env.sh" || exit 1
source "$ART/scripts/expected.sh"
source "$ART/scripts/caches.sh"

[ "$DEVICES" = all ] && DEVICES="$(devices_list | paste -sd' ' -)"
if [ ${#ROWS[@]} -gt 0 ]; then
  [ -z "$DEVICES" ] || { echo "check_picks.sh: name rows or --device, not both (a row id names its device)" >&2; exit 2; }
  for R in "${ROWS[@]}"; do row_device "$R" > /dev/null || exit 2; done
else
  DEVICES="${DEVICES:-rtx5090}"
  for D in $DEVICES; do
    device_field "$D" gpu_arch > /dev/null || exit 2
    mapfile -t -O ${#ROWS[@]} ROWS < <(device_rows "$D")
  done
fi

# A toolkit too old for the device's architecture cannot compile its rows;
# that says nothing about the decision, so it is reported as such.
cuda_targets() {   # <sm_XY>
  "$CUDA/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_${1#sm_}"
}

OUT="$ART/picks_check"; mkdir -p "$OUT"
printf 'device\trow\tselect\tresult\tseconds\tcache\tapplied\n' > "$OUT/picks.tsv"
echo "compiler: $POSEIDON_CLANGXX"
echo "plugin:   $(bash scripts/detect.sh plugin) (md5 $(md5sum < "$(bash scripts/detect.sh plugin)" | cut -c1-8))"
echo "CUDA:     $CUDA ($("$CUDA/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p'))"
echo "caches:   $([ "${POSEIDON_SHIPPED_CACHE:-1}" = 0 ] && echo "not seeded (--no-shipped-cache)" || echo "seeded from devices/<device>/caches/")"
FAIL=0 LAST_DEV="" NSAME=0 NROWS=0
summary() { [ -n "$LAST_DEV" ] && echo "== $LAST_DEV: $NSAME of $NROWS rows SAME"; NSAME=0 NROWS=0; }
for R in "${ROWS[@]}"; do
  D="$(row_device "$R")"
  if [ "$D" != "$LAST_DEV" ]; then
    summary; LAST_DEV="$D"
    echo "== $D ($(device_field "$D" gpu_arch)): $(device_field "$D" description)"
  fi
  NROWS=$((NROWS + 1))
  B="$(expected_field "$R" bench)" || { FAIL=1; continue; }
  ARCH="$(device_field "$D" gpu_arch)"
  ( row_env "$R" ) || exit 1
  T0=$(date +%s) got="" cache=""
  if ! cuda_targets "$ARCH"; then
    res=COMPILE-FAILED
    echo "check_picks.sh: the CUDA toolkit at $CUDA cannot compile for $ARCH (nvcc --list-gpu-arch has no compute_${ARCH#sm_});" >&2
    echo "check_picks.sh: $D rows need a toolkit that targets $ARCH (sm_103: CUDA 12.9 or newer). Set CUDA=<toolkit>." >&2
  elif ( row_env "$R" && cd "benchmarks/$B" && bash scripts/rows.sh "$R" "$OUT/$R.log" ) > "$OUT/$R.out" 2>&1; then
    want="$(expected_field "$R" pin | tr ',' '\n' | sed '/^$/d' | sort -u)"
    got="$(pins_of_log "$OUT/$R.log")"
    if [ "$want" = "$got" ]; then res=SAME; NSAME=$((NSAME + 1)); else res=DIFFERENT; fi
    cache="$(cache_summary "$OUT/$R.log" "$OUT/$R.out")"
  else
    res=COMPILE-FAILED
  fi
  [ "$res" = SAME ] || FAIL=1
  s=$(( $(date +%s) - T0 ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$D" "$R" "$(expected_field "$R" select)" "$res" "$s" "$cache" \
    "$(echo "$got" | paste -sd',' -)" >> "$OUT/picks.tsv"
  printf '%-24s %-15s %4ss  %s\n' "$R" "$res" "$s" "$cache"
  [ "$res" = COMPILE-FAILED ] && cat "$OUT/$R.out" "$OUT/$R.log" 2>/dev/null | grep -a -m3 -i 'error' | sed 's/^/    /'
done
summary
echo "wrote picks_check/picks.tsv"
exit $FAIL
