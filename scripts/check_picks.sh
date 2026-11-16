#!/usr/bin/env bash
# check_picks.sh: compile-only check, without a GPU, that each decision row of
# expected/picks*.tsv applies its configuration; logs and picks.tsv go to picks_check/.
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ART"

usage() {
  cat <<'EOF'
Usage: scripts/check_picks.sh [--device rtx5090|b300|gh200|all] [row id ...]
  --device D   check every row of device D (default: rtx5090); a named row implies its device
  row id       check only these rows of expected/picks*.tsv
EOF
}
DEVICES="" ROWS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICES="${2:?--device needs a device name}"; shift 2 ;;
    --device=*) DEVICES="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "check_picks.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    *) ROWS+=("$1"); shift ;;
  esac
done

export POSEIDON_ARTIFACT_NO_GPU=1 PROFILES=shipped GENGA_HERBIE=replay CUDA_VISIBLE_DEVICES=
unset GPU_ID
export COST_MODEL="${COST_MODEL:-$ART/devices/rtx5090/cost_models/cm_sm_120_RTX5090.csv}"
source "$ART/env.sh" || exit 1
source "$ART/scripts/expected.sh"

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

cuda_targets() {   # <sm_XY>
  "$CUDA/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_${1#sm_}"
}

OUT="$ART/picks_check"; mkdir -p "$OUT"
printf 'device\trow\tselect\tresult\tseconds\tapplied\n' > "$OUT/picks.tsv"
echo "compiler: $POSEIDON_CLANGXX"
echo "plugin:   $(bash scripts/detect.sh plugin) (md5 $(md5sum < "$(bash scripts/detect.sh plugin)" | cut -c1-8))"
echo "CUDA:     $CUDA ($("$CUDA/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p'))"
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
  T0=$(date +%s) got=""
  if ! cuda_targets "$ARCH"; then
    res=COMPILE-FAILED
    echo "check_picks.sh: the CUDA toolkit at $CUDA cannot compile for $ARCH (nvcc --list-gpu-arch has no compute_${ARCH#sm_});" >&2
    echo "check_picks.sh: $D rows need a toolkit that targets $ARCH (sm_103: CUDA 12.9 or newer). Set CUDA=<toolkit>." >&2
  elif ( row_env "$R" && cd "benchmarks/$B" && bash scripts/rows.sh "$R" "$OUT/$R.log" ) > "$OUT/$R.out" 2>&1; then
    want="$(expected_field "$R" pin | tr ',' '\n' | sed '/^$/d' | sort -u)"
    got="$(pins_of_log "$OUT/$R.log")"
    if [ "$want" = "$got" ]; then res=SAME; NSAME=$((NSAME + 1)); else res=DIFFERENT; fi
  else
    res=COMPILE-FAILED
  fi
  [ "$res" = SAME ] || FAIL=1
  s=$(( $(date +%s) - T0 ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$D" "$R" "$(expected_field "$R" select)" "$res" "$s" \
    "$(echo "$got" | paste -sd',' -)" >> "$OUT/picks.tsv"
  printf '%-24s %-15s %4ss\n' "$R" "$res" "$s"
  [ "$res" = COMPILE-FAILED ] && cat "$OUT/$R.out" "$OUT/$R.log" 2>/dev/null | grep -a -m3 -i 'error' | sed 's/^/    /'
done
summary
echo "wrote picks_check/picks.tsv"
exit $FAIL
