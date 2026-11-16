#!/usr/bin/env bash
# explain.sh: compile one decision row without a GPU, as check_picks.sh does, with
# -poseidon-explain, and print the candidates considered, applied and rejected
# (files in explain/<row>/).
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ART"
usage() {
  cat <<'EOF'
Usage: scripts/explain.sh <row id> [poseidon-explain options]
       scripts/explain.sh --list
  --list               every row id, per device
  --format F           report (default), source, diff, remarks, fpcore or json
  --all                list every alternative
  --max N              alternatives listed per FP subgraph (default 8)
  --function <name>    only the sites with <name> in their name
EOF
}
case "${1:-}" in
  ""|-h|--help) usage; [ -n "${1:-}" ]; exit $? ;;
esac

export POSEIDON_ARTIFACT_NO_GPU=1 PROFILES=shipped GENGA_HERBIE=replay CUDA_VISIBLE_DEVICES= ROWS_SOLVE_FRESH=1
unset GPU_ID
export COST_MODEL="${COST_MODEL:-$ART/devices/rtx5090/cost_models/cm_sm_120_RTX5090.csv}"
source "$ART/env.sh" || exit 1
source "$ART/scripts/expected.sh"

if [ "$1" = --list ]; then
  for d in $(devices_list); do
    echo "$d ($(device_field "$d" gpu_arch)):"
    for r in $(device_rows "$d"); do printf '  %-24s %s\n' "$r" "$(expected_field "$r" what | cut -c1-90)"; done
  done
  exit 0
fi
ROW="$1"; shift
B="$(expected_field "$ROW" bench)" || { echo "explain.sh: '$ROW' is no row; scripts/explain.sh --list names them" >&2; exit 2; }
EXPLAIN="$(bash scripts/detect.sh poseidon_explain)" || exit 1

OUT="$ART/explain/$ROW"
rm -rf "$OUT"; mkdir -p "$OUT/json"
echo "explain: compiling $ROW ($B, device $(row_device "$ROW")) with -poseidon-explain ..." >&2
if ! ( row_env "$ROW" && cd "benchmarks/$B" &&
       bash scripts/rows.sh "$ROW" "$OUT/compile.log" -poseidon-explain="$OUT/json" -gline-tables-only ) > "$OUT/rows.out" 2>&1; then
  echo "explain.sh: the compile of $ROW failed; see $OUT/compile.log and $OUT/rows.out" >&2
  grep -a -m3 -i error "$OUT/compile.log" "$OUT/rows.out" >&2
  exit 1
fi
ls "$OUT"/json/*.explain.json > /dev/null 2>&1 ||
  { echo "explain.sh: the compile of $ROW wrote no explanation (see $OUT/compile.log)" >&2; exit 1; }

want="$(expected_field "$ROW" pin | tr ',' '\n' | sed '/^$/d' | sort -u)"
got="$(pins_of_log "$OUT/compile.log")"
"$PY" "$EXPLAIN" "$OUT/json" > "$OUT/report.txt" 2>&1
{
  echo "== $ROW: $(expected_field "$ROW" what)"
  echo "   selected by $(expected_field "$ROW" select), cost model $(basename "$(row_cost_model "$ROW")")"
  if [ "$want" = "$got" ]; then echo "   applied: the configuration of the row (SAME)"
  else echo "   applied: a DIFFERENT configuration than the row's"; fi
  echo
} >&2
"$PY" "$EXPLAIN" "$OUT/json" "$@"
