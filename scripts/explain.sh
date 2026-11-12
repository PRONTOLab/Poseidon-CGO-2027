#!/usr/bin/env bash
# explain.sh <row id> [poseidon-explain options...]
#
# Shows what the compiler decided for one decision row (any device: the row
# id names it, as in expected/picks*.tsv), compile-only and without a GPU:
# the row is compiled exactly as ./run_artifact.sh --picks compiles it (the
# device's saved cost model and architecture, the shipped profiles and Herbie
# answers), with -poseidon-explain, and poseidon-explain prints, per FP
# subgraph, the candidates considered, the one applied and the ones rejected,
# each with its modelled cost (a share of the site's profiled cost) and
# modelled error, then the Pareto frontier with the flag that selects each
# point.
#
# Options after the row id go to poseidon-explain: --format
# report|remarks|json (default report), --all (every alternative), --max N,
# --function <name>. The compile's log and the
# explanation files stay in explain/<row>/ (report.txt: the default report).
#
#   scripts/explain.sh --list             # every row id, per device
#   scripts/explain.sh ozp-tau1e-7
#   scripts/explain.sh t1-drift-gh200 --all
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ART"
usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in
  ""|-h|--help) usage; [ -n "${1:-}" ]; exit $? ;;
esac

export POSEIDON_ARTIFACT_NO_GPU=1 PROFILES=shipped GENGA_HERBIE=replay CUDA_VISIBLE_DEVICES= ROWS_SOLVE_FRESH=1
unset GPU_ID
export COST_MODEL="${COST_MODEL:-$ART/cost_models/cm_sm_120_RTX5090.csv}"
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
       bash scripts/rows.sh "$ROW" "$OUT/compile.log" -poseidon-explain="$OUT/json" ) > "$OUT/rows.out" 2>&1; then
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
