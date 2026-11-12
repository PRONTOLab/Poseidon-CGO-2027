#!/bin/bash
# Energy-drift TIME SERIES (Fig. 4). Integrates the Kepler drift (N=65536) for a
# sweep of step counts under three configurations and records the mean relative
# orbital-energy drift:
#   FP64     = kepler_gold (exact Kepler -> machine-floor drift, the reference)
#   naive    = the plain FP32 downcast: the cheapest point of the Herbie-off
#              frontier (run_kepler_herbieoff.sh), override with B_NAIVE=<share>
#   poseidon = the drift row of the expected-results table (the Newton step as a
#              binary32 4-term series + the f/g block as a two-component
#              expansion), selected at that row's frontier share on the shipped
#              model; override with B_POSEIDON=<share>
# Writes kepler_timeseries.tsv (steps  fp64  naive_fp32  poseidon). The drift is a
# deterministic accuracy trajectory (timing-irrelevant), so each point is a single run.
source "$(dirname "$0")/common.sh"
SRC=genga_kepler.cu

[ -s cache_kep_herbieoff/budgets.txt ] || { echo "[timeseries] run scripts/run_kepler_herbieoff.sh first"; exit 1; }
B_NAIVE="${B_NAIVE:-$(tr ',' '\n' < cache_kep_herbieoff/budgets.txt | sort -g | head -1)}"
if [ -z "${B_POSEIDON:-}" ]; then
  if [ "$ON_SHIPPED_MODEL" = 1 ]; then
    B_POSEIDON="$(expected_share genga-drift-algebraic)"
  else
    echo "[timeseries] this device's own cost model: the drift row's share is the RTX 5090 model's."
    echo "[timeseries] Pick the Poseidon point from kepler_full_g0.tsv and rerun with B_POSEIDON=<share>."
    exit 0
  fi
fi
N=65536
STEPS="100 300 1000 3000 10000 30000 100000 300000 500000"
mkdir -p logs_ts

mapfile -t FP_NAIVE < <(kepler_herbieoff_flags)
mapfile -t FP_POS < <(kepler_flags cache_kep_rolled)
echo "[build] naive FP32 (Herbie-off frontier head, share $B_NAIVE)"
bash ../../scripts/pt.sh solve_compile "kepler-timeseries b=$B_NAIVE" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP_NAIVE[@]}" \
  -poseidon-cost-budget="$B_NAIVE" "$SRC" -o kepler_ts_fp32.exe > logs_ts/build_naive.log 2>&1 \
  || { echo "naive build failed (logs_ts/build_naive.log)"; exit 1; }
echo "  pin: $(grep -a -m1 '^\[poseidon\] pin: ' logs_ts/build_naive.log | sed 's/^\[poseidon\] pin: //')"
echo "[build] Poseidon (share $B_POSEIDON)"
bash ../../scripts/pt.sh solve_compile "kepler-timeseries b=$B_POSEIDON" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP_POS[@]}" \
  -poseidon-cost-budget="$B_POSEIDON" "$SRC" -o kepler_ts_pos.exe > logs_ts/build_poseidon.log 2>&1 \
  || { echo "Poseidon build failed (logs_ts/build_poseidon.log)"; exit 1; }
expected_report genga-drift-algebraic logs_ts/build_poseidon.log

# Never discard stderr and never write an empty cell silently: keep stderr in
# logs_ts/, retry once on an empty parse, and warn loudly.
drift() {  # <exe> <mode:--gold|""> <steps>
  local out="" try
  for try in 1 2; do
    out=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 600 ./"$1" --integrate $2 --steps=$3 $N \
          2>>"logs_ts/err_${1%.exe}_s$3.log" \
          | grep -oE "Mean energy drift = \S+" | grep -oE "[0-9.eE+-]+$")
    [ -n "$out" ] && break
    echo "[timeseries] WARNING: empty drift ($1 $2 steps=$3, try $try) — stderr in logs_ts/" >&2
  done
  printf '%s' "$out"
}

TSV=kepler_timeseries.tsv; printf "steps\tfp64\tnaive_fp32\tposeidon\n" > "$TSV"
echo "[sweep] steps: $STEPS"
for S in $STEPS; do
  f64=$(drift kepler_ts_pos.exe  --gold $S)   # gold kernel = pure FP64
  nai=$(drift kepler_ts_fp32.exe ""      $S)
  pos=$(drift kepler_ts_pos.exe  ""      $S)
  printf "%s\t%s\t%s\t%s\n" "$S" "$f64" "$nai" "$pos" >> "$TSV"
  printf "  steps=%-7s fp64=%-12s naive=%-12s poseidon=%s\n" "$S" "$f64" "$nai" "$pos"
done
awk -F'\t' 'NR>1 {for (i=2;i<=NF;i++) if ($i=="") bad=1} END {exit bad}' "$TSV" \
  || echo "[timeseries] WARNING: $TSV contains empty cells — see logs_ts/ for the per-run stderr"
echo "[done] wrote $TSV"
