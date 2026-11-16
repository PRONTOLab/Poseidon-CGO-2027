#!/bin/bash
# Energy drift over step counts for FP64, naive FP32 and Poseidon (Fig. 4) -> kepler_timeseries.tsv.
# B_NAIVE / B_POSEIDON override the two frontier shares; needs run_kepler_herbieoff.sh first.
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

drift() {  # <exe> <mode:--gold|""> <steps>: DRIFT = the mean energy drift
  genga_run "timeseries $1 ${2:-} steps=$3" DRIFT=acc:"Mean energy drift" \
    -- 600 ./"$1" --integrate $2 --steps="$3" $N
}

TSV=kepler_timeseries.tsv; printf "steps\tfp64\tnaive_fp32\tposeidon\n" > "$TSV"
echo "[sweep] steps: $STEPS"
for S in $STEPS; do
  drift kepler_ts_pos.exe  --gold $S; f64=$DRIFT
  drift kepler_ts_fp32.exe ""      $S; nai=$DRIFT
  drift kepler_ts_pos.exe  ""      $S; pos=$DRIFT
  printf "%s\t%s\t%s\t%s\n" "$S" "$f64" "$nai" "$pos" >> "$TSV"
  printf "  steps=%-7s fp64=%-12s naive=%-12s poseidon=%s\n" "$S" "$f64" "$nai" "$pos"
done
echo "[done] wrote $TSV"
