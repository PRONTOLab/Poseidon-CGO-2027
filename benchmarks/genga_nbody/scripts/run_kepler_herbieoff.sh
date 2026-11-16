#!/bin/bash
# Drift frontier with algebraic rewrites disabled (Fig. 3 right, ablation) -> kepler_herbieoff_g0.tsv.
# Needs run_kepler.sh first (profile, tmp_kep/kepler-baseline.exe).
source "$(dirname "$0")/common.sh"
SRC=genga_kepler.cu
mapfile -t FP < <(kepler_herbieoff_flags)

echo "[1/3] Herbie-off solve (budget 0) -> cache_kep_herbieoff/budgets.txt"
rm -rf cache_kep_herbieoff reports_kep_ho tmp_kep_ho logs_kep_ho; mkdir -p cache_kep_herbieoff tmp_kep_ho logs_kep_ho
bash ../../scripts/pt.sh solve_compile "kepler-herbieoff b=0" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP[@]}" -poseidon-cost-budget=0 \
  -poseidon-report-path=reports_kep_ho "$SRC" -o tmp_kep_ho/kho-0.exe
echo "  $(tr ',' '\n' < cache_kep_herbieoff/budgets.txt | wc -l) frontier points"

echo "[2/3] compile one binary per frontier point (parallel)"
BUDS=$(tr ',' ' ' < cache_kep_herbieoff/budgets.txt)
pids=0
for B in $BUDS; do
  ( bash ../../scripts/pt.sh solve_compile "kepler-herbieoff b=$B" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP[@]}" -poseidon-cost-budget="$B" \
      "$SRC" -o "tmp_kep_ho/kho-$B.exe" > "logs_kep_ho/kho-$B.log" 2>&1 ) &
  pids=$((pids+1)); [ $((pids % 12)) -eq 0 ] && wait
done
wait
echo "  compiled $(ls tmp_kep_ho/kho-*.exe 2>/dev/null | wc -l) binaries"

echo "[3/3] measure -> kepler_herbieoff_g0.tsv"
mean_time "kepler-herbieoff F64 baseline" 120 tmp_kep/kepler-baseline.exe; BT=$MEAN_MS; echo "  F64 baseline single-step (N=1M) = $BT ms"
TSV=kepler_herbieoff_g0.tsv; printf "budget\tspeedup\tEdrift\tL2\tpin\n" > "$TSV"
for B in $BUDS; do
  exe=tmp_kep_ho/kho-$B.exe; need_exe "$exe" "$B" "logs_kep_ho/kho-$B.log"
  mean_time "kepler-herbieoff share $B" 120 "$exe"; ms=$MEAN_MS
  genga_run "kepler-herbieoff share $B, accuracy" l2=acc:"L2 relative error" -- 120 "$exe"
  genga_run "kepler-herbieoff share $B, energy drift" dr=acc:"Mean energy drift" -- 180 "$exe" --integrate --steps=20000 65536
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')")
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs_kep_ho/kho-$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\t%s\n" "$B" "$sp" "$dr" "$l2" "$pin" >> "$TSV"
  printf "  %-22s speedup=%-7s Edrift=%-12s L2=%s\n" "$B" "$sp" "$dr" "$l2"
done
echo "[done] wrote $TSV"
