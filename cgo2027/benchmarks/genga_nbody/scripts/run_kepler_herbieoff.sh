#!/bin/bash
# HERBIE-OFF ablation on the KEPLER drift: the optimization pass run on
# genga_kepler.cu with algebraic rewrites disabled (-poseidon-enable-herbie=0 ->
# candidate set = FP32 + expansion + FP64), SAME rolled+split decomposition and
# the same profile as the full solve. Sweep the frontier, measure single-step
# time/L2 (N=1M, avg of NUM_RUNS) + energy drift (N=65536, 20k steps). Writes
# kepler_herbieoff_g0.tsv, overlaid as the dashed ablation curve on Fig. 3
# (right). Its cheapest point is the naive FP32 downcast.
# Requires run_kepler.sh to have produced its profile + tmp_kep/kepler-baseline.exe.
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
avg() { "$PY" -c "import sys;v=[float(x) for x in sys.argv[1:] if x];print(f'{sum(v)/len(v):.6f}' if v else 'nan')" "$@"; }
grab() { grep -oE "$1 = \S+" | grep -oE "[0-9.eE+-]+$"; }
bt=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_kep/kepler-baseline.exe 2>>measure_err.log | grab "Elapsed time"; done)
BT=$(avg $bt); echo "  F64 baseline single-step (N=1M) = $BT ms"
TSV=kepler_herbieoff_g0.tsv; printf "budget\tspeedup\tEdrift\tL2\tpin\n" > "$TSV"
for B in $BUDS; do
  exe=tmp_kep_ho/kho-$B.exe; [ -f "$exe" ] || continue
  ts=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" 2>>measure_err.log | grab "Elapsed time"; done)
  ms=$(avg $ts)
  l2=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" 2>>measure_err.log | grab "L2 relative error")
  dr=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 180 "$exe" --integrate --steps=20000 65536 2>>measure_err.log | grab "Mean energy drift")
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')" 2>/dev/null || echo nan)
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs_kep_ho/kho-$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\t%s\n" "$B" "$sp" "$dr" "$l2" "$pin" >> "$TSV"
  printf "  %-22s speedup=%-7s Edrift=%-12s L2=%s\n" "$B" "$sp" "$dr" "$l2"
done
echo "[done] wrote $TSV"
