#!/bin/bash
# Expansion-OFF ablation on the FORCE kick: the optimization pass run on
# genga_nbody.cu with floating-point expansion disabled
# (-poseidon-enable-multifloat=0 -> candidate set = FP32 + FP64 only), same
# profile / cost model as the full force solve. Sweep the frontier, measure L2 +
# wall clock (avg of NUM_RUNS) at N=65536.
# Writes force_mfoff.tsv, overlaid as the dashed ablation curve on Fig. 3 (left).
# Requires run_force.sh to have produced its profile and tmp/genga_nbody-baseline.exe.
source "$(dirname "$0")/common.sh"
SRC=genga_nbody.cu
N=65536
mapfile -t FPOPT < <(force_flags -poseidon-enable-multifloat=0 -poseidon-cache=cache_mfoff)

echo "[1/3] expansion-off solve (budget 0) -> cache_mfoff/budgets.txt"
rm -rf cache_mfoff reports_mfoff tmp_mfoff; mkdir -p cache_mfoff tmp_mfoff
bash ../../scripts/pt.sh solve_compile "force-mfoff b=0" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPOPT[@]}" -poseidon-cost-budget=0 \
  -poseidon-report-path=reports_mfoff "$SRC" -o tmp_mfoff/mfoff-0.exe
echo "  frontier shares: $(cat cache_mfoff/budgets.txt)"

avg() { "$PY" -c "import sys;v=[float(x) for x in sys.argv[1:] if x];print(f'{sum(v)/len(v):.6f}' if v else 'nan')" "$@"; }
bt=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" tmp/genga_nbody-baseline.exe $N 2>>measure_err.log | grep -oE "Elapsed time = \S+" | grep -oE "[0-9.e+-]+$"; done)
BT=$(avg $bt); echo "  F64 baseline = $BT ms"

echo "[2/3] compile + measure each frontier point (expansion off: FP32+F64)"
TSV=force_mfoff.tsv; printf "budget\tspeedup\tL2\tpin\n" > "$TSV"
mkdir -p logs_mfoff
for B in $(tr ',' ' ' < cache_mfoff/budgets.txt); do
  exe=tmp_mfoff/mfoff-$B.exe
  bash ../../scripts/pt.sh solve_compile "force-mfoff b=$B" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPOPT[@]}" -poseidon-cost-budget="$B" "$SRC" -o "$exe" \
    > "logs_mfoff/compile_$B.log" 2>&1 || { echo "  $B: compile FAIL (logs_mfoff/compile_$B.log)"; continue; }
  ts=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" $N 2>>measure_err.log | grep -oE "Elapsed time = \S+" | grep -oE "[0-9.e+-]+$"; done)
  ms=$(avg $ts)
  l2=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" $N 2>>measure_err.log | grep -oE "L2 relative error = \S+" | grep -oE "[0-9.e+-]+$")
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')" 2>/dev/null || echo nan)
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs_mfoff/compile_$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\n" "$B" "$sp" "$l2" "$pin" >> "$TSV"
  printf "  %-22s speedup=%-8s L2=%-14s %s\n" "$B" "$sp" "$l2" "$pin"
done
echo "[3/3] wrote $TSV"
