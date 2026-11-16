#!/bin/bash
# Force frontier with expansions disabled (Fig. 3 left, ablation) -> force_mfoff.tsv.
# Needs run_force.sh first (profile, tmp/genga_nbody-baseline.exe).
source "$(dirname "$0")/common.sh"
SRC=genga_nbody.cu
N=65536
mapfile -t FPOPT < <(force_flags -poseidon-enable-multifloat=0 -poseidon-cache=cache_mfoff)

echo "[1/3] expansion-off solve (budget 0) -> cache_mfoff/budgets.txt"
rm -rf cache_mfoff reports_mfoff tmp_mfoff; mkdir -p cache_mfoff tmp_mfoff
bash ../../scripts/pt.sh solve_compile "force-mfoff b=0" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPOPT[@]}" -poseidon-cost-budget=0 \
  -poseidon-report-path=reports_mfoff "$SRC" -o tmp_mfoff/mfoff-0.exe
echo "  frontier shares: $(cat cache_mfoff/budgets.txt)"

mean_time "force-mfoff F64 baseline" 120 tmp/genga_nbody-baseline.exe $N; BT=$MEAN_MS; echo "  F64 baseline = $BT ms"

echo "[2/3] compile + measure each frontier point (expansion off: FP32+F64)"
TSV=force_mfoff.tsv; printf "budget\tspeedup\tL2\tpin\n" > "$TSV"
mkdir -p logs_mfoff
for B in $(tr ',' ' ' < cache_mfoff/budgets.txt); do
  exe=tmp_mfoff/mfoff-$B.exe
  bash ../../scripts/pt.sh solve_compile "force-mfoff b=$B" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPOPT[@]}" -poseidon-cost-budget="$B" "$SRC" -o "$exe" \
    > "logs_mfoff/compile_$B.log" 2>&1
  need_exe "$exe" "$B" "logs_mfoff/compile_$B.log"
  mean_time "force-mfoff share $B" 120 "$exe" $N; ms=$MEAN_MS
  genga_run "force-mfoff share $B, accuracy" l2=acc:"L2 relative error" -- 120 "$exe" $N
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')")
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs_mfoff/compile_$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\n" "$B" "$sp" "$l2" "$pin" >> "$TSV"
  printf "  %-22s speedup=%-8s L2=%-14s %s\n" "$B" "$sp" "$l2" "$pin"
done
echo "[3/3] wrote $TSV"
