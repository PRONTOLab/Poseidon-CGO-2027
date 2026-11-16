#!/bin/bash
# run_surrogate.sh [N ...]: the Ozaki-II runtime at fixed nm across deploy sizes -> surrogate.tsv.
# SURROGATE_SIZES, METHODS (default cublas nm8 nm9 nm10), SURROGATE_TSV, REPS override the sweep.
set -e
source "$(dirname "$0")/common.sh"
REPS="${REPS:-$NUM_RUNS}"
TSV="${SURROGATE_TSV:-surrogate.tsv}"
NLIST="${*:-${SURROGATE_SIZES:-1024 2048 4096 8192}}"
MLIST="${METHODS:-cublas nm8 nm9 nm10}"

run_avg() {  # <launch> <exe args...>: R = first RMSD, T = mean time_ms
  local launch="$1" sum=0 t r i; shift; R=""; T=""
  for i in $(seq $REPS); do
    measure "$launch, run $i of $REPS" t=time:time_ms r=acc:RMSD -- env CUDA_VISIBLE_DEVICES="$GPU_ID" "$@"
    [ -z "$R" ] && R=$r
    sum=$("$PY" -c "print($sum + $t)")
  done
  T=$("$PY" -c "print(f'{$sum / $REPS:.3f}')")
}

printf "N\tmethod\trmsd\ttime_ms\tspeedup_naive\n" > $TSV
echo "baseline = NAIVE scalar-FP64 GEMM (1x); cuBLAS DGEMM recorded as reference."
echo "profile is fixed at N=512 -> tau=1e-7 selects nm8 (row ozp-tau1e-7); deploy across N:"
printf "%-6s %-8s %-12s %-10s %-18s\n" N method RMSD time_ms "speedup-vs-naive / bar"
for NV in $NLIST; do
  [ -s "data/synth${NV}_h.mtx" ] || { echo "missing data/synth${NV}_h.mtx; run: bash scripts/gen_data.sh $NV"; exit 1; }
  bash ../../scripts/pt.sh baseline_compile "surrogate N=$NV" -- "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$NV "$OZAKI_RT" surrogate_xN.cu -o "$TMPD"/sx_$NV.exe 2>/dev/null
  H=data/synth${NV}_h.mtx; K=data/synth${NV}_k.mtx
  run_avg "surrogate N=$NV fp64" "$TMPD"/sx_$NV.exe $H $K fp64
  tn=$T; rn=$R
  printf "%s\t%s\t%s\t%s\t%s\n" $NV fp64 "$rn" "$tn" "1.000" >> $TSV
  printf "%-6s %-8s %-12s %-10s %-18s\n" $NV fp64 "$rn" "$tn" "(BASELINE 1x)"
  for M in $MLIST; do
    run_avg "surrogate N=$NV $M" "$TMPD"/sx_$NV.exe $H $K $M
    r=$R; t=$T
    sp=$("$PY" -c "print(f'{$tn/$t:.3f}')")
    printf "%s\t%s\t%s\t%s\t%s\n" $NV $M "$r" "$t" "$sp" >> $TSV
    tag=$("$PY" -c "print(f'{$tn/$t:.2f}x  ' + ('CLEARS' if $r<=1e-7 else 'MISS'))")
    printf "%-6s %-8s %-12s %-10s %-18s\n" $NV $M "$r" "$t" "$tag"
  done
done
echo "wrote $TSV"
[ "$TSV" != surrogate.tsv ] || bash "$ART/scripts/text_results.sh" deployed-sizes
