#!/bin/bash
# Surrogate-generality sweep (SCVH, eval §5.2). The per-matmul precision Poseidon
# selects from the CHEAP N=512 profile (chemically-sufficient = Ozaki nm9) must
# clear the chemistry bar (RMSD<=1e-7) at EVERY deploy scale; one rung cheaper
# (nm8) misses everywhere. This is a pure-RUNTIME sweep: it links the in-tree
# Ozaki-II runtime ($OZAKI_RT) and calls it directly at the fixed nm, so no
# compiler solve is needed here (the choice was made once by the N=512 profile).
# Runs fp64(naive) / cublas / nm8 / nm9 / nm10 across N and writes surrogate.tsv
# with speedups over the NAIVE scalar-FP64 GEMM (the baseline shared with the
# money plot). nm10 is carried as the next rung up, so the table brackets the
# pick on both sides. Wall-clock = AVERAGE over REPS runs; RMSD is reproducible
# to ~0.1% (the INT8 residue GEMM is exact, but cuBLAS may reassociate).
#   Usage: GPU_ID=1 bash scripts/run_surrogate.sh [N ...]   (default 1024 2048 4096 8192)
#          METHODS="cublas nm9" bash scripts/run_surrogate.sh    (override the ladder)
#          SURROGATE_SIZES="1024 2048 4096" bash scripts/run_surrogate.sh
# SURROGATE_SIZES is the environment form of the positional list, for the same
# reason SURROGATE_TSV and METHODS exist in the environment: run_all.sh calls
# this script with no arguments, and N=8192 is about two thirds of the sweep's
# wall clock, so a site with a node-hour ceiling needs a way to name the sizes
# without editing the pipeline.
set -e
source "$(dirname "$0")/common.sh"
REPS="${REPS:-$NUM_RUNS}"
# SURROGATE_TSV lets a re-measurement of one N block write to its own file:
# the sweep truncates its output, so pointing a partial re-run at surrogate.tsv
# would drop every other N.
TSV="${SURROGATE_TSV:-surrogate.tsv}"
NLIST="${*:-${SURROGATE_SIZES:-1024 2048 4096 8192}}"
MLIST="${METHODS:-cublas nm8 nm9 nm10}"

run_avg() {  # <exe args...> -> sets R (rmsd, first run) and T (mean time_ms over REPS)
  local sum=0 t; R=""; T=""
  for i in $(seq $REPS); do
    local out; out=$(CUDA_VISIBLE_DEVICES=$GPU_ID "$@")
    t=$(echo "$out" | sed -n 's/.*time_ms=\([0-9.]*\).*/\1/p')
    [ -z "$R" ] && R=$(echo "$out" | sed -n 's/.*RMSD=\([0-9.eE+-]*\).*/\1/p')
    sum=$("$PY" -c "print($sum + $t)")
  done
  T=$("$PY" -c "print(f'{$sum / $REPS:.3f}')")
}

printf "N\tmethod\trmsd\ttime_ms\tspeedup_naive\n" > $TSV
echo "baseline = NAIVE scalar-FP64 GEMM (1x); cuBLAS DGEMM recorded as reference."
echo "profile is fixed at N=512 -> selects nm9 (chemically sufficient); deploy across N:"
printf "%-6s %-8s %-12s %-10s %-18s\n" N method RMSD time_ms "speedup-vs-naive / bar"
for NV in $NLIST; do
  [ -s "data/synth${NV}_h.mtx" ] || { echo "missing data/synth${NV}_h.mtx — run: bash scripts/gen_data.sh $NV"; exit 1; }
  bash ../../scripts/pt.sh baseline_compile "surrogate N=$NV" -- "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$NV "$OZAKI_RT" surrogate_xN.cu -o "$TMPD"/sx_$NV.exe 2>/dev/null
  H=data/synth${NV}_h.mtx; K=data/synth${NV}_k.mtx
  run_avg "$TMPD"/sx_$NV.exe $H $K fp64
  tn=$T; rn=$R
  printf "%s\t%s\t%s\t%s\t%s\n" $NV fp64 "$rn" "$tn" "1.000" >> $TSV
  printf "%-6s %-8s %-12s %-10s %-18s\n" $NV fp64 "$rn" "$tn" "(BASELINE 1x)"
  for M in $MLIST; do
    run_avg "$TMPD"/sx_$NV.exe $H $K $M
    r=$R; t=$T
    sp=$("$PY" -c "print(f'{$tn/$t:.3f}')")
    printf "%s\t%s\t%s\t%s\t%s\n" $NV $M "$r" "$t" "$sp" >> $TSV
    tag=$("$PY" -c "print(f'{$tn/$t:.2f}x  ' + ('CLEARS' if $r<=1e-7 else 'MISS'))")
    printf "%-6s %-8s %-12s %-10s %-18s\n" $NV $M "$r" "$t" "$tag"
  done
done
echo "wrote $TSV"
