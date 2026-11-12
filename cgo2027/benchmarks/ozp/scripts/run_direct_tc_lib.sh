#!/bin/bash
# Library-dispatch reference for the direct tensor-core raises at N=APPLY_N:
# the same operand precision as the in-kernel WMMA raises in pareto.tsv (FP16,
# BF16, TF32 inputs; FP32 accumulation) but computed by one cuBLAS GEMM
# (ozp_poseidon.cu, OZP_CUBLAS_TC). These are reference points like cuBLAS
# DGEMM, not Poseidon candidates. Speedups are over the scalar-FP64 baseline
# re-measured in this session; every executable averages 5 timed runs.
#   -> <outdir>/direct_tc_lib.tsv  (tag candidate L2 RMSD Energy time_ms speedup traj)
#   -> <outdir>/traj_LIB_<f16|bf16|tf32>.tsv  (per-iteration convergence)
# Usage: bash scripts/run_direct_tc_lib.sh [outdir]   (default: bench dir)
set -uo pipefail
source "$(dirname "$0")/common.sh"
OUT="${1:-$BENCH_DIR}"; mkdir -p "$OUT"
HMTX=data/synth${APPLY_N}_h.mtx
KMTX=data/synth${APPLY_N}_k.mtx
[ -s "$HMTX" ] || { echo "missing $HMTX -- run: bash scripts/gen_data.sh $APPLY_N"; exit 1; }

run_one () {  # <exe> <log> -> "<L2>\t<RMSD>\t<Energy>\t<time_ms>"
  local out; out=$(CUDA_VISIBLE_DEVICES=$GPU_ID "$1" "$HMTX" "$KMTX")
  echo "$out" > "$2"
  local l2 rm en t
  l2=$(echo "$out" | sed -n 's/.*L2 relative error = \([0-9.eEnaif+-]*\).*/\1/p')
  rm=$(echo "$out" | sed -n 's/.*RMSD = \([0-9.eEnaif+-]*\).*/\1/p')
  en=$(echo "$out" | sed -n 's/.*Energy abs error = \([0-9.eEnaif+-]*\).*/\1/p')
  t=$(echo "$out"  | sed -n 's/.*Elapsed time = \([0-9.]*\).*/\1/p')
  echo -e "${l2:-NA}\t${rm:-NA}\t${en:-NA}\t${t:-NA}"
}
trace_one () { CUDA_VISIBLE_DEVICES=$GPU_ID "$1" "$HMTX" "$KMTX" --trace \
  | awk '/^iter/{print $2"\t"$4}' > "$OUT/traj_$2.tsv"; }

echo "### baseline scalar FP64 @N=$APPLY_N (this session) ###"
"$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPD"/dtl_base.exe \
  || { echo "FATAL: baseline compile failed"; exit 1; }
BASE=$(run_one "$TMPD"/dtl_base.exe "$OUT/run_direct_tc_lib_baseline.log"); BT=$(echo "$BASE" | cut -f4)
echo "  baseline: ${BT} ms (RMSD $(echo "$BASE" | cut -f2))"

TSV="$OUT/direct_tc_lib.tsv"
printf 'tag\tcandidate\tL2_vs_Kref\tRMSD\tEnergy\ttime_ms\tspeedup\ttraj\n' > "$TSV"
printf 'FP64\tnaive-f64-baseline\t%s\t1.000\tFP64\n' "$BASE" >> "$TSV"
for spec in "1:f16:cublas_gemmex_f16/f32" "2:bf16:cublas_gemmex_bf16/f32" "3:tf32:cublas_gemmex_tf32/f32"; do
  IFS=: read -r MODE TAG CAND <<< "$spec"
  rm -f "$TMPD"/dtl_$TAG.exe
  echo "### cuBLAS direct TC: $CAND ###"
  "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N -DOZP_CUBLAS_TC=$MODE ozp_poseidon.cu -o "$TMPD"/dtl_$TAG.exe \
    2>"$OUT/build_direct_tc_lib_$TAG.log" || { echo "  BUILD FAILED ($TAG)"; tail -5 "$OUT/build_direct_tc_lib_$TAG.log"; continue; }
  R=$(run_one "$TMPD"/dtl_$TAG.exe "$OUT/run_direct_tc_lib_$TAG.log"); OT=$(echo "$R" | cut -f4)
  SP=$("$PY" -c "print(f'{$BT/$OT:.3f}')" 2>/dev/null || echo NA)
  trace_one "$TMPD"/dtl_$TAG.exe "LIB_$TAG"
  printf 'LIB_%s\t%s\t%s\t%s\tLIB_%s\n' "$TAG" "$CAND" "$R" "$SP" "$TAG" | tee -a "$TSV"
done
echo "BASELINE_THIS_SESSION_MS=$BT" | tee -a "$TSV"
echo "=== done: $TSV ==="
column -t -s $'\t' "$TSV"
