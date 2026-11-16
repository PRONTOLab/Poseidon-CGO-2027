#!/bin/bash
# run_direct_tc_lib.sh [outdir]: cuBLAS FP16/BF16/TF32 GEMMs as reference points
# -> direct_tc_lib.tsv, traj_LIB_*.tsv.
set -uo pipefail
source "$(dirname "$0")/common.sh"
OUT="${1:-$BENCH_DIR}"; mkdir -p "$OUT"
HMTX=data/synth${APPLY_N}_h.mtx
KMTX=data/synth${APPLY_N}_k.mtx
[ -s "$HMTX" ] || { echo "missing $HMTX -- run: bash scripts/gen_data.sh $APPLY_N"; exit 1; }

echo "### baseline scalar FP64 @N=$APPLY_N (this session) ###"
"$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPD"/dtl_base.exe \
  || { echo "FATAL: baseline compile failed"; exit 1; }
run_one "direct-tc-lib: scalar FP64 baseline" "$TMPD"/dtl_base.exe; BASE="$RUN_ROW"; BT=$(echo "$BASE" | cut -f4)
printf '%s\n' "$MEASURE_OUT" > "$OUT/run_direct_tc_lib_baseline.log"
echo "  baseline: ${BT} ms (RMSD $(echo "$BASE" | cut -f2))"

TSV="$OUT/direct_tc_lib.tsv"
printf 'tag\tcandidate\tL2_vs_Kref\tRMSD\tEnergy\ttime_ms\tspeedup\ttraj\n' > "$TSV"
printf 'FP64\tnaive-f64-baseline\t%s\t1.000\tFP64\n' "$BASE" >> "$TSV"
for spec in "1:f16:cublas_gemmex_f16/f32" "2:bf16:cublas_gemmex_bf16/f32" "3:tf32:cublas_gemmex_tf32/f32"; do
  IFS=: read -r MODE TAG CAND <<< "$spec"
  rm -f "$TMPD"/dtl_$TAG.exe
  echo "### cuBLAS direct TC: $CAND ###"
  "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N -DOZP_CUBLAS_TC=$MODE ozp_poseidon.cu -o "$TMPD"/dtl_$TAG.exe \
    2>"$OUT/build_direct_tc_lib_$TAG.log" || { echo "  BUILD FAILED ($TAG)"; tail -5 "$OUT/build_direct_tc_lib_$TAG.log"
      printf 'LIB_%s\tBUILD_FAIL\tNA\tNA\tNA\tNA\tNA\tLIB_%s\n' "$TAG" "$TAG" >> "$TSV"; continue; }
  run_one "direct-tc-lib $CAND" "$TMPD"/dtl_$TAG.exe; R="$RUN_ROW"; OT=$(echo "$R" | cut -f4)
  printf '%s\n' "$MEASURE_OUT" > "$OUT/run_direct_tc_lib_$TAG.log"
  SP=$("$PY" -c "print(f'{$BT/$OT:.3f}')")
  trace_one "direct-tc-lib $CAND, trace" "$TMPD"/dtl_$TAG.exe "$OUT/traj_LIB_$TAG.tsv"
  printf 'LIB_%s\t%s\t%s\t%s\tLIB_%s\n' "$TAG" "$CAND" "$R" "$SP" "$TAG" | tee -a "$TSV"
done
echo "BASELINE_THIS_SESSION_MS=$BT" | tee -a "$TSV"
echo "=== done: $TSV ==="
column -t -s $'\t' "$TSV" 2>/dev/null || cat "$TSV"
end_if_build_failed "$TSV" 2
