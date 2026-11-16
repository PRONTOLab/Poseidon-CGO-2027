#!/bin/bash
# In-kernel tensor-core raises and FP16TCEC, each applied by name -> pareto.tsv, traj_*.tsv.
# PARETO_CANDS='<tag>=<name>;...' replaces the candidate list.
set -e
source "$(dirname "$0")/common.sh"

HMTX=data/synth${APPLY_N}_h.mtx
KMTX=data/synth${APPLY_N}_k.mtx

PARETO_CANDS="${PARETO_CANDS:-M0_1=wmma m16n16k16 f16/f32;M0_2=wmma m16n16k16 bf16/f32;M0_3=wmma m16n16k8 tf32/f32;M0_4=tcec n=2 wmma m16n16k16 f16/f32;M0_5=tcec n=3 wmma m16n16k16 f16/f32;M0_6=tcec n=5 wmma m16n16k16 f16/f32}"

bash "$(dirname "$0")/profile_only.sh" pareto
bash "$(dirname "$0")/gen_data.sh" "$APPLY_N"

echo "### baseline scalar FP64 (no -DPOSEIDON) @N=$APPLY_N ###"
bash ../../scripts/pt.sh baseline_compile pareto -- "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPD/ozp_base.exe"
run_one "pareto: scalar FP64 baseline" "$TMPD/ozp_base.exe"; BASE="$RUN_ROW"; BT=$(echo "$BASE" | cut -f4)
trace_one "pareto: scalar FP64 baseline, trace" "$TMPD/ozp_base.exe" traj_FP64.tsv

printf 'tag\tcandidate\tL2_vs_Kref\tRMSD\tEnergy\ttime_ms\tspeedup\n' > pareto.tsv
printf 'FP64\tnaive-f64-baseline\t%s\t1.000\n' "$BASE" >> pareto.tsv

IFS=';' read -ra CANDS <<< "$PARETO_CANDS"
for E in "${CANDS[@]}"; do
  C="${E%%=*}"; NAME="${E#*=}"
  echo "### build+run $C ($NAME) @N=$APPLY_N ###"
  rm -rf cache
  if bash ../../scripts/pt.sh solve_compile "pareto $C" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" -DOZP_N=$APPLY_N \
       "-poseidon-apply-rewrites=M0=$NAME" ozp_poseidon.cu -o "$TMPD/ozp_$C.exe" \
       > "$TMPD/ozp_build_$C.log" 2>&1; then
    SEL=$(pick_of_log "$TMPD/ozp_build_$C.log" | tr ' ' '_')
    run_one "pareto $C ($NAME)" "$TMPD/ozp_$C.exe"; R="$RUN_ROW"; OT=$(echo "$R" | cut -f4)
    SP=$("$PY" -c "print(f'{$BT/$OT:.3f}')")
    trace_one "pareto $C ($NAME), trace" "$TMPD/ozp_$C.exe" "traj_$C.tsv"
    printf '%s\t%s\t%s\t%s\n' "$C" "${SEL:-?}" "$R" "$SP" | tee -a pareto.tsv
  else
    echo "  BUILD FAILED for $C (see tmp/ozp_build_$C.log):"; tail -4 "$TMPD/ozp_build_$C.log"
    printf '%s\tBUILD_FAIL\tNA\tNA\tNA\n' "$C" >> pareto.tsv
  fi
done

echo "=== DONE: pareto.tsv ==="
column -t -s $'\t' pareto.tsv 2>/dev/null || cat pareto.tsv
end_if_build_failed pareto.tsv 2
