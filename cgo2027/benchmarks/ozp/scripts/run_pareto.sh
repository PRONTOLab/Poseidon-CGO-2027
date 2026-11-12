#!/bin/bash
# ozp convergence / Pareto data: the in-kernel candidate ladder (direct
# tensor-core raises + FP16TCEC) that stalls above the chemistry bar, plus the
# scalar-FP64 baseline. Profile once @PROF_N (SCVH), build baseline @APPLY_N, then
# for each candidate apply that one rewrite by name (-poseidon-apply-rewrites)
# and record its speed/accuracy point + per-iteration convergence trajectory.
#   -> pareto.tsv  (tag candidate L2 RMSD Energy time_ms speedup)
#   -> traj_<tag>.tsv, traj_FP64.tsv  (per-iteration RMSD trajectories)
# The Ozaki-II (>F32) rungs come from run_dispatch_rungs.sh. Run via bash.
set -e
source "$(dirname "$0")/common.sh"

HMTX=data/synth${APPLY_N}_h.mtx
KMTX=data/synth${APPLY_N}_k.mtx

# The candidates, by the compiler's own name for each (a name the pass does not
# propose stops the compile with the list of the names it does). The tags name
# the output files (traj_<tag>.tsv), as the figure script expects them.
#   PARETO_CANDS='<tag>=<name>;...' replaces the list.
PARETO_CANDS="${PARETO_CANDS:-M0_1=wmma m16n16k16 f16/f32;M0_2=wmma m16n16k16 bf16/f32;M0_3=wmma m16n16k8 tf32/f32;M0_4=tcec n=2 wmma m16n16k16 f16/f32;M0_5=tcec n=3 wmma m16n16k16 f16/f32;M0_6=tcec n=5 wmma m16n16k16 f16/f32}"

run_one () {  # <exe> -> "<L2>\t<RMSD>\t<Energy>\t<time_ms>"
  local out; out=$(CUDA_VISIBLE_DEVICES=$GPU_ID "$1" "$HMTX" "$KMTX")
  local l2 rm en t
  l2=$(echo "$out" | sed -n 's/.*L2 relative error = \([0-9.eEnaif+-]*\).*/\1/p')
  rm=$(echo "$out" | sed -n 's/.*RMSD = \([0-9.eEnaif+-]*\).*/\1/p')
  en=$(echo "$out" | sed -n 's/.*Energy abs error = \([0-9.eEnaif+-]*\).*/\1/p')
  t=$(echo "$out"  | sed -n 's/.*Elapsed time = \([0-9.]*\).*/\1/p')
  echo -e "${l2:-NA}\t${rm:-NA}\t${en:-NA}\t${t:-NA}"
}
trace_one () {
  CUDA_VISIBLE_DEVICES=$GPU_ID "$1" "$HMTX" "$KMTX" --trace \
    | awk '/^iter/{print $2"\t"$4}' > "traj_$2.tsv"
}

# The same two phases the cost-model bootstrap runs, from one place.
bash "$(dirname "$0")/profile_only.sh" pareto
bash "$(dirname "$0")/gen_data.sh" "$APPLY_N"

echo "### baseline scalar FP64 (no -DPOSEIDON) @N=$APPLY_N ###"
bash ../../scripts/pt.sh baseline_compile pareto -- "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPD/ozp_base.exe"
BASE=$(run_one "$TMPD/ozp_base.exe"); BT=$(echo "$BASE" | cut -f4)
trace_one "$TMPD/ozp_base.exe" FP64

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
    R=$(run_one "$TMPD/ozp_$C.exe"); OT=$(echo "$R" | cut -f4)
    SP=$("$PY" -c "print(f'{$BT/$OT:.3f}')" 2>/dev/null || echo NA)
    trace_one "$TMPD/ozp_$C.exe" "$C"
    printf '%s\t%s\t%s\t%s\n' "$C" "${SEL:-?}" "$R" "$SP" | tee -a pareto.tsv
  else
    echo "  BUILD FAILED for $C (see tmp/ozp_build_$C.log):"; tail -4 "$TMPD/ozp_build_$C.log"
    printf '%s\tBUILD_FAIL\tNA\tNA\tNA\n' "$C" >> pareto.tsv
  fi
done

echo "=== DONE — pareto.tsv ==="
column -t -s $'\t' pareto.tsv 2>/dev/null || cat pareto.tsv
