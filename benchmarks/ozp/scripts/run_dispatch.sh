#!/bin/bash
# Frontier of host-dispatched products (Ozaki-II, TCEC, direct library GEMMs), one
# build per frontier share -> dispatch.tsv, traj_OZII_*.tsv.
set -e
source "$(dirname "$0")/common.sh"

HMTX=data/synth${APPLY_N}_h.mtx
KMTX=data/synth${APPLY_N}_k.mtx
[ -s "$HMTX" ] || { echo "missing $HMTX; run: bash scripts/gen_data.sh $APPLY_N"; exit 1; }

if [ "${REUSE_PROFILE:-0}" = 1 ] && compgen -G 'fpprofile/*.fpprofile' >/dev/null; then
  echo "### REUSE_PROFILE=1: reusing ./fpprofile (N=$PROF_N) ###"
else
  echo "### profile-gen @N=$PROF_N ###"
  bash ../../scripts/pt.sh profile_compile dispatch -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" -DOZP_N=$PROF_N -poseidon-profile-generate \
    ozp_poseidon.cu -o "$TMPD/dr_prof.exe"
  rm -rf fpprofile
  bash ../../scripts/pt.sh profile_run dispatch -- env CUDA_VISIBLE_DEVICES=$GPU_ID POSEIDON_PROFILE_DIR=./fpprofile \
    "$TMPD/dr_prof.exe" data/synth${PROF_N}_h.mtx data/synth${PROF_N}_k.mtx --profiling
fi

echo "### baseline scalar FP64 @N=$APPLY_N (this session) ###"
bash ../../scripts/pt.sh baseline_compile dispatch -- "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPD/dr_base.exe"
run_one "dispatch: scalar FP64 baseline" "$TMPD/dr_base.exe"; BASE="$RUN_ROW"; BT=$(echo "$BASE" | cut -f4)
echo "  baseline: ${BT} ms (RMSD $(echo "$BASE" | cut -f2))"

echo "### solve once @budget0 -> cache table + frontier shares (profile ./$PROF) ###"
# budgets.txt is write-once and must be gone before this solve.
if [ "${REUSE_PROFILE:-0}" = 1 ]; then
  rm -f cache/budgets.txt cache/*.ozdispatch
else
  rm -rf cache
fi
bash ../../scripts/pt.sh solve_compile "dispatch b=0" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
  -poseidon-cost-budget=0 ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "$TMPD/dr_v0.exe" > "$TMPD/dr_v0.log" 2>&1
BPS=$(cat cache/budgets.txt)
echo "  frontier shares: $BPS"

printf 'budget\tpick\tL2_vs_Kref\tRMSD\tEnergy\ttime_ms\tspeedup\ttraj\n' > dispatch.tsv
IFS=',' read -ra ARR <<< "$BPS"
for B in "${ARR[@]}"; do
  [ "$B" = "0" ] && continue
  # An in-kernel pick writes no descriptor; a leftover one would host-dispatch the previous pick.
  rm -f cache/*.ozdispatch
  # Removed first so a failed compile cannot leave an earlier run's binary to be timed.
  rm -f "$TMPD/dr_$B.exe"
  log=$(bash ../../scripts/pt.sh solve_compile "dispatch b=$B" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
        -poseidon-print -poseidon-cost-budget=$B \
        ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "$TMPD/dr_$B.exe" 2>&1 || true)
  echo "$log" > "$TMPD/dr_$B.log"
  if [ ! -s "$TMPD/dr_$B.exe" ]; then
    echo "COMPILE FAILED at budget $B -- recording the failure and moving on:" >&2
    echo "$log" | grep -i error | head -5 >&2
    printf '%s\tCOMPILE-FAILED\tNA\tNA\tNA\tNA\tNA\tNA\n' "$B" | tee -a dispatch.tsv
    continue
  fi
  # A frontier point whose accuracy is refused (domain error INF) applies nothing.
  if [ -z "$(pins_of_log "$TMPD/dr_$B.log")" ]; then
    echo "  share $B: no configuration is selected at this frontier point (its accuracy is refused); skipped"
    printf '%s\tNOT-SELECTABLE\tNA\tNA\tNA\tNA\tNA\tNA\n' "$B" >> dispatch.tsv
    continue
  fi
  pick=$(pick_of_log "$TMPD/dr_$B.log" | sed -E 's/^(ozaki-ii nm=[0-9]+|direct-dispatch cublas [a-z0-9/]+|[a-z0-9-]+ [a-z0-9/]+).*/\1/; s/ /_/g')
  [ -z "$pick" ] && pick="baseline"
  # plot_ozp.py finds Ozaki-II traces by nm.
  if [[ "$pick" == *nm=* ]]; then nm=${pick##*nm=}; tag="OZII_nm${nm}"; else tag="OZII_${B}"; fi
  run_one "dispatch share $B ($pick)" "$TMPD/dr_$B.exe"; R="$RUN_ROW"; OT=$(echo "$R" | cut -f4)
  SP=$("$PY" -c "print(f'{$BT/$OT:.3f}')")
  trace_one "dispatch share $B ($pick), trace" "$TMPD/dr_$B.exe" "traj_$tag.tsv"
  printf '%s\t%s\t%s\t%s\t%s\n' "$B" "$pick" "$R" "$SP" "$tag" | tee -a dispatch.tsv
done
echo "BASELINE_THIS_SESSION_MS=$BT" | tee -a dispatch.tsv
echo "=== done: dispatch.tsv + traj_OZII_*.tsv ==="
end_if_build_failed dispatch.tsv 2
