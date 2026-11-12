#!/bin/bash
# ozp frontier rungs (the FP64-grade-AND-fast Ozaki-II points, and the direct
# and error-corrected library dispatches the same frontier carries). Solve the
# DP once (budget 0, caching the table + the frontier's shares in
# cache/budgets.txt), then at each frontier point build with the host-dispatch
# runtimes poseidon-clang++ links, record the solver's pick + whole-app
# RMSD-vs-Kref + time (the executable already averages 5 timed runs
# internally), and --trace the per-iteration convergence.
#   -> dispatch_rungs.tsv  (budget pick L2 RMSD Energy time_ms speedup traj)
#   -> traj_OZII_nm<nm>.tsv  (per-iteration convergence, nm-tagged for plot_ozp)
# Run via bash. Speedups are over the scalar-FP64 baseline re-measured this session.
set -e
source "$(dirname "$0")/common.sh"

HMTX=data/synth${APPLY_N}_h.mtx
KMTX=data/synth${APPLY_N}_k.mtx
[ -s "$HMTX" ] || { echo "missing $HMTX — run: bash scripts/gen_data.sh $APPLY_N"; exit 1; }

run_one () {  # <exe> -> "<L2>\t<RMSD>\t<Energy>\t<time_ms>"
  local out; out=$(CUDA_VISIBLE_DEVICES=$GPU_ID "$1" "$HMTX" "$KMTX")
  local l2 rm en t
  l2=$(echo "$out" | sed -n 's/.*L2 relative error = \([0-9.eEnaif+-]*\).*/\1/p')
  rm=$(echo "$out" | sed -n 's/.*RMSD = \([0-9.eEnaif+-]*\).*/\1/p')
  en=$(echo "$out" | sed -n 's/.*Energy abs error = \([0-9.eEnaif+-]*\).*/\1/p')
  t=$(echo "$out"  | sed -n 's/.*Elapsed time = \([0-9.]*\).*/\1/p')
  echo -e "${l2:-NA}\t${rm:-NA}\t${en:-NA}\t${t:-NA}"
}
trace_one () { CUDA_VISIBLE_DEVICES=$GPU_ID "$1" "$HMTX" "$KMTX" --trace \
  | awk '/^iter/{print $2"\t"$4}' > "traj_$2.tsv"; }

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
BASE=$(run_one "$TMPD/dr_base.exe"); BT=$(echo "$BASE" | cut -f4)
echo "  baseline: ${BT} ms (RMSD $(echo "$BASE" | cut -f2))"

echo "### solve once @budget0 -> cache table + frontier shares (profile ./$PROF) ###"
# budgets.txt is write-once, so it must be gone before this solve or the
# previous candidate set's frontier is replayed. Under REUSE_PROFILE the
# content-addressed accuracy memo (cache/matmulacc_*) is kept; table.json is
# keyed by every solve input and re-solves when one differs.
if [ "${REUSE_PROFILE:-0}" = 1 ]; then
  rm -f cache/budgets.txt cache/*.ozdispatch
else
  rm -rf cache
fi
bash ../../scripts/pt.sh solve_compile "dispatch b=0" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
  -poseidon-cost-budget=0 ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "$TMPD/dr_v0.exe" > "$TMPD/dr_v0.log" 2>&1
BPS=$(cat cache/budgets.txt)
echo "  frontier shares: $BPS"

printf 'budget\tpick\tL2_vs_Kref\tRMSD\tEnergy\ttime_ms\tspeedup\ttraj\n' > dispatch_rungs.tsv
IFS=',' read -ra ARR <<< "$BPS"
for B in "${ARR[@]}"; do
  [ "$B" = "0" ] && continue
  # A breakpoint that selects an IN-KERNEL candidate writes no descriptor, so a
  # leftover from the previous (dispatch) breakpoint would host-dispatch a
  # kernel this budget never picked. The DP table + accuracy memo are kept: the
  # whole point of the breakpoint sweep is to reuse one solve across budgets.
  rm -f cache/*.ozdispatch
  # The output path is keyed by budget, so a FAILED compile here would otherwise
  # leave run_one timing the binary a PREVIOUS run left at tmp/dr_$B.exe and
  # emitting a full row of plausible numbers for a program this budget never
  # built. Deleting it first makes its presence below proof that THIS compile
  # produced it.
  rm -f "$TMPD/dr_$B.exe"
  log=$(bash ../../scripts/pt.sh solve_compile "dispatch b=$B" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
        -poseidon-print -poseidon-cost-budget=$B \
        ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "$TMPD/dr_$B.exe" 2>&1 || true)
  echo "$log" > "$TMPD/dr_$B.log"
  if [ ! -s "$TMPD/dr_$B.exe" ]; then
    echo "COMPILE FAILED at budget $B -- recording the failure and moving on:" >&2
    echo "$log" | grep -i error | head -5 >&2
    printf '%s\tCOMPILE-FAILED\tNA\tNA\tNA\tNA\tNA\tNA\n' "$B" | tee -a dispatch_rungs.tsv
    continue
  fi
  # A frontier point whose accuracy the model refuses (an FP16 format below
  # the profiled magnitudes: domain error INF) is listed in budgets.txt but no
  # budget selects it; that compile applies nothing. Record it, measure nothing.
  if [ -z "$(pins_of_log "$TMPD/dr_$B.log")" ]; then
    echo "  share $B: no configuration is selected at this frontier point (its accuracy is refused); skipped"
    printf '%s\tNOT-SELECTABLE\tNA\tNA\tNA\tNA\tNA\tNA\n' "$B" >> dispatch_rungs.tsv
    continue
  fi
  pick=$(pick_of_log "$TMPD/dr_$B.log" | sed -E 's/^(ozaki-ii nm=[0-9]+|direct-dispatch cublas [a-z0-9/]+|[a-z0-9-]+ [a-z0-9/]+).*/\1/; s/ /_/g')
  [ -z "$pick" ] && pick="baseline"
  # nm-tag the trajectory (OZII_nm8..nm14) so plot_ozp finds it; non-Ozaki picks
  # fall back to the budget tag.
  if [[ "$pick" == *nm=* ]]; then nm=${pick##*nm=}; tag="OZII_nm${nm}"; else tag="OZII_${B}"; fi
  R=$(run_one "$TMPD/dr_$B.exe"); OT=$(echo "$R" | cut -f4)
  SP=$("$PY" -c "print(f'{$BT/$OT:.3f}')" 2>/dev/null || echo NA)
  trace_one "$TMPD/dr_$B.exe" "$tag"
  printf '%s\t%s\t%s\t%s\t%s\n' "$B" "$pick" "$R" "$SP" "$tag" | tee -a dispatch_rungs.tsv
done
echo "BASELINE_THIS_SESSION_MS=$BT" | tee -a dispatch_rungs.tsv
echo "=== done: dispatch_rungs.tsv + traj_OZII_*.tsv ==="
