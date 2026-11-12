#!/bin/bash
# ozp call-site accuracy-target auto-pick (eval §5.2). Profile once @PROF_N
# (SCVH), then for each accuracy target tau build @APPLY_N with
# `-poseidon-tau=<tau>`: the SOLVER picks, per matmul, the cheapest tensor-core
# rewrite whose estimated domain error clears tau, at COMPILE TIME, and
# host-dispatches it. Reports the deployed density-matrix RMSD and the speedup
# over scalar FP64 and over cuBLAS DGEMM (the library FP64 baseline), and
# writes autopick.tsv, which the figure script circles. The picks the shipped
# RTX 5090 model makes are rows ozp-tau1e-6/1e-7/1e-12 of EXPECTED_RESULTS.md.
#   Usage: GPU_ID=1 bash scripts/run_autopick.sh [tau ...]   (default 1e-6 1e-7 1e-12)
#
# REUSE_PROFILE=1 (opt-in; default 0 = the clean-room flow above) keeps an
# existing ./fpprofile and ./cache instead of rebuilding them. What must NOT
# survive a tau is cache/*.ozdispatch: it is a one-shot descriptor handed from
# the device compile to the host compile, and a solve that picks an in-kernel
# candidate writes none, so a leftover from the previous tau would silently
# host-dispatch the wrong scheme. Under REUSE_PROFILE that single file is
# removed per tau in place of the whole tree.
set -e
source "$(dirname "$0")/common.sh"

bash "$(dirname "$0")/gen_data.sh" "$PROF_N" "$APPLY_N"
HA=data/synth${APPLY_N}_h.mtx; KA=data/synth${APPLY_N}_k.mtx

if [ "${REUSE_PROFILE:-0}" = 1 ] && compgen -G 'fpprofile/*.fpprofile' >/dev/null; then
  echo "### REUSE_PROFILE=1: reusing ./fpprofile (N=$PROF_N) and ./cache ###"
  rm -f cache/*.ozdispatch
else
  echo "### profile-gen @ surrogate N=$PROF_N (the choice transfers to N=$APPLY_N) ###"
  bash ../../scripts/pt.sh profile_compile autopick -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" -DOZP_N=$PROF_N -poseidon-profile-generate \
    ozp_poseidon.cu -o "$TMPD/reb_prof.exe" > "$TMPD/reb_prof.log" 2>&1
  rm -rf fpprofile cache
  bash ../../scripts/pt.sh profile_run autopick -- env CUDA_VISIBLE_DEVICES=$GPU_ID POSEIDON_PROFILE_DIR=./fpprofile \
    "$TMPD/reb_prof.exe" data/synth${PROF_N}_h.mtx data/synth${PROF_N}_k.mtx --profiling >/dev/null
fi
echo "### solves read the profile in ./$PROF (PROFILES=$PROFILES) ###"

echo "### baselines @N=$APPLY_N: cuBLAS DGEMM and scalar FP64 ###"
bash ../../scripts/pt.sh baseline_compile autopick -- "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N "$OZAKI_RT" surrogate_xN.cu -o "$TMPD/reb_cub.exe" > "$TMPD/reb_cub.log" 2>&1
CT=$(CUDA_VISIBLE_DEVICES=$GPU_ID "$TMPD/reb_cub.exe" $HA $KA cublas | sed -n 's/.*time_ms=\([0-9.]*\).*/\1/p')
bash ../../scripts/pt.sh baseline_compile autopick -- "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPD/reb_base.exe" > "$TMPD/reb_base.log" 2>&1
BT=$(CUDA_VISIBLE_DEVICES=$GPU_ID "$TMPD/reb_base.exe" $HA $KA | sed -n 's/.*Elapsed time = \([0-9.]*\).*/\1/p')
echo "  cuBLAS DGEMM: ${CT} ms   scalar FP64: ${BT} ms"

printf 'tau\tpick\trung_label\tmodelled_domain_err\tdeployed_RMSD\tL2_vs_Kref\ttime_ms\tx_scalar_fp64\tx_cublas_dgemm\n' > autopick.tsv
printf "  %-8s %-40s %-13s %-10s %-9s %s\n" tau "compile-time pick" RMSD time "x FP64" "x cuBLAS"
TOLS=("$@")
[ ${#TOLS[@]} -eq 0 ] && TOLS=(1e-6 1e-7 1e-12)
for TOL in "${TOLS[@]}"; do
  if [ "${REUSE_PROFILE:-0}" = 1 ]; then rm -f cache/*.ozdispatch; else rm -rf cache; fi
  rm -f "$TMPD/reb_$TOL.exe"
  bash ../../scripts/pt.sh solve_compile "autopick tau=$TOL" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
    -poseidon-tau=$TOL -poseidon-print \
    ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "$TMPD/reb_$TOL.exe" > "$TMPD/reb_$TOL.log" 2>&1 \
    || { echo "  $TOL BUILD FAILED (tmp/reb_$TOL.log)"; continue; }
  case "$TOL" in 1e-6|1e-7|1e-12) expected_report "ozp-tau$TOL" "$TMPD/reb_$TOL.log" ;; esac
  pick=$(pick_of_log "$TMPD/reb_$TOL.log"); [ -z "$pick" ] && pick="baseline F64"
  dom=$(sed -n 's/.*error-budget: matmul -> .*(domain err \([0-9.eE+-]*\).*/\1/p' "$TMPD/reb_$TOL.log" | head -1)
  rung=$(echo "$pick" | grep -oE 'nm=[0-9]+' || echo "$pick" | cut -d' ' -f1)
  res=$(CUDA_VISIBLE_DEVICES=$GPU_ID "$TMPD/reb_$TOL.exe" $HA $KA)
  r=$(echo "$res" | sed -n 's/.*RMSD = \([0-9.eE+-]*\).*/\1/p')
  l2=$(echo "$res" | sed -n 's/.*L2 relative error = \([0-9.eE+-]*\).*/\1/p')
  t=$(echo "$res" | sed -n 's/.*Elapsed time = \([0-9.]*\).*/\1/p')
  sp=$("$PY" -c "print(f'{$BT/$t:.3f}')" 2>/dev/null || echo NA)
  sc=$("$PY" -c "print(f'{$CT/$t:.3f}')" 2>/dev/null || echo NA)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$TOL" "$pick" "$rung" "${dom:-NA}" "${r:-NA}" "${l2:-NA}" "${t:-NA}" "$sp" "$sc" >> autopick.tsv
  printf "  %-8s %-40s %-13s %-10s %-9s %s\n" "$TOL" "$pick" "$r" "${t}ms" "${sp}x" "${sc}x"
done
echo "wrote autopick.tsv"
