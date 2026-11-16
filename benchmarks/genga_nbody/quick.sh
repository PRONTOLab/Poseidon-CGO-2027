#!/bin/bash
# Quick run: one force frontier point and a 2000-step drift accuracy check.
set -e
source "$(dirname "$0")/scripts/common.sh"
rm -f overhead.tsv
mkdir -p tmp_quick
head_share() { tr ',' '\n' < "$1/budgets.txt" | sort -g | head -1; }

echo "======== FORCE quick run (genga_nbody.cu) ========"
bash ../../scripts/pt.sh profile_compile force -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" -poseidon-profile-generate genga_nbody.cu -o genga_nbody_profiled.exe
rm -rf fpprofile
bash ../../scripts/pt.sh profile_run force -- env CUDA_VISIBLE_DEVICES="$GPU_ID" POSEIDON_PROFILE_DIR=./fpprofile ./genga_nbody_profiled.exe --profiling
compare_profile fpprofile force
rm -f cache/table.json cache/budgets.txt; mkdir -p cache
mapfile -t FF < <(force_flags -poseidon-cache=cache)
bash ../../scripts/pt.sh baseline_compile force -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" genga_nbody.cu -o tmp_quick/base.exe
bash ../../scripts/pt.sh solve_compile "force b=0" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FF[@]}" -poseidon-cost-budget=0 genga_nbody.cu -o tmp_quick/b0.exe
B="$(head_share cache)"
bash ../../scripts/pt.sh solve_compile "force b=$B" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FF[@]}" -poseidon-cost-budget="$B" \
  genga_nbody.cu -o tmp_quick/fp32.exe > tmp_quick/fp32.log 2>&1
expected_report genga-force-fp32 tmp_quick/fp32.log
genga_run "quick: force F64 baseline" -- 120 tmp_quick/base.exe 65536; BT=$GENGA_MS
genga_run "quick: force FP32" L2=acc:"L2 relative error" -- 120 tmp_quick/fp32.exe 65536; OT=$GENGA_MS
printf '%s\n' "$MEASURE_OUT" > tmp_quick/fp32.out
echo "  force FP32: baseline=${BT}ms opt=${OT}ms speedup=$("$PY" -c "print(f'{$BT/$OT:.1f}x')") L2=$L2  (expect ~51x, 5.056976e-06)"

echo "======== DRIFT quick run (genga_kepler.cu, Herbie answers: $GENGA_HERBIE) ========"
bash ../../scripts/pt.sh profile_compile kepler -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" -poseidon-profile-generate genga_kepler.cu -o kepler_profiled.exe
rm -rf fpprofile_kep_rolled
bash ../../scripts/pt.sh profile_run kepler -- env CUDA_VISIBLE_DEVICES="$GPU_ID" POSEIDON_PROFILE_DIR=./fpprofile_kep_rolled ./kepler_profiled.exe --profiling
compare_profile fpprofile_kep_rolled kepler
rm -rf cache_kep_herbieoff; mkdir -p cache_kep_herbieoff cache_kep_rolled
rm -f cache_kep_rolled/table.json cache_kep_rolled/budgets.txt
mapfile -t KH < <(kepler_herbieoff_flags)
mapfile -t KC < <(kepler_flags cache_kep_rolled)
bash ../../scripts/pt.sh solve_compile "kepler-naive b=0" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${KH[@]}" -poseidon-cost-budget=0 genga_kepler.cu -o tmp_quick/kn0.exe
BN="$(head_share cache_kep_herbieoff)"
bash ../../scripts/pt.sh solve_compile "kepler-naive b=$BN" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${KH[@]}" -poseidon-cost-budget="$BN" \
  genga_kepler.cu -o tmp_quick/naive.exe > tmp_quick/naive.log 2>&1
expected_report genga-drift-naive tmp_quick/naive.log
if [ "$GENGA_HERBIE_USABLE" = 0 ]; then
  herbie_skip_note "the Herbie-rewritten drift kernel"
  genga_run "quick: drift naive FP32" NAI=acc:"Mean energy drift" -- 600 tmp_quick/naive.exe --integrate --steps=2000 65536
  echo "  drift @2000 steps: naive-FP32=$NAI"
  echo "======== quick.sh OK ========"
  exit 0
fi
# A share is specific to one cost model; elsewhere tau = 1e-11 selects the same configuration.
if [ "$ON_SHIPPED_MODEL" = 1 ]; then SEL="-poseidon-cost-budget=$(expected_share genga-drift-algebraic)"; else SEL=-poseidon-tau=1e-11; fi
bash ../../scripts/pt.sh solve_compile "kepler ${SEL#-poseidon-}" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${KC[@]}" "$SEL" \
  genga_kepler.cu -o tmp_quick/drift.exe > tmp_quick/drift.log 2>&1
echo "  $(herbie_summary tmp_quick/drift.log)"
expected_report genga-drift-algebraic tmp_quick/drift.log
genga_run "quick: drift naive FP32" NAI=acc:"Mean energy drift" -- 600 tmp_quick/naive.exe --integrate --steps=2000 65536
genga_run "quick: drift Poseidon" POS=acc:"Mean energy drift" -- 600 tmp_quick/drift.exe --integrate --steps=2000 65536
echo "  drift @2000 steps: naive-FP32=$NAI  Poseidon=$POS  (expect Poseidon about 1000x smaller)"
echo "======== quick.sh OK ========"
