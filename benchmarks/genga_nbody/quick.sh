#!/bin/bash
# Kick-the-tires: verify the Poseidon toolchain runs end-to-end on this GPU in a
# few minutes, without the full frontier sweeps. Builds and checks ONE force
# point (the cheapest frontier point, all-FP32, expect ~51x at L2 ~5e-6) and the
# drift naive-vs-Poseidon accuracy at 2000 steps (expect naive ~1e-5, Poseidon
# far below it). Not a paper measurement — a smoke test.
set -e
source "$(dirname "$0")/scripts/common.sh"
# fresh compilation-overhead log for this run
rm -f overhead.tsv
mkdir -p tmp_quick
head_share() { tr ',' '\n' < "$1/budgets.txt" | sort -g | head -1; }

echo "======== FORCE smoke test (genga_nbody.cu) ========"
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
BT=$(CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_quick/base.exe 65536 | grep -oE "Elapsed time = \S+" | grep -oE "[0-9.e+-]+$")
OT=$(CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_quick/fp32.exe 65536 | tee tmp_quick/fp32.out | grep -oE "Elapsed time = \S+" | grep -oE "[0-9.e+-]+$")
L2=$(grep -oE "L2 relative error = \S+" tmp_quick/fp32.out | grep -oE "[0-9.e+-]+$")
echo "  force FP32: baseline=${BT}ms opt=${OT}ms speedup=$("$PY" -c "print(f'{$BT/$OT:.1f}x')") L2=$L2  (expect ~51x, 5.056976e-06)"

echo "======== DRIFT smoke test (genga_kepler.cu, Herbie answers: $GENGA_HERBIE) ========"
bash ../../scripts/pt.sh profile_compile kepler -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" -poseidon-profile-generate genga_kepler.cu -o kepler_profiled.exe
rm -rf fpprofile_kep_rolled
bash ../../scripts/pt.sh profile_run kepler -- env CUDA_VISIBLE_DEVICES="$GPU_ID" POSEIDON_PROFILE_DIR=./fpprofile_kep_rolled ./kepler_profiled.exe --profiling
compare_profile fpprofile_kep_rolled kepler
rm -rf cache_kep_herbieoff; mkdir -p cache_kep_herbieoff cache_kep_rolled
rm -f cache_kep_rolled/table.json cache_kep_rolled/budgets.txt
mapfile -t KH < <(kepler_herbieoff_flags)
mapfile -t KC < <(kepler_flags cache_kep_rolled)
# Naive FP32: the cheapest point of the rewrite-free frontier, the plain downcast.
bash ../../scripts/pt.sh solve_compile "kepler-naive b=0" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${KH[@]}" -poseidon-cost-budget=0 genga_kepler.cu -o tmp_quick/kn0.exe
BN="$(head_share cache_kep_herbieoff)"
bash ../../scripts/pt.sh solve_compile "kepler-naive b=$BN" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${KH[@]}" -poseidon-cost-budget="$BN" \
  genga_kepler.cu -o tmp_quick/naive.exe > tmp_quick/naive.log 2>&1
expected_report genga-drift-naive tmp_quick/naive.log
if [ "$GENGA_HERBIE_USABLE" = 0 ]; then
  herbie_skip_note "the Herbie-rewritten drift kernel"
  NAI=$(CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_quick/naive.exe --integrate --steps=2000 65536 | grep -oE "Mean energy drift = \S+" | grep -oE "[0-9.e+-]+$")
  echo "  drift @2000 steps: naive-FP32=$NAI"
  echo "======== quick.sh OK ========"
  exit 0
fi
# Poseidon: the drift row of EXPECTED_RESULTS.md on the shipped model; on this
# device's own model, the most accurate rewrite that costs no more than the
# plain downcast.
if [ "$ON_SHIPPED_MODEL" = 1 ]; then BP="$(expected_share genga-drift-algebraic)"; else BP="$BN"; fi
bash ../../scripts/pt.sh solve_compile "kepler b=$BP" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${KC[@]}" -poseidon-cost-budget="$BP" \
  genga_kepler.cu -o tmp_quick/knee.exe > tmp_quick/knee.log 2>&1
echo "  $(herbie_summary tmp_quick/knee.log)"
[ "$ON_SHIPPED_MODEL" = 1 ] && expected_report genga-drift-algebraic tmp_quick/knee.log
NAI=$(CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_quick/naive.exe --integrate --steps=2000 65536 | grep -oE "Mean energy drift = \S+" | grep -oE "[0-9.e+-]+$")
POS=$(CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_quick/knee.exe  --integrate --steps=2000 65536 | grep -oE "Mean energy drift = \S+" | grep -oE "[0-9.e+-]+$")
echo "  drift @2000 steps: naive-FP32=$NAI  Poseidon=$POS  (Poseidon should be ~100-1000x smaller)"
echo "======== quick.sh OK ========"
