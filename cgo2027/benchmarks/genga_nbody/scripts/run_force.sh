#!/bin/bash
# FORCE kick (genga_nbody.cu) — the floating-point expansion / precision axis.
# Profile (N=256) -> solve (expansion + FP32, Herbie off) -> compile one binary
# per frontier point -> measure L2 vs FP64 gold + wall clock (avg of NUM_RUNS)
# at N=65536.  Writes force_full_g0.tsv (budget speedup L2).
#
# The whole-kernel two-component expansion lands near-FP64 accuracy at ~4.5x;
# the all-FP32 point is ~51x.
source "$(dirname "$0")/common.sh"
SRC=genga_nbody.cu
N=65536

echo "[1/4] profile (N=256, reverse-mode AD)"
bash ../../scripts/pt.sh profile_compile force -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" -poseidon-profile-generate "$SRC" -o genga_nbody_profiled.exe
rm -rf fpprofile
bash ../../scripts/pt.sh profile_run force -- env CUDA_VISIBLE_DEVICES="$GPU_ID" POSEIDON_PROFILE_DIR=./fpprofile ./genga_nbody_profiled.exe --profiling \
  || { echo "PROFILE RUN FAILED (exit $?) — aborting; a silent profile failure yields a degenerate flat-cost solve"; exit 1; }
[ -d fpprofile ] && [ -n "$(ls -A fpprofile 2>/dev/null)" ] \
  || { echo "fpprofile/ is missing or empty after the profile run — aborting"; exit 1; }
compare_profile fpprofile force

echo "[2/4] enumeration solve (budget 0: DP table + frontier shares in cache/budgets.txt)"
# budgets.txt is written only when absent, so it is cleared before each
# enumeration. Budget 0 is device-neutral: the produced binary is unused; only
# the enumerated cache/budgets.txt (this model's frontier) matters.
mapfile -t FPOPT < <(force_flags -poseidon-cache=cache)
rm -f cache/table.json cache/budgets.txt; rm -rf reports; mkdir -p cache
bash ../../scripts/pt.sh solve_compile "force b=0" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPOPT[@]}" \
  -poseidon-cost-budget=0 -poseidon-report-path=reports "$SRC" -o genga_nbody_opt.exe
echo "  frontier shares: $(cat cache/budgets.txt)"

echo "[3/4] compile one binary per frontier point"
rm -rf tmp logs; mkdir -p tmp logs
bash ../../scripts/pt.sh baseline_compile force -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "$SRC" -o tmp/genga_nbody-baseline.exe
for B in $(tr ',' ' ' < cache/budgets.txt); do
  bash ../../scripts/pt.sh solve_compile "force b=$B" -- "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPOPT[@]}" -poseidon-cost-budget="$B" \
    "$SRC" -o "tmp/genga_nbody-$B.exe" >"logs/compile_$B.log" 2>&1 \
    && echo "  $B OK" || echo "  $B FAIL (logs/compile_$B.log)"
done

echo "[4/4] measure (N=$N, avg of $NUM_RUNS) -> force_full_g0.tsv"
avg() { "$PY" -c "import sys;v=[float(x) for x in sys.argv[1:] if x];print(f'{sum(v)/len(v):.6f}' if v else 'nan')" "$@"; }
bt=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" tmp/genga_nbody-baseline.exe $N 2>>measure_err.log | grep -oE "Elapsed time = \S+" | grep -oE "[0-9.e+-]+$"; done)
BT=$(avg $bt); echo "  F64 baseline = $BT ms"
TSV=force_full_g0.tsv; printf "budget\tspeedup\tL2\tpin\n" > "$TSV"
for B in $(tr ',' ' ' < cache/budgets.txt); do
  exe=tmp/genga_nbody-$B.exe; [ -f "$exe" ] || continue
  ts=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" $N 2>>measure_err.log | grep -oE "Elapsed time = \S+" | grep -oE "[0-9.e+-]+$"; done)
  ms=$(avg $ts)
  l2=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" $N 2>>measure_err.log | grep -oE "L2 relative error = \S+" | grep -oE "[0-9.e+-]+$")
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')" 2>/dev/null || echo nan)
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs/compile_$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\n" "$B" "$sp" "$l2" "$pin" >> "$TSV"
  printf "  %-22s speedup=%-8s L2=%-14s %s\n" "$B" "$sp" "$l2" "$pin"
done
echo "[done] wrote $TSV"
