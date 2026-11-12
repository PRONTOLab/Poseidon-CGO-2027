#!/bin/bash
# KEPLER drift (genga_kepler.cu) — the Herbie algebraic-rewrite axis.
# Profile (N=256, rolled Newton loop) -> solve (Herbie ON, rolled + chain-split,
# Herbie answers replayed from herbie_answers/kepler unless GENGA_HERBIE=run)
# -> compile one binary per frontier point -> measure single-step L2 + wall
# clock (N=1M, avg of NUM_RUNS) AND orbital-energy drift (N=65536, 20k steps).
# Writes kepler_full_g0.tsv.
#
# The 1-cos / sin-x cancellations of the Newton step are rewritten
# algebraically (a 4-term series), holding the FP32 kernel to ~5e-8 energy
# drift (vs ~1.4e-4 for the plain downcast) at a ~5.4x speedup.
source "$(dirname "$0")/common.sh"
SRC=genga_kepler.cu
CACHE=cache_kep_rolled

echo "[1/4] profile (N=256, ROLLED Newton loop)"
bash ../../scripts/pt.sh profile_compile kepler -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" -poseidon-profile-generate "$SRC" -o kepler_profiled.exe
rm -rf fpprofile_kep_rolled
bash ../../scripts/pt.sh profile_run kepler -- env CUDA_VISIBLE_DEVICES="$GPU_ID" POSEIDON_PROFILE_DIR=./fpprofile_kep_rolled ./kepler_profiled.exe --profiling
echo "  profile files: $(ls fpprofile_kep_rolled 2>/dev/null | wc -l)"
compare_profile fpprofile_kep_rolled kepler

mapfile -t FP < <(kepler_flags "$CACHE" -poseidon-report-path=reports_kep_rolled)
echo "[2/4] solve (Herbie ON, rolled + split; GENGA_HERBIE=$GENGA_HERBIE, profile $(solve_profile kepler fpprofile_kep_rolled))"
# Only the derived DP table and the write-once budgets.txt are dropped; Herbie
# results a GENGA_HERBIE=run search cached here are kept.
rm -f "$CACHE"/table.json "$CACHE"/budgets.txt; rm -rf reports_kep_rolled; mkdir -p "$CACHE"
bash ../../scripts/pt.sh solve_compile "kepler b=0" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP[@]}" -poseidon-cost-budget=0 \
  "$SRC" -o kepler_rolled_opt.exe 2>&1 | tee solve_kepler.log | grep -a -E 'Herbie answers:|DP table contains|error' || true
[ -s "$CACHE"/budgets.txt ] || { echo "kepler solve failed (solve_kepler.log)"; exit 1; }
echo "  $(tr ',' '\n' < "$CACHE"/budgets.txt | wc -l) frontier points; $(herbie_summary solve_kepler.log)"

echo "[3/4] compile one Herbie-on binary per frontier point (parallel, reuse $CACHE)"
rm -rf tmp_kep logs_kep; mkdir -p tmp_kep logs_kep
bash ../../scripts/pt.sh baseline_compile kepler -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "$SRC" -o tmp_kep/kepler-baseline.exe   # F64 baseline (no Poseidon)
BUDS=$(tr ',' ' ' < "$CACHE"/budgets.txt)
pids=0
for B in $BUDS; do
  ( bash ../../scripts/pt.sh solve_compile "kepler b=$B" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP[@]}" -poseidon-cost-budget="$B" \
      "$SRC" -o "tmp_kep/kepler-$B.exe" >"logs_kep/kepler-$B.log" 2>&1 ) &
  pids=$((pids+1)); [ $((pids % 12)) -eq 0 ] && wait
done
wait
echo "  compiled $(ls tmp_kep/kepler-*.exe 2>/dev/null | wc -l) binaries"

echo "[4/4] measure -> kepler_full_g0.tsv (speedup@N=1M, Edrift@N=65536/20k steps)"
avg() { "$PY" -c "import sys;v=[float(x) for x in sys.argv[1:] if x];print(f'{sum(v)/len(v):.6f}' if v else 'nan')" "$@"; }
grab() { grep -oE "$1 = \S+" | grep -oE "[0-9.eE+-]+$"; }
bt=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_kep/kepler-baseline.exe 2>>measure_err.log | grab "Elapsed time"; done)
BT=$(avg $bt); echo "  F64 baseline single-step (N=1M) = $BT ms"
TSV=kepler_full_g0.tsv; printf "budget\tspeedup\tEdrift\tL2\tpin\n" > "$TSV"
for B in $BUDS; do
  exe=tmp_kep/kepler-$B.exe; [ -f "$exe" ] || continue
  ts=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" 2>>measure_err.log | grab "Elapsed time"; done)
  ms=$(avg $ts)
  l2=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" 2>>measure_err.log | grab "L2 relative error")
  dr=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 180 "$exe" --integrate --steps=20000 65536 2>>measure_err.log | grab "Mean energy drift")
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')" 2>/dev/null || echo nan)
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs_kep/kepler-$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\t%s\n" "$B" "$sp" "$dr" "$l2" "$pin" >> "$TSV"
  printf "  %-22s speedup=%-7s Edrift=%-12s L2=%s\n" "$B" "$sp" "$dr" "$l2"
done
echo "[done] wrote $TSV"
