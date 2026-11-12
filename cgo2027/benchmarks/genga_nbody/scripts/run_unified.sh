#!/bin/bash
# UNIFIED force-then-drift single solve (genga_unified.cu): one annotated site
# over a combined kick+drift step. A single DP solve picks floating-point
# expansion / FP32 on the force (kick) subgraphs AND Herbie rewrites on the
# Kepler drift subgraphs -> one Pareto front for the whole step. Profile (N=256)
# -> solve (Herbie ON, rolled+split, Herbie answers replayed from
# herbie_answers/unified unless GENGA_HERBIE=run) -> measure a spanning subset
# of the frontier plus every point where the modelled error first drops
# tenfold: single-step L2 (pos+vel vs FP64 gold) + wall clock (avg of
# NUM_RUNS). Writes unified.tsv; the expected-results table names the
# FP64-grade point by its frontier share. No figure.
source "$(dirname "$0")/common.sh"
SRC=genga_unified.cu
CACHE=cache_unified

echo "[1/4] profile unified (N=256, AD over kick+drift)"
bash ../../scripts/pt.sh profile_compile unified -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" -poseidon-profile-generate "$SRC" -o unified_profiled.exe
rm -rf fpprofile_unified
bash ../../scripts/pt.sh profile_run unified -- env CUDA_VISIBLE_DEVICES="$GPU_ID" POSEIDON_PROFILE_DIR=./fpprofile_unified ./unified_profiled.exe --profiling
echo "  profile files: $(ls fpprofile_unified 2>/dev/null | wc -l)"
compare_profile fpprofile_unified unified

mapfile -t FP < <(unified_flags "$CACHE" -poseidon-report-path=reports_unified)
echo "[2/4] solve unified (Herbie ON; GENGA_HERBIE=$GENGA_HERBIE, profile $(solve_profile unified fpprofile_unified))"
rm -f "$CACHE"/table.json "$CACHE"/budgets.txt; rm -rf reports_unified; mkdir -p "$CACHE"
bash ../../scripts/pt.sh solve_compile "unified b=0" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP[@]}" -poseidon-cost-budget=0 \
  "$SRC" -o genga_unified_opt.exe 2>&1 | tee solve_unified.log | grep -a -E 'Herbie answers:|DP table contains|error' || true
[ -s "$CACHE"/budgets.txt ] || { echo "unified solve failed (solve_unified.log)"; exit 1; }
echo "  $(tr ',' '\n' < "$CACHE"/budgets.txt | wc -l) frontier points; $(herbie_summary solve_unified.log)"

echo "[3/4] compile baseline + spanning subset + modelled-accuracy steps (parallel, reuse $CACHE)"
rm -rf tmp_uni logs_uni; mkdir -p tmp_uni logs_uni
bash ../../scripts/pt.sh baseline_compile unified -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "$SRC" -o tmp_uni/uni-baseline.exe
# The frontier with each point's modelled error comes from the explanation file
# the solve wrote (-poseidon-report-path implies -poseidon-explain).
EXTRA=""
[ "$ON_SHIPPED_MODEL" = 1 ] && EXTRA="$("$PY" -c "import sys; print(repr(float(sys.argv[1])))" "$(expected_share genga-unified)")"
SUBSET=$("$PY" - "$EXTRA" <<'PY'
import glob, json, sys
ex = glob.glob("reports_unified/*explain.json")
assert len(ex) == 1, "expected one explanation file in reports_unified/, found %d" % len(ex)
pts = sorted((p["budget"], p["modelled_rel_error"]) for p in json.load(open(ex[0]))["pareto"])
buds = [b for b, _ in pts]
stride = max(1, len(buds) // 12)
sub = set(buds[::stride]) | {buds[0], buds[-1]}
last = None
for b, e in pts:              # from the cheapest (least accurate) point up
    if last is None or e < 0.1 * last:
        sub.add(b); last = e
if sys.argv[1]:
    sub.add(float(sys.argv[1]))
print(" ".join(repr(b) for b in sorted(sub)))
PY
)
pids=0
for B in $SUBSET; do
  ( bash ../../scripts/pt.sh solve_compile "unified b=$B" -- "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FP[@]}" -poseidon-cost-budget="$B" \
      "$SRC" -o "tmp_uni/uni-$B.exe" >"logs_uni/uni-$B.log" 2>&1 ) &
  pids=$((pids+1)); [ $((pids % 12)) -eq 0 ] && wait
done
wait
echo "  compiled $(ls tmp_uni/uni-*.exe 2>/dev/null | wc -l) binaries"
[ -n "$EXTRA" ] && expected_report genga-unified "logs_uni/uni-$EXTRA.log"

echo "[4/4] measure unified front (single-step L2 pos+vel, avg of $NUM_RUNS) -> unified.tsv"
avg() { "$PY" -c "import sys;v=[float(x) for x in sys.argv[1:] if x];print(f'{sum(v)/len(v):.6f}' if v else 'nan')" "$@"; }
grab() { grep -oE "$1 = \S+" | grep -oE "[0-9.eE+-]+$"; }
bt=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" tmp_uni/uni-baseline.exe 2>>measure_err.log | grab "Elapsed time"; done)
BT=$(avg $bt)
TSV=unified.tsv; printf "budget\tstep_ms\tspeedup\tL2\tpin\n" > "$TSV"
printf "%22s %9s %8s %12s\n" budget step_ms speedup L2
printf "%22s %9.4f %8s %12s\n" baseline-F64 "$BT" 1.00x "$(CUDA_VISIBLE_DEVICES=$GPU_ID tmp_uni/uni-baseline.exe 2>>measure_err.log | grab 'L2 relative error')"
for B in $SUBSET; do
  exe=tmp_uni/uni-$B.exe; [ -f "$exe" ] || continue
  ts=$(for i in $(seq "$NUM_RUNS"); do CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" 2>>measure_err.log | grab "Elapsed time"; done)
  ms=$(avg $ts)
  l2=$(CUDA_VISIBLE_DEVICES="$GPU_ID" timeout 120 "$exe" 2>>measure_err.log | grab "L2 relative error")
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')" 2>/dev/null || echo nan)
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs_uni/uni-$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\t%s\n" "$B" "$ms" "$sp" "$l2" "$pin" >> "$TSV"
  printf "%22s %9s %7sx %12s\n" "$B" "$ms" "$sp" "$l2"
done
echo "[done] wrote $TSV (the expected-results row genga-unified is the point at share ${EXTRA:-<device-specific>})"
