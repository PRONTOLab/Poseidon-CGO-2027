#!/bin/bash
# Force-then-drift kernel in one solve: a frontier subset -> unified.tsv.
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
# Measured: an even subset, every tenfold drop in modelled error, and the expected row.
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
for b, e in pts:
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
mean_time "unified F64 baseline" 120 tmp_uni/uni-baseline.exe; BT=$MEAN_MS
genga_run "unified F64 baseline, accuracy" bl2=acc:"L2 relative error" -- 120 tmp_uni/uni-baseline.exe
TSV=unified.tsv; printf "budget\tstep_ms\tspeedup\tL2\tpin\n" > "$TSV"
printf "%22s %9s %8s %12s\n" budget step_ms speedup L2
printf "%22s %9.4f %8s %12s\n" baseline-F64 "$BT" 1.00x "$bl2"
for B in $SUBSET; do
  exe=tmp_uni/uni-$B.exe; need_exe "$exe" "$B" "logs_uni/uni-$B.log"
  mean_time "unified share $B" 120 "$exe"; ms=$MEAN_MS
  genga_run "unified share $B, accuracy" l2=acc:"L2 relative error" -- 120 "$exe"
  sp=$("$PY" -c "print(f'{$BT/$ms:.3f}')")
  pin=$(grep -a -m1 '^\[poseidon\] pin: ' "logs_uni/uni-$B.log" | sed 's/^\[poseidon\] pin: //')
  printf "%s\t%s\t%s\t%s\t%s\n" "$B" "$ms" "$sp" "$l2" "$pin" >> "$TSV"
  printf "%22s %9s %7sx %12s\n" "$B" "$ms" "$sp" "$l2"
done
echo "[done] wrote $TSV"
[ -z "$EXTRA" ] || bash "$ART/scripts/text_results.sh" force-then-drift
