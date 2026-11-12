#!/usr/bin/env bash
# replay.sh <out dir> [row ...]: the B300 rows of the replay pack, measured.
#
# Per row of the pack's picks_b300.tsv (decided on the sender's machine by
# solve_b300_locally.sh), with the saved B300 cost model of that row, the
# shipped profiles and, for GENGA drift, the pack's B300 Herbie answers:
#   1. compile the pack's configuration by name (-poseidon-apply-rewrites, no
#      -poseidon-tau, no -poseidon-cost-budget of its own): the compiler's
#      manual selector applies those candidates and runs no DP solve; a name
#      the compile cannot price (e.g. a Herbie answer it would have to search
#      for) is an LLVM ERROR, and no Herbie binary exists at the path the
#      flags give, so no search can start;
#   2. refuse the row (FAILED, not measured) unless the compile log shows
#      exactly that: the applied configuration ([poseidon] pin:) equals the
#      pack's, every explanation file says selector "manual", the Herbie
#      summary reports 0 fresh searches, and no search was attempted;
#   3. build and run the baselines and measure, as the benchmark's own
#      pipeline script does for that kind of row.
# Writes <out dir>/replay.tsv and <out dir>/rows/<row>/.
#
# REPLAY_NO_GPU=1: steps 1 and 2 only (the local docker test).
set -uo pipefail
OUT="${1:?usage: replay.sh <out dir> [row ...]}"; shift
TC="${TC:-/toolchain}"; BUNDLE="${BUNDLE:-/bundle}"; WORK="${WORK:-/work}"
PACK="${PACK:-$WORK/replay_pack}"
if [ ! -f "$PACK/picks_b300.tsv" ] && [ -f "$BUNDLE/replay_pack.tar.gz" ]; then
  rm -rf "$PACK" && mkdir -p "$(dirname "$PACK")" && tar -C "$(dirname "$PACK")" -xzf "$BUNDLE/replay_pack.tar.gz"
fi
NO_GPU="${REPLAY_NO_GPU:-0}"
NUM_RUNS="${NUM_RUNS:-5}"
ART="$WORK/cgo2027"
ROWS="$PACK/b300_rows.tsv"; CMDIR="$PACK/cost_models"
log() { echo "[replay $(date -u +%H:%M:%S)] $*"; }
[ -f "$PACK/picks_b300.tsv" ] || { log "no replay pack at $PACK"; exit 99; }
[ -f "$TC/poseidon.stamp" ] || { log "no toolchain in $TC: run the build stage first"; exit 99; }
# The compiler in the toolchain Volume must be the bundle's (a build of an
# earlier bundle would replay these names with another compiler).
case "$(cat "$TC/poseidon.stamp")" in
  "src=$(sha256sum "$BUNDLE/poseidon-src.tar.gz" | cut -d' ' -f1) "*) ;;
  *) log "the toolchain in $TC was built from another poseidon-src.tar.gz: run the build stage first"; exit 99 ;;
esac
mkdir -p "$OUT/rows"
rm -rf "$ART" && mkdir -p "$WORK/tmp" && tar -C "$WORK" -xzf "$BUNDLE/artifact.tar.gz"
export POSEIDON_BUILD="$TC/poseidon-build" TMPDIR="$WORK/tmp"
if [ "$NO_GPU" = 1 ]; then export POSEIDON_ARTIFACT_NO_GPU=1 CUDA_VISIBLE_DEVICES=; unset GPU_ID
else export GPU_ID=0; fi
PY="${PY:-$(command -v python3)}"; export PY
source "$(dirname "$0")/b300_lib.sh"
cp -f "$PACK/MANIFEST" "$OUT/replay_pack_MANIFEST"; cp -f "$PACK/picks_b300.tsv" "$OUT/"

pick_field() { awk -F'\t' -v id="$1" -v col="$2" 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
  $1 == id { print $(c[col]); exit }' "$PACK/picks_b300.tsv"; }
avg() { "$PY" -c "import sys;v=[float(x) for x in sys.argv[1:] if x];print(f'{sum(v)/len(v):.6f}' if v else 'nan')" "$@"; }
grab() { grep -oE "$1 = \S+" | grep -oE "[0-9.eE+-]+$" | head -1; }
ratio() { "$PY" -c "print(f'{float(\"$1\")/float(\"$2\"):.3f}')" 2>/dev/null || echo nan; }

printf 'row\tstatus\tpin\tbaseline_ms\tms\tspeedup\taccuracy\n' > "$OUT/replay.tsv"
ROW_IDS=("$@")
[ ${#ROW_IDS[@]} -gt 0 ] || mapfile -t ROW_IDS < <(awk -F'\t' 'NR > 1 {print $1}' "$PACK/picks_b300.tsv")
FAILS=0
for row in "${ROW_IDS[@]}"; do
  R="$OUT/rows/$row"; mkdir -p "$R"
  pin="$(pick_field "$row" pin)"
  if [ -z "$pin" ]; then
    log "$row: the pack has no configuration for this row; skipped"
    printf '%s\tNO-PICK\t\t\t\t\t\n' "$row" >> "$OUT/replay.tsv"; continue
  fi
  BD="$(bench_dir "$row")"
  # The local solve's caches (accuracy memos, Herbie answers it read).
  for c in $(cache_dirs "$row"); do
    rm -rf "${BD:?}/$c"; [ -d "$PACK/rows/$row/$c" ] && cp -r "$PACK/rows/$row/$c" "$BD/$c"
  done
  # GENGA drift: the B300 answer pack replaces the shipped (RTX 5090)
  # answers, and is read through -poseidon-import-herbie-answers only.
  if [ "$row" = t1-drift-b300 ]; then
    [ -d "$PACK/herbie_answers_b300/kepler" ] || { log "$row: the pack has no B300 Herbie answers"; FAILS=$((FAILS + 1)); continue; }
    rm -rf "$BD/herbie_answers/kepler" && cp -r "$PACK/herbie_answers_b300/kepler" "$BD/herbie_answers/kepler"
    rm -f "$BD"/cache_b300_kep/cachedHerbie*
  fi
  exe="$WORK/tmp/$row.exe"; rm -f "$exe"
  log "$row: compiling the pack's configuration by name"
  compile_row "$row" "$exe" "$R/compile.log" "$R/explain" -poseidon-apply-rewrites="$pin"
  # ---- the guard ------------------------------------------------------------
  why=""
  [ -s "$exe" ] || why="the compile failed"
  [ "$(pins_of "$R/compile.log")" = "$(echo "$pin" | tr ',' '\n' | sed '/^$/d' | sort -u)" ] ||
    why="${why:+$why; }the applied configuration differs from the pack"
  sel="$("$PY" -c 'import glob,json,sys; print(" ".join(sorted({json.load(open(f))["target"].get("selector","?") for f in glob.glob(sys.argv[1]+"/*.explain.json")})))' "$R/explain" 2>/dev/null)"
  [ "$sel" = manual ] || why="${why:+$why; }selector '$sel', not 'manual' (a solve ran)"
  fresh="$(herbie_fresh "$R/compile.log")"
  [ "${fresh:-0}" = 0 ] || why="${why:+$why; }$fresh Herbie searches"
  grep -a -q -E 'posix_spawn|Execution failed' "$R/compile.log" && why="${why:+$why; }a Herbie search was attempted"
  if [ -n "$why" ]; then
    log "$row: FAILED: $why (see rows/$row/compile.log)"
    printf '%s\tFAILED: %s\t%s\t\t\t\t\n' "$row" "$why" "$pin" >> "$OUT/replay.tsv"
    FAILS=$((FAILS + 1)); continue
  fi
  log "$row: compile applied the pack's configuration, selector manual, no Herbie search"
  if [ "$NO_GPU" = 1 ]; then
    printf '%s\tCOMPILED\t%s\t\t\t\t\n' "$row" "$pin" >> "$OUT/replay.tsv"; continue
  fi
  # ---- baseline and measurement, as the benchmark's pipeline does ---------------
  bt="" ms="" acc=""
  case "$(row_field "$row" bench)" in
    ozp) (
        b300_env "$row"; source "$ART/benchmarks/ozp/scripts/common.sh" 2>/dev/null
        bash scripts/gen_data.sh "$APPLY_N" > "$R/gen_data.log" 2>&1
        HA=data/synth${APPLY_N}_h.mtx; KA=data/synth${APPLY_N}_k.mtx
        "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N "$OZAKI_RT" surrogate_xN.cu -o "$TMPD/cub.exe" > "$R/cublas_build.log" 2>&1
        "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPD/base.exe" > "$R/baseline_build.log" 2>&1
        # NUM_RUNS launches of each program; every launch is kept in runs.txt.
        ct=$(for i in $(seq "$NUM_RUNS"); do "$TMPD/cub.exe" $HA $KA cublas | tee -a "$R/cublas.out" | sed -n 's/.*time_ms=\([0-9.]*\).*/\1/p'; done)
        b=$(for i in $(seq "$NUM_RUNS"); do "$TMPD/base.exe" $HA $KA | tee -a "$R/baseline.out" | grab "Elapsed time"; done)
        t=$(for i in $(seq "$NUM_RUNS"); do "$exe" $HA $KA | tee -a "$R/run.out" | grab "Elapsed time"; done)
        res=$("$exe" $HA $KA)
        CT="$(avg $ct)"; T="$(avg $t)"
        echo "cublas_ms: $(echo $ct) | baseline_ms: $(echo $b) | ms: $(echo $t)" > "$R/runs.txt"
        echo "$(avg $b)|$T|RMSD=$(echo "$res" | grab RMSD);L2=$(echo "$res" | grab 'L2 relative error');cublas_ms=$CT;x_cublas=$(ratio "$CT" "$T")" > "$R/measure.txt" ) ;;
    chase_eigensolver) (
        b300_env "$row"; source "$ART/benchmarks/chase_eigensolver/scripts/common.sh" 2>/dev/null
        chase_make baseline BENCH_DEFS="$DEP_DEFS" > "$R/baseline_build.log" 2>&1
        cp "$exe" chase_opt.exe
        # One unrecorded launch of the baseline first: the first program of a
        # job that loads cuSOLVER pays a one-time start-up inside solve_ms.
        ./chase_baseline.exe > "$R/warmup.out" 2>&1
        # Two interleaved sets of NUM_RUNS launches each, every launch
        # recorded; the cell is the first set (set 2 shows the repeatability).
        ms1() { timeout 600 "$1" 2>&1 | tee -a "$R/all_runs.out" | grep -oE "solve_ms = [0-9.]+" | grep -oE "[0-9.]+$"; }
        o1=$(for i in $(seq "$NUM_RUNS"); do ms1 ./chase_opt.exe; done)
        b1=$(for i in $(seq "$NUM_RUNS"); do ms1 ./chase_baseline.exe; done)
        o2=$(for i in $(seq "$NUM_RUNS"); do ms1 ./chase_opt.exe; done)
        b2=$(for i in $(seq "$NUM_RUNS"); do ms1 ./chase_baseline.exe; done)
        ./chase_opt.exe > "$R/run.out" 2>&1
        l2=$(grep "L2 vs planted" "$R/run.out" | grep -oE "= [0-9.e+-]+" | head -1 | tr -d '= ')
        lk=$(grep -oE "locked [0-9]+/[0-9]+" "$R/run.out" | tail -1)
        { echo "warmup_baseline_ms: $(grep -oE "solve_ms = [0-9.]+" "$R/warmup.out" | grep -oE "[0-9.]+$")"
          echo "set1 opt_ms: $(echo $o1) mean $(avg $o1)"; echo "set1 baseline_ms: $(echo $b1) mean $(avg $b1)"
          echo "set2 opt_ms: $(echo $o2) mean $(avg $o2)"; echo "set2 baseline_ms: $(echo $b2) mean $(avg $b2)"; } > "$R/runs.txt"
        echo "$(avg $b1)|$(avg $o1)|eigL2=$l2;$lk" > "$R/measure.txt" ) ;;
    genga_nbody) (
        b300_env "$row"; source "$ART/benchmarks/genga_nbody/scripts/common.sh" 2>/dev/null
        base="$TMPDIR/$row-base.exe"
        if [ "${row#*force}" != "$row" ]; then
          "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" genga_nbody.cu -o "$base" > "$R/baseline_build.log" 2>&1
          args=(65536)
        else
          "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" genga_kepler.cu -o "$base" > "$R/baseline_build.log" 2>&1
          args=()
        fi
        [ -s "$base" ] || { echo "the FP64 baseline did not build" > "$R/err.log"; exit 1; }
        # The baseline is timed before and after the rewritten program.
        b=$(for i in $(seq "$NUM_RUNS"); do "$base" "${args[@]}" 2>>"$R/err.log" | grab "Elapsed time"; done)
        t=$(for i in $(seq "$NUM_RUNS"); do timeout 120 "$exe" "${args[@]}" 2>>"$R/err.log" | grab "Elapsed time"; done)
        b2=$(for i in $(seq "$NUM_RUNS"); do "$base" "${args[@]}" 2>>"$R/err.log" | grab "Elapsed time"; done)
        l2=$(timeout 120 "$exe" "${args[@]}" 2>>"$R/err.log" | grab "L2 relative error")
        acc="L2=$l2"
        if [ "${row#*force}" = "$row" ]; then
          dr=$(timeout 300 "$exe" --integrate --steps=20000 65536 2>>"$R/err.log" | grab "Mean energy drift")
          acc="Edrift=$dr;L2=$l2"
        fi
        echo "baseline_ms before: $(echo $b) mean $(avg $b) | ms: $(echo $t) mean $(avg $t) | baseline_ms after: $(echo $b2) mean $(avg $b2)" > "$R/runs.txt"
        echo "$(avg $b)|$(avg $t)|$acc" > "$R/measure.txt" ) ;;
  esac
  IFS='|' read -r bt ms acc < "$R/measure.txt" 2>/dev/null || true
  printf '%s\tMEASURED\t%s\t%s\t%s\t%s\t%s\n' "$row" "$pin" "$bt" "$ms" "$(ratio "$bt" "$ms")" "$acc" >> "$OUT/replay.tsv"
  log "$row: baseline ${bt} ms, ${ms} ms, speedup $(ratio "$bt" "$ms"), $acc"
done
cut -f1,2,4-7 "$OUT/replay.tsv"
[ "$FAILS" = 0 ]
