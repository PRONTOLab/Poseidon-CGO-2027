#!/usr/bin/env bash
# table1.sh [--device D] [--out <dir>] [row ...]: measure the Table 1 rows of a GPU on that GPU.
# Each row's configuration (the pin column of expected/picks*.tsv) is compiled by name, without a
# solve or a Herbie search, and timed against the scalar FP64 program -> <out>/table1.tsv.
# D: the device of this GPU's compute capability (default) or rtx5090, b300, gh200.
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ART/scripts/expected.sh"
DEV="" OUT="" ROW_IDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEV="${2:?--device needs a device}"; shift ;;
    --out) OUT="${2:?--out needs a directory}"; shift ;;
    -h|--help) sed -n '2,5p' "$0" | sed 's/^# //'; exit 0 ;;
    -*) echo "table1.sh: unknown option $1" >&2; exit 2 ;;
    *) ROW_IDS+=("$1") ;;
  esac
  shift
done
if [ -z "$DEV" ]; then
  ARCH="$(bash "$ART/scripts/detect.sh" gpu_arch)" || exit 1
  DEV="$(awk -F'\t' -v a="$ARCH" 'NR > 1 && $2 == a {print $1; exit}' "$_DEVICES_TSV")"
  [ -n "$DEV" ] || { echo "table1.sh: Table 1 has no column for this GPU ($ARCH); --device rtx5090|b300|gh200 names one" >&2; exit 2; }
fi
device_field "$DEV" gpu_arch > /dev/null || exit 2
# The Table 1 rows: every row of the B300 and GH200 picks; the five RTX 5090 rows of Table 1.
if [ ${#ROW_IDS[@]} = 0 ]; then
  case "$DEV" in
    rtx5090) ROW_IDS=(genga-force-expansion genga-drift-algebraic ozp-tau1e-7 ozp-tau1e-6 rchfsi-bf16-nm12) ;;
    *) mapfile -t ROW_IDS < <(device_rows "$DEV") ;;
  esac
fi
OUT="${OUT:-$ART/table1/$DEV}"
mkdir -p "$OUT/rows" && OUT="$(cd "$OUT" && pwd)"
TMPX="$(mktemp -d)"; trap 'rm -rf "$TMPX"' EXIT
NUM_RUNS="${NUM_RUNS:-5}"
log() { echo "[table1 $(date +%H:%M:%S)] $*"; }

# The row's device settings; a decision depends on the shipped profiles and Herbie answers only.
row_settings() { row_env "$1" && export PROFILES=shipped GENGA_HERBIE=replay; }
genga_cache() { case "$1" in *force*) echo "cache_table1_${DEV}_force" ;; *) echo "cache_table1_${DEV}_kep" ;; esac; }

# compile_row <row> <exe> <log> <reports dir> [compiler flags...]
compile_row() {
  local row="$1" exe="$2" log="$3" rep="$4"; shift 4
  (
    row_settings "$row" || exit 1
    case "$(expected_field "$row" bench)" in
      ozp)
        source "$ART/benchmarks/ozp/scripts/common.sh" > /dev/null 2>&1 || exit 1
        mkdir -p cache; rm -f cache/*.ozdispatch; rm -rf "$rep"
        "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N="$APPLY_N" \
          -poseidon-print -poseidon-report-path="$rep" "$@" ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "$exe" > "$log" 2>&1 ;;
      chase_eigensolver)
        source "$ART/benchmarks/chase_eigensolver/scripts/common.sh" > /dev/null 2>&1 || exit 1
        # Flags with spaces reach `make solve` quoted in a response file.
        local rsp="$TMPX/solve.rsp" a
        : > "$rsp"
        for a in "$@"; do
          "$PY" -c 'import sys; a=sys.argv[1]; print(chr(34) + a.replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34)) + chr(34))' "$a" >> "$rsp"
        done
        rm -rf reports
        make solve BENCH_DEFS="$DEP_DEFS" BUDGET=0 PROFDIR="$PROFSRC" EXTRA_SOLVE="@$rsp" > /dev/null 2>&1
        cp solve.log "$log"
        rm -rf "$rep"; [ -d reports ] && cp -r reports "$rep"
        [ -s chase_opt.exe ] && cp chase_opt.exe "$exe" ;;
      genga_nbody)
        source "$ART/benchmarks/genga_nbody/scripts/common.sh" > /dev/null 2>&1 || exit 1
        rm -rf "$rep"
        local c FPF; c="$(genga_cache "$row")"
        case "$row" in
          *force*)
            mapfile -t FPF < <(force_flags -poseidon-cache="$c")
            "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPF[@]}" -poseidon-report-path="$rep" "$@" \
              genga_nbody.cu -o "$exe" > "$log" 2>&1 ;;
          *)
            mapfile -t FPF < <(kepler_flags "$c")
            "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FPF[@]}" -poseidon-report-path="$rep" "$@" \
              genga_kepler.cu -o "$exe" > "$log" 2>&1 ;;
        esac ;;
      *) echo "table1.sh: no benchmark for row $row" >&2; exit 2 ;;
    esac
  )
}

pins_of() { grep -a '^\[poseidon\] pin: ' "$1" | sed 's/^\[poseidon\] pin: //' | tr ',' '\n' | sed '/^$/d' | sort -u; }
herbie_fresh() { grep -a -oE 'Herbie answers: [0-9]+ fresh' "$1" | grep -oE '[0-9]+' | paste -sd+ | bc 2>/dev/null || echo 0; }
avg() { "$PY" -c "import sys;v=[float(x) for x in sys.argv[1:]];print(f'{sum(v)/len(v):.6f}')" "$@"; }
ratio() { "$PY" -c "print(f'{float(\"$1\")/float(\"$2\"):.3f}')" 2>/dev/null || echo nan; }
# runs <launch> <log> <var>=<time|acc>:<label> -- <command...>: NUM_RUNS checked launches; values in RUNS.
runs() {
  local _rl="$1" _rlog="$2" _rspec="$3" _ri _rn
  shift 4
  _rn="${_rspec%%=*}"; RUNS=()
  for _ri in $(seq "$NUM_RUNS"); do
    measure "$_rl, run $_ri of $NUM_RUNS" "$_rspec" -- "$@"
    printf '%s\n' "$MEASURE_OUT" >> "$_rlog"
    RUNS+=("${!_rn}")
  done
}
PY="$(bash "$ART/scripts/detect.sh" py)" || exit 1

log "device $DEV ($(device_field "$DEV" description)), rows: ${ROW_IDS[*]}"
printf 'row\tstatus\tpin\tbaseline_ms\tms\tspeedup\taccuracy\n' > "$OUT/table1.tsv"
FAILS=0
for row in "${ROW_IDS[@]}"; do
  R="$OUT/rows/$row"; rm -rf "$R"; mkdir -p "$R"
  pin="$(expected_field "$row" pin)" || { FAILS=$((FAILS + 1)); continue; }
  bench="$(expected_field "$row" bench)"
  [ "$bench" = genga_nbody ] && rm -rf "$ART/benchmarks/genga_nbody/$(genga_cache "$row")"
  exe="$TMPX/$row.exe"
  log "$row: compiling its configuration by name"
  compile_row "$row" "$exe" "$R/compile.log" "$R/explain" -poseidon-apply-rewrites="$pin"
  why=""
  [ -s "$exe" ] || why="the compile failed"
  [ "$(pins_of "$R/compile.log")" = "$(echo "$pin" | tr ',' '\n' | sed '/^$/d' | sort -u)" ] ||
    why="${why:+$why; }the applied configuration differs from the row"
  sel="$("$PY" -c 'import glob,json,sys; print(" ".join(sorted({json.load(open(f))["target"].get("selector","?") for f in glob.glob(sys.argv[1]+"/*.explain.json")})))' "$R/explain" 2>/dev/null)"
  [ "$sel" = manual ] || why="${why:+$why; }selector '$sel', not 'manual' (a solve ran)"
  fresh="$(herbie_fresh "$R/compile.log")"
  [ "${fresh:-0}" = 0 ] || why="${why:+$why; }$fresh Herbie searches"
  if [ -n "$why" ]; then
    log "$row: FAILED: $why (see $R/compile.log)"
    printf '%s\tFAILED: %s\t%s\t\t\t\t\n' "$row" "$why" "$pin" >> "$OUT/table1.tsv"
    FAILS=$((FAILS + 1)); continue
  fi
  log "$row: applied exactly its configuration, no solve, no Herbie search; measuring"
  rm -f "$R/measure.txt"
  case "$bench" in
    ozp) (
        row_settings "$row"; source "$ART/benchmarks/ozp/scripts/common.sh" > /dev/null 2>&1
        bash scripts/gen_data.sh "$APPLY_N" > "$R/gen_data.log" 2>&1
        HA=data/synth${APPLY_N}_h.mtx; KA=data/synth${APPLY_N}_k.mtx
        "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N "$OZAKI_RT" surrogate_xN.cu -o "$TMPX/cub.exe" > "$R/cublas_build.log" 2>&1
        "$CLANG" "${COMMON_CUDA[@]}" -DOZP_N=$APPLY_N ozp_poseidon.cu -o "$TMPX/base.exe" > "$R/baseline_build.log" 2>&1
        runs "$row cuBLAS DGEMM" "$R/cublas.out" x=time:time_ms -- "$TMPX/cub.exe" $HA $KA cublas; ct=("${RUNS[@]}")
        runs "$row FP64 baseline" "$R/baseline.out" x=time:"Elapsed time" -- "$TMPX/base.exe" $HA $KA; b=("${RUNS[@]}")
        runs "$row" "$R/run.out" x=time:"Elapsed time" -- "$exe" $HA $KA; t=("${RUNS[@]}")
        measure "$row accuracy" _t=time:"Elapsed time" rmsd=acc:RMSD l2=acc:"L2 relative error" -- "$exe" $HA $KA
        CT="$(avg "${ct[@]}")"; T="$(avg "${t[@]}")"
        echo "cublas_ms: ${ct[*]} | baseline_ms: ${b[*]} | ms: ${t[*]}" > "$R/runs.txt"
        echo "$(avg "${b[@]}")|$T|RMSD=$rmsd;L2=$l2;cublas_ms=$CT;x_cublas=$(ratio "$CT" "$T")" > "$R/measure.txt" ) ;;
    chase_eigensolver) (
        row_settings "$row"; source "$ART/benchmarks/chase_eigensolver/scripts/common.sh" > /dev/null 2>&1
        chase_make baseline BENCH_DEFS="$DEP_DEFS" > "$R/baseline_build.log" 2>&1
        cp "$exe" chase_opt.exe
        # The first program that loads cuSOLVER pays a one-time start-up inside solve_ms.
        measure "$row warm-up baseline" w=time:solve_ms -- ./chase_baseline.exe
        printf '%s\n' "$MEASURE_OUT" > "$R/warmup.out"
        runs "$row" "$R/all_runs.out" x=time:solve_ms -- timeout 600 ./chase_opt.exe; o=("${RUNS[@]}")
        runs "$row baseline" "$R/all_runs.out" x=time:solve_ms -- timeout 600 ./chase_baseline.exe; b=("${RUNS[@]}")
        measure "$row accuracy" _t=time:solve_ms l2=acc:"eigval L2 vs planted" -- ./chase_opt.exe
        printf '%s\n' "$MEASURE_OUT" > "$R/run.out"
        lk=$(grep -oE "locked [0-9]+/[0-9]+" "$R/run.out" | tail -1)
        [ -n "$lk" ] || measure_fail "$row accuracy" "it printed no 'locked <n>/<m>'"
        echo "opt_ms: ${o[*]} | baseline_ms: ${b[*]}" > "$R/runs.txt"
        echo "$(avg "${b[@]}")|$(avg "${o[@]}")|eigL2=$l2;$lk" > "$R/measure.txt" ) ;;
    genga_nbody) (
        row_settings "$row"; source "$ART/benchmarks/genga_nbody/scripts/common.sh" > /dev/null 2>&1
        base="$TMPX/$row-base.exe"
        if [ "${row#*force}" != "$row" ]; then
          "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" genga_nbody.cu -o "$base" > "$R/baseline_build.log" 2>&1
          args=(65536)
        else
          "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" genga_kepler.cu -o "$base" > "$R/baseline_build.log" 2>&1
          args=()
        fi
        [ -s "$base" ] || { echo "the FP64 baseline did not build" > "$R/err.log"; exit 1; }
        runs "$row FP64 baseline" "$R/runs.out" x=time:"Elapsed time" -- timeout 120 "$base" "${args[@]}"; b=("${RUNS[@]}")
        runs "$row" "$R/runs.out" x=time:"Elapsed time" -- timeout 120 "$exe" "${args[@]}"; t=("${RUNS[@]}")
        measure "$row accuracy" _t=time:"Elapsed time" l2=acc:"L2 relative error" -- timeout 120 "$exe" "${args[@]}"
        acc="L2=$l2"
        if [ "${row#*force}" = "$row" ]; then
          measure "$row energy drift" _t=time:"Elapsed time" dr=acc:"Mean energy drift" -- timeout 300 "$exe" --integrate --steps=20000 65536
          acc="Edrift=$dr;L2=$l2"
        fi
        echo "baseline_ms: ${b[*]} | ms: ${t[*]}" > "$R/runs.txt"
        echo "$(avg "${b[@]}")|$(avg "${t[@]}")|$acc" > "$R/measure.txt" ) ;;
  esac
  if [ $? != 0 ] || [ ! -s "$R/measure.txt" ]; then
    log "$row: FAILED: a measurement did not pass its checks (see the message above)"
    printf '%s\tFAILED: measurement\t%s\t\t\t\t\n' "$row" "$pin" >> "$OUT/table1.tsv"
    FAILS=$((FAILS + 1)); continue
  fi
  IFS='|' read -r bt ms acc < "$R/measure.txt"
  printf '%s\tMEASURED\t%s\t%s\t%s\t%s\t%s\n' "$row" "$pin" "$bt" "$ms" "$(ratio "$bt" "$ms")" "$acc" >> "$OUT/table1.tsv"
  log "$row: baseline ${bt} ms, ${ms} ms, speedup $(ratio "$bt" "$ms"), $acc"
done
echo
echo "==== Table 1, $DEV column (speedup over scalar FP64; $OUT/table1.tsv) ===="
awk -F'\t' 'NR > 1 { printf "%-24s %-10s %s  %s\n", $1, ($2 == "MEASURED" ? $6 "x" : $2), "", $7 }' "$OUT/table1.tsv"
[ "$FAILS" = 0 ]
