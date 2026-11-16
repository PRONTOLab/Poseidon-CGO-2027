#!/usr/bin/env bash
# run_artifact.sh: single-command reproduction driver (./run_artifact.sh -h).
set -uo pipefail
cd "$(dirname "$0")"
usage() {
  cat <<'EOF'
Usage: ./run_artifact.sh [mode] [--force]
  (no mode)                 setup check, cost model, every benchmark, figures, results bundle
  --quick                   setup check, cost model, one strategy per benchmark
  --check                   setup check only
  --picks [--device D]      compile-only check of the expected decisions; D: rtx5090 (default), b300, gh200, all; no GPU
  --explain <row> [opts]    what the compiler considered, applied and rejected for one decision row; no GPU
  --explain --list          list the decision rows
  --table1 [--device D]     measure the Table 1 rows of this GPU (D: its device by default), each compiled by name
  --bootstrap-only          setup check and cost model, then stop
  --calibrate               measure this GPU's cost model even where the shipped one applies
  --from-scratch            full run or --quick from nothing precomputed: measured cost model, fresh profiles, Herbie searches
  --force                   start although a benchmark of this artifact is already running
Environment:
  BENCHES                   benchmarks to run (default: genga_nbody ozp chase_eigensolver)
  GPU_ID                    GPU to use (default: first of CUDA_VISIBLE_DEVICES, else 0)
  GENGA_HERBIE=auto|replay|run   GENGA drift and unified solves: auto (default) replays the shipped Herbie answers on a shipped cost model and searches with Herbie otherwise
  HERBIE_ANSWERS=<dir>      answer pack the GENGA solves replay instead of the device's
  HERBIE=<binary>           Herbie for GENGA_HERBIE=run and --from-scratch
  PROFILES=fresh|shipped    solves read profiles made on this GPU (default) or benchmarks/*/profiles/
  SURROGATE_SIZES=<list>    sizes of the ozp surrogate sweep (default: 1024 2048 4096 8192)
  NUM_RUNS=<n>              timed runs averaged per measurement (default: 5)
  POSEIDON_BUILD, CLANG, CUDA, PY, COST_MODEL, GPU_ARCH   toolchain and device; also read from local.env
EOF
}

MODE=full BOOTSTRAP_ARGS="" FORCE_RUN=0 FROM_SCRATCH=0 PICK_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE=quick ;;
    --check) MODE=check ;;
    --picks) MODE=picks ;;
    --device) [ $# -ge 2 ] || { echo "run_artifact: --device needs a device (rtx5090, b300, gh200, all)" >&2; exit 1; }
              PICK_ARGS+=(--device "$2"); shift ;;
    --device=*) PICK_ARGS+=(--device "${1#*=}") ;;
    --explain) shift; exec bash scripts/explain.sh "$@" ;;
    --table1) shift; exec bash scripts/table1.sh "$@" ;;
    --from-scratch) FROM_SCRATCH=1 ;;
    --bootstrap-only) MODE=bootstrap ;;
    --calibrate) BOOTSTRAP_ARGS=--calibrate ;;
    --force) FORCE_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
  shift
done
if [ ${#PICK_ARGS[@]} -gt 0 ] && [ "$MODE" != picks ]; then
  echo "run_artifact: --device selects the devices of --picks; a GPU run uses the GPU it runs on" >&2; exit 1
fi

if [ "$MODE" = picks ]; then
  exec bash scripts/check_picks.sh "${PICK_ARGS[@]}"
fi

if [ "$FROM_SCRATCH" = 1 ]; then
  case "$MODE" in full|quick) ;; *) echo "run_artifact: --from-scratch is a GPU run (full or --quick)" >&2; exit 1 ;; esac
  BOOTSTRAP_ARGS="--calibrate --fresh-profile"
  export PROFILES=fresh GENGA_HERBIE=run
  _BUILD="$(bash scripts/detect.sh build_dir)" || exit 1
  HERBIE="${HERBIE:-$(sed -n 's/^POSEIDON_HERBIE_BINARY:FILEPATH=//p' "$_BUILD/CMakeCache.txt")}"
  if [ -z "$HERBIE" ] || [ ! -x "$HERBIE" ] ||
     ! "$HERBIE" improve --help 2>&1 | grep -q -- '--num-taylor-terms'; then
    echo "run_artifact: --from-scratch runs Herbie, and needs one that accepts --num-taylor-terms." >&2
    echo "run_artifact: none at '${HERBIE:-<unset>}' (HERBIE, else the build's POSEIDON_HERBIE_BINARY)." >&2
    echo "run_artifact: Build it from sources/herbie with sources/herbie-num-taylor-terms.patch:" >&2
    echo "run_artifact:   scripts/build_toolchain.sh --with-herbie --prefix <prefix>   (Racket 8.15, Rust 1.85.1)" >&2
    echo "run_artifact: then HERBIE=<prefix>/herbie/bin/herbie ./run_artifact.sh --from-scratch" >&2
    exit 1
  fi
  export HERBIE
  echo "==== from scratch: Herbie $HERBIE; removing earlier solve caches ===="
  for _c in benchmarks/*/cache*/; do
    [ -d "$_c" ] && { echo "  rm -r $_c"; rm -r -- "${_c:?}"; }
  done
fi

# Subshell: the cost model resolved here is final only after the bootstrap.
echo "==== setup check ===="
(
  set -e
  POSEIDON_ARTIFACT_NO_MODEL=1 source ./env.sh
  PLUGIN="$(bash scripts/detect.sh plugin)"
  echo "  POSEIDON_ROOT = $POSEIDON_ROOT"
  echo "  DRIVER        = $POSEIDON_CLANGXX"
  echo "  PLUGIN        = $PLUGIN (md5 $(md5sum < "$PLUGIN" | cut -c1-8))"
  echo "  CLANG         = $CLANG ($("$CLANG" --version | head -1))"
  echo "  CUDA          = $CUDA ($("$CUDA/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p'))"
  echo "  GPU           = id $GPU_ID: $(nvidia-smi -i "$GPU_ID" --query-gpu=name,compute_cap --format=csv,noheader)"
  echo "  PY            = $PY"
  if [ -n "$COST_MODEL" ]; then echo "  COST_MODEL    = $COST_MODEL"
  else echo "  COST_MODEL    = none yet for $GPU_ARCH: the bootstrap measures this GPU"; fi
  "$PY" -c 'import numpy, matplotlib'
) || {
  echo "setup check FAILED: fix the reported item (environment variable or local.env pin; ./run_artifact.sh -h)." >&2
  echo "On a cluster login node with no GPU visible, run inside a GPU allocation (slurm/run_artifact.sbatch)." >&2
  exit 1
}
[ "$MODE" = check ] && { echo "==== setup check OK ===="; exit 0; }

# A second copy of a benchmark overwrites the first one's outputs and shares its GPU silently.
running_in_benchmarks () {
  local p c
  [ -d /proc ] || return 0
  for p in /proc/[0-9]*; do
    p="${p#/proc/}"
    c="$(readlink "/proc/$p/cwd" 2>/dev/null)" || continue
    case "$c" in "$PWD"/benchmarks/*)
      printf '  pid %-8s started %s   %s\n' "$p" \
        "$(ps -o lstart= -p "$p" 2>/dev/null | tr -s ' ')" \
        "$(tr -d '\0' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-70)" ;;
    esac
  done
}
STALE="$(running_in_benchmarks)"
if [ -n "$STALE" ]; then
  echo "==== a benchmark of this artifact is ALREADY RUNNING ====" >&2
  printf '%s\n' "$STALE" >&2
  if [ "$FORCE_RUN" = 1 ]; then
    echo "run_artifact: --force given; starting anyway. Two copies of a benchmark" >&2
    echo "run_artifact: write the same output files and share the same GPU, so the" >&2
    echo "run_artifact: numbers from this run are not trustworthy." >&2
  else
    echo "run_artifact: refusing to start. Stop them first (kill the pids above)," >&2
    echo "run_artifact: or pass --force if they belong to another copy of the artifact." >&2
    exit 1
  fi
fi

echo "==== cost model ===="
bash scripts/bootstrap_cost_model.sh $BOOTSTRAP_ARGS || exit 1
[ "$MODE" = bootstrap ] && exit 0

source ./env.sh || exit 1
GPU_NAME="$(bash scripts/detect.sh gpu_name)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RES="results/${GPU_NAME}_${GPU_ARCH}_${STAMP}"
mkdir -p logs figures "$RES"

OWN_MODEL=0
if ! cmp -s "$COST_MODEL" "$(bash scripts/detect.sh shipped_cost_model)"; then
  OWN_MODEL=1
  echo "NOTE: this run prices with a cost model measured on this device (${COST_MODEL#"$PWD"/})."
  echo "      Its decisions are this model's own; they are reported, not compared with the"
  echo "      RTX 5090's. After the benchmarks the run checks the shipped decisions"
  echo "      compile-only (the same as ./run_artifact.sh --picks --device ...), and its"
  echo "      speedups are this GPU's, not judged against the RTX 5090 numbers."
fi

BENCHES="${BENCHES:-genga_nbody ozp chase_eigensolver}"
bench_cmd() {
  case "${1}_${MODE}" in
    *_quick)      echo "bash quick.sh" ;;
    *)            echo "bash run_all.sh" ;;
  esac
}

# The lock makes a second copy refuse instead of interleaving its outputs with the first's.
run_bench () {   # $1 = bench, $2 = command
  cd "benchmarks/$1" || exit 1
  if command -v flock >/dev/null 2>&1; then
    exec 9>".run.lock"
    if ! flock -n 9; then
      echo "run_artifact: benchmark '$1' is already running in this directory."
      [ -s .run.pid ] && echo "  holder: pid $(cat .run.pid) started $(ps -o lstart= -p "$(cat .run.pid)" 2>/dev/null | tr -s ' ')"
      echo "  Two copies overwrite each other's outputs and share the GPU."
      exit 97
    fi
    # $$ is the driver's pid even inside a subshell; $BASHPID is this runner's.
    echo "${BASHPID:-$$}" > .run.pid
  fi
  eval "$2"
}

FAILED=""
for b in $BENCHES; do
  CMD="$(bench_cmd "$b")"
  echo "==== [$b] $CMD (log: logs/$b.log) ===="
  if ( run_bench "$b" "$CMD" ) > "logs/$b.log" 2>&1; then
    echo "==== [$b] OK ===="
  else
    echo "==== [$b] FAILED (tail of logs/$b.log below) ===="
    tail -20 "logs/$b.log"
    FAILED="$FAILED $b"
  fi
done

# The shipped models' decisions are compile-time, so they are checked on any GPU.
PICKS_RC=""
if [ "$OWN_MODEL" = 1 ]; then
  _DEVS=all; [ "$MODE" = quick ] && _DEVS=rtx5090
  echo "==== decisions of the shipped cost models, compile-only (--picks --device $_DEVS; log: logs/picks.log) ===="
  ( unset COST_MODEL GPU_ARCH; bash scripts/check_picks.sh --device "$_DEVS" ) > logs/picks.log 2>&1
  PICKS_RC=$?
  grep -E '^== ' logs/picks.log
fi

echo "==== collecting results into $RES ===="
{
  echo "date: $(date -Is)   host: $(hostname)   slurm_job: ${SLURM_JOB_ID:-none}"
  echo "mode: $MODE   benches: $BENCHES   failed:${FAILED:- none}"
  echo "gpu: id $GPU_ID: $(nvidia-smi -i "$GPU_ID" --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>/dev/null)"
  echo "clang: $("$CLANG" --version | head -1)"
  echo "cuda: $("$CUDA/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p')"
  echo "cost_model: $COST_MODEL (md5 $(md5sum < "$COST_MODEL" | cut -c1-8))"
  echo "poseidon_root: $POSEIDON_ROOT"
  echo "poseidon_driver: $POSEIDON_CLANGXX"
  echo "poseidon_plugin: $(bash scripts/detect.sh plugin) (md5 $(md5sum < "$(bash scripts/detect.sh plugin)" | cut -c1-8))"
  echo "genga_herbie: ${GENGA_HERBIE:-auto}   profiles: ${PROFILES:-fresh}"
} > "$RES/config.txt"
grep -h '^\[expected\]' logs/*.log > "$RES/expected_picks.txt" 2>/dev/null || true
[ -f picks_check/picks.tsv ] && [ -n "$PICKS_RC" ] && cp picks_check/picks.tsv "$RES/picks_check.tsv"
cp -r logs "$RES/"
[ -n "$(ls figures 2>/dev/null)" ] && cp -r figures "$RES/"
cp "$COST_MODEL" "$RES/"
[ -f cost_models/target.env ] && cp cost_models/target.env "$RES/"
[ -f cost_models/bootstrap_timing.txt ] && cp cost_models/bootstrap_timing.txt "$RES/"
for d in benchmarks/*/; do
  b="$(basename "$d")"
  files="$(find "$d" -maxdepth 1 \( -name '*.tsv' -o -name 'drift.csv' -o -name 'site_sensitivity.txt' \) 2>/dev/null)"
  [ -n "$files" ] || continue
  mkdir -p "$RES/data/$b"
  echo "$files" | xargs -I{} cp {} "$RES/data/$b/"
  [ -d "$d/traces" ] && cp -r "$d/traces" "$RES/data/$b/"
done
tar -czf "results_${GPU_NAME}_${GPU_ARCH}_${STAMP}.tar.gz" -C results "$(basename "$RES")"
echo "==== results bundle: results_${GPU_NAME}_${GPU_ARCH}_${STAMP}.tar.gz ===="

if [ -n "$FAILED" ]; then
  echo "==== DONE with failures:$FAILED (see logs/) ===="
  exit 1
fi
[ -s "$RES/expected_picks.txt" ] && { echo "==== configurations checked against expected/picks.tsv ===="; cat "$RES/expected_picks.txt"; }
if [ "$OWN_MODEL" = 1 ]; then
  echo "==== this GPU priced with its own model ===="
  echo "  decisions of the shipped models (device-independent): $(grep -E '^== ' logs/picks.log | sed 's/^== //' | paste -sd';' -)"
  [ "$PICKS_RC" = 0 ] || echo "  a shipped decision did not reproduce: see logs/picks.log and $RES/picks_check.tsv"
  echo "  this GPU's own decisions: the [expected] lines above (reported, not compared)"
  echo "  speedups: this GPU's own, in the benchmark tables"
  [ "$PICKS_RC" = 0 ] || exit 1
fi
echo "==== DONE. Expected results: section Expected Results of README.md. ===="
