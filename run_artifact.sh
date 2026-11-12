#!/usr/bin/env bash
# run_artifact.sh: single-command reproduction driver.
#
#   ./run_artifact.sh                  # preflight + cost model + full paper
#                                      #   pipeline + results bundle (hours)
#   ./run_artifact.sh --quick          # preflight + cost model + smoke tests
#   ./run_artifact.sh --check          # preflight only (nothing runs on the GPU)
#   ./run_artifact.sh --picks          # compile-only check of every RTX 5090
#                                      #   decision of EXPECTED_RESULTS.md (no GPU)
#   ./run_artifact.sh --picks --device rtx5090|b300|gh200|all
#                                      # the same for the saved cost models of
#                                      #   the paper's Table 1 devices (no GPU)
#   ./run_artifact.sh --explain <row> [--all|--format F]
#                                      # what the compiler considered, applied
#                                      #   and rejected for one row (no GPU);
#                                      #   --explain --list names the rows
#   ./run_artifact.sh --bootstrap-only # preflight + cost model, then stop
#   ./run_artifact.sh --calibrate      # measure this GPU's own cost model even
#                                      #   where the shipped one applies, then run
#   ./run_artifact.sh --from-scratch   # use nothing precomputed: measured cost
#                                      #   model, fresh profiles, Herbie searches,
#                                      #   no shipped solve caches (needs Herbie
#                                      #   with --num-taylor-terms); with --quick too
#   ./run_artifact.sh --no-shipped-cache  # (with any mode) solve without the
#                                      #   shipped DP tables and accuracy memos
#   ./run_artifact.sh --force          # start even though a benchmark of this
#                                      #   artifact is already running (unsafe)
#
#   BENCHES="ozp chase_eigensolver" ./run_artifact.sh   # subset of benchmarks
#   PARALLEL_GPUS="0 1 2 3" ./run_artifact.sh           # one bench per GPU (multi-GPU box)
#   GPU_ID=1 ./run_artifact.sh                          # the GPU to use (default 0,
#                                                       #   or Slurm's CUDA_VISIBLE_DEVICES)
#   GENGA_HERBIE=run, PROFILES=shipped                  # see README: Choices
#
# PARALLEL_GPUS names the devices this run owns and gives each benchmark one of
# them; with more benchmarks than devices they run in waves, never two at once
# on one device. BENCHES order decides which benchmark gets which device.
#
# The cost model is settled for the selected GPU BEFORE any experiment
# (scripts/bootstrap_cost_model.sh): on an sm_120 GPU (RTX 5090 class) the
# shipped RTX 5090 model is checked against the device and used, unless
# --calibrate; on any other GPU this device is measured with poseidon-calibrate
# (~15-30 min) into cost_models/measured/ and cost_models/target.env records it.
# A run priced with a measured model makes that model's decisions: it checks
# the shipped decisions compile-only at the end (--picks --device all; the
# quick run: rtx5090) and reports its speedups as this GPU's, unjudged.
# Results (logs, TSVs, figures, the cost model, a config snapshot) are collected
# into results/<device>_<timestamp>/ and tarred for transfer.
#
# On Slurm systems, submit from the artifact root inside a GPU allocation:
#   sbatch -A <account> slurm/run_artifact.sbatch [--quick]           # one GPU
#   sbatch -A <account> slurm/run_artifact_parallel.sbatch [--quick]  # one per GPU
set -uo pipefail
cd "$(dirname "$0")"
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

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
    --no-shipped-cache) export POSEIDON_SHIPPED_CACHE=0 ;;
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

# ---- from scratch: nothing precomputed --------------------------------------
# The cost model is measured on this GPU (from the ozp surrogate profile this
# run makes), every benchmark profiles on this GPU, GENGA's drift and unified
# solves search with Herbie instead of replaying answers, and no shipped or
# earlier solve cache is read.
if [ "$FROM_SCRATCH" = 1 ]; then
  case "$MODE" in full|quick) ;; *) echo "run_artifact: --from-scratch is a GPU run (full or --quick)" >&2; exit 1 ;; esac
  BOOTSTRAP_ARGS="--calibrate --fresh-profile"
  export PROFILES=fresh GENGA_HERBIE=run POSEIDON_SHIPPED_CACHE=0
  _BUILD="$(bash scripts/detect.sh build_dir)" || exit 1
  HERBIE="${HERBIE:-$(sed -n 's/^POSEIDON_HERBIE_BINARY:FILEPATH=//p' "$_BUILD/CMakeCache.txt")}"
  if [ -z "$HERBIE" ] || [ ! -x "$HERBIE" ] ||
     ! "$HERBIE" improve --help 2>&1 | grep -q -- '--num-taylor-terms'; then
    echo "run_artifact: --from-scratch runs Herbie, and needs one that accepts --num-taylor-terms." >&2
    echo "run_artifact: none at '${HERBIE:-<unset>}' (HERBIE, else the build's POSEIDON_HERBIE_BINARY)." >&2
    echo "run_artifact: Build it from sources/herbie with sources/herbie-num-taylor-terms.patch:" >&2
    echo "run_artifact:   scripts/build_toolchain.sh --with-herbie --prefix <prefix>   (Racket 8.15, Rust 1.85.1)" >&2
    echo "run_artifact:   or: scripts/docker.sh build --herbie   (docker/Dockerfile.herbie)" >&2
    echo "run_artifact: then HERBIE=<prefix>/herbie/bin/herbie ./run_artifact.sh --from-scratch" >&2
    exit 1
  fi
  export HERBIE
  echo "==== from scratch: Herbie $HERBIE; removing earlier solve caches ===="
  for _c in benchmarks/*/cache*/; do
    [ -d "$_c" ] && { echo "  rm -r $_c"; rm -r -- "${_c:?}"; }
  done
fi

# On a multi-GPU box PARALLEL_GPUS names the devices the caller owns. The
# preflight and the cost-model bootstrap run on ONE device, and detect.sh
# defaults that to id 0, which on a shared box is a GPU the caller may neither
# own nor have to itself. Take the first of the list instead, unless GPU_ID was
# pinned explicitly.
if [ -z "${GPU_ID:-}" ] && [ -n "${PARALLEL_GPUS:-}" ]; then
  read -ra _BOOT_GPUS <<< "$PARALLEL_GPUS"
  export GPU_ID="${_BOOT_GPUS[0]}"
fi

# ---- preflight (subshell: nothing resolved here may leak into the run, the
# ---- cost model in particular is only final after the bootstrap) ------------
echo "==== preflight ===="
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
  echo "preflight FAILED: fix the reported item (environment variable or local.env pin; README: Configuration)." >&2
  echo "On a cluster login node with no GPU visible, run inside a GPU allocation (slurm/run_artifact.sbatch)." >&2
  exit 1
}
[ "$MODE" = check ] && { echo "==== preflight OK ===="; exit 0; }

# ---- nothing of this artifact may already be running -----------------------
# A second copy of a benchmark in the same directory overwrites the first one's
# outputs and shares its GPU, and neither is reported: it looks like a slow run,
# not like a collision. The cheapest reliable check is which processes are
# sitting inside benchmarks/ right now.
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
    echo "run_artifact: or pass --force if you know they are someone else's copy." >&2
    exit 1
  fi
fi


# ---- cost model bootstrap (own process; may write cost_models/target.env) ----
echo "==== cost model ===="
bash scripts/bootstrap_cost_model.sh $BOOTSTRAP_ARGS || exit 1
[ "$MODE" = bootstrap ] && exit 0

# Only now is the configuration final for this device.
source ./env.sh || exit 1
GPU_NAME="$(bash scripts/detect.sh gpu_name)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RES="results/${GPU_NAME}_${GPU_ARCH}_${STAMP}"
mkdir -p logs figures "$RES"

OWN_MODEL=0
if ! cmp -s "$COST_MODEL" "$(bash scripts/detect.sh shipped_cost_model)"; then
  OWN_MODEL=1
  echo "NOTE: this run prices with a cost model measured on this device ($(basename "$COST_MODEL"))."
  echo "      Its decisions are this model's own; they are reported, not compared with the"
  echo "      RTX 5090's. After the benchmarks the run checks the shipped decisions"
  echo "      compile-only (the same as ./run_artifact.sh --picks --device ...), and its"
  echo "      speedups are this GPU's, not judged against the RTX 5090 numbers."
fi

# ---- benchmarks -------------------------------------------------------------
BENCHES="${BENCHES:-fig2_drift genga_nbody ozp chase_eigensolver}"
bench_cmd() {
  case "${1}_${MODE}" in
    fig2_drift_*) echo "make all" ;;
    *_quick)      echo "bash quick.sh" ;;
    *)            echo "bash run_all.sh" ;;
  esac
}

# One runner per benchmark directory, enforced with a non-blocking lock: the
# second copy refuses instead of silently interleaving its outputs with the
# first's. .run.pid records the holder so the refusal can name it.
run_bench () {   # $1 = bench, $2 = GPU id ("" to leave GPU_ID alone), $3 = command
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
  if [ -n "$2" ]; then GPU_ID="$2" eval "$3"; else eval "$3"; fi
}

FAILED=""
if [ -n "${PARALLEL_GPUS:-}" ]; then
  # One benchmark per GPU, in waves of at most one benchmark per device: the
  # k-th benchmark of a wave gets the k-th device, and the surplus waits for a
  # device to come free. Every running benchmark therefore has its device to
  # itself and its kernel timings stay comparable with a sequential run; only
  # host-side compiles overlap, and the results bundle is unchanged. Waves
  # matter because a device list SHORTER than the benchmark list is the normal
  # case (four benchmarks, a two- or three-GPU allocation): two benchmarks
  # sharing a device time each other's kernels, and nothing in the output says
  # so. Benchmark order decides the mapping, so name BENCHES in the order the
  # devices should be handed out.
  read -ra _GPUS <<< "$PARALLEL_GPUS"
  read -ra _BENCHES <<< "$BENCHES"
  _NG=${#_GPUS[@]}
  _WAVES=$(( (${#_BENCHES[@]} + _NG - 1) / _NG ))
  for (( s=0; s<${#_BENCHES[@]}; s+=_NG )); do
    [ "$_WAVES" -gt 1 ] && echo "==== wave $(( s/_NG + 1 )) of $_WAVES ===="
    _PIDS=(); _NAMES=()
    for (( k=0; k<_NG && s+k<${#_BENCHES[@]}; k++ )); do
      b="${_BENCHES[$((s+k))]}"; g="${_GPUS[$k]}"
      CMD="$(bench_cmd "$b")"
      echo "==== [$b] $CMD on GPU $g (log: logs/$b.log) ===="
      ( run_bench "$b" "$g" "$CMD" ) > "logs/$b.log" 2>&1 &
      _PIDS+=($!); _NAMES+=("$b")
    done
    for j in "${!_PIDS[@]}"; do
      if wait "${_PIDS[$j]}"; then
        echo "==== [${_NAMES[$j]}] OK ===="
      else
        echo "==== [${_NAMES[$j]}] FAILED (tail of logs/${_NAMES[$j]}.log below) ===="
        tail -20 "logs/${_NAMES[$j]}.log"
        FAILED="$FAILED ${_NAMES[$j]}"
      fi
    done
  done
else
  for b in $BENCHES; do
    CMD="$(bench_cmd "$b")"
    echo "==== [$b] $CMD (log: logs/$b.log) ===="
    if ( run_bench "$b" "" "$CMD" ) > "logs/$b.log" 2>&1; then
      echo "==== [$b] OK ===="
    else
      echo "==== [$b] FAILED (tail of logs/$b.log below) ===="
      tail -20 "logs/$b.log"
      FAILED="$FAILED $b"
    fi
  done
fi

# ---- device-independent checks on a GPU priced with its own model -----------
# The decisions of the shipped models are a compile-time function of the
# compiler, the profiles and those models, so they are checked here as on any
# machine; the GPU's own decisions and timings above are its own.
PICKS_RC=""
if [ "$OWN_MODEL" = 1 ]; then
  _DEVS=all; [ "$MODE" = quick ] && _DEVS=rtx5090
  echo "==== decisions of the shipped cost models, compile-only (--picks --device $_DEVS; log: logs/picks.log) ===="
  ( unset COST_MODEL GPU_ARCH; bash scripts/check_picks.sh --device "$_DEVS" ) > logs/picks.log 2>&1
  PICKS_RC=$?
  grep -E '^== ' logs/picks.log
fi

# ---- results bundle ---------------------------------------------------------
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
  echo "genga_herbie: ${GENGA_HERBIE:-replay}   profiles: ${PROFILES:-fresh}"
} > "$RES/config.txt"
# Every row whose configuration a run checked against EXPECTED_RESULTS.md.
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
[ -s "$RES/expected_picks.txt" ] && { echo "==== configurations checked against EXPECTED_RESULTS.md ===="; cat "$RES/expected_picks.txt"; }
if [ "$OWN_MODEL" = 1 ]; then
  echo "==== this GPU priced with its own model ===="
  echo "  decisions of the shipped models (device-independent): $(grep -E '^== ' logs/picks.log | sed 's/^== //' | paste -sd';' -)"
  [ "$PICKS_RC" = 0 ] || echo "  a shipped decision did not reproduce: see logs/picks.log and $RES/picks_check.tsv"
  echo "  this GPU's own decisions: the [expected] lines above (reported, not compared)"
  echo "  speedups: this GPU's own, in the benchmark tables; EXPECTED_RESULTS.md's speedup"
  echo "  tolerances are for an RTX 5090. What carries over is the paper's qualitative claims."
  [ "$PICKS_RC" = 0 ] || exit 1
fi
echo "==== DONE. Compare the bundle against EXPECTED_RESULTS.md. ===="
