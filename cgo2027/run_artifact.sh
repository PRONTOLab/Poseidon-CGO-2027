#!/usr/bin/env bash
# run_artifact.sh — single-command reproduction driver.
#
#   ./run_artifact.sh                  # preflight + cost model + full paper
#                                      #   pipeline + results bundle (hours)
#   ./run_artifact.sh --quick          # preflight + cost model + smoke tests
#   ./run_artifact.sh --check          # preflight only (nothing runs on the GPU)
#   ./run_artifact.sh --picks          # compile-only check of every decision in
#                                      #   EXPECTED_RESULTS.md (no GPU needed)
#   ./run_artifact.sh --bootstrap-only # preflight + cost model, then stop
#   ./run_artifact.sh --calibrate      # measure this GPU's own cost model even
#                                      #   where the shipped one applies, then run
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
# Results (logs, TSVs, figures, the cost model, a config snapshot) are collected
# into results/<device>_<timestamp>/ and tarred for transfer.
#
# On Slurm systems, submit from the artifact root inside a GPU allocation:
#   sbatch -A <account> slurm/run_artifact.sbatch [--quick]           # one GPU
#   sbatch -A <account> slurm/run_artifact_parallel.sbatch [--quick]  # one per GPU
set -uo pipefail
cd "$(dirname "$0")"

MODE=full BOOTSTRAP_ARGS="" FORCE_RUN=0
for _a in "$@"; do
  case "$_a" in
    --quick) MODE=quick ;;
    --check) MODE=check ;;
    --picks) MODE=picks ;;
    --bootstrap-only) MODE=bootstrap ;;
    --calibrate) BOOTSTRAP_ARGS=--calibrate ;;
    --force) FORCE_RUN=1 ;;
    *) sed -n '2,24p' "$0"; exit 1 ;;
  esac
done

if [ "$MODE" = picks ]; then
  exec bash scripts/check_picks.sh
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
  echo "preflight FAILED — fix the reported item (environment variable or local.env pin; README: Configuration)." >&2
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

if ! cmp -s "$COST_MODEL" "$(bash scripts/detect.sh shipped_cost_model)"; then
  echo "NOTE: this run prices with a cost model measured on this device ($(basename "$COST_MODEL"))."
  echo "      The solver's picks are that model's and may differ from EXPECTED_RESULTS.md,"
  echo "      whose picks are the shipped RTX 5090 model's; every run prints, per row,"
  echo "      whether the configuration differs. See EXPECTED_RESULTS.md: Own cost model."
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
echo "==== DONE. Compare the bundle against EXPECTED_RESULTS.md. ===="
