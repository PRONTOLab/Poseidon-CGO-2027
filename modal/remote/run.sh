#!/usr/bin/env bash
# run.sh <out dir>: the artifact on this GPU, one benchmark at a time, saving
# after each one; a second call with the same <out dir> resumes.
#
# For each benchmark of BENCHES (default: the four run_artifact.sh runs, in
# its order) that has no DONE marker in <out dir>/<bench>/:
#   BENCHES=<bench> ./run_artifact.sh [--quick]
# exactly as slurm/run_artifact.sbatch runs it, then save into <out dir>/<bench>/
# (whatever the benchmark's exit status):
#   run_artifact.log        the driver's output (tee)
#   results_*.tar.gz        run_artifact.sh's own results bundle
#   bench_state.tar.gz      the benchmark directory as the run left it: TSVs,
#                           traces, logs, fresh profiles (fpprofile*/), Herbie
#                           and DP caches (cache*/: Herbie answers, table.json,
#                           budgets.txt, accuracy memos), reports; without
#                           executables, objects and ozp's generated matrices
#   cost_models/            the cost model this run priced with
#   DONE                    exit status and wall time
# The cost model is measured once (run_artifact.sh's bootstrap: this GPU has no
# shipped model) and kept in <out dir>/state/, so a resumed run prices every
# benchmark with the same model.
#
# A benchmark is stopped DEADLINE_MARGIN seconds before DEADLINE (epoch
# seconds; the Modal function's own timeout) so that its partial state is
# saved; it is then marked TIMEOUT, not DONE, and a resumed call runs it again.
#
# Environment: TC, BUNDLE, BENCHES, MODE (full|quick), GENGA_HERBIE
# (auto|replay|run; auto = run when this device has no shipped model and a
# Herbie was built, else replay), DEADLINE, DEADLINE_MARGIN, WORK.
set -uo pipefail
OUT="${1:?usage: run.sh <out dir>}"
TC="${TC:-/toolchain}"
BUNDLE="${BUNDLE:-/bundle}"
WORK="${WORK:-/work}"
MODE="${MODE:-full}"
BENCHES="${BENCHES:-fig2_drift genga_nbody ozp chase_eigensolver}"
GENGA_HERBIE_MODE="${GENGA_HERBIE:-auto}"
DEADLINE="${DEADLINE:-0}"
DEADLINE_MARGIN="${DEADLINE_MARGIN:-2700}"
ART="$WORK/artifact"
STATE="$OUT/state"
log() { echo "[run $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

[ -f "$TC/poseidon.stamp" ] || { log "no toolchain in $TC: run the build stage first"; exit 99; }
mkdir -p "$OUT" "$STATE" "$OUT/provenance"

# ---- a fresh artifact tree at a fixed path ----------------------------------
# The bootstrap records the measured model by absolute path
# (cost_models/target.env), so the tree lives at the same path in every
# container of this run.
rm -rf "$ART" && mkdir -p "$WORK" && tar -C "$WORK" -xzf "$BUNDLE/artifact.tar.gz"
cd "$ART"
mkdir -p "$WORK/tmp"
export TMPDIR="$WORK/tmp"
export POSEIDON_BUILD="$TC/poseidon-build"
export GPU_ID=0
# PY: detect.sh picks the python on PATH that imports numpy and matplotlib.
case "$MODE" in quick) RA_ARGS=(--quick) ;; full) RA_ARGS=() ;;
  *) log "MODE=$MODE: use full or quick"; exit 2 ;; esac

# ---- provenance (once per container) ------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
{
  echo "container_start: $STAMP"
  echo "mode: $MODE   benches: $BENCHES   genga_herbie: $GENGA_HERBIE_MODE"
  echo "llvm: $(cat "$TC/llvm.stamp" 2>/dev/null)"
  echo "poseidon: $(cat "$TC/poseidon.stamp")"
  echo "herbie: $(cat "$TC/herbie.stamp" 2>/dev/null || echo none)"
  echo "nvcc: $(/usr/local/cuda/bin/nvcc --version | sed -n 's/^Cuda compilation tools, //p')"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,power.limit,clocks.max.sm --format=csv
  echo "cpus: $(nproc)   mem: $(free -g | awk '/^Mem:/ {print $2 " GiB"}')"
} > "$OUT/provenance/container_$STAMP.txt" 2>&1
nvidia-smi -q > "$OUT/provenance/nvidia-smi-q_$STAMP.txt" 2>&1 || true
cp -f "$BUNDLE/MANIFEST" "$OUT/provenance/MANIFEST"
cp -f "$POSEIDON_BUILD/CMakeCache.txt" "$OUT/provenance/poseidon-CMakeCache.txt"
cat "$OUT/provenance/container_$STAMP.txt"

# ---- the cost model of this run ---------------------------------------------
if [ -f "$STATE/cost_models/target.env" ]; then
  log "restoring this run's cost model from $STATE/cost_models"
  cp -a "$STATE/cost_models/." cost_models/
else
  log "measuring this GPU's cost model (run_artifact.sh --bootstrap-only)"
  ./run_artifact.sh --bootstrap-only 2>&1 | tee "$OUT/bootstrap.log"
  rc=${PIPESTATUS[0]}
  mkdir -p "$STATE/cost_models"
  for f in cost_models/target.env cost_models/bootstrap_timing.txt; do
    [ -f "$f" ] && cp -f "$f" "$STATE/cost_models/"
  done
  [ -d cost_models/measured ] && cp -a cost_models/measured "$STATE/cost_models/"
  if [ "$rc" != 0 ]; then
    log "the cost-model bootstrap failed (exit $rc); see bootstrap.log"
    rm -rf "$STATE/cost_models"
    exit 98
  fi
  # On an sm_120 card the shipped model applies and nothing is measured.
  [ -f "$STATE/cost_models/target.env" ] || rm -rf "$STATE/cost_models"
fi

# ---- GENGA's Herbie mode ------------------------------------------------------
# The shipped Herbie answers were searched against the RTX 5090 model's
# platform; with this device's own model the drift and unified solves ask
# questions those answers do not hold, so they need a Herbie search.
HERBIE_BIN="$TC/herbie/bin/herbie"
if [ "$GENGA_HERBIE_MODE" = auto ]; then
  if [ -f cost_models/target.env ] && [ -x "$HERBIE_BIN" ]; then GENGA_HERBIE_MODE=run; else GENGA_HERBIE_MODE=replay; fi
fi
export GENGA_HERBIE="$GENGA_HERBIE_MODE"
[ "$GENGA_HERBIE" = run ] && export HERBIE="$HERBIE_BIN"
log "GENGA_HERBIE=$GENGA_HERBIE${HERBIE:+ HERBIE=$HERBIE}"

save_bench() {   # $1 = bench, $2 = status line
  local b="$1" d="$OUT/$1"
  mkdir -p "$d"
  cp -f logs/"$b".log "$d/" 2>/dev/null || true
  find . -maxdepth 1 -name 'results_*.tar.gz' -newer "$d/.started" -exec cp -f {} "$d/" \;
  tar -C benchmarks -czf "$d/bench_state.tar.gz" \
      --exclude='*.exe' --exclude='*.o' --exclude='*.a' --exclude='*.so' --exclude='*.mtx' \
      --exclude="$b/data" --exclude='.run.lock' "$b" 2>/dev/null || true
  rm -rf "$d/cost_models" && mkdir -p "$d/cost_models"
  cp -a cost_models/. "$d/cost_models/" 2>/dev/null || true
  echo "$2" > "$d/${3:-DONE}"
}

FAILED=""
for b in $BENCHES; do
  d="$OUT/$b"
  if [ -f "$d/DONE" ]; then log "$b: already done ($(cat "$d/DONE")); skipped"; continue; fi
  mkdir -p "$d" && rm -f "$d/TIMEOUT" && touch "$d/.started"
  LIMIT=()
  if [ "$DEADLINE" -gt 0 ]; then
    left=$(( DEADLINE - $(date +%s) - DEADLINE_MARGIN ))
    if [ "$left" -le 600 ]; then log "$b: not enough time left in this call; stopping (resume with the same run id)"; break; fi
    LIMIT=(timeout --signal=TERM --kill-after=120 "$left")
  fi
  log "$b: BENCHES=$b ./run_artifact.sh ${RA_ARGS[*]-}"
  T0=$(date +%s)
  BENCHES="$b" "${LIMIT[@]}" ./run_artifact.sh "${RA_ARGS[@]}" 2>&1 | tee "$d/run_artifact.log"
  rc=${PIPESTATUS[0]}
  dt=$(( $(date +%s) - T0 ))
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    log "$b: stopped at the deadline after ${dt}s; saving its partial state"
    save_bench "$b" "timeout after ${dt}s" TIMEOUT
    FAILED="$FAILED $b(timeout)"
    break
  fi
  save_bench "$b" "exit $rc after ${dt}s"
  [ "$rc" = 0 ] || FAILED="$FAILED $b"
  log "$b: exit $rc after ${dt}s; saved into $d"
done

{
  echo "run: $(basename "$OUT")   mode: $MODE   genga_herbie: $GENGA_HERBIE"
  for b in $BENCHES; do
    if [ -f "$OUT/$b/DONE" ]; then echo "$b: $(cat "$OUT/$b/DONE")"
    elif [ -f "$OUT/$b/TIMEOUT" ]; then echo "$b: $(cat "$OUT/$b/TIMEOUT") (resume to rerun)"
    else echo "$b: not run yet (resume)"; fi
  done
} | tee "$OUT/SUMMARY.txt"
[ -z "$FAILED" ] || { log "failed:$FAILED"; exit 1; }
exit 0
