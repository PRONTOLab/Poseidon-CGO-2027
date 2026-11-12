#!/usr/bin/env bash
# detect.sh — single source of truth for artifact configuration.
#
# Both config.mk (via the generated .detected.mk fragment) and env.sh (via
# `detect.sh export`) resolve every knob through this script, so make targets
# and directly-invoked pipeline scripts always agree.
#
# Resolution order for every variable:
#   caller environment > local.env (user pins) > cost_models/target.env
#   (written by scripts/bootstrap_cost_model.sh) > auto-detection
# Both files use guard-style assignments (VAR="${VAR:-...}"), so a variable
# already set in the environment always wins.
#
# The cost model. On a GPU of the architecture the shipped model was measured
# on (cost_models/cm_sm_120_RTX5090.csv, native_arch sm_120) that model is the
# default. Anywhere else there is no default: another device's prices give that
# device's decisions, so COST_MODEL resolves empty and every consumer stops with
# the command that measures this device (run_artifact.sh does it by itself).
#
# POSEIDON_ARTIFACT_NO_GPU=1 resolves without touching a GPU (no nvidia-smi):
# GPU_ARCH defaults to the shipped model's architecture and GPU_ID to "none".
# The compile-only pick check (scripts/check_picks.sh) runs that way.
#
# Usage:
#   detect.sh export    # print `export VAR=...` lines (eval'd by env.sh)
#   detect.sh mkfrag    # print a `VAR ?= ...` make fragment (config.mk)
#   detect.sh <var>     # print one resolved value (poseidon_root, build_dir,
#                       #   clang, cuda, gpu_id, gpu_arch, gpu_name, py,
#                       #   poseidon_clangxx, poseidon_calibrate, plugin,
#                       #   cost_model, shipped_cost_model)
set -o pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -f "$ART/local.env" ] && . "$ART/local.env"
[ -f "$ART/cost_models/target.env" ] && . "$ART/cost_models/target.env"

fail() { echo "detect.sh: error: $*" >&2; exit 1; }
warn() { echo "detect.sh: warning: $*" >&2; }

# The RTX 5090 model the paper's decisions were solved with, and the Herbie
# platform generated from it, ship with the artifact.
SHIPPED_COST_MODEL="$ART/cost_models/cm_sm_120_RTX5090.csv"

model_arch() {   # $1 = cost model csv -> its native_arch header
  sed -n 's/^# native_arch=//p' "$1" 2>/dev/null | head -1 | tr -d ' '
}

resolve_gpu_id() {
  if [ -n "${GPU_ID:-}" ]; then echo "$GPU_ID"; return; fi
  [ "${POSEIDON_ARTIFACT_NO_GPU:-0}" = 1 ] && { echo none; return; }
  # Under Slurm (or any pre-constrained session) CUDA_VISIBLE_DEVICES is the
  # authority; selecting any other index would land outside the allocation.
  # Index or UUID entries both work for CUDA_VISIBLE_DEVICES and nvidia-smi -i.
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "${CUDA_VISIBLE_DEVICES%%,*}"; return; fi
  echo 0
}

# The Enzyme build tree that configured with -DENZYME_POSEIDON=ON: it holds the
# Poseidon plugin, the driver and the Enzyme plugin side by side, and its
# CMakeCache names the source tree.
resolve_build_dir() {
  [ -n "${POSEIDON_BUILD:-}" ] || fail "set POSEIDON_BUILD=<Enzyme build tree configured with -DENZYME_POSEIDON=ON> (README: Setup)"
  [ -f "$POSEIDON_BUILD/CMakeCache.txt" ] || fail "POSEIDON_BUILD=$POSEIDON_BUILD is not a build tree"
  (cd "$POSEIDON_BUILD" && pwd)
}

# The source tree supplies the public header and the GEMM runtime sources the
# reference arms (plain clang, no Poseidon) compile; the driver itself takes
# everything else from its own build tree.
resolve_poseidon_root() {
  local r="${POSEIDON_ROOT:-}"
  if [ -z "$r" ]; then
    local b; b="$(resolve_build_dir)" || exit 1
    r="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$b/CMakeCache.txt")/Poseidon"
  fi
  [ -f "$r/runtime/ozaki/ozaki_rt.cu" ] && [ -f "$r/../include/poseidon/poseidon.h" ] ||
    fail "$r is not the enzyme/Poseidon source tree of the build (no runtime/ozaki/ozaki_rt.cu or ../include/poseidon/poseidon.h); set POSEIDON_ROOT"
  (cd "$r" && pwd)
}

# The driver and the calibration tool are built next to each other; either the
# build tree or an installed Poseidon supplies both.
resolve_poseidon_tool() {   # $1 = build dir, $2 = tool name
  local bin
  bin="$(ls -t "$1"/Poseidon/bin/"$2" 2>/dev/null | head -1)"
  [ -n "$bin" ] || bin="$(command -v "$2" || true)"
  [ -n "$bin" ] || fail "$2 not found under $1/Poseidon/bin or on PATH - build the compiler first (README: Setup)"
  echo "$bin"
}

# Recorded in config.txt only: poseidon-clang++ loads the plugin of the tree it
# runs from, and nothing here passes a plugin path to it.
resolve_plugin() {   # $1 = driver
  ls "$(dirname "$1")"/../lib/Poseidon-*.so 2>/dev/null | head -1
}

resolve_clang() {   # $1 = build dir
  if [ -n "${CLANG:-}" ]; then
    [ -x "$CLANG" ] || fail "CLANG=$CLANG is not executable"
    echo "$CLANG"; return
  fi
  # The build already knows which LLVM it was compiled against; its clang++ is
  # the only one guaranteed plugin-ABI-compatible.
  local llvmdir cand
  llvmdir="$(sed -n 's/^LLVM_DIR:[A-Z]*=//p' "$1/CMakeCache.txt" | head -1)"
  cand="${llvmdir%/lib/cmake/llvm}/bin/clang++"
  [ -x "$cand" ] && { echo "$cand"; return; }
  cand="$(command -v clang++ || true)"
  if [ -n "$cand" ]; then
    warn "using clang++ from PATH ($cand); it must be the LLVM build the Poseidon plugin was compiled against"
    echo "$cand"; return
  fi
  fail "no clang++ found; set CLANG to the clang++ of the LLVM build matching the Poseidon plugin"
}

resolve_cuda() {
  if [ -n "${CUDA:-}" ]; then echo "$CUDA"; return; fi
  # CUDA_HOME/CUDA_PATH cover module-managed systems (e.g. NERSC cudatoolkit).
  local c
  for c in "${CUDA_HOME:-}" "${CUDA_PATH:-}"; do
    [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { echo "$c"; return; }
  done
  c="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)"
  [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { echo "$c"; return; }
  [ -x /usr/local/cuda/bin/nvcc ] && { echo /usr/local/cuda; return; }
  c="$(command -v nvcc || true)"
  [ -n "$c" ] && { dirname "$(dirname "$c")"; return; }
  fail "CUDA toolkit not found; set CUDA=/path/to/cuda (or module load cudatoolkit)"
}

resolve_gpu_arch() {   # $1 = gpu id
  if [ -n "${GPU_ARCH:-}" ]; then echo "$GPU_ARCH"; return; fi
  if [ "${POSEIDON_ARTIFACT_NO_GPU:-0}" = 1 ]; then
    model_arch "${COST_MODEL:-$SHIPPED_COST_MODEL}"; return
  fi
  command -v nvidia-smi >/dev/null 2>&1 ||
    fail "nvidia-smi not found; set GPU_ARCH (e.g. sm_80) — on a cluster, run inside a GPU allocation"
  local cc
  cc="$(nvidia-smi -i "$1" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  [ -n "$cc" ] || fail "cannot query compute capability of GPU '$1' (no GPU visible? wrong GPU_ID? on a cluster login node, run inside a GPU allocation or set GPU_ARCH)"
  echo "sm_${cc/./}"
}

resolve_gpu_name() {   # $1 = gpu id
  local n=""
  [ "${POSEIDON_ARTIFACT_NO_GPU:-0}" = 1 ] ||
    n="$(nvidia-smi -i "$1" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  [ -n "$n" ] || { echo unknown; return; }
  echo "$n" | sed -E 's/NVIDIA|GeForce|Tesla|Quadro//g; s/[^A-Za-z0-9]//g'
}

resolve_py() {
  local p
  if [ -n "${PY:-}" ]; then
    "$PY" -c 'import numpy, matplotlib' >/dev/null 2>&1 ||
      fail "PY=$PY cannot import numpy+matplotlib"
    echo "$PY"; return
  fi
  for p in python3 python; do
    command -v "$p" >/dev/null 2>&1 || continue
    "$p" -c 'import numpy, matplotlib' >/dev/null 2>&1 && { command -v "$p"; return; }
  done
  fail "no python with numpy+matplotlib found; set PY (e.g. a conda env's python)"
}

# Empty when this device has no model yet; never another device's model.
resolve_cost_model() {   # $1 = gpu arch
  if [ -n "${COST_MODEL:-}" ]; then
    [ -f "$COST_MODEL" ] || fail "COST_MODEL=$COST_MODEL does not exist"
    echo "$COST_MODEL"; return
  fi
  [ -f "$SHIPPED_COST_MODEL" ] || fail "the shipped cost model $SHIPPED_COST_MODEL is missing"
  if [ -n "$1" ] && [ "$1" = "$(model_arch "$SHIPPED_COST_MODEL")" ]; then
    echo "$SHIPPED_COST_MODEL"; return
  fi
  echo ""
}

resolve_all() {
  ROOT_="$(resolve_poseidon_root)" || exit 1
  BUILD_="$(resolve_build_dir)" || exit 1
  CLANG_="$(resolve_clang "$BUILD_")" || exit 1
  CUDA_="$(resolve_cuda)" || exit 1
  GID_="$(resolve_gpu_id)"
  ARCH_="$(resolve_gpu_arch "$GID_")" || exit 1
  PY_="$(resolve_py)" || exit 1
  PCLANG_="$(resolve_poseidon_tool "$BUILD_" poseidon-clang++)" || exit 1
  PCAL_="$(resolve_poseidon_tool "$BUILD_" poseidon-calibrate)" || exit 1
  CM_="$(resolve_cost_model "$ARCH_")" || exit 1
  PINC_="$(cd "$ROOT_/.." && pwd)/include"
  OZRT_="$ROOT_/runtime/ozaki/ozaki_rt.cu"
  TCRT_="$ROOT_/runtime/tcec/tcec_rt.cu"
  DRRT_="$ROOT_/runtime/direct/direct_rt.cu"
}

VARS="POSEIDON_ROOT CLANG CUDA GPU_ARCH GPU_ID PY POSEIDON_CLANGXX POSEIDON_CALIBRATE POSEIDON_INC OZAKI_RT TCEC_RT DIRECT_RT COST_MODEL"
value_of() {
  case "$1" in
    POSEIDON_ROOT) echo "$ROOT_" ;;   CLANG) echo "$CLANG_" ;;   CUDA) echo "$CUDA_" ;;
    GPU_ARCH) echo "$ARCH_" ;;        GPU_ID) echo "$GID_" ;;     PY) echo "$PY_" ;;
    POSEIDON_CLANGXX) echo "$PCLANG_" ;; POSEIDON_CALIBRATE) echo "$PCAL_" ;;
    POSEIDON_INC) echo "$PINC_" ;;    OZAKI_RT) echo "$OZRT_" ;;  TCEC_RT) echo "$TCRT_" ;;
    DIRECT_RT) echo "$DRRT_" ;;       COST_MODEL) echo "$CM_" ;;
  esac
}

cmd_export() {
  resolve_all
  local v; for v in $VARS; do printf 'export %s=%q\n' "$v" "$(value_of "$v")"; done
}

cmd_mkfrag() {
  resolve_all
  # Make fragment; paths with whitespace are unsupported by make itself.
  local v; for v in $VARS; do printf '%s ?= %s\n' "$v" "$(value_of "$v")"; done
}

case "${1:-}" in
  export) cmd_export ;;
  mkfrag) cmd_mkfrag ;;
  poseidon_root) resolve_poseidon_root ;;
  build_dir) resolve_build_dir ;;
  poseidon_clangxx) resolve_poseidon_tool "$(resolve_build_dir)" poseidon-clang++ ;;
  poseidon_calibrate) resolve_poseidon_tool "$(resolve_build_dir)" poseidon-calibrate ;;
  plugin) resolve_plugin "$(resolve_poseidon_tool "$(resolve_build_dir)" poseidon-clang++)" ;;
  clang) resolve_clang "$(resolve_build_dir)" ;;
  cuda) resolve_cuda ;;
  gpu_id) resolve_gpu_id ;;
  gpu_arch) resolve_gpu_arch "$(resolve_gpu_id)" ;;
  gpu_name) resolve_gpu_name "$(resolve_gpu_id)" ;;
  py) resolve_py ;;
  cost_model) resolve_cost_model "$(resolve_gpu_arch "$(resolve_gpu_id)" 2>/dev/null || true)" ;;
  shipped_cost_model) echo "$SHIPPED_COST_MODEL" ;;
  *) fail "unknown query '${1:-}' (see header for usage)" ;;
esac
