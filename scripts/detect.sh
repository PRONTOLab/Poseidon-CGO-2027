#!/usr/bin/env bash
# detect.sh export|mkfrag|sources|<var>: resolve the artifact configuration
# (environment > local.env > cost_models/target.env > auto-detection).
set -o pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ -f "$ART/local.env" ] && . "$ART/local.env"
[ -f "$ART/cost_models/target.env" ] && . "$ART/cost_models/target.env"

fail() { echo "detect.sh: error: $*" >&2; exit 1; }
warn() { echo "detect.sh: warning: $*" >&2; }

SOURCES_DIR="$ART/sources"

SHIPPED_COST_MODEL="$ART/devices/rtx5090/cost_models/cm_sm_120_RTX5090.csv"

model_arch() {   # $1 = cost model csv -> its native_arch header
  sed -n 's/^# native_arch=//p' "$1" 2>/dev/null | head -1 | tr -d ' '
}

resolve_gpu_id() {
  if [ -n "${GPU_ID:-}" ]; then echo "$GPU_ID"; return; fi
  [ "${POSEIDON_ARTIFACT_NO_GPU:-0}" = 1 ] && { echo none; return; }
  # Under Slurm, an index outside CUDA_VISIBLE_DEVICES lands outside the allocation.
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "${CUDA_VISIBLE_DEVICES%%,*}"; return; fi
  echo 0
}

resolve_build_dir() {
  [ -n "${POSEIDON_BUILD:-}" ] || fail "set POSEIDON_BUILD=<Enzyme build tree configured with -DENZYME_POSEIDON=ON> (README: Build From Source)"
  [ -f "$POSEIDON_BUILD/CMakeCache.txt" ] || fail "POSEIDON_BUILD=$POSEIDON_BUILD is not a build tree"
  (cd "$POSEIDON_BUILD" && pwd)
}

# The reference programs (plain clang) compile the header and GEMM runtimes of the source tree.
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

resolve_poseidon_tool() {   # $1 = build dir, $2 = tool name
  local bin
  bin="$(ls -t "$1"/Poseidon/bin/"$2" 2>/dev/null | head -1)"
  [ -n "$bin" ] || bin="$(command -v "$2" || true)"
  [ -n "$bin" ] || fail "$2 not found under $1/Poseidon/bin or on PATH - build the compiler first (README: Build From Source)"
  echo "$bin"
}

# A build tree does not copy poseidon-explain next to the driver; an installation does.
resolve_poseidon_explain() {
  local b r
  b="$(resolve_build_dir)" || exit 1
  for r in "$b/Poseidon/bin/poseidon-explain" "$(command -v poseidon-explain || true)" \
           "$(resolve_poseidon_root)/tools/explain/poseidon-explain"; do
    [ -n "$r" ] && [ -f "$r" ] && { echo "$r"; return; }
  done
  fail "poseidon-explain not found in $b/Poseidon/bin, on PATH or under the source tree's tools/explain"
}

# Recorded only: poseidon-clang++ loads the plugin of its own tree.
resolve_plugin() {   # $1 = driver
  ls "$(dirname "$1")"/../lib/Poseidon-*.so 2>/dev/null | head -1
}

resolve_clang() {   # $1 = build dir
  if [ -n "${CLANG:-}" ]; then
    [ -x "$CLANG" ] || fail "CLANG=$CLANG is not executable"
    echo "$CLANG"; return
  fi
  # Only the clang++ of the LLVM the build used is plugin-ABI-compatible.
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
    fail "nvidia-smi not found; set GPU_ARCH (e.g. sm_80); on a cluster, run inside a GPU allocation"
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

# Empty when this device has no model yet: another device's model gives that device's decisions.
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
  local v; for v in $VARS; do printf '%s ?= %s\n' "$v" "$(value_of "$v")"; done
}

case "${1:-}" in
  export) cmd_export ;;
  mkfrag) cmd_mkfrag ;;
  poseidon_root) resolve_poseidon_root ;;
  build_dir) resolve_build_dir ;;
  poseidon_clangxx) resolve_poseidon_tool "$(resolve_build_dir)" poseidon-clang++ ;;
  poseidon_calibrate) resolve_poseidon_tool "$(resolve_build_dir)" poseidon-calibrate ;;
  poseidon_explain) resolve_poseidon_explain ;;
  sources) [ -d "$SOURCES_DIR" ] || fail "no sources/ directory at $SOURCES_DIR"; (cd "$SOURCES_DIR" && pwd) ;;
  plugin) resolve_plugin "$(resolve_poseidon_tool "$(resolve_build_dir)" poseidon-clang++)" ;;
  clang) resolve_clang "$(resolve_build_dir)" ;;
  cuda) resolve_cuda ;;
  gpu_id) resolve_gpu_id ;;
  gpu_arch) resolve_gpu_arch "$(resolve_gpu_id)" ;;
  gpu_name) resolve_gpu_name "$(resolve_gpu_id)" ;;
  py) resolve_py ;;
  cost_model) resolve_cost_model "$(resolve_gpu_arch "$(resolve_gpu_id)" 2>/dev/null || true)" ;;
  shipped_cost_model) echo "$SHIPPED_COST_MODEL" ;;
  *) fail "unknown query '${1:-}'" ;;
esac
