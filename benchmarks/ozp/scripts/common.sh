#!/bin/bash
# Shared configuration for the ozp pipeline scripts. Config (compiler, CUDA,
# GPU_ID, python, the Ozaki-II runtime source, and the artifact-wide COST_MODEL)
# comes from the artifact-wide env.sh, which auto-detects the toolchain
# (scripts/detect.sh) and is overridable from the environment or local.env, so
# `make` and a direct `bash scripts/<x>.sh` see the same values.
#
# PROFILES=fresh (default) | shipped: the solves read this run's N=512 profile
# (fpprofile/) or the shipped one (profiles/surrogate512/, with which the
# expected-results table's decisions were made).

set -o pipefail

ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ART/env.sh"
source "$ART/scripts/expected.sh"

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROF_N="${PROF_N:-512}"       # SCVH surrogate profile size
APPLY_N="${APPLY_N:-2048}"    # deploy size (paper figures)
NUM_RUNS="${NUM_RUNS:-5}"     # avg-of-5 wall-clock (artifact convention)
PROFILES="${PROFILES:-fresh}"
case "$PROFILES" in
  fresh) PROF=fpprofile ;;
  shipped) PROF=profiles/surrogate512 ;;
  *) echo "PROFILES=$PROFILES: use fresh or shipped" >&2; exit 2 ;;
esac

cd "$BENCH_DIR"
# Executables and their logs stay inside the benchmark directory.
TMPD="$BENCH_DIR/tmp"; mkdir -p "$TMPD"

# ── compiler flag groups (bash arrays; zsh mis-splits unquoted flag vars) ────
# poseidon-clang++ supplies the two pass plugins, Poseidon's include directory,
# -ffp-contract=on, the profiler under -poseidon-profile-generate and the GEMM
# runtimes under -poseidon-profile-use, so only what is this benchmark's own is
# spelled out here.
#  -fcuda-rdc : the annotated kernels are compiled as relocatable device code.
# The reference and baseline arms are built by plain clang++, so the include
# directory Poseidon's header lives in is named here as well.
COMMON_CUDA=( -x cuda --cuda-gpu-arch="$GPU_ARCH" --cuda-path="$CUDA" -fcuda-rdc
  -O3 -Wno-deprecated-declarations -I"$POSEIDON_INC"
  -L"$CUDA/lib64" -lcudart -lcublas -lstdc++ -lm )

# The source selects its annotated kernel on -DPOSEIDON.
POSE=( -DPOSEIDON )

# Solver flags shared by every ozp solve: matmul-only (no Herbie, no scalar PT),
# tensor-core raising on, and COST_MODEL (the shipped RTX 5090 model, or the
# one the bootstrap measured on this device).
FP=( -poseidon-profile-use=./"$PROF" -poseidon-enable-herbie=0
  -poseidon-enable-pt=0 -poseidon-cache=cache
  -poseidon-cost-model="$COST_MODEL"
  -poseidon-loose-coverage -poseidon-raise-wmma )

# Host-side GEMM substitution. The solver may pick any of the three dispatch
# families; poseidon-clang++ links all three runtimes (and cuMpSGEMM when the
# compiler was configured with it).
DISPATCH=( -poseidon-ozaki-host-dispatch )
DISPATCH_RT=()

# The applied matrix-product scheme of a compile, from its pin line.
pick_of_log() { pins_of_log "$1" | sed 's/^.*:M0=//' | head -1; }

echo "[config] POSEIDON_ROOT=$POSEIDON_ROOT GPU_ID=$GPU_ID arch=$GPU_ARCH" >&2
echo "[config] COST_MODEL=${COST_MODEL#$ART/}  NUM_RUNS=$NUM_RUNS  PROFILES=$PROFILES" >&2
