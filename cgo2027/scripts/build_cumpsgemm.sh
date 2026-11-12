#!/usr/bin/env bash
# build_cumpsgemm.sh [dir] — fetch and build cuMpSGEMM, the error-corrected FP16
# tensor-core GEMM the compiler's TCEC dispatch runtime delegates to.
#
#   bash scripts/build_cumpsgemm.sh            # into deps/cuMpSGEMM of the artifact
#   bash scripts/build_cumpsgemm.sh /some/dir
#
# Then configure the compiler with
#   -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT=<dir>
# (README: Setup). The results of EXPECTED_RESULTS.md come from that build.
#
# cuMpSGEMM: https://github.com/enp1s0/cuMpSGEMM (MIT License, see LICENSE in
# the checkout), at the commit below, with its submodules at the commits that
# commit records (checked below). It is built as its own CMakeLists.txt says
# (Release; the file sets CMAKE_CUDA_ARCHITECTURES to 80, so the library holds
# sm_80 code and PTX that the driver compiles for the device at load; on an
# RTX 5090 that measured faster than a native sm_120 build of this commit).
# Needs git, CMake >= 3.18 and the CUDA toolkit (CUDA or CUDA_HOME, else nvcc
# on PATH).
set -euo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${1:-$ART/deps/cuMpSGEMM}"
URL=https://github.com/enp1s0/cuMpSGEMM.git
COMMIT=42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b
CUDA="${CUDA:-${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc || echo /nonexistent/bin/nvcc)")")}}"
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc: set CUDA or CUDA_HOME to the CUDA toolkit" >&2; exit 1; }
# path  commit (the submodule commits 42dd92e records)
SUBMODULES="submodules/cuGEMM-Mx2x2 1f68ec6dc1a2b741c552030e5bbe4ad33d398876
submodules/cutf 97339d83d3fb091aa75f90495cf9a6fa9c1f4ff2
submodules/wmma_extension 03772a7ff08220185fae81487513e885e6962759"

if [ ! -d "$DIR/.git" ]; then
  git clone "$URL" "$DIR"
fi
git -C "$DIR" checkout --quiet "$COMMIT"
git -C "$DIR" submodule update --init --recursive
while read -r path commit; do
  [ "$(git -C "$DIR/$path" rev-parse HEAD)" = "$commit" ] ||
    { echo "$path is not at $commit" >&2; exit 1; }
done <<< "$SUBMODULES"
cmake -S "$DIR" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc"
cmake --build "$DIR/build" -j "$(nproc)"
ls "$DIR/build/libcumpsgemm.so" >/dev/null
echo "cuMpSGEMM $COMMIT built in $DIR/build; configure the compiler with"
echo "  -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT=$DIR"
