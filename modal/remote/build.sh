#!/usr/bin/env bash
# build.sh <llvm|herbie|poseidon|all>: the toolchain the artifact runs on, built
# once into $TC (a Modal Volume) and reused by every later stage.
#
#   llvm      llvm-project at LLVM_COMMIT (clang, lld; X86 and NVPTX), installed
#             into $TC/llvm. Built on the container's local disk, installed when
#             done; the stamp $TC/llvm.stamp names the configuration.
#   herbie    (all: only with WITH_HERBIE=1) Herbie from $BUNDLE/herbie-src.tar.gz, the way
#             enzyme/Poseidon/cmake/Herbie.cmake builds it (cargo + raco exe +
#             raco distribute), into $TC/herbie; stamp $TC/herbie.stamp.
#   poseidon  Enzyme with -DENZYME_POSEIDON=ON from $BUNDLE/poseidon-src.tar.gz,
#             source in $TC/poseidon-src, build tree in $TC/poseidon-build (the
#             artifact's POSEIDON_BUILD); stamp $TC/poseidon.stamp.
#
# Each part is skipped when its stamp matches; a changed tarball rebuilds that
# part only. Environment: TC, BUNDLE, JOBS (default: nproc), CUDA_ARCHS
# (CMAKE_CUDA_ARCHITECTURES of the runtimes and calibrators), FORCE (space
# separated parts to rebuild), LLVM_PREFIX (an existing LLVM build or install
# to build Poseidon against instead of $TC/llvm; local testing only),
# LLVM_CONFIGURE_ONLY=1 (stop the llvm part after cmake; local testing only).
set -euo pipefail

TC="${TC:-/toolchain}"
BUNDLE="${BUNDLE:-/bundle}"
JOBS="${JOBS:-$(nproc)}"
CUDA_ARCHS="${CUDA_ARCHS:-103-real}"
CUDA="${CUDA:-/usr/local/cuda}"
FORCE="${FORCE:-}"
LLVM_COMMIT=9d618fc0243103263ebf9bdcd08ef25e871d9c78
LLVM_PROJECTS="clang;lld"
LLVM_TARGETS="X86;NVPTX"
SCRATCH="${SCRATCH:-/tmp/poseidon-build-scratch}"
export PATH="/opt/racket/bin:${CARGO_HOME:-/opt/rust/cargo}/bin:$CUDA/bin:$PATH"

log() { echo "[build $(date -u +%H:%M:%S)] $*"; }
forced() { case " $FORCE " in *" $1 "*|*" all "*) return 0 ;; esac; return 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
stamp_ok() {   # $1 = stamp file, $2 = expected content
  [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]
}
mkdir -p "$TC" "$SCRATCH"

build_llvm() {
  local want="commit=$LLVM_COMMIT projects=$LLVM_PROJECTS targets=$LLVM_TARGETS type=Release assertions=OFF"
  if ! forced llvm && stamp_ok "$TC/llvm.stamp" "$want"; then
    log "llvm: up to date ($TC/llvm)"; return
  fi
  log "llvm: fetching $LLVM_COMMIT"
  local src="$SCRATCH/llvm-project-$LLVM_COMMIT" bld="$SCRATCH/llvm-build"
  if [ ! -f "$src/llvm/CMakeLists.txt" ]; then
    curl -fsSL --retry 5 -o "$SCRATCH/llvm.tar.gz" \
      "https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz"
    log "llvm: tarball sha256 $(sha "$SCRATCH/llvm.tar.gz")"
    tar -C "$SCRATCH" -xzf "$SCRATCH/llvm.tar.gz"
    rm -f "$SCRATCH/llvm.tar.gz"
  fi
  # The local build this mirrors: Release, no assertions, plugins on (the
  # Enzyme and Poseidon plugins load into this clang), compiled by clang 18.
  # mlir and AMDGPU are left out: ENZYME_MLIR=OFF, and Poseidon's AMDGPU arms
  # are built only with POSEIDON_ROCM_PATH. lld stays: Enzyme builds its lld
  # plugin when lld is present, as the local build did.
  cmake -G Ninja -S "$src/llvm" -B "$bld" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_LINKER=lld \
    -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS" \
    -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_PLUGINS=ON \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_PARALLEL_LINK_JOBS=8 \
    -DCMAKE_INSTALL_PREFIX="$TC/llvm"
  [ "${LLVM_CONFIGURE_ONLY:-0}" = 1 ] && { log "llvm: configured (LLVM_CONFIGURE_ONLY)"; return; }
  log "llvm: building with $JOBS jobs"
  ninja -C "$bld" -j "$JOBS"
  rm -rf "$TC/llvm" "$TC/llvm.stamp"
  ninja -C "$bld" install > "$SCRATCH/llvm-install.log"
  "$TC/llvm/bin/clang++" --version | head -1
  echo "$want" > "$TC/llvm.stamp"
  rm -rf "$bld" "$src"
  log "llvm: installed into $TC/llvm"
}

build_herbie() {
  local tb="$BUNDLE/herbie-src.tar.gz"
  [ -f "$tb" ] || { log "herbie: $tb missing; skipped"; return; }
  local want="src=$(sha "$tb") racket=$(racket -e '(display (version))') rust=$(rustc --version | cut -d' ' -f2)"
  if ! forced herbie && stamp_ok "$TC/herbie.stamp" "$want" && [ -x "$TC/herbie/bin/herbie" ]; then
    log "herbie: up to date ($TC/herbie)"; return
  fi
  local src="$SCRATCH/herbie"
  rm -rf "$src" && tar -C "$SCRATCH" -xzf "$tb"
  cd "$src"
  log "herbie: building (cargo, raco)"
  # The commands of enzyme/Poseidon/cmake/Herbie.cmake, in the source tree.
  cargo build --release --manifest-path=egg-herbie/Cargo.toml
  raco pkg install --auto --no-docs --batch ./egg-herbie
  raco pkg update --auto --no-docs --batch fpbench rival rival3 ||
    raco pkg install --auto --no-docs --batch fpbench rival rival3
  mkdir -p herbie-compiled
  raco exe -o herbie --orig-exe --embed-dlls --vv src/main.rkt
  raco distribute herbie-compiled herbie
  rm -rf "$TC/herbie" "$TC/herbie.stamp"
  cp -a herbie-compiled "$TC/herbie"
  cd /
  "$TC/herbie/bin/herbie" improve --help 2>&1 | grep -q -- '--num-taylor-terms' ||
    log "herbie: WARNING this Herbie has no --num-taylor-terms (GENGA would need GENGA_TAYLOR_TERMS=)"
  echo "$want" > "$TC/herbie.stamp"
  rm -rf "$src"
  log "herbie: installed into $TC/herbie"
}

# cuMpSGEMM, the error-corrected GEMM backend the TCEC runtime delegates to
# (-DPOSEIDON_TCEC_USE_CUMPSGEMM=ON), from $BUNDLE/cumpsgemm-src.tar.gz into
# $TC/cumpsgemm (the build in $TC/cumpsgemm/build is what Poseidon's runtime,
# calibrators and applications link). Its CMakeLists hard-sets
# CMAKE_CUDA_ARCHITECTURES 80; that one line is set to this device's
# architecture in the unpacked copy, as the artifact's build_cumpsgemm.sh did.
# Only the shared library target is built (the rule library links libcuda,
# which a build container without a driver does not have).
build_cumpsgemm() {
  local tb="$BUNDLE/cumpsgemm-src.tar.gz" arch="${CUDA_ARCHS%%-*}"
  arch="${arch%%;*}"
  local want="src=$(sha "$tb") arch=$arch cuda=$("$CUDA/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p')"
  if ! forced cumpsgemm && stamp_ok "$TC/cumpsgemm.stamp" "$want"; then
    log "cumpsgemm: up to date ($TC/cumpsgemm)"; return
  fi
  rm -rf "$TC/cumpsgemm" "$TC/cumpsgemm.stamp"
  tar -C "$TC" -xzf "$tb" && mv "$TC/cuMpSGEMM" "$TC/cumpsgemm"
  local cml="$TC/cumpsgemm/CMakeLists.txt"
  [ "$(grep -c -E '^SET\(CMAKE_CUDA_ARCHITECTURES 80\)$' "$cml")" = 1 ] ||
    { echo "build: cuMpSGEMM's CMakeLists no longer has exactly one SET(CMAKE_CUDA_ARCHITECTURES 80)" >&2; exit 1; }
  sed -i "s/^SET(CMAKE_CUDA_ARCHITECTURES 80)\$/SET(CMAKE_CUDA_ARCHITECTURES $arch)/" "$cml"
  log "cumpsgemm: building for sm_$arch with $JOBS jobs"
  cmake -S "$TC/cumpsgemm" -B "$TC/cumpsgemm/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" > "$SCRATCH/cumpsgemm-cmake.log" 2>&1 ||
    { tail -20 "$SCRATCH/cumpsgemm-cmake.log"; exit 1; }
  make -C "$TC/cumpsgemm/build" -j "$JOBS" cumpsgemm
  local n; n="$("$CUDA/bin/cuobjdump" --list-elf "$TC/cumpsgemm/build/libcumpsgemm.so" | grep -c "sm_$arch")"
  [ "$n" -gt 0 ] || { echo "build: libcumpsgemm.so carries no sm_$arch code" >&2; exit 1; }
  log "cumpsgemm: libcumpsgemm.so carries $n sm_$arch cubin(s)"
  echo "$want" > "$TC/cumpsgemm.stamp"
}

build_poseidon() {
  local tb="$BUNDLE/poseidon-src.tar.gz"
  local llvm="${LLVM_PREFIX:-$TC/llvm}"
  [ -f "$llvm/lib/cmake/llvm/LLVMConfig.cmake" ] ||
    { echo "build: no LLVM at $llvm (run the llvm part first)" >&2; exit 1; }
  # An external LLVM is named by its clang's version line, not by its path.
  local llvm_id
  if [ -n "${LLVM_PREFIX:-}" ]; then llvm_id="external:$("$llvm/bin/clang" --version | head -1)"
  else llvm_id="$(cat "$TC/llvm.stamp")"; fi
  local cumps="${WITH_CUMPSGEMM:-1}"
  local cumps_id=off
  if [ "$cumps" = 1 ]; then
    [ -f "$TC/cumpsgemm.stamp" ] || { echo "build: no cuMpSGEMM in $TC (run the cumpsgemm part first)" >&2; exit 1; }
    cumps_id="$(cat "$TC/cumpsgemm.stamp")"
  fi
  local want="src=$(sha "$tb") llvm=[$llvm_id] cumpsgemm=[$cumps_id] cuda_archs=$CUDA_ARCHS cuda=$("$CUDA/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p')"
  if ! forced poseidon && stamp_ok "$TC/poseidon.stamp" "$want"; then
    log "poseidon: up to date ($TC/poseidon-build)"; return
  fi
  rm -rf "$TC/poseidon-src" "$TC/poseidon-build" "$TC/poseidon.stamp"
  mkdir -p "$TC/poseidon-src"
  tar -C "$TC/poseidon-src" --strip-components=1 -xzf "$tb"
  # As the local build is configured (its CMakeCache): Ninja, Enzyme's
  # default (empty) build type, the pinned clang as C/C++ compiler, nvcc for
  # the runtimes and calibrators, Poseidon and the clang plugin on, MLIR off.
  # The Herbie path is where the herbie part installs it; the artifact replays
  # shipped answers or runs it (GENGA_HERBIE), and the calibrator reads the
  # module names its platform files are written in from it.
  log "poseidon: configuring against $llvm (CUDA architectures $CUDA_ARCHS)"
  cmake -G Ninja -S "$TC/poseidon-src/enzyme" -B "$TC/poseidon-build" \
    -DCMAKE_C_COMPILER="$llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="$llvm/bin/clang++" \
    -DLLVM_DIR="$llvm/lib/cmake/llvm" \
    -DENZYME_POSEIDON=ON -DENZYME_CLANG=ON -DENZYME_MLIR=OFF \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DPOSEIDON_HERBIE_BINARY="$TC/herbie/bin/herbie" \
    $([ "$cumps" = 1 ] && echo "-DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT=$TC/cumpsgemm")
  log "poseidon: building with $JOBS jobs"
  ninja -C "$TC/poseidon-build" -j "$JOBS"
  ls -l "$TC/poseidon-build/Poseidon/bin/poseidon-clang++" "$TC/poseidon-build/Poseidon/lib/"Poseidon-*.so \
        "$TC/poseidon-build/Enzyme/"ClangEnzyme-*.so
  echo "$want" > "$TC/poseidon.stamp"
  log "poseidon: built in $TC/poseidon-build"
}

case "${1:-all}" in
  llvm) build_llvm ;;
  cumpsgemm) build_cumpsgemm ;;
  herbie) build_herbie ;;
  poseidon) build_poseidon ;;
  # Herbie in its own process: a failed Herbie build must not stop the
  # compiler build (only the GENGA drift and unified solves on a GPU without
  # shipped Herbie answers need it).
  # WITH_HERBIE=1 only for the full on-device flow: the replay run never
  # searches (its Herbie answers are shipped).
  all) build_llvm
       if [ "${WITH_HERBIE:-0}" = 1 ]; then
         bash "$0" herbie || log "herbie: FAILED (only a GENGA search on the device needs it)"
       else
         log "herbie: not built (WITH_HERBIE=0)"
       fi
       [ "${WITH_CUMPSGEMM:-1}" = 1 ] && build_cumpsgemm
       build_poseidon ;;
  *) echo "usage: build.sh <llvm|herbie|cumpsgemm|poseidon|all>" >&2; exit 2 ;;
esac
log "done: $*"
