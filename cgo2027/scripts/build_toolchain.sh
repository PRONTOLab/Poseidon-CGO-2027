#!/usr/bin/env bash
# build_toolchain.sh: build what the artifact compiles with, from the
# repository's sources, into one prefix (README: Setup, B, as one script).
#
#   scripts/build_toolchain.sh --prefix <dir> [options]
#
# Steps, in order (each is skipped when its stamp in <dir>/.stamps says it is
# done for the same inputs, so a rerun resumes after the last finished step):
#   llvm       LLVM 9d618fc (clang, lld; X86 and NVPTX) from GitHub's source
#              tarball, checked against its sha256, installed in <dir>/llvm
#   cumpsgemm  cuMpSGEMM 42dd92e from sources/cuMpSGEMM, built in
#              <dir>/cuMpSGEMM as its CMakeLists says (sm_80 code and PTX)
#   compiler   Enzyme with Poseidon from sources/Enzyme, in <dir>/poseidon-build
#   herbie     only with --with-herbie: Herbie from sources/herbie with
#              sources/herbie-num-taylor-terms.patch, in <dir>/herbie (the path
#              the compiler step names as POSEIDON_HERBIE_BINARY); needed only
#              to search yourself (GENGA_HERBIE=run, --from-scratch)
#
# Options:
#   --prefix <dir>        where everything goes (required)
#   --llvm <dir>          use this LLVM 9d618fc (an install or build tree with
#                         bin/clang++ and lib/cmake/llvm) instead of building it
#   --cuda <dir>          the CUDA toolkit (default: CUDA, CUDA_HOME, then nvcc
#                         on PATH, then /usr/local/cuda)
#   --cuda-archs <list>   CMAKE_CUDA_ARCHITECTURES of the Poseidon runtimes and
#                         calibrators (default: the visible GPU's, e.g. 120-real;
#                         required when no GPU is visible)
#   --jobs <n>            parallel build jobs (default: all cores)
#   --with-herbie         also build Herbie (needs racket, raco and cargo:
#                         Racket 8.15 and Rust 1.85.1)
#   --only <step>         run one step (llvm, cumpsgemm, compiler, herbie)
#   --keep-llvm-sources   keep the LLVM source and build trees (7 GB) after
#                         the install
#   -h, --help            this text
#
# Prints the `export POSEIDON_BUILD=...` line to use at the end.
set -euo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRCS="$(bash "$ART/scripts/detect.sh" sources)" || exit 1
LLVM_COMMIT=9d618fc0243103263ebf9bdcd08ef25e871d9c78
LLVM_SHA256=57eb91d0f644d74671300ea3e154fc22c9070aa87656844f0795c637693b4bdb

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "build_toolchain.sh: $*" >&2; exit 1; }
say() { echo "[toolchain $(date +%H:%M:%S)] $*"; }

P="" LLVM="" CUDA_DIR="" ARCHS="" JOBS="" HERBIE=0 ONLY="" KEEP_LLVM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) P="${2:?--prefix needs a directory}"; shift ;;
    --llvm) LLVM="${2:?--llvm needs a directory}"; shift ;;
    --cuda) CUDA_DIR="${2:?--cuda needs a directory}"; shift ;;
    --cuda-archs) ARCHS="${2:?--cuda-archs needs a list}"; shift ;;
    --jobs) JOBS="${2:?--jobs needs a number}"; shift ;;
    --with-herbie) HERBIE=1 ;;
    --only) ONLY="${2:?--only needs a step}"; shift ;;
    --keep-llvm-sources) KEEP_LLVM=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument '$1'" ;;
  esac
  shift
done
[ -n "$P" ] || { usage >&2; die "--prefix <dir> is required"; }
case "$ONLY" in ""|llvm|cumpsgemm|compiler|herbie) ;; *) die "--only takes llvm, cumpsgemm, compiler or herbie, not '$ONLY'" ;; esac
[ "$ONLY" = herbie ] && HERBIE=1
mkdir -p "$P" && P="$(cd "$P" && pwd)"
JOBS="${JOBS:-$(nproc)}"
STAMPS="$P/.stamps"; mkdir -p "$STAMPS"
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
done_for() { [ -f "$STAMPS/$1" ] && [ "$(cat "$STAMPS/$1")" = "$2" ]; }   # <step> <inputs>

need() {   # <program> <why>
  command -v "$1" > /dev/null 2>&1 || die "'$1' not found on PATH: $2"
}
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }   # <have> <need>

# ---- prerequisites ------------------------------------------------------------
need cmake "CMake 3.24 or newer builds LLVM, cuMpSGEMM and the compiler"
version_ge "$(cmake --version | sed -n 's/^cmake version //p')" 3.24 ||
  die "CMake $(cmake --version | sed -n 's/^cmake version //p') is too old; 3.24 or newer is needed"
need ninja "Ninja builds LLVM and the compiler"
need make "make builds cuMpSGEMM"
if [ -z "$CUDA_DIR" ]; then
  for c in "${CUDA:-}" "${CUDA_HOME:-}" "$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /x/x/x)")")" /usr/local/cuda; do
    [ -n "$c" ] && [ -x "$c/bin/nvcc" ] && { CUDA_DIR="$c"; break; }
  done
fi
[ -n "$CUDA_DIR" ] && [ -x "$CUDA_DIR/bin/nvcc" ] ||
  die "no CUDA toolkit found (nvcc): pass --cuda <dir> or set CUDA_HOME (CUDA 12.x or 13.x)"
CUDA_DIR="$(cd "$CUDA_DIR" && pwd)"
if [ -z "$ARCHS" ] && want compiler; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')" || cc=""
  [ -n "$cc" ] || die "no GPU visible to nvidia-smi: pass --cuda-archs (120-real for the RTX 5090 class, 90-real for H100/GH200, 103-real for B300)"
  ARCHS="${cc}-real"
fi

if want llvm && [ -z "$LLVM" ]; then
  need curl "the LLVM step downloads GitHub's source tarball of $LLVM_COMMIT"
  need sha256sum "the LLVM step checks the tarball"
  need tar "the LLVM step unpacks the tarball"
  need python3 "LLVM's build runs Python"
  command -v c++ > /dev/null 2>&1 || command -v clang++ > /dev/null 2>&1 || command -v g++ > /dev/null 2>&1 ||
    die "no C++ compiler (c++, clang++ or g++) on PATH; LLVM needs a C++17 compiler"
fi
if want compiler; then
  need git "Enzyme's CMake configure step requires git (nothing is cloned)"
  need python3 "the compiler's build runs Python"
  printf '#include <mpfr.h>\n#include <gmp.h>\n' | "${CXX:-c++}" -E -x c++ - > /dev/null 2>&1 ||
    die "mpfr.h/gmp.h not found by ${CXX:-c++}: install MPFR and GMP development files (libmpfr-dev libgmp-dev)"
  [ -f "$SRCS/Enzyme/enzyme/CMakeLists.txt" ] ||
    die "sources/Enzyme is empty: git submodule update --init --recursive"
fi
if want cumpsgemm; then
  for d in cutf wmma_extension cuGEMM-Mx2x2; do
    [ -n "$(ls -A "$SRCS/cuMpSGEMM/submodules/$d" 2>/dev/null)" ] ||
      die "sources/cuMpSGEMM/submodules/$d is empty: git submodule update --init --recursive"
  done
fi
if [ "$HERBIE" = 1 ] && want herbie; then
  need racket "the Herbie step needs Racket 8.15 (https://download.racket-lang.org/releases/8.15/)"
  need raco "the Herbie step needs Racket 8.15's raco"
  need cargo "the Herbie step needs Rust 1.85.1 (rustup toolchain install 1.85.1)"
  need patch "the Herbie step applies sources/herbie-num-taylor-terms.patch"
  [ -f "$SRCS/herbie/src/main.rkt" ] || die "sources/herbie is empty: git submodule update --init --recursive"
  [ -f "$SRCS/herbie-num-taylor-terms.patch" ] || die "sources/herbie-num-taylor-terms.patch is missing"
  rv="$(racket --version | grep -oE '[0-9]+\.[0-9]+' | head -1)"
  [ "$rv" = 8.15 ] || echo "build_toolchain.sh: warning: Racket $rv; the shipped answers were searched with a Herbie built on Racket 8.15" >&2
  cv="$(cargo --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  [ "$cv" = 1.85.1 ] || echo "build_toolchain.sh: warning: cargo $cv; the shipped answers were searched with a Herbie built with Rust 1.85.1" >&2
fi

# ---- llvm -----------------------------------------------------------------------
if [ -n "$LLVM" ]; then
  [ -d "$LLVM" ] || die "--llvm $LLVM is not a directory"
  LLVM="$(cd "$LLVM" && pwd)"
  [ -x "$LLVM/bin/clang++" ] && [ -f "$LLVM/lib/cmake/llvm/LLVMConfig.cmake" ] ||
    die "--llvm $LLVM has no bin/clang++ and lib/cmake/llvm/LLVMConfig.cmake"
  say "LLVM: $LLVM ($("$LLVM/bin/clang++" --version | head -1))"
  # A revision recorded in the LLVM is part of the compiler's key of every DP
  # table; the shipped tables were made with one that records none.
  if grep -q '^#define LLVM_REVISION' "$LLVM/include/llvm/Support/VCSRevision.h" 2>/dev/null; then
    echo "build_toolchain.sh: note: this LLVM records its git revision, which keys the compiler's DP tables;" >&2
    echo "build_toolchain.sh: the shipped DP tables (built from the tarball, no revision) will be re-solved, visibly." >&2
  fi
else
  LLVM="$P/llvm"
  if want llvm && ! done_for llvm "$LLVM_COMMIT"; then
    SRC="$P/src"; mkdir -p "$SRC"
    if [ ! -d "$SRC/llvm-project-$LLVM_COMMIT/llvm" ]; then
      if [ ! -f "$SRC/llvm.tar.gz" ]; then
        say "LLVM: downloading the source tarball of $LLVM_COMMIT"
        curl -fL --retry 5 -o "$SRC/llvm.tar.gz.part" "https://github.com/llvm/llvm-project/archive/$LLVM_COMMIT.tar.gz"
        mv "$SRC/llvm.tar.gz.part" "$SRC/llvm.tar.gz"
      fi
      echo "$LLVM_SHA256  $SRC/llvm.tar.gz" | sha256sum -c - ||
        { rm -f "$SRC/llvm.tar.gz"; die "the LLVM tarball does not have the pinned sha256; deleted, rerun to download again"; }
      say "LLVM: unpacking"
      tar -xzf "$SRC/llvm.tar.gz" -C "$SRC"
    fi
    CC_ARGS=()
    if command -v clang > /dev/null && command -v clang++ > /dev/null && command -v ld.lld > /dev/null; then
      CC_ARGS=(-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld)
    fi
    say "LLVM: configure and build ($JOBS jobs; about 9 minutes at 96 jobs)"
    cmake -G Ninja -S "$SRC/llvm-project-$LLVM_COMMIT/llvm" -B "$SRC/llvm-build" \
      -DCMAKE_BUILD_TYPE=Release "${CC_ARGS[@]}" \
      -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_PLUGINS=ON \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
      -DLLVM_INSTALL_UTILS=ON -DLLVM_PARALLEL_LINK_JOBS=8 \
      -DCMAKE_INSTALL_PREFIX="$LLVM"
    ninja -C "$SRC/llvm-build" -j "$JOBS"
    ninja -C "$SRC/llvm-build" install > /dev/null
    # lit from the same tree, so the compiler's lit tests can run.
    rm -rf "$LLVM/share/lit"; mkdir -p "$LLVM/share"
    cp -a "$SRC/llvm-project-$LLVM_COMMIT/llvm/utils/lit" "$LLVM/share/lit"
    printf '#!/bin/sh\nexec python3 "%s/share/lit/lit.py" "$@"\n' "$LLVM" > "$LLVM/bin/llvm-lit"
    chmod +x "$LLVM/bin/llvm-lit"
    echo "$LLVM_COMMIT" > "$STAMPS/llvm"
    [ "$KEEP_LLVM" = 1 ] || rm -rf "${SRC:?}/llvm-build" "${SRC:?}/llvm-project-$LLVM_COMMIT" "${SRC:?}/llvm.tar.gz"
    say "LLVM: installed in $LLVM"
  fi
fi

# ---- cuMpSGEMM -----------------------------------------------------------------
CUMPS="$P/cuMpSGEMM"
CUMPS_ID="42dd92ea1c0e3d603bbb3f8f354f53e0967a9f4b nvcc=$("$CUDA_DIR/bin/nvcc" --version | tail -1)"
if want cumpsgemm && ! done_for cumpsgemm "$CUMPS_ID"; then
  say "cuMpSGEMM: copying sources/cuMpSGEMM to $CUMPS and building the shared library"
  rm -rf "${CUMPS:?}"
  cp -a "$SRCS/cuMpSGEMM" "$CUMPS"
  cmake -S "$CUMPS" -B "$CUMPS/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_DIR/bin/nvcc"
  # Only the shared library: the rule library links libcuda, absent without a driver.
  make -C "$CUMPS/build" -j "$JOBS" cumpsgemm
  [ -f "$CUMPS/build/libcumpsgemm.so" ] || die "cuMpSGEMM built no libcumpsgemm.so"
  echo "$CUMPS_ID" > "$STAMPS/cumpsgemm"
fi

# ---- the compiler --------------------------------------------------------------
BUILD="$P/poseidon-build"
if want compiler; then
  [ -x "$LLVM/bin/clang++" ] || die "no LLVM at $LLVM: run the llvm step, or pass --llvm <dir>"
  [ -f "$CUMPS/build/libcumpsgemm.so" ] || die "no cuMpSGEMM at $CUMPS: run the cumpsgemm step first"
  LIT_ARGS=(); [ -x "$LLVM/bin/llvm-lit" ] && LIT_ARGS=(-DLLVM_EXTERNAL_LIT="$LLVM/bin/llvm-lit")
  say "compiler: configure and build sources/Enzyme (CUDA archs $ARCHS)"
  cmake -G Ninja -S "$SRCS/Enzyme/enzyme" -B "$BUILD" \
    -DCMAKE_C_COMPILER="$LLVM/bin/clang" -DCMAKE_CXX_COMPILER="$LLVM/bin/clang++" \
    -DLLVM_DIR="$LLVM/lib/cmake/llvm" "${LIT_ARGS[@]}" \
    -DENZYME_POSEIDON=ON -DENZYME_CLANG=ON -DENZYME_MLIR=OFF \
    -DCMAKE_CUDA_COMPILER="$CUDA_DIR/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
    -DPOSEIDON_HERBIE_BINARY="$P/herbie/bin/herbie" \
    -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT="$CUMPS"
  ninja -C "$BUILD" -j "$JOBS"
  [ -x "$BUILD/Poseidon/bin/poseidon-clang++" ] || die "the compiler build has no Poseidon/bin/poseidon-clang++"
fi

# ---- Herbie --------------------------------------------------------------------
if [ "$HERBIE" = 1 ] && want herbie; then
  HERBIE_ID="$(git -C "$SRCS/herbie" rev-parse HEAD 2>/dev/null || echo unknown) $(sha256sum < "$SRCS/herbie-num-taylor-terms.patch" | cut -c1-16)"
  if ! done_for herbie "$HERBIE_ID"; then
    # A copy is patched; the submodule's checkout stays as it is.
    HS="$P/src/herbie"
    say "Herbie: copying sources/herbie, applying the --num-taylor-terms patch, building"
    rm -rf "${HS:?}"; mkdir -p "$P/src"
    cp -a "$SRCS/herbie" "$HS"
    rm -rf "$HS/.git"
    ( cd "$HS" && patch -p1 < "$SRCS/herbie-num-taylor-terms.patch" )
    ( cd "$HS"
      cargo build --release --manifest-path=egg-herbie/Cargo.toml
      raco pkg install --auto --no-docs --batch ./egg-herbie
      raco pkg update --auto --no-docs --batch fpbench rival rival3 ||
        raco pkg install --auto --no-docs --batch fpbench rival rival3
      mkdir -p herbie-compiled
      raco exe -o herbie --orig-exe --embed-dlls --vv src/main.rkt
      raco distribute herbie-compiled herbie )
    rm -rf "${P:?}/herbie"
    cp -a "$HS/herbie-compiled" "$P/herbie"
    "$P/herbie/bin/herbie" improve --help 2>&1 | grep -q -- '--num-taylor-terms' ||
      die "the built Herbie does not accept --num-taylor-terms"
    echo "$HERBIE_ID" > "$STAMPS/herbie"
    say "Herbie: $P/herbie/bin/herbie"
  fi
fi

echo
echo "Done. Use this toolchain with:"
echo "  export POSEIDON_BUILD=$BUILD"
[ -x "$P/herbie/bin/herbie" ] && echo "  export HERBIE=$P/herbie/bin/herbie     # for GENGA_HERBIE=run and --from-scratch"
echo "and a python with numpy and matplotlib (PY=<python> if it is not python3 on PATH)."
