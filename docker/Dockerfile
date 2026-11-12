# One image with everything the artifact needs: CUDA 12.9, LLVM at the pinned
# commit (clang, lld; X86 and NVPTX), cuMpSGEMM, the Poseidon compiler built
# from the sources/Enzyme submodule, Python with numpy and matplotlib, and the
# artifact at /artifact with POSEIDON_BUILD and PY preset.
#
# Build from the repository root, cloned with its submodules (git clone
# --recursive, or git submodule update --init --recursive):
#
#   docker build -f docker/Dockerfile -t poseidon-cgo2027 .
#
# Run on the GPU (Docker with the NVIDIA Container Toolkit):
#
#   docker run --rm -it --gpus all poseidon-cgo2027            # or
#   docker run --rm -it --device nvidia.com/gpu=0 poseidon-cgo2027   # CDI
#   ./run_artifact.sh --check
#
# Downloads at build time: the base image, the LLVM source tarball of
# LLVM_COMMIT from GitHub (checked against LLVM_SHA256), Ubuntu packages and
# two Python wheels. Everything else comes from sources/.
#
# Build arguments:
#   JOBS        parallel build jobs (default: all cores)
#   CUDA_ARCHS  CMAKE_CUDA_ARCHITECTURES of the Poseidon runtimes and
#               calibrators; 120-real is the RTX 5090 class (compute
#               capability 12.0). Another GPU needs its own, e.g. 90-real
#               for an H100, 80-real for an A100.
#   LLVM_SHA256 empty skips the tarball check.

ARG CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04

# ---- common runtime and build dependencies ----------------------------------
FROM ${CUDA_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl xz-utils file bc time procps make cmake ninja-build \
      libmpfr-dev libgmp-dev zlib1g-dev libzstd-dev libxml2-dev liblzma-dev \
      python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/py \
 && /opt/py/bin/pip install --no-cache-dir numpy==2.4.2 matplotlib==3.10.9 \
 && /opt/py/bin/python -c 'import numpy, matplotlib'
ENV PATH=/opt/py/bin:$PATH

# ---- LLVM at the commit the compiler is built against ------------------------
# Release, no assertions, plugins on (the Poseidon and Enzyme plugins load into
# this clang). Only the install tree leaves this stage. lit is copied from the
# same source tree so that the compiler's lit tests can run in the image.
FROM base AS llvm
ARG LLVM_COMMIT=9d618fc0243103263ebf9bdcd08ef25e871d9c78
ARG LLVM_SHA256=57eb91d0f644d74671300ea3e154fc22c9070aa87656844f0795c637693b4bdb
ARG JOBS=
RUN apt-get update \
 && apt-get install -y --no-install-recommends clang lld \
 && rm -rf /var/lib/apt/lists/*
RUN set -eux; cd /tmp; \
    curl -fsSL --retry 5 -o llvm.tar.gz \
      "https://github.com/llvm/llvm-project/archive/${LLVM_COMMIT}.tar.gz"; \
    if [ -n "${LLVM_SHA256}" ]; then echo "${LLVM_SHA256}  llvm.tar.gz" | sha256sum -c -; fi; \
    tar -xzf llvm.tar.gz; rm -f llvm.tar.gz; \
    cmake -G Ninja -S "llvm-project-${LLVM_COMMIT}/llvm" -B llvm-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DLLVM_USE_LINKER=lld \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_ENABLE_PLUGINS=ON \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
      -DLLVM_INSTALL_UTILS=ON \
      -DLLVM_PARALLEL_LINK_JOBS=8 \
      -DCMAKE_INSTALL_PREFIX=/opt/llvm; \
    ninja -C llvm-build -j "${JOBS:-$(nproc)}"; \
    du -sh "llvm-project-${LLVM_COMMIT}" llvm-build; \
    ninja -C llvm-build install > /dev/null; \
    du -sh /opt/llvm; \
    mkdir -p /opt/llvm/share; \
    cp -a "llvm-project-${LLVM_COMMIT}/llvm/utils/lit" /opt/llvm/share/lit; \
    printf '#!/bin/sh\nexec python3 /opt/llvm/share/lit/lit.py "$@"\n' > /opt/llvm/bin/llvm-lit; \
    chmod +x /opt/llvm/bin/llvm-lit; \
    rm -rf /tmp/llvm-build "/tmp/llvm-project-${LLVM_COMMIT}"; \
    /opt/llvm/bin/clang++ --version

# ---- cuMpSGEMM and the Poseidon compiler -------------------------------------
FROM base AS poseidon
ARG JOBS=
ARG CUDA_ARCHS=120-real
# Enzyme's configure step declares (but this build never runs) git downloads
# of BLAS sources for its optional bitcode loader; CMake needs git to exist.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
COPY --from=llvm /opt/llvm /opt/llvm
COPY sources/cuMpSGEMM /opt/poseidon/cumpsgemm
COPY sources/Enzyme /opt/poseidon/src
# cuMpSGEMM 42dd92e with its submodules, built as its CMakeLists says (sm_80
# code and PTX; the shipped RTX 5090 cost model's error-corrected dispatch row
# was measured against that build). Only the shared library is built: the
# rule library links libcuda, which a build without a GPU driver lacks.
RUN set -eux; \
    for d in cutf wmma_extension cuGEMM-Mx2x2; do \
      [ -n "$(ls -A /opt/poseidon/cumpsgemm/submodules/$d 2>/dev/null)" ] || \
        { echo "sources/cuMpSGEMM/submodules/$d is empty: git submodule update --init --recursive" >&2; exit 1; }; \
    done; \
    cmake -S /opt/poseidon/cumpsgemm -B /opt/poseidon/cumpsgemm/build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc; \
    make -C /opt/poseidon/cumpsgemm/build -j "${JOBS:-$(nproc)}" cumpsgemm; \
    find /opt/poseidon/cumpsgemm/build -mindepth 1 -maxdepth 1 ! -name libcumpsgemm.so -exec rm -rf {} +; \
    ls -l /opt/poseidon/cumpsgemm/build/libcumpsgemm.so
# The compiler, configured as the reference build: Ninja, Enzyme's default
# (empty) build type, the pinned clang, Poseidon and the clang plugin on, the
# TCEC runtime delegating to cuMpSGEMM. No Herbie is built: the artifact
# replays the shipped Herbie answers by default; docker/Dockerfile.herbie puts
# one at the path given here. Object files are deleted afterwards; the build
# tree keeps what the driver, the scripts and a rebuild read.
RUN set -eux; \
    [ -f /opt/poseidon/src/enzyme/CMakeLists.txt ] || \
      { echo "sources/Enzyme is empty: git submodule update --init --recursive" >&2; exit 1; }; \
    cmake -G Ninja -S /opt/poseidon/src/enzyme -B /opt/poseidon/build \
      -DCMAKE_C_COMPILER=/opt/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/llvm/bin/clang++ \
      -DLLVM_DIR=/opt/llvm/lib/cmake/llvm \
      -DLLVM_EXTERNAL_LIT=/opt/llvm/bin/llvm-lit \
      -DENZYME_POSEIDON=ON -DENZYME_CLANG=ON -DENZYME_MLIR=OFF \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DPOSEIDON_HERBIE_BINARY=/opt/herbie/bin/herbie \
      -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON \
      -DPOSEIDON_CUMPSGEMM_ROOT=/opt/poseidon/cumpsgemm; \
    ninja -C /opt/poseidon/build -j "${JOBS:-$(nproc)}"; \
    du -sh /opt/poseidon/build /opt/poseidon/cumpsgemm; \
    find /opt/poseidon/build -name '*.o' -delete; \
    du -sh /opt/poseidon/build; \
    ls -l /opt/poseidon/build/Poseidon/bin/poseidon-clang++ /opt/poseidon/build/Poseidon/lib/Poseidon-*.so

# ---- the artifact, without the submodules the stages above were built from ----
FROM base AS artifact
COPY . /artifact
RUN rm -rf /artifact/sources

# ---- the image ---------------------------------------------------------------
FROM base
COPY --from=llvm /opt/llvm /opt/llvm
COPY --from=poseidon /opt/poseidon /opt/poseidon
COPY --from=artifact /artifact /artifact
# cuMpSGEMM carries sm_80 code and PTX, which the driver compiles for the GPU
# on first use; a large cache keeps that result across the artifact's runs.
ENV POSEIDON_BUILD=/opt/poseidon/build \
    PY=/opt/py/bin/python \
    CUDA_CACHE_MAXSIZE=4294967296 \
    PATH=/opt/poseidon/build/Poseidon/bin:/opt/llvm/bin:/opt/py/bin:/usr/local/cuda/bin:$PATH
WORKDIR /artifact
CMD ["/bin/bash"]
