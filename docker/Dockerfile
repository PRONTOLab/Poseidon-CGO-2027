# The artifact image: CUDA 12.9, the pinned LLVM, cuMpSGEMM, the compiler, Herbie and Python, built
# from the repository root with its submodules: docker build -f docker/Dockerfile -t poseidon-cgo2027 .
# Build args: JOBS, CUDA_ARCHS (default 120-real, the RTX 5090), LLVM_SHA256 (empty skips the check).

ARG CUDA_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04

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

# Plugins on: the Poseidon and Enzyme plugins load into this clang.
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
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
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

FROM base AS poseidon
ARG JOBS=
ARG CUDA_ARCHS=120-real
# Enzyme's configure step needs git to exist; nothing is cloned.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git patch \
 && rm -rf /var/lib/apt/lists/*
COPY --from=llvm /opt/llvm /opt/llvm
COPY sources/cuMpSGEMM /opt/poseidon/cumpsgemm
COPY sources/cumpsgemm-explicit-gemm-instantiation.patch /opt/poseidon/
COPY sources/Enzyme /opt/poseidon/src
# A compute_80 binary per CUDA_ARCHS avoids JIT at first use; only the shared library is built,
# since the rule library links libcuda, absent without a GPU driver.
RUN set -eux; \
    for d in cutf wmma_extension cuGEMM-Mx2x2; do \
      [ -n "$(ls -A /opt/poseidon/cumpsgemm/submodules/$d 2>/dev/null)" ] || \
        { echo "sources/cuMpSGEMM/submodules/$d is empty: git submodule update --init --recursive" >&2; exit 1; }; \
    done; \
    ( cd /opt/poseidon/cumpsgemm \
      && patch -p1 < /opt/poseidon/cumpsgemm-explicit-gemm-instantiation.patch ); \
    gencode="$(printf '%s\n' "${CUDA_ARCHS}" | tr ';' '\n' | sed 's/-real$//' | grep -vx 80 \
      | sed 's/^/-gencode arch=compute_80,code=sm_/' | tr '\n' ' ')"; \
    cmake -S /opt/poseidon/cumpsgemm -B /opt/poseidon/cumpsgemm/build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      "-DCMAKE_CUDA_FLAGS=${gencode}"; \
    make -C /opt/poseidon/cumpsgemm/build -j "${JOBS:-$(nproc)}" cumpsgemm; \
    find /opt/poseidon/cumpsgemm/build -mindepth 1 -maxdepth 1 ! -name libcumpsgemm.so -exec rm -rf {} +; \
    ls -l /opt/poseidon/cumpsgemm/build/libcumpsgemm.so
# Empty CMAKE_CUDA_FLAGS_RELEASE keeps nvcc's default flags for the CUDA sources.
RUN set -eux; \
    [ -f /opt/poseidon/src/enzyme/CMakeLists.txt ] || \
      { echo "sources/Enzyme is empty: git submodule update --init --recursive" >&2; exit 1; }; \
    cmake -G Ninja -S /opt/poseidon/src/enzyme -B /opt/poseidon/build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS_RELEASE= \
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

FROM base
COPY --from=llvm /opt/llvm /opt/llvm
COPY --from=poseidon /opt/poseidon /opt/poseidon
ENV RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo
RUN set -eux; \
    curl -fsSL --retry 5 -o /tmp/racket.sh \
      https://download.racket-lang.org/releases/8.15/installers/racket-8.15-x86_64-linux-cs.sh; \
    sh /tmp/racket.sh --in-place --dest /opt/racket; rm -f /tmp/racket.sh; \
    curl -fsSL --retry 5 -o /tmp/rustup-init \
      https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init; \
    chmod +x /tmp/rustup-init; \
    /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.85.1; \
    rm -f /tmp/rustup-init
ENV PATH=/opt/racket/bin:/opt/rust/cargo/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends patch \
 && rm -rf /var/lib/apt/lists/*
COPY sources/herbie /tmp/herbie
COPY sources/herbie-num-taylor-terms.patch /tmp/herbie-num-taylor-terms.patch
RUN set -eux; \
    [ -f /tmp/herbie/src/main.rkt ] || \
      { echo "sources/herbie is empty: git submodule update --init --recursive" >&2; exit 1; }; \
    cd /tmp/herbie; patch -p1 < /tmp/herbie-num-taylor-terms.patch; \
    cargo build --release --manifest-path=egg-herbie/Cargo.toml; \
    raco pkg install --auto --no-docs --batch ./egg-herbie; \
    raco pkg install --auto --no-docs --batch --name fpbench \
      https://github.com/FPBench/FPBench.git#3189f754b3716a17238c6f7aab1dfd056bd6785b; \
    raco pkg install --auto --no-docs --batch --name rival \
      https://github.com/herbie-fp/rival.git#4d2334a05338be8df9c6924e30534d693f83267d; \
    raco pkg install --auto --no-docs --batch --name rival3 \
      https://github.com/herbie-fp/rival3/releases/download/v1.2.1/rival3-racket.zip; \
    mkdir -p herbie-compiled; \
    raco exe -o herbie --orig-exe --embed-dlls --vv src/main.rkt; \
    raco distribute herbie-compiled herbie; \
    cp -a herbie-compiled /opt/herbie; \
    cd /; rm -rf /tmp/herbie /tmp/herbie-num-taylor-terms.patch; \
    /opt/herbie/bin/herbie improve --help 2>&1 | grep -q -- '--num-taylor-terms'
RUN mkdir -p /artifact
ENV POSEIDON_BUILD=/opt/poseidon/build \
    PY=/opt/py/bin/python \
    HERBIE=/opt/herbie/bin/herbie \
    PATH=/opt/poseidon/build/Poseidon/bin:/opt/llvm/bin:/opt/py/bin:/usr/local/cuda/bin:$PATH
WORKDIR /artifact
CMD ["/bin/bash"]
