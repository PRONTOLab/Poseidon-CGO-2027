#!/usr/bin/env bash
# docker.sh: build the artifact's container and run its modes in it; outputs land in ./out.
set -euo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRCS="$(bash "$ART/scripts/detect.sh" sources)" || exit 1
CTX="$ART"
usage() {
  cat <<'EOF'
Usage: scripts/docker.sh build [docker build options]
       scripts/docker.sh run [run_artifact.sh options]
       scripts/docker.sh picks [--device rtx5090|b300|gh200|all]
       scripts/docker.sh explain <row id> [poseidon-explain options]
       scripts/docker.sh shell
  build      build the image sbrantq/poseidon-cgo2027 (docker/Dockerfile)
  run        ./run_artifact.sh on the GPU; the results bundle goes to ./out
  picks      ./run_artifact.sh --picks without a GPU; ./out/picks_check
  explain    ./run_artifact.sh --explain without a GPU; ./out/explain/<row>
  shell      an interactive shell in the container
  IMAGE      image to run (default: sbrantq/poseidon-cgo2027)
  GPU_ID     the one GPU to pass (default: all)
  MEMORY     container memory cap (default: 24g)
  BENCHES, GENGA_HERBIE, PROFILES, SURROGATE_SIZES, NUM_RUNS   passed to the container when set
EOF
}
die() { echo "docker.sh: $*" >&2; exit 1; }
[ $# -gt 0 ] || { usage >&2; exit 1; }
CMD="$1"; shift
case "$CMD" in -h|--help) usage; exit 0 ;; esac

command -v docker > /dev/null 2>&1 || die "docker not found on PATH (README: Docker Image needs Docker with the NVIDIA Container Toolkit)"
docker info > /dev/null 2>&1 || die "the Docker daemon is not reachable (is it running, and may this user use it?)"
IMAGE="${IMAGE:-sbrantq/poseidon-cgo2027}"
have_image() {
  docker image inspect "$IMAGE" > /dev/null 2>&1 ||
    die "no image $IMAGE: docker pull sbrantq/poseidon-cgo2027 (or scripts/docker.sh build)"
}
GPU_ARGS=(--gpus all)
[ -n "${GPU_ID:-}" ] && GPU_ARGS=(--gpus "device=$GPU_ID")
ENV_ARGS=()
for v in BENCHES GENGA_HERBIE PROFILES SURROGATE_SIZES NUM_RUNS; do
  [ -n "${!v:-}" ] && ENV_ARGS+=(-e "$v=${!v}")
done
# The container runs a fresh copy of this repository's files (sources/ is built into the image).
ctr() {   # <docker run options...> -- <command...>
  local opts=()
  while [ "$1" != -- ]; do opts+=("$1"); shift; done; shift
  # Files the container writes to /out (./out on the host) are handed to the calling user.
  docker run --rm --memory "${MEMORY:-24g}" "${opts[@]}" -e HOST_IDS="$(id -u):$(id -g)" \
    -v "$ART:/repo:ro" "$IMAGE" bash -c '
    tar -C /repo --exclude=./.git --exclude=./sources --exclude=./out --exclude=./toolchain \
        --exclude=./results --exclude=./picks_check --exclude=./explain --exclude=./modal/bundle \
        -cf - . | tar -C /artifact -xf - || exit 1
    "$@"; rc=$?
    [ -d /out ] && chown -R "$HOST_IDS" /out 2>/dev/null
    exit $rc' ctr "$@"
}

case "$CMD" in
  build)
    [ -f "$SRCS/Enzyme/enzyme/CMakeLists.txt" ] ||
      die "sources/Enzyme is empty: git submodule update --init --recursive"
    # The image's CUDA code is compiled for the compute capability of this machine's GPU
    # (12.0, the RTX 5090, without a GPU) unless a CUDA_ARCHS build argument is given.
    ARCH_ARGS=()
    case " $* " in
      *CUDA_ARCHS=*) ;;
      *) cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
                 -i "$(case "${GPU_ID:-0}" in *[!0-9]*) echo 0 ;; *) echo "${GPU_ID:-0}" ;; esac)" 2>/dev/null |
               head -1 | tr -d ' .')"
         if [ -n "$cc" ]; then
           echo "docker.sh: building for compute capability ${cc%?}.${cc: -1} (the GPU of this machine)" >&2
           ARCH_ARGS=(--build-arg "CUDA_ARCHS=${cc}-real")
         fi ;;
    esac
    docker build -f "$ART/docker/Dockerfile" -t sbrantq/poseidon-cgo2027 "${ARCH_ARGS[@]}" "$@" "$CTX"
    exit 0 ;;
  run)
    have_image; mkdir -p "$PWD/out"
    ctr -it "${GPU_ARGS[@]}" "${ENV_ARGS[@]}" -v "$PWD/out:/out" -- \
      bash -c './run_artifact.sh "$@"; rc=$?; cp results_*.tar.gz /out/ 2>/dev/null
               [ -d table1 ] && cp -r table1 /out/; exit $rc' run_artifact "$@" ;;
  picks)
    have_image; mkdir -p "$PWD/out"
    ctr -v "$PWD/out:/out" -- \
      bash -c './run_artifact.sh --picks "$@"; rc=$?; rm -rf /out/picks_check; cp -r picks_check /out/ 2>/dev/null
               for d in benchmarks/*/cache*/; do mkdir -p "/out/picks_check/caches/${d%/}"; cp -r "$d." "/out/picks_check/caches/${d%/}/"; done
               exit $rc' picks "$@" ;;
  explain)
    have_image
    [ $# -gt 0 ] || die "explain needs a row id (scripts/docker.sh explain --list names them)"
    mkdir -p "$PWD/out"
    ctr -v "$PWD/out:/out" -- \
      bash -c './run_artifact.sh --explain "$@"; rc=$?; [ -d explain ] && { mkdir -p /out/explain; cp -r explain/. /out/explain/; }; exit $rc' explain "$@" ;;
  shell)
    have_image; mkdir -p "$PWD/out"
    command -v nvidia-smi > /dev/null 2>&1 || GPU_ARGS=()
    ctr -it "${GPU_ARGS[@]}" "${ENV_ARGS[@]}" -v "$PWD/out:/out" -- bash ;;
  *) usage >&2; die "unknown command '$CMD'" ;;
esac
