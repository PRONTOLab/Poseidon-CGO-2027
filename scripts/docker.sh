#!/usr/bin/env bash
# docker.sh build|run|picks|explain [...]: the artifact's container (README:
# Setup, A) in one word each. Run it from anywhere inside the repository.
#
#   scripts/docker.sh build [--herbie] [docker build options...]
#       builds poseidon-cgo2027 from docker/Dockerfile (and, with --herbie,
#       poseidon-cgo2027:herbie from docker/Dockerfile.herbie on top of it)
#   scripts/docker.sh run [run_artifact.sh options...]
#       ./run_artifact.sh in the container on the GPU (--gpus all, or
#       GPU_ID=<n> for one); results land in ./out on the host
#   scripts/docker.sh picks [--device rtx5090|b300|gh200|all]
#       ./run_artifact.sh --picks in the container, without a GPU; the
#       per-row logs and picks.tsv land in ./out/picks_check
#   scripts/docker.sh explain <row> [poseidon-explain options...]
#       ./run_artifact.sh --explain <row> in the container, without a GPU;
#       the report, compile log and JSON land in ./out/explain/<row>
#   scripts/docker.sh shell
#       an interactive shell in the container on the GPU
#
# IMAGE=<name> selects another image (default poseidon-cgo2027, or
# poseidon-cgo2027:herbie when it exists). Containers are capped at
# MEMORY (default 24g): a Herbie search takes up to 13 GB.
set -euo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRCS="$(bash "$ART/scripts/detect.sh" sources)" || exit 1
CTX="$ART"   # the build context: the repository root
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "docker.sh: $*" >&2; exit 1; }
[ $# -gt 0 ] || { usage >&2; exit 1; }
CMD="$1"; shift
case "$CMD" in -h|--help) usage; exit 0 ;; esac

command -v docker > /dev/null 2>&1 || die "docker not found on PATH (README: Setup, A needs Docker with the NVIDIA Container Toolkit)"
docker info > /dev/null 2>&1 || die "the Docker daemon is not reachable (is it running, and may this user use it?)"
if [ -z "${IMAGE:-}" ]; then
  IMAGE=poseidon-cgo2027
  docker image inspect poseidon-cgo2027:herbie > /dev/null 2>&1 && IMAGE=poseidon-cgo2027:herbie
fi
have_image() {
  docker image inspect "$IMAGE" > /dev/null 2>&1 ||
    die "no image $IMAGE: scripts/docker.sh build (or docker load -i <the deposit's image file>)"
}
GPU_ARGS=(--gpus all)
[ -n "${GPU_ID:-}" ] && GPU_ARGS=(--gpus "device=$GPU_ID")
# The variables of README: Configuration and Choices that a run reads, handed
# to the container when they are set here.
ENV_ARGS=()
for v in BENCHES GENGA_HERBIE PROFILES SURROGATE_SIZES NUM_RUNS PARALLEL_GPUS; do
  [ -n "${!v:-}" ] && ENV_ARGS+=(-e "$v=${!v}")
done
ctr() {   # <docker run options...> -- <command...>
  local opts=()
  while [ "$1" != -- ]; do opts+=("$1"); shift; done; shift
  docker run --rm --memory "${MEMORY:-24g}" "${opts[@]}" "$IMAGE" "$@"
}

case "$CMD" in
  build)
    WITH_HERBIE=0 ARGS=()
    for a in "$@"; do [ "$a" = --herbie ] && WITH_HERBIE=1 || ARGS+=("$a"); done
    [ -f "$SRCS/Enzyme/enzyme/CMakeLists.txt" ] ||
      die "sources/Enzyme is empty: git submodule update --init --recursive"
    docker build -f "$ART/docker/Dockerfile" -t poseidon-cgo2027 "${ARGS[@]}" "$CTX"
    [ "$WITH_HERBIE" = 1 ] && docker build -f "$ART/docker/Dockerfile.herbie" -t poseidon-cgo2027:herbie "$CTX"
    exit 0 ;;
  run)
    have_image; mkdir -p "$PWD/out"
    ctr -it "${GPU_ARGS[@]}" "${ENV_ARGS[@]}" -v "$PWD/out:/out" -- \
      bash -c './run_artifact.sh "$@"; rc=$?; cp results_*.tar.gz /out/ 2>/dev/null; exit $rc' run_artifact "$@" ;;
  picks)
    have_image; mkdir -p "$PWD/out"
    ctr -v "$PWD/out:/out" -- \
      bash -c './run_artifact.sh --picks "$@"; rc=$?; rm -rf /out/picks_check; cp -r picks_check /out/ 2>/dev/null; exit $rc' picks "$@" ;;
  explain)
    have_image
    [ $# -gt 0 ] || die "explain needs a row id (scripts/docker.sh explain --list names them)"
    mkdir -p "$PWD/out"
    ctr -v "$PWD/out:/out" -- \
      bash -c './run_artifact.sh --explain "$@"; rc=$?; [ -d explain ] && { mkdir -p /out/explain; cp -r explain/. /out/explain/; }; exit $rc' explain "$@" ;;
  shell)
    have_image; mkdir -p "$PWD/out"
    # On the GPU when this machine has one; a shell without it otherwise.
    command -v nvidia-smi > /dev/null 2>&1 || GPU_ARGS=()
    ctr -it "${GPU_ARGS[@]}" "${ENV_ARGS[@]}" -v "$PWD/out:/out" -- bash ;;
  *) usage >&2; die "unknown command '$CMD'" ;;
esac
