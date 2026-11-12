#!/usr/bin/env bash
# solve_b300_locally.sh [row ...]: decide every B300 row on this machine, no GPU,
# and write the B300 replay pack (./replay_pack) that the Modal run replays.
#
#   LLVM_TREE=<build or install tree of LLVM 9d618fc> [LLVM_ALIAS=<path>] \
#   HERBIE_DIST=<pinned Herbie dist> ./solve_b300_locally.sh
#
# In the docker image test/docker_test.sh builds (poseidon-b300-test:latest,
# CUDA 13.1, compiling for sm_103), as this user, with this directory mounted
# at /m and the LLVM tree read-only: builds the compiler of
# bundle/poseidon-src.tar.gz (skipped when bundle/MANIFEST's tarball is
# unchanged), unpacks bundle/artifact.tar.gz, and runs remote/solve_rows.sh
# over b300_rows.tsv with the saved B300 cost models in b300_inputs/.
# LLVM_ALIAS: a second path the LLVM tree is mounted at (an LLVM build
# configured under another path, e.g. through a symlink, names that path in its
# CMake files). Rerun unchanged after make_bundle.sh picks up a new compiler.
#
# Output: replay_pack/ (picks_b300.tsv, compare_b300.tsv, MANIFEST,
# cost_models/, rows/<row>/), then run make_bundle.sh so the pack is shipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${LLVM_TREE:?set LLVM_TREE to the LLVM 9d618fc build or install tree}"
IMG="${IMG:-poseidon-b300-test:latest}"
CTR=poseidon-b300-solve
SRCROOT="$(dirname "$LLVM_TREE")"
docker image inspect "$IMG" >/dev/null 2>&1 ||
  { echo "no image $IMG: build it with test/docker_test.sh <llvm tree> image" >&2; exit 1; }
mkdir -p "$HERE/.solve-work/home"
docker rm -f "$CTR" >/dev/null 2>&1 || true
MOUNTS=(-v "$HERE:/m" -v "$SRCROOT:$SRCROOT:ro")
[ -n "${LLVM_ALIAS:-}" ] && MOUNTS+=(-v "$SRCROOT:$LLVM_ALIAS:ro")
# The pinned Herbie (upstream 73ba1fe76 + --num-taylor-terms) for the B300
# drift search: HERBIE_DIST=<its raco-distribute directory>.
[ -n "${HERBIE_DIST:-}" ] && MOUNTS+=(-v "$HERBIE_DIST:/herbie-dist:ro")
# Herbie searches run in this container: memory capped at 24 GiB.
docker run --rm --name "$CTR" --memory 24g --user "$(id -u):$(id -g)" -e HOME=/m/.solve-work/home \
  -e LLVM_PREFIX="$LLVM_TREE" -e JOBS="${JOBS:-32}" "${MOUNTS[@]}" "$IMG" \
  bash /m/remote/solve_rows.sh "$@" 2>&1 | tee "$HERE/solve_b300.log"
# The log is kept next to the pack; it names no local path.
sed -i -e "s#$SRCROOT#<llvm-tree-root>#g" ${HERBIE_DIST:+-e "s#$HERBIE_DIST#<herbie-dist>#g"} ${LLVM_ALIAS:+-e "s#$LLVM_ALIAS#<llvm-tree-root>#g"} "$HERE/solve_b300.log"
