#!/usr/bin/env bash
# replay_compile_test.sh <llvm tree> [llvm alias]: remote/replay.sh with
# REPLAY_NO_GPU=1 (every row of the bundled replay pack compiled by name for
# sm_103 and checked: applied configuration, selector "manual", no Herbie
# search), in the CUDA 13.1 test image, using the toolchain
# solve_b300_locally.sh built in .solve-work/tc. Output: .solve-work/replay-out.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L="${1:?usage: replay_compile_test.sh <llvm tree> [llvm alias]}"; SRCROOT="$(dirname "$L")"
MOUNTS=(-v "$HERE:/m" -v "$SRCROOT:$SRCROOT:ro")
[ -n "${2:-}" ] && MOUNTS+=(-v "$SRCROOT:$2:ro")
docker run --rm --name poseidon-b300-replaytest --user "$(id -u):$(id -g)" -e HOME=/m/.solve-work/home \
  "${MOUNTS[@]}" poseidon-b300-test:latest bash -c '
    rm -rf /m/.solve-work/replay-out /m/.solve-work/replay-work
    TC=/m/.solve-work/tc BUNDLE=/m/bundle WORK=/m/.solve-work/replay-work REPLAY_NO_GPU=1 \
      bash /m/remote/replay.sh /m/.solve-work/replay-out' 2>&1 | grep -v -E '^(==|$)|NVIDIA|CUDA Version|governed|license|pulling|NGC|WARNING: The|Use the NVIDIA|cloud-native|Container image' |
  tee "$HERE/test/logs/replay_compile.log"
