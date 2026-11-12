#!/usr/bin/env bash
# docker_test.sh <llvm-tree> [step ...]: the Modal container recipe, locally,
# without a GPU and without Modal.
#
#   image     build the image poseidon_b300.py defines (same base, apt and pip
#             packages, Racket and Rust install commands; Python from apt in a
#             venv instead of Modal's add_python)
#   start     start the test container (sources mounted read-only)
#   herbie    remote/build.sh herbie
#   poseidon  remote/build.sh poseidon against <llvm-tree> (an existing build of
#             the pinned LLVM, mounted read-only at the same path)
#   smoke     remote/smoke.sh with SMOKE_NO_GPU=1 (compiles, no runs)
#   picks     the artifact's compile-only decision check (run_artifact.sh --picks)
#   herbierun GENGA's drift row solved by a Herbie search, as on a measured GPU
#   llvm      remote/build.sh llvm with LLVM_CONFIGURE_ONLY=1 (fetch + configure)
#   stop      remove the container
# Default steps: image start herbie poseidon smoke picks llvm stop.
# LLVM_ALIAS=<path>: also mount the LLVM source root there (an LLVM build
# configured under another path, e.g. through a symlink, names that path in
# its CMake package files).
# Logs: test/logs/<step>.log. Containers and images are named poseidon-b300-*.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_TREE="${1:?usage: docker_test.sh <llvm-tree> [step ...]}"; shift
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(image start herbie poseidon smoke picks llvm stop)
IMG=poseidon-b300-test:latest
CTR=poseidon-b300-test
LOGS="$HERE/test/logs"; mkdir -p "$LOGS"
PYV="$HERE/.venv/bin/python"
JOBS="${JOBS:-32}"
LLVM_SRC_ROOT="$(dirname "$LLVM_TREE")"   # LLVMConfig.cmake names the source tree too

ex() { docker exec -e TC=/toolchain -e BUNDLE=/bundle "$@"; }

step_image() {
  local ctx; ctx="$(mktemp -d "$HERE/test/.ctx.XXXXXX")"
  ( cd "$HERE" && "$PYV" - "$ctx/Dockerfile" ) <<'PY'
import sys, poseidon_b300 as p
apt = " ".join(p.APT_PACKAGES)
env = " ".join(f"{k}={v}" for k, v in p.RUST_ENV.items())
open(sys.argv[1], "w").write(f"""FROM {p.CUDA_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {apt} python3 python3-venv
RUN python3 -m venv /opt/py && /opt/py/bin/pip install {" ".join(p.PIP_PACKAGES)}
ENV PATH=/opt/py/bin:$PATH
RUN {p.RACKET_INSTALL}
RUN {p.RUST_INSTALL}
ENV {env}
""")
PY
  cat "$ctx/Dockerfile"
  docker build -t "$IMG" "$ctx"; local rc=$?
  rm -rf "$ctx"; return $rc
}
step_start() {
  docker rm -f "$CTR" >/dev/null 2>&1
  docker run -d --name "$CTR" \
    -v "$HERE/bundle:/bundle:ro" -v "$HERE/remote:/opt/poseidon-b300:ro" \
    -v "$LLVM_SRC_ROOT:$LLVM_SRC_ROOT:ro" \
    ${LLVM_ALIAS:+-v "$LLVM_SRC_ROOT:$LLVM_ALIAS:ro"} "$IMG" sleep infinity
}
step_herbie()   { ex "$CTR" bash /opt/poseidon-b300/build.sh herbie; }
step_poseidon() { ex -e LLVM_PREFIX="$LLVM_TREE" -e JOBS="$JOBS" "$CTR" bash /opt/poseidon-b300/build.sh poseidon; }
step_smoke()    { ex -e SMOKE_NO_GPU=1 "$CTR" bash /opt/poseidon-b300/smoke.sh /toolchain/smoke-out; }
step_picks() {
  ex "$CTR" bash -c 'set -e; rm -rf /work && mkdir -p /work/tmp && tar -C /work -xzf /bundle/artifact.tar.gz
    cd /work/cgo2027 && export POSEIDON_BUILD=/toolchain/poseidon-build TMPDIR=/work/tmp PY=$(command -v python3)
    ./run_artifact.sh --picks; rc=$?; cat picks_check/picks.tsv; exit $rc'
}
# GENGA's B300 path on the CPU: a cost model that is not byte-identical to the
# shipped one (the shipped one plus a comment line) makes common.sh derive the
# x1e6 copy and its Herbie platform with this build's calibrator and Herbie,
# as on a measured device; then the drift row is solved with a Herbie search
# (GENGA_HERBIE=run) from the shipped profile, and its configuration compared
# with the expected one.
step_herbierun() {
  ex "$CTR" bash -c 'cd /work/cgo2027 || exit 1
    export POSEIDON_BUILD=/toolchain/poseidon-build TMPDIR=/work/tmp POSEIDON_ARTIFACT_NO_GPU=1 \
           PROFILES=shipped GENGA_HERBIE=run HERBIE=/toolchain/herbie/bin/herbie
    { cat cost_models/cm_sm_120_RTX5090.csv; echo "# local test copy"; } > /work/cm_sm_120_localtest.csv
    export COST_MODEL=/work/cm_sm_120_localtest.csv
    source scripts/expected.sh
    R=genga-drift-algebraic
    ( cd benchmarks/genga_nbody && rm -rf cache_kep_rolled measured && bash scripts/rows.sh $R /work/herbierun.log ) || { tail -30 /work/herbierun.log; exit 1; }
    grep -a -E "Herbie answers|platform" /work/herbierun.log | head
    want="$(expected_field $R pin | tr , "\n" | sed "/^$/d" | sort -u)"
    got="$(pins_of_log /work/herbierun.log)"
    echo "expected: $want"; echo "applied:  $got"
    [ "$want" = "$got" ] && echo "herbierun: SAME" || { echo "herbierun: DIFFERENT"; exit 1; }'
}
step_llvm() { ex -e TC=/tmp/tc-llvm-test -e LLVM_CONFIGURE_ONLY=1 "$CTR" bash /opt/poseidon-b300/build.sh llvm; }
step_stop() { docker rm -f "$CTR"; }

for s in "${STEPS[@]}"; do
  echo "==== docker_test: $s ($(date -u +%H:%M:%S)) ===="
  T0=$(date +%s)
  "step_$s" > "$LOGS/$s.log" 2>&1; rc=$?
  echo "==== docker_test: $s exit $rc after $(( $(date +%s) - T0 )) s (log: test/logs/$s.log) ===="
  tail -5 "$LOGS/$s.log"
  if [ "$rc" != 0 ] && [ "$s" != stop ]; then echo "==== docker_test: stopping after a failed step ===="; break; fi
done
