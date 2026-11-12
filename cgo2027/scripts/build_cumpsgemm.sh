#!/usr/bin/env bash
# build_cumpsgemm.sh [prefix]: build cuMpSGEMM, the error-corrected FP16
# tensor-core GEMM the compiler's TCEC dispatch runtime delegates to, from the
# sources/cuMpSGEMM submodule (42dd92e and its submodules), into
# <prefix>/cuMpSGEMM (default prefix: deps/ of the artifact). It is the
# cumpsgemm step of scripts/build_toolchain.sh; extra options go to it
# (--cuda <dir>, --cuda-archs <list>, --jobs <n>).
#
# Then configure the compiler with
#   -DPOSEIDON_TCEC_USE_CUMPSGEMM=ON -DPOSEIDON_CUMPSGEMM_ROOT=<prefix>/cuMpSGEMM
# (or let scripts/build_toolchain.sh do that). The results of
# EXPECTED_RESULTS.md come from that build.
set -euo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
PREFIX="$ART/deps"
[ $# -gt 0 ] && [ "${1#-}" = "$1" ] && { PREFIX="$1"; shift; }
exec bash "$ART/scripts/build_toolchain.sh" --prefix "$PREFIX" --only cumpsgemm "$@"
