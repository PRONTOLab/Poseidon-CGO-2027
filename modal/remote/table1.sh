#!/usr/bin/env bash
# table1.sh <out dir> [row ...]: the B300 column of Table 1 on this B300 (scripts/table1.sh of the
# artifact, with the toolchain of the build stage).
set -uo pipefail
OUT="${1:?usage: table1.sh <out dir> [row ...]}"; shift
TC="${TC:-/toolchain}"; BUNDLE="${BUNDLE:-/bundle}"; WORK="${WORK:-/work}"
log() { echo "[table1 $(date -u +%H:%M:%S)] $*"; }
[ -f "$TC/poseidon.stamp" ] || { log "no toolchain in $TC: run the build stage first"; exit 99; }
# A build of an earlier bundle would compile these configurations with another compiler.
case "$(cat "$TC/poseidon.stamp")" in
  "src=$(sha256sum "$BUNDLE/poseidon-src.tar.gz" | cut -d' ' -f1) "*) ;;
  *) log "the toolchain in $TC was built from another poseidon-src.tar.gz: run the build stage first"; exit 99 ;;
esac
ART="$WORK/artifact"
rm -rf "$ART" && mkdir -p "$WORK/tmp" && tar -C "$WORK" -xzf "$BUNDLE/artifact.tar.gz"
export POSEIDON_BUILD="$TC/poseidon-build" TMPDIR="$WORK/tmp" GPU_ID=0
exec bash "$ART/scripts/table1.sh" --device b300 --out "$OUT" "$@"
