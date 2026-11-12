#!/usr/bin/env bash
# make_caches.sh [--device rtx5090|b300|gh200|all]: regenerate the shipped
# solve caches, devices/<device>/caches/ (see scripts/caches.sh).
#
# Per device: the benchmarks' cache directories the device's rows use are
# emptied of DP tables and accuracy memos, every row is compiled by
# scripts/check_picks.sh --no-shipped-cache (so each table and memo entry is
# computed by this compiler, here), and what the compiles wrote is copied to
# devices/<device>/caches/<bench>/<cache dir>/: the GENGA tables (table.json)
# and the ozp and R-ChFSI accuracy memos (matmulacc_*.json). The ozp table is
# not kept: one table.json holds one entry per function, and every device's
# ozp rows solve the same function in the same directory.
#
# An entry is used by another build only when every part of its key matches,
# the compiler's included: the digest of the pass sources and the LLVM version
# and revision. The shipped caches were made in the container of
# docker/Dockerfile, whose LLVM (built from the source tarball) records no
# revision, as does an LLVM built by scripts/build_toolchain.sh.
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ART"
DEVICES=rtx5090
case "${1:-}" in
  --device) DEVICES="${2:?--device needs a device}" ;;
  --device=*) DEVICES="${1#*=}" ;;
  -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "make_caches.sh: unknown argument $1" >&2; exit 2 ;;
esac
export POSEIDON_ARTIFACT_NO_GPU=1 COST_MODEL="${COST_MODEL:-$ART/cost_models/cm_sm_120_RTX5090.csv}"
source "$ART/env.sh" || exit 1
source "$ART/scripts/expected.sh"
[ "$DEVICES" = all ] && DEVICES="$(devices_list | paste -sd' ' -)"

dirs_of() {   # <device> <bench>: the cache directories its rows write
  local sfx=""; [ "$1" = rtx5090 ] || sfx="_$1"
  case "$2" in
    genga_nbody) echo "cache$sfx cache_kep_rolled$sfx cache_unified$sfx"; [ -z "$sfx" ] && echo cache_kep_herbieoff ;;
    ozp|chase_eigensolver) echo cache ;;
  esac
}

RC=0
for D in $DEVICES; do
  device_field "$D" gpu_arch > /dev/null || exit 2
  BENCHES="$(for r in $(device_rows "$D"); do expected_field "$r" bench; done | sort -u)"
  for B in $BENCHES; do
    for c in $(dirs_of "$D" "$B"); do
      rm -f "benchmarks/$B/$c/table.json" "benchmarks/$B/$c/budgets.txt" "benchmarks/$B/$c"/matmulacc_*.json
    done
  done
  echo "== $D: solving every row without shipped caches"
  bash scripts/check_picks.sh --no-shipped-cache --device "$D" || RC=1
  DST="$ART/devices/$D/caches"
  rm -rf "${DST:?}"
  for B in $BENCHES; do
    for c in $(dirs_of "$D" "$B"); do
      S="benchmarks/$B/$c"
      [ -d "$S" ] || continue
      mkdir -p "$DST/$B/$c"
      [ "$B" = genga_nbody ] && [ -f "$S/table.json" ] && cp "$S/table.json" "$DST/$B/$c/"
      cp "$S"/matmulacc_*.json "$DST/$B/$c/" 2>/dev/null
      rmdir "$DST/$B/$c" 2>/dev/null
    done
  done
  mkdir -p "$DST"
  {
    echo "compiler_plugin_md5=$(md5sum < "$(bash scripts/detect.sh plugin)" | cut -c1-32)"
    echo "clang=$("$CLANG" --version | head -1)"
    echo "cuda=$("$CUDA/bin/nvcc" --version | sed -n 's/^Cuda compilation tools, //p')"
    echo "host=$(uname -m)"
    echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    (cd "$DST" && find . -type f ! -name 'MANIFEST*' | sort | xargs -r md5sum)
  } > "$DST/MANIFEST.tmp"
  mv "$DST/MANIFEST.tmp" "$DST/MANIFEST"
  echo "== $D: wrote devices/$D/caches/ ($(find "$DST" -type f | wc -l) files)"
done
exit $RC
