#!/usr/bin/env bash
# text_results.sh [--dir <dir>] [force-then-drift|deployed-sizes]: the measured numbers of the
# results without a figure, from <dir>/genga_nbody/unified.tsv and <dir>/ozp/surrogate.tsv.
# <dir>: benchmarks/ (default; the tables of the last run) or the data/ directory of a results bundle.
set -uo pipefail
ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="$ART/benchmarks" WHICH=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) D="${2:?--dir needs a directory}"; shift ;;
    force-then-drift|deployed-sizes) WHICH="$1" ;;
    -h|--help) sed -n '2,3p' "$0" | sed 's/^# //'; exit 0 ;;
    *) echo "text_results.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
source "$ART/scripts/expected.sh"

if [ "$WHICH" != deployed-sizes ]; then
  U="$D/genga_nbody/unified.tsv"
  if [ -s "$U" ]; then
    awk -F'\t' -v b="$(expected_share genga-unified)" 'NR > 1 && $1 + 0 == b + 0 { found = 1
      printf "Sec. 4.2, force-then-drift (share %s): %.2fx, L2 %.2e;", b, $3, $4
      print " compare with the row \"GENGA force-then-drift, one solve\" of Expected Results in README.md" }
      END { if (!found) print "Sec. 4.2, force-then-drift: no point at the share of row genga-unified (a measured cost model decides other points)" }' "$U"
  else
    echo "Sec. 4.2, force-then-drift: no $U yet (benchmarks/genga_nbody/run_force_then_drift.sh)"
  fi
fi

if [ "$WHICH" != force-then-drift ]; then
  T="$D/ozp/surrogate.tsv"
  if [ -s "$T" ]; then
    awk -F'\t' '$2 == "nm8" && $1 == 1024 {lo = $5} $2 == "nm8" && $1 == 8192 {hi = $5; r = $3}
      END { if (lo != "" && hi != "") {
              printf "Sec. 4.3, deployed sizes, Ozaki-II nm = 8: %.1fx at N = 1024, %.1fx and RMSD %.1e at N = 8192;", lo, hi, r
              print " compare with the row \"ozp, Ozaki-II nm = 8 deployed at N = 1024 and N = 8192\" of Expected Results in README.md"
            } else print "Sec. 4.3, deployed sizes: the table has no nm8 rows at both N = 1024 and N = 8192" }' "$T"
  else
    echo "Sec. 4.3, deployed sizes: no $T yet (benchmarks/ozp/run_deployed_sizes.sh)"
  fi
fi
