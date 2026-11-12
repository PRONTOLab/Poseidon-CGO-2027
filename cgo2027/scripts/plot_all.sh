#!/usr/bin/env bash
# plot_all.sh — regenerate EVERY paper figure, from one entry point.
#
#   ./scripts/plot_all.sh          # all figures into figures/
#   ./scripts/plot_all.sh ozp      # just one benchmark's figures
#
# Why this exists: the figures were reachable only as `make figures` inside each
# benchmark directory, so redrawing the paper meant knowing which directories
# to visit and in what state each had to be. This is the single
# entry point over the same per-benchmark scripts -- it adds no plotting logic
# of its own and changes no figure.
#
# It needs NO environment variables and NO GPU. Every plotting script falls back
# to the shipped `golden/rtx5090/` data when a benchmark has not been run here, so this
# works on a fresh unpack and is the fastest way to check that the Python side
# of the artifact is healthy before committing hours to a full run. plot_ozp.py
# draws Fig. 5, plot_chase.py Fig. 6 and genga_figures.py Fig. 3/4.
# Run one directly with --out (genga: --out-dir) and --data-dir / --trace-dir
# to plot a different bundle somewhere else.
#
# IT OVERWRITES figures/. The shipped PDFs are drawn from golden/rtx5090/; a
# redraw is not byte-identical (matplotlib stamps a creation date). Copy
# figures/ aside first if you need to compare against what shipped.
#
# One benchmark per line: <bench dir> <script> [script...]
set -uo pipefail
cd "$(dirname "$0")/.."
# Plotting needs python only, not the compiler or a GPU.
PY="$(bash scripts/detect.sh py)" || { echo "plot_all: no python with numpy+matplotlib (set PY)" >&2; exit 1; }
mkdir -p figures

BENCHES="fig2_drift genga_nbody ozp chase_eigensolver"
scripts_for() {
  case "$1" in
    fig2_drift)        echo "plot_drift.py" ;;
    genga_nbody)       echo "genga_figures.py" ;;
    ozp)               echo "plot_ozp.py plot_surrogate.py" ;;
    chase_eigensolver) echo "plot_chase.py" ;;
    *)                 echo "" ;;
  esac
}

[ $# -gt 0 ] && BENCHES="$*"
LOGDIR="$(mktemp -d)"; trap 'rm -rf "$LOGDIR"' EXIT
FAILED=""
for b in $BENCHES; do
  ss="$(scripts_for "$b")"
  if [ -z "$ss" ]; then
    echo "plot_all: no plotting script registered for '$b'" >&2
    FAILED="$FAILED $b"; continue
  fi
  for s in $ss; do
    printf '==== %s / %s ' "$b" "$s"
    log="$LOGDIR/${b}_${s}.log"
    if ( cd "benchmarks/$b" && "$PY" "$s" ) > "$log" 2>&1; then
      echo "OK"
    else
      echo "FAILED"
      sed 's/^/       /' "$log" | tail -8
      FAILED="$FAILED $b/$s"
    fi
  done
done

echo
echo "figures in $(pwd)/figures:"
ls -1 figures/*.pdf 2>/dev/null | sed 's|^|  |' || echo "  (none)"
[ -z "$FAILED" ] || { echo "plot_all: FAILED:$FAILED" >&2; exit 1; }
echo "plot_all: all figures regenerated"
