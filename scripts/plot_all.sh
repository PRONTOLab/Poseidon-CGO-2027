#!/usr/bin/env bash
# plot_all.sh [bench...]: redraw the paper figures into figures/, without a GPU; a benchmark
# not run here is plotted from its recorded/rtx5090/ data.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="$(bash scripts/detect.sh py)" || { echo "plot_all: no python with numpy+matplotlib (set PY)" >&2; exit 1; }
mkdir -p figures

BENCHES="genga_nbody ozp chase_eigensolver"
scripts_for() {
  case "$1" in
    genga_nbody)       echo "genga_figures.py" ;;
    ozp)               echo "plot_ozp.py" ;;
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
