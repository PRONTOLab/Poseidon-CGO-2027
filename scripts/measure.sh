#!/usr/bin/env bash
# measure.sh: checks a launch passes before its numbers are recorded; a failed check exits 1.
# A CUDA start-up failure can print zeros and exit 0, so a missing or non-positive time stops the step.

measure_fail() {   # <launch> <what>
  echo "measure: $1: $2. The step stops; nothing of this launch is recorded." >&2
  if [ -n "${MEASURE_OUT:-}" ]; then
    echo "measure: the last lines it printed:" >&2
    printf '%s\n' "$MEASURE_OUT" | tail -n 12 | sed 's/^/    /' >&2
  fi
  exit 1
}

measure_parse() {   # <launch> <spec...>
  local _ml="$1" _ms _mv _mk _mn _mx
  shift
  for _ms in "$@"; do
    _mv="${_ms%%=*}" _mk="${_ms#*=}"
    _mn="${_mk#*:}" _mk="${_mk%%:*}"
    case "$_mk" in time|acc) ;; *) echo "measure: spec '$_ms' is not <var>=<time|acc>:<label>" >&2; exit 2 ;; esac
    _mx="$(printf '%s\n' "${MEASURE_OUT:-}" | sed -n "s/.*$_mn[ =]*\([^ ]*\).*/\1/p" | head -1)"
    [ -n "$_mx" ] || measure_fail "$_ml" "it printed no '$_mn'"
    if [ "$_mk" = time ]; then
      awk -v v="$_mx" 'BEGIN {
          if (v !~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/) exit 1
          x = v + 0; if (x - x != 0) exit 1 }' ||
        measure_fail "$_ml" "'$_mn' is $_mx, not a finite number"
      awk -v v="$_mx" 'BEGIN { exit !(v + 0 > 0) }' ||
        measure_fail "$_ml" "'$_mn' is $_mx; a time must be positive"
    else
      awk -v v="$_mx" 'BEGIN {
          if (v ~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/) exit 0
          exit !(tolower(v) ~ /^[-+]?(nan|inf|infinity)$/) }' ||
        measure_fail "$_ml" "'$_mn' is $_mx, not a number"
    fi
    printf -v "$_mv" '%s' "$_mx"
  done
}

# Call from the script's own shell, not in $(...) or a pipeline: a failed check is an exit.
measure() {   # <launch> <spec...> -- <command...>
  local _ml="$1" _mspecs=() _mrc
  shift
  while [ $# -gt 0 ] && [ "$1" != -- ]; do _mspecs+=("$1"); shift; done
  [ "${1:-}" = -- ] && [ $# -ge 2 ] || { echo "measure: $_ml: no command after --" >&2; exit 2; }
  shift
  MEASURE_OUT="$("$@" 2>&1)"
  _mrc=$?
  MEASURE_STATUS=$_mrc
  [ "$_mrc" != 0 ] && [[ " ${MEASURE_ALLOW:-} " == *" $_mrc "* ]] && return 0
  [ "$_mrc" = 0 ] || measure_fail "$_ml" "it exited with status $_mrc ($*)"
  measure_parse "$_ml" "${_mspecs[@]}"
}
