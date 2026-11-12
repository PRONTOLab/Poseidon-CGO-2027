#!/usr/bin/env bash
# expected.sh — read the expected decisions (expected/picks.tsv). Source it.
#
# expected/picks.tsv holds, per decision row of EXPECTED_RESULTS.md, how the
# row's configuration is selected and the configuration the compiler picks
# there with the shipped RTX 5090 cost model, spelled as the compiler's own
# `[poseidon] pin:` value (what -poseidon-apply-rewrites takes):
#   id      the row id used in EXPECTED_RESULTS.md
#   bench   benchmark directory
#   select  tau=<t> | share=<s> (a frontier point's -poseidon-cost-budget) |
#           head (the cheapest frontier point) | zero (budget 0)
#   what    the configuration in words
#   pin     the applied candidates, items joined by ',' (sites in any order)
# A change of the compiler that moves a decision changes this file (and the
# table generated from it), not the scripts.
_EXPECTED_TSV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/expected/picks.tsv"

expected_field() {   # <id> <column name>
  awk -F'\t' -v id="$1" -v col="$2" '
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $1 == id { print $(c[col]); found = 1; exit }
    END { if (!found) exit 1 }' "$_EXPECTED_TSV" ||
    { echo "expected.sh: no row '$1' in $_EXPECTED_TSV" >&2; return 1; }
}

# The row's -poseidon-cost-budget / -poseidon-tau; fails (and says so) when the
# row is not selected that way, so an empty value never reaches a compile.
_expected_sel() {   # <id> <share|tau>
  local v; v="$(expected_field "$1" select | sed -n "s/^$2=//p")"
  [ -n "$v" ] || { echo "expected.sh: row '$1' is not selected by $2" >&2; return 1; }
  echo "$v"
}
expected_share() { _expected_sel "$1" share; }
expected_tau()   { _expected_sel "$1" tau; }

# The applied candidates of a compile, from its `[poseidon] pin:` lines: every
# non-empty item, one per line, sorted (a joint solve prints one line per site).
pins_of_log() {   # <solve log>
  grep -a '^\[poseidon\] pin: ' "$1" | sed 's/^\[poseidon\] pin: //' | tr ',' '\n' |
    sed '/^$/d' | sort -u
}

# expected_report <id> <solve log>: one line saying whether the compile applied
# the row's expected configuration. Never fails: a moved pick is reported, and
# the measurement that follows is of what the compiler actually picked.
expected_report() {
  local want got
  want="$(expected_field "$1" pin | tr ',' '\n' | sed '/^$/d' | sort -u)" || return 0
  got="$(pins_of_log "$2")"
  if [ "$want" = "$got" ]; then
    echo "[expected] $1: the compiler applied the configuration of EXPECTED_RESULTS.md"
  else
    echo "[expected] $1: DIFFERENT configuration than EXPECTED_RESULTS.md (expected with the shipped RTX 5090 model):"
    echo "[expected]   expected: $(echo "$want" | paste -sd',' -)"
    echo "[expected]   applied:  $(echo "$got" | paste -sd',' -)"
  fi
}
