#!/usr/bin/env bash
# expected.sh: read the expected decisions (expected/picks*.tsv) and the
# devices they were made for (devices/devices.tsv). Source it.
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
# expected/picks_<device>.tsv holds the same for another device's saved cost
# models (the paper's Table 1 columns), with one more column:
#   cost_model  the model the row is solved with, relative to the artifact
#               root (GENGA rows: the model's x1e6 copy)
# Row ids are unique across the files, so a row id names its device.
# A change of the compiler that moves a decision changes these files (and the
# tables generated from them), not the scripts.
_EXPECTED_ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_DEVICES_TSV="$_EXPECTED_ART/devices/devices.tsv"

_tsv_field() {   # <id> <column name> <files...>: the first row with that id
  local id="$1" col="$2"; shift 2
  awk -F'\t' -v id="$id" -v col="$col" '
    FNR == 1 { delete c; for (i = 1; i <= NF; i++) c[$i] = i; next }
    $1 == id { if (col in c) print $(c[col]); found = 1; exit }
    END { if (!found) exit 1 }' "$@"
}

expected_field() {   # <id> <column name>
  _tsv_field "$1" "$2" "$_EXPECTED_ART"/expected/picks*.tsv ||
    { echo "expected.sh: no row '$1' in expected/picks*.tsv" >&2; return 1; }
}

device_field() {   # <device> <column name>
  _tsv_field "$1" "$2" "$_DEVICES_TSV" ||
    { echo "expected.sh: no device '$1' in devices/devices.tsv (devices: $(devices_list | paste -sd' ' -))" >&2; return 1; }
}
devices_list() { awk -F'\t' 'NR > 1 {print $1}' "$_DEVICES_TSV"; }
device_rows() {   # <device>: its row ids, in file order
  local f; f="$(device_field "$1" picks)" || return 1
  awk -F'\t' 'NR > 1 {print $1}' "$_EXPECTED_ART/$f"
}
row_device() {   # <id>: the device whose picks file holds the row
  local d
  for d in $(devices_list); do
    device_rows "$d" | grep -qxF "$1" && { echo "$d"; return 0; }
  done
  echo "expected.sh: no row '$1' in expected/picks*.tsv" >&2; return 1
}
# The cost model a row is solved with: its own column, else its device's.
row_cost_model() {   # <id>
  local m d
  m="$(expected_field "$1" cost_model)" || return 1
  if [ -z "$m" ]; then d="$(row_device "$1")" && m="$(device_field "$d" cost_model)" || return 1; fi
  echo "$_EXPECTED_ART/$m"
}

# Exports what a compile of the row is priced and targeted with: its cost
# model, its device's architecture and Herbie answers. Fails on missing data.
row_env() {   # <id>
  local d m arch ma
  d="$(row_device "$1")" && m="$(row_cost_model "$1")" && arch="$(device_field "$d" gpu_arch)" || return 1
  [ -f "$m" ] || { echo "expected.sh: the cost model $m of row $1 is missing" >&2; return 1; }
  ma="$(sed -n 's/^# native_arch=//p' "$m" | head -1 | tr -d ' ')"
  [ "$ma" = "$arch" ] || { echo "expected.sh: row $1: $m was measured on $ma, its device $d is $arch" >&2; return 1; }
  export COST_MODEL="$m" GPU_ARCH="$arch" HERBIE_ANSWERS="$_EXPECTED_ART/$(device_field "$d" herbie_answers)"
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
# The expected configuration is a function of the row's cost model: a compile
# priced with another model (a GPU's own measured one) makes that model's
# decision, which is reported as such and not compared; ./run_artifact.sh
# --picks checks the expected decisions from the shipped models on any machine.
expected_report() {
  local want got model
  want="$(expected_field "$1" pin | tr ',' '\n' | sed '/^$/d' | sort -u)" || return 0
  got="$(pins_of_log "$2")"
  model="$(row_cost_model "$1")" || return 0
  if [ -n "${COST_MODEL:-}" ] && ! cmp -s "$COST_MODEL" "$model" &&
     ! { [ "$(row_device "$1")" = rtx5090 ] && cmp -s "$COST_MODEL" "$_EXPECTED_ART/cost_models/x1e6/cm_native_x1e6.csv"; }; then
    echo "[expected] $1: decided with this run's cost model ($(basename "$COST_MODEL")), applied: $(echo "$got" | paste -sd',' -)"
    echo "[expected]   not compared: the row's configuration is the decision of $(basename "$model") (./run_artifact.sh --picks checks it)"
    return 0
  fi
  if [ "$want" = "$got" ]; then
    echo "[expected] $1: the compiler applied the configuration of EXPECTED_RESULTS.md"
  else
    echo "[expected] $1: DIFFERENT configuration than EXPECTED_RESULTS.md (expected with $(basename "$model")):"
    echo "[expected]   expected: $(echo "$want" | paste -sd',' -)"
    echo "[expected]   applied:  $(echo "$got" | paste -sd',' -)"
  fi
}
