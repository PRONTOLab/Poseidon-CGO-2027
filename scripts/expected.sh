#!/usr/bin/env bash
# expected.sh: source it to read expected/picks*.tsv (per decision row: bench, select, what,
# pin) and devices/devices.tsv.
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
row_device() {   # <id>: the device of the picks file holding the row
  local d
  for d in $(devices_list); do
    device_rows "$d" | grep -qxF "$1" && { echo "$d"; return 0; }
  done
  echo "expected.sh: no row '$1' in expected/picks*.tsv" >&2; return 1
}
row_cost_model() {   # <id>: its device's cost model
  local d m
  d="$(row_device "$1")" && m="$(device_field "$d" cost_model)" || return 1
  echo "$_EXPECTED_ART/$m"
}

row_env() {   # <id>
  local d m arch ma
  d="$(row_device "$1")" && m="$(row_cost_model "$1")" && arch="$(device_field "$d" gpu_arch)" || return 1
  [ -f "$m" ] || { echo "expected.sh: the cost model $m of row $1 is missing" >&2; return 1; }
  ma="$(sed -n 's/^# native_arch=//p' "$m" | head -1 | tr -d ' ')"
  [ "$ma" = "$arch" ] || { echo "expected.sh: row $1: $m was measured on $ma, its device $d is $arch" >&2; return 1; }
  export COST_MODEL="$m" GPU_ARCH="$arch" HERBIE_ANSWERS="$_EXPECTED_ART/$(device_field "$d" herbie_answers)"
}

# Fails when the row is not selected that way, so an empty value never reaches a compile.
_expected_sel() {   # <id> <share|tau>
  local v; v="$(expected_field "$1" select | sed -n "s/^$2=//p")"
  [ -n "$v" ] || { echo "expected.sh: row '$1' is not selected by $2" >&2; return 1; }
  echo "$v"
}
expected_share() { _expected_sel "$1" share; }
expected_tau()   { _expected_sel "$1" tau; }

pins_of_log() {   # <solve log>
  grep -a '^\[poseidon\] pin: ' "$1" | sed 's/^\[poseidon\] pin: //' | tr ',' '\n' |
    sed '/^$/d' | sort -u
}

# Never fails: a moved pick is reported, and a compile priced with another cost model is not compared.
expected_report() {   # <id> <solve log>
  local want got model
  want="$(expected_field "$1" pin | tr ',' '\n' | sed '/^$/d' | sort -u)" || return 0
  got="$(pins_of_log "$2")"
  model="$(row_cost_model "$1")" || return 0
  if [ -n "${COST_MODEL:-}" ] && ! cmp -s "$COST_MODEL" "$model"; then
    echo "[expected] $1: decided with this run's cost model (${COST_MODEL#"$_EXPECTED_ART"/}), applied: $(echo "$got" | paste -sd',' -)"
    echo "[expected]   not compared: the row's configuration is the decision of ${model#"$_EXPECTED_ART"/} (./run_artifact.sh --picks checks it)"
    return 0
  fi
  if [ "$want" = "$got" ]; then
    echo "[expected] $1: the compiler applied the expected configuration (expected/picks.tsv)"
  else
    echo "[expected] $1: DIFFERENT configuration than expected/picks.tsv (expected with ${model#"$_EXPECTED_ART"/}):"
    echo "[expected]   expected: $(echo "$want" | paste -sd',' -)"
    echo "[expected]   applied:  $(echo "$got" | paste -sd',' -)"
  fi
}
