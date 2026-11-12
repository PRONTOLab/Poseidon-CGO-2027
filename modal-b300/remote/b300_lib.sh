#!/usr/bin/env bash
# b300_lib.sh: one compile of one B300 row with the artifact's own flags, shared
# by the local solve (solve_rows.sh) and the replay on the B300 (replay.sh), so
# both compile with the same command line. Source it after setting:
#   ART       the unpacked artifact (…/cgo2027)
#   CMDIR     directory holding the saved B300 cost models
#   ROWS      b300_rows.tsv
# The rows read the profiles the artifact ships (PROFILES=shipped). GENGA rows
# price with the x1e6 copy of the saved B300 model (rev2) and the Herbie
# platform generated from it (cost_models/genga_x1e6/, made by the local
# solve); the drift row's Herbie answers are the B300 answer pack, searched
# locally for sm_103 against that platform. GENGA_HERBIE_MODE=run (local solve
# only) searches with $HERBIE; otherwise answers are imported and no Herbie
# binary exists at the path given, so a question without an answer stops the
# compile. COMPILE_WRAP: a command prefix for the compile (e.g. /usr/bin/time).

row_field() {   # <row> <column>
  awk -F'\t' -v id="$1" -v col="$2" '
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $1 == id { print $(c[col]); found = 1; exit }
    END { if (!found) exit 1 }' "$ROWS"
}

# Whatever env.sh resolves must not depend on a GPU or on the shipped model.
b300_env() {   # <row>
  export PROFILES=shipped GENGA_HERBIE="${GENGA_HERBIE_MODE:-replay}" GPU_ARCH=sm_103
  export COST_MODEL="$CMDIR/$(row_field "$1" cost_model)"
  [ -f "$COST_MODEL" ] || { echo "b300_lib: no cost model $COST_MODEL" >&2; return 1; }
}

# compile_row <row> <exe> <log> <reports dir> [compiler flags...]
# Runs in a subshell; the compile's output goes to <log>.
compile_row() {
  local row="$1" exe="$2" log="$3" rep="$4"; shift 4
  local bench; bench="$(row_field "$row" bench)"
  (
    b300_env "$row" || exit 1
    case "$bench" in
      ozp)
        source "$ART/benchmarks/ozp/scripts/common.sh" 2>/dev/null || exit 1
        mkdir -p cache; rm -f cache/*.ozdispatch; rm -rf "$rep"
        "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N="$APPLY_N" \
          -poseidon-print -poseidon-report-path="$rep" "$@" ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "$exe" > "$log" 2>&1 ;;
      chase_eigensolver)
        source "$ART/benchmarks/chase_eigensolver/scripts/common.sh" 2>/dev/null || exit 1
        # The artifact's own `make solve` (its flags), at BUDGET (default 0);
        # extra flags travel in a response file through EXTRA_SOLVE.
        local rsp; rsp="$(mktemp)"; local b=0 a
        : > "$rsp"
        for a in "$@"; do
          case "$a" in -poseidon-cost-budget=*) b="${a#*=}" ;;
            *) "$PY" -c 'import sys; a=sys.argv[1]; print(chr(34) + a.replace(chr(92), chr(92)*2).replace(chr(34), chr(92)+chr(34)) + chr(34))' "$a" >> "$rsp" ;; esac
        done
        rm -rf reports
        make solve BENCH_DEFS="$DEP_DEFS" BUDGET="$b" PROFDIR="$PROFSRC" EXTRA_SOLVE="@$rsp" > /dev/null 2>&1
        cp solve.log "$log"; rm -f "$rsp"
        rm -rf "$rep"; [ -d reports ] && cp -r reports "$rep"
        [ -s chase_opt.exe ] && cp chase_opt.exe "$exe" ;;
      genga_nbody)
        source "$ART/benchmarks/genga_nbody/scripts/common.sh" 2>/dev/null || exit 1
        rm -rf "$rep"
        local FPF
        case "$row" in
          *force*)
            mapfile -t FPF < <(force_flags -poseidon-cache=cache_b300_force)
            "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${FPF[@]}" -poseidon-report-path="$rep" "$@" \
              genga_nbody.cu -o "$exe" > "$log" 2>&1 ;;
          *drift*)
            mapfile -t FPF < <(kepler_flags cache_b300_kep)
            ${COMPILE_WRAP:-} "$POSEIDON_CLANGXX" "${KEPLER_CUDA[@]}" "${FPF[@]}" -poseidon-report-path="$rep" "$@" \
              genga_kepler.cu -o "$exe" > "$log" 2>&1 ;;
          *) echo "b300_lib: no GENGA kernel for row $row" >&2; exit 2 ;;
        esac ;;
      *) echo "b300_lib: unknown bench $bench" >&2; exit 2 ;;
    esac
  )
}

# The benchmark directory a row compiles in (its caches live there).
bench_dir() { echo "$ART/benchmarks/$(row_field "$1" bench)"; }
cache_dirs() {   # <row>: the solve caches of that row, relative to bench_dir
  case "$(row_field "$1" bench)" in
    ozp|chase_eigensolver) echo cache ;;
    genga_nbody) case "$1" in *force*) echo cache_b300_force ;; *) echo cache_b300_kep ;; esac ;;
  esac
}

# The applied configuration a compile printed, one item per line, sorted.
pins_of() { grep -a '^\[poseidon\] pin: ' "$1" | sed 's/^\[poseidon\] pin: //' | tr ',' '\n' | sed '/^$/d' | sort -u; }
# Herbie searches a compile started (the "fresh" count of its summary line).
herbie_fresh() { grep -a -oE 'Herbie answers: [0-9]+ fresh' "$1" | grep -oE '[0-9]+' | paste -sd+ | bc 2>/dev/null || echo 0; }
