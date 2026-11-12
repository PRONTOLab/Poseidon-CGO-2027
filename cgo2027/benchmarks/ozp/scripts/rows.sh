#!/bin/bash
# rows.sh <row id> <log> — compile the configuration of one ozp row of
# EXPECTED_RESULTS.md exactly as the pipeline scripts do (common.sh's flag
# groups, deploy size APPLY_N), writing the compile's output to <log>. No binary
# is run, so no GPU is needed: the compile-only pick check
# (../../scripts/check_picks.sh) calls this for every row. The content-addressed
# accuracy memo in cache/ is kept across rows (the decision does not depend on
# it); the one-shot dispatch descriptors are not.
set -o pipefail
source "$(dirname "$0")/common.sh"
ROW="${1:?row id}"; LOG="${2:?log file}"
mkdir -p tmp_rows cache
rm -f cache/*.ozdispatch
solve() { "$POSEIDON_CLANGXX" "${COMMON_CUDA[@]}" "${POSE[@]}" "${FP[@]}" "${DISPATCH[@]}" -DOZP_N=$APPLY_N \
            -poseidon-print "$@" ozp_poseidon.cu "${DISPATCH_RT[@]}" -o "tmp_rows/$ROW.exe"; }
case "$(expected_field "$ROW" select)" in
  tau=*) T="$(expected_tau "$ROW")" || exit 1; solve -poseidon-tau="$T" > "$LOG" 2>&1 ;;
  share=*) B="$(expected_share "$ROW")" || exit 1; solve -poseidon-cost-budget="$B" > "$LOG" 2>&1 ;;
  *) echo "rows.sh: no ozp row '$ROW'" >&2; exit 2 ;;
esac
