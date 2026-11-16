#!/bin/bash
# Sensitivity-guided frontier, sensitivity-blind ablation (Kappa lines removed) and FP64 baseline
# -> guided.tsv, blind.tsv, ablation.tsv, strategies.tsv, traces/.
# GUIDED_BUDGETS replaces the measured shares.
set -u
source "$(dirname "$0")/common.sh"
mkdir -p logs traces

echo "=== 1. profile @ surrogate ($PROF_DEFS), with each site's condition number ==="
chase_make profile PROF_DEFS="$PROF_DEFS" > logs/profile.log 2>&1 || { echo "PROFILE FAILED (see logs/profile.log)"; exit 1; }
[ -d fpprofile ] || { echo "no fpprofile/"; exit 1; }
grep -lq "^Kappa =" fpprofile/*.fpprofile || { echo "NO MEASURED Kappa IN ANY PROFILE"; exit 1; }
echo "--- Kappa per site (filter = hemm_body must be the smallest) ---"
grep -H "^Kappa" fpprofile/*.fpprofile
echo "--- solves read ./$PROFSRC (PROFILES=$PROFILES) ---"
# budgets.txt is write-once and must be re-enumerated for a new profile.
rm -f cache/budgets.txt

rm -rf fpprofile_blind
cp -r "$PROFSRC" fpprofile_blind
sed -i '/^Kappa/d' fpprofile_blind/*.fpprofile

ROWS="rchfsi-tf32tcec rchfsi-bf16tcec rchfsi-bf16tcec-nm14 rchfsi-bf16-nm12"
row_of_share() {   # <share> -> the expected row it is the frontier point of, if any
  [ "$ON_SHIPPED_MODEL" = 1 ] || return 0
  local r; for r in $ROWS; do
    "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)" \
      "$1" "$(expected_share "$r")" && { echo "$r"; return 0; }
  done
}

if [ -z "$GUIDED_BUDGETS" ]; then
  echo "=== enumerating this model's frontier ==="
  timeout 1800 make GPU_ID="$GPU_ID" solve BENCH_DEFS="$DEP_DEFS" BUDGET=0 PROFDIR="$PROFSRC" \
    > logs/solve_enumerate.log 2>&1 || { echo "ENUMERATION SOLVE FAILED (logs/solve_enumerate.log)"; exit 1; }
  [ -s cache/budgets.txt ] || { echo "no cache/budgets.txt after enumeration solve"; exit 1; }
  # Keeps the shallowest non-zero points (the filter-only band) that even spacing can skip.
  EXTRA=""
  [ "$ON_SHIPPED_MODEL" = 1 ] && for r in $ROWS; do EXTRA="$EXTRA $(expected_share "$r")"; done
  GUIDED_BUDGETS=$("$PY" - $EXTRA <<'PY'
import os, sys
b = sorted({float(x) for x in open('cache/budgets.txt').read().replace(',', ' ').split()})
cap = int(os.environ.get('MAX_POINTS', '12'))
shallow = int(os.environ.get('SHALLOW_KEEP', '4'))
if len(b) > cap:
    keep = {0, len(b) - 1}
    nz = [i for i, v in enumerate(b) if v != 0]
    keep.update(nz[-shallow:])
    rest = [i for i in range(len(b)) if i not in keep]
    fill = cap - len(keep)
    if fill > 0 and rest:
        keep.update(rest[round(j * (len(rest) - 1) / max(fill - 1, 1))] for j in range(fill))
    b = [b[i] for i in sorted(keep)]
b = sorted(set(b) | {float(x) for x in sys.argv[1:]})
print(' '.join(repr(v) for v in b))
PY
)
  echo "  guided shares: $GUIDED_BUDGETS"
fi

measure_point() {   # $1=share $2=guided|blind [$3=expected row id]
  local B="$1" KIND="$2" ROW="${3:-}"
  local PD="$PROFSRC"; [ "$KIND" = blind ] && PD=./fpprofile_blind
  # timeout needs an executable, not the chase_make function.
  timeout 1800 make GPU_ID="$GPU_ID" solve BENCH_DEFS="$DEP_DEFS" BUDGET="$B" PROFDIR="$PD" > logs/solve_${KIND}_${B}.log 2>&1
  cp solve.log logs/dp_${KIND}_${B}.log 2>/dev/null
  local sch; sch=$(solve_schemes)
  [ -n "$ROW" ] && expected_report "$ROW" solve.log | tee -a logs/expected.log
  local tmp; tmp=$(mktemp)
  run_avg5 "$KIND share $B" ./chase_opt.exe "$tmp"
  grep -E "EIGRESID|eigval L2" "$tmp" > "traces/eigresid_${KIND}_${B}.txt"
  [ -n "$ROW" ] && cp "traces/eigresid_${KIND}_${B}.txt" "traces/eigresid_${ROW}.txt"
  rm -f "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\n" "$B" "$sch" "$EIG_L2" "$LOCKED" "$AVG_MS" | tee -a ${KIND}.tsv
}

echo "=== 2. sensitivity-guided strategies ==="
: > logs/expected.log
printf "budget\tschemes\teigL2\tlocked\tsolve_ms_avg5\n" > guided.tsv
for B in $GUIDED_BUDGETS; do measure_point "$B" guided "$(row_of_share "$B")"; done

echo "=== 3. sensitivity-blind (Kappa removed) ablation ==="
printf "budget\tschemes\teigL2\tlocked\tsolve_ms_avg5\n" > blind.tsv
for B in ${BLIND_BUDGETS:-$GUIDED_BUDGETS}; do
  [ "$B" = "$(expected_share rchfsi-bf16-nm12 2>/dev/null)" ] && [ "$ON_SHIPPED_MODEL" = 1 ] && continue  # measured in step 4
  measure_point "$B" blind
done

# The ablation row: the cheapest guided share that locks 64/64 (the rchfsi-bf16-nm12 row on the shipped model).
if [ "$ON_SHIPPED_MODEL" = 1 ] && [ -z "${ABLATION_SHARE:-}" ]; then
  ABLATION_SHARE=$(expected_share rchfsi-bf16-nm12)
else
  ABLATION_SHARE=${ABLATION_SHARE:-$(awk -F'\t' 'NR>1 && $4=="locked 64/64" {print $1}' guided.tsv | sort -g | head -1)}
fi
if [ -z "$ABLATION_SHARE" ]; then
  echo "=== 4. ablation: SKIPPED, no fully locked guided strategy on this device ==="
else
  echo "=== 4. ablation at the cheapest fully locked guided strategy (share $ABLATION_SHARE) ==="
  grep -q "^${ABLATION_SHARE}$(printf '\t')" guided.tsv || measure_point "$ABLATION_SHARE" guided "$(row_of_share "$ABLATION_SHARE")"
  if ! grep -q "^${ABLATION_SHARE}$(printf '\t')" blind.tsv; then
    if [ "$ON_SHIPPED_MODEL" = 1 ]; then measure_point "$ABLATION_SHARE" blind rchfsi-blind; else measure_point "$ABLATION_SHARE" blind; fi
  fi
  { printf "strategy\tbudget\tschemes\teigL2\tlocked\tsolve_ms_avg5\n"
    awk -F'\t' -v k="$ABLATION_SHARE" 'NR>1 && $1==k {printf "sensitivity-guided\t%s\n", $0}' guided.tsv
    awk -F'\t' -v k="$ABLATION_SHARE" 'NR>1 && $1==k {printf "sensitivity-blind\t%s\n", $0}' blind.tsv
  } > ablation.tsv
  echo "--- ablation ---"; cat ablation.tsv
fi

echo "=== 5. FP64 baseline (deploy) + residual trace (avg-of-5) ==="
chase_make baseline BENCH_DEFS="$DEP_DEFS" > logs/base_deploy.log 2>&1
tmp=$(mktemp)
run_avg5 "FP64 baseline (deploy)" ./chase_baseline.exe "$tmp"
grep -E "EIGRESID|eigval L2" "$tmp" > traces/eigresid_fp64.txt
rm -f "$tmp"
fp64_ms=$AVG_MS

{ printf "budget\tschemes\teigL2\tlocked\tsolve_ms\n"
  printf "0\tall-FP64\t%s\tlocked 64/64\t%s\n" \
    "$(grep -oE 'eigval L2 vs planted = [0-9.e+-]+' traces/eigresid_fp64.txt | grep -oE '[0-9.e+-]+$')" "${fp64_ms:-?}"
  tail -n +2 guided.tsv | while IFS=$'\t' read -r b s l lk ms; do
    printf "%s\t%s\t%s\t%s\t%s\n" "$b" "$s" "$l" "$lk" "$ms"; done
} > strategies.tsv

echo "=== DONE ==="
echo "--- guided ---"; cat guided.tsv
echo "--- blind ---"; cat blind.tsv
[ -s ablation.tsv ] && { echo "--- ablation ---"; cat ablation.tsv; }
[ -s logs/expected.log ] && { echo "--- expected rows (as reported above) ---"; sed 's/^\[expected\] */  /' logs/expected.log; }
echo "fp64 baseline solve_ms=${fp64_ms:-?}"
