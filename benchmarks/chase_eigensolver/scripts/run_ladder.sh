#!/bin/bash
# Sensitivity-guided ladder + sensitivity-blind ablation for eval §5.3.
#
#   1. profile   @ N=512 NEV=32  (fpprofile/, with each site's Kappa)
#   2. ON  arm   deploy @ N=2048 NEV=64, joint DP, sweep the ladder (avg-of-5)
#   3. OFF arm   the same shares against a copy of the profile with the Kappa
#                lines removed (sensitivity-blind)
#   4. the ablation row: both arms at the guided knee
#   5. FP64 baseline row + the per-eigenpair residual traces the figure needs
#
# The ladder (shares of the profiled cost) is an evenly spaced sample of this
# model's frontier and, on the shipped RTX 5090 model, the frontier points of
# the R-ChFSI rows of EXPECTED_RESULTS.md (expected/picks.tsv); LADDER_BUDGETS
# replaces it. Each measured row records the configuration the solver applied
# (its pin lines) and, for an expected row, whether it is the expected one.
#
# Outputs (fresh; the shipped RTX 5090 run is in golden/rtx5090/):
#   scvh_probe_on.tsv, scvh_probe_off.tsv, ablation_knee.tsv, chase_money_data.tsv
#   traces/eigresid_fp64.txt, traces/eigresid_<arm>_<share>.txt and, for the
#   expected rows, traces/eigresid_<row id>.txt (what plot_chase.py draws); logs/
set -u
source "$(dirname "$0")/common.sh"
mkdir -p logs traces

echo "=== 1. profile @ surrogate ($PROF_DEFS), with each site's condition number ==="
chase_make profile PROF_DEFS="$PROF_DEFS" > logs/profile.log 2>&1 || { echo "PROFILE FAILED (see logs/profile.log)"; exit 1; }
[ -d fpprofile ] || { echo "no fpprofile/"; exit 1; }
grep -lq "^Kappa" fpprofile/*.fpprofile || { echo "NO Kappa IN ANY PROFILE"; exit 1; }
echo "--- Kappa per site (filter = hemm_body must be the smallest) ---"
grep -H "^Kappa" fpprofile/*.fpprofile
echo "--- solves read ./$PROFSRC (PROFILES=$PROFILES) ---"
# profile changed -> the write-once DP breakpoint cache must be re-enumerated.
rm -f cache/budgets.txt

# The sensitivity-blind arm is the SAME profile with the probe's header lines
# removed, so the two arms differ in nothing else.
rm -rf fpprofile_blind
cp -r "$PROFSRC" fpprofile_blind
sed -i '/^Kappa/d' fpprofile_blind/*.fpprofile

ROWS="rchfsi-tf32tcec rchfsi-bf16tcec rchfsi-bf16tcec-nm14 rchfsi-knee"
row_of_share() {   # <share> -> the expected row it is the frontier point of, if any
  [ "$ON_SHIPPED_MODEL" = 1 ] || return 0
  local r; for r in $ROWS; do
    "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)" \
      "$1" "$(expected_share "$r")" && { echo "$r"; return 0; }
  done
}

if [ -z "$LADDER_BUDGETS" ]; then
  echo "=== enumerating this model's frontier ==="
  timeout 1800 make GPU_ID="$GPU_ID" solve BENCH_DEFS="$DEP_DEFS" BUDGET=0 PROFDIR="$PROFSRC" \
    > logs/solve_enumerate.log 2>&1 || { echo "ENUMERATION SOLVE FAILED (logs/solve_enumerate.log)"; exit 1; }
  [ -s cache/budgets.txt ] || { echo "no cache/budgets.txt after enumeration solve"; exit 1; }
  # Sampling always keeps the shallowest non-zero frontier points: by DP
  # construction they are the lowest-accuracy-cost single-site configurations
  # (the filter-only band), and plain even spacing can skip the whole band
  # when most points cluster at the deep end.
  EXTRA=""
  [ "$ON_SHIPPED_MODEL" = 1 ] && for r in $ROWS; do EXTRA="$EXTRA $(expected_share "$r")"; done
  LADDER_BUDGETS=$("$PY" - $EXTRA <<'PY'
import os, sys
b = sorted({float(x) for x in open('cache/budgets.txt').read().replace(',', ' ').split()})
cap = int(os.environ.get('LADDER_MAX_POINTS', '12'))
shallow = int(os.environ.get('LADDER_SHALLOW_KEEP', '4'))
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
  echo "  ladder: $LADDER_BUDGETS"
fi

measure() {   # $1=share $2=arm(on|off) [$3=expected row id]
  local B="$1" ARM="$2" ROW="${3:-}"
  local PD="$PROFSRC"; [ "$ARM" = off ] && PD=./fpprofile_blind
  # NB: `timeout` needs a real executable, not the chase_make shell function.
  timeout 1800 make GPU_ID="$GPU_ID" solve BENCH_DEFS="$DEP_DEFS" BUDGET="$B" PROFDIR="$PD" > logs/solve_${ARM}_${B}.log 2>&1
  cp solve.log logs/dp_${ARM}_${B}.log 2>/dev/null
  local sch; sch=$(solve_schemes)
  [ -n "$ROW" ] && expected_report "$ROW" solve.log | tee -a logs/expected.log
  local tmp; tmp=$(mktemp)
  read avg l2 lk < <(run_avg5 "$tmp")
  grep -E "EIGRESID|eigval L2" "$tmp" > "traces/eigresid_${ARM}_${B}.txt"
  [ -n "$ROW" ] && cp "traces/eigresid_${ARM}_${B}.txt" "traces/eigresid_${ROW}.txt"
  rm -f "$tmp"
  printf "%s\t%s\t%s\t%s\t%s\n" "$B" "$sch" "$l2" "$lk" "$avg" | tee -a scvh_probe_${ARM}.tsv
}

echo "=== 2. sensitivity-guided (Kappa present) ladder ==="
: > logs/expected.log
printf "budget\tschemes\teigL2\tlocked\tsolve_ms_avg5\n" > scvh_probe_on.tsv
for B in $LADDER_BUDGETS; do measure "$B" on "$(row_of_share "$B")"; done

echo "=== 3. sensitivity-blind (Kappa removed) ablation ==="
printf "budget\tschemes\teigL2\tlocked\tsolve_ms_avg5\n" > scvh_probe_off.tsv
for B in ${LADDER_OFF_BUDGETS:-$LADDER_BUDGETS}; do
  [ "$B" = "$(expected_share rchfsi-knee 2>/dev/null)" ] && [ "$ON_SHIPPED_MODEL" = 1 ] && continue  # measured in step 4
  measure "$B" off
done

# ---- 4. the ablation row, at the knee --------------------------------------
# The paper states the ablation at ONE budget: the sensitivity-guided KNEE, the
# cheapest frontier point at which the guided solve still locks 64/64. On the
# shipped model that is the knee row of EXPECTED_RESULTS.md; elsewhere it is
# derived from the guided arm. Both arms are reported at exactly it.
if [ "$ON_SHIPPED_MODEL" = 1 ] && [ -z "${KNEE:-}" ]; then
  KNEE=$(expected_share rchfsi-knee)
else
  KNEE=${KNEE:-$(awk -F'\t' 'NR>1 && $4=="locked 64/64" {print $1}' scvh_probe_on.tsv | sort -g | head -1)}
fi
if [ -z "$KNEE" ]; then
  echo "=== 4. ablation at the knee: SKIPPED, no fully-locked guided row on this device ==="
else
  echo "=== 4. ablation at the sensitivity-guided knee (share $KNEE) ==="
  grep -q "^${KNEE}$(printf '\t')" scvh_probe_on.tsv || measure "$KNEE" on "$(row_of_share "$KNEE")"
  if ! grep -q "^${KNEE}$(printf '\t')" scvh_probe_off.tsv; then
    if [ "$ON_SHIPPED_MODEL" = 1 ]; then measure "$KNEE" off rchfsi-blind; else measure "$KNEE" off; fi
  fi
  { printf "arm\tbudget\tschemes\teigL2\tlocked\tsolve_ms_avg5\n"
    awk -F'\t' -v k="$KNEE" 'NR>1 && $1==k {printf "sensitivity-guided\t%s\n", $0}' scvh_probe_on.tsv
    awk -F'\t' -v k="$KNEE" 'NR>1 && $1==k {printf "sensitivity-blind\t%s\n", $0}' scvh_probe_off.tsv
  } > ablation_knee.tsv
  echo "--- ablation at the knee ---"; cat ablation_knee.tsv
fi

echo "=== 5. FP64 baseline (deploy) + residual trace (avg-of-5) ==="
chase_make baseline BENCH_DEFS="$DEP_DEFS" > logs/base_deploy.log 2>&1
fp64_sum=0; fp64_n=0
for r in $(seq 1 "$NUM_RUNS"); do
  o=$(CUDA_VISIBLE_DEVICES=$GPU_ID ./chase_baseline.exe 2>&1)
  [ "$r" = 1 ] && echo "$o" | grep -E "EIGRESID|eigval L2" > traces/eigresid_fp64.txt
  m=$(echo "$o" | grep -oE "solve_ms = [0-9.]+" | grep -oE "[0-9.]+$")
  [ -n "$m" ] && { fp64_sum=$(num_add "$fp64_sum" "$m"); fp64_n=$((fp64_n+1)); }
done
fp64_ms=$([ "$fp64_n" -gt 0 ] && num_div "$fp64_sum" "$fp64_n" || echo "?")

# ── assemble chase_money_data.tsv (FP64 + the guided ladder) ─────────────────
{ printf "budget\tschemes\teigL2\tlocked\tsolve_ms\n"
  printf "0\tall-FP64\t%s\tlocked 64/64\t%s\n" \
    "$(grep -oE 'eigval L2 vs planted = [0-9.e+-]+' traces/eigresid_fp64.txt | grep -oE '[0-9.e+-]+$')" "${fp64_ms:-?}"
  tail -n +2 scvh_probe_on.tsv | while IFS=$'\t' read -r b s l lk ms; do
    printf "%s\t%s\t%s\t%s\t%s\n" "$b" "$s" "$l" "$lk" "$ms"; done
} > chase_money_data.tsv

echo "=== DONE ==="
echo "--- guided ---"; cat scvh_probe_on.tsv
echo "--- blind ---"; cat scvh_probe_off.tsv
[ -s ablation_knee.tsv ] && { echo "--- ablation at the knee ---"; cat ablation_knee.tsv; }
[ -s logs/expected.log ] && { echo "--- expected rows ---"; cat logs/expected.log; }
echo "fp64 baseline solve_ms=${fp64_ms:-?}"
