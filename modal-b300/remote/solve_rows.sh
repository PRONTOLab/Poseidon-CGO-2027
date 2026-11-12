#!/usr/bin/env bash
# solve_rows.sh: the B300 rows solved compile-only (no GPU), writing the replay
# pack. Run by ../solve_b300_locally.sh inside the CUDA 13.1 container.
#
# Per row of b300_rows.tsv, with the saved B300 cost model of that row, the
# shipped profiles and (GENGA drift) the shipped Herbie answers:
#   tau=<t>        one solve at that accuracy target (the artifact's autopick)
#   match=<pin>    the frontier solve at budget 0 (the artifact's enumeration
#                  step), then the frontier point whose configuration is <pin>
#                  (a row id instead of a pin: that row's pin in the artifact's
#                  expected/picks.tsv), compiled at its share. The paper chose
#                  these rows from measured frontiers, so the local step checks
#                  that the paper's configuration is on the B300 model's
#                  frontier and fixes its share; if it is not, the row records
#                  the frontier and DIFFERENT.
#   pin=<value>    that configuration by name (-poseidon-apply-rewrites): a
#                  paper configuration the solver no longer picks, replayed
#                  next to the row with the solver's pick
# Writes $OUT: picks_b300.tsv, compare_b300.tsv, cost_models/, rows/<row>/
# (solve logs, explanation files, the solve caches, apply.txt = the pin value).
set -uo pipefail
M="${MDIR:-/m}"
W="$M/.solve-work"
OUT="${OUT:-$M/replay_pack}"
export TC="$W/tc" BUNDLE="$M/bundle" CUDA_ARCHS=103-real SCRATCH="$W/scratch"
export JOBS="${JOBS:-32}"
ROWS="$M/b300_rows.tsv"
TOOLS="$M/remote/pack_tools.py"
PY="${PY:-$(command -v python3)}"; export PY
log() { echo "[solve $(date -u +%H:%M:%S)] $*"; }

bash "$M/remote/build.sh" cumpsgemm || exit 1
bash "$M/remote/build.sh" poseidon || exit 1
# Herbie for the drift search: the pinned binary mounted at /herbie-dist
# (copied to where the compiler was configured to find it).
if [ -x /herbie-dist/bin/herbie ] && ! cmp -s /herbie-dist/bin/herbie "$TC/herbie/bin/herbie"; then
  rm -rf "$TC/herbie" && cp -a /herbie-dist "$TC/herbie"
fi
export HERBIE="$TC/herbie/bin/herbie"
COMMIT="$(sed -n 's/^compiler_commit=//p' "$BUNDLE/MANIFEST")"

rm -rf "$W/art" "$OUT.new" && mkdir -p "$W/art" "$W/tmp"
# With rows named, the other rows of an existing pack are kept.
if [ $# -gt 0 ] && [ -f "$OUT/picks_b300.tsv" ]; then cp -a "$OUT" "$OUT.new"; fi
mkdir -p "$OUT.new/rows" "$OUT.new/cost_models"
tar -C "$W/art" -xzf "$BUNDLE/artifact.tar.gz"
ART="$W/art/cgo2027"
CMDIR="$OUT.new/cost_models"
cp "$M"/b300_inputs/*.csv "$CMDIR/"
export POSEIDON_BUILD="$TC/poseidon-build" POSEIDON_ARTIFACT_NO_GPU=1 TMPDIR="$W/tmp" CUDA_VISIBLE_DEVICES=
unset GPU_ID
source "$M/remote/b300_lib.sh"
source "$ART/scripts/expected.sh"

# GENGA prices with the saved B300 model (rev2) scaled x1e6 and searches
# against the Herbie platform generated from that copy: both made once here,
# by the artifact's own common.sh, and shipped in the pack.
if [ ! -f "$CMDIR/genga_x1e6/cm_native_x1e6.herbie.rkt" ]; then
  ( export COST_MODEL="$CMDIR/cm_sm_103_B300SXM6AC.rev2.csv" GPU_ARCH=sm_103 PROFILES=shipped GENGA_HERBIE=run
    rm -rf "$ART/benchmarks/genga_nbody/measured"
    source "$ART/benchmarks/genga_nbody/scripts/common.sh" ) || { log "deriving the GENGA x1e6 model failed"; exit 1; }
  mkdir -p "$CMDIR/genga_x1e6"
  cp "$ART/benchmarks/genga_nbody/measured/cm_native_x1e6.csv" "$ART/benchmarks/genga_nbody/measured/cm_native_x1e6.herbie.rkt" "$CMDIR/genga_x1e6/"
fi

[ -f "$OUT.new/picks_b300.tsv" ] ||
  printf 'row\tbench\tcost_model\tselect\tshare\tpin\tmodelled_cost_after\tmodelled_rel_error\therbie_fresh\n' > "$OUT.new/picks_b300.tsv"
[ -f "$OUT.new/compare_b300.tsv" ] ||
  printf 'row\tpaper\tsaved_pick\tlocal_pick\tverdict\tdetail\n' > "$OUT.new/compare_b300.tsv"

ROW_IDS=("$@")
[ ${#ROW_IDS[@]} -gt 0 ] || mapfile -t ROW_IDS < <(awk -F'\t' 'NR > 1 {print $1}' "$ROWS")
for row in "${ROW_IDS[@]}"; do
  log "$row"
  R="$OUT.new/rows/$row"; rm -rf "$R"; mkdir -p "$R"
  for t in picks_b300 compare_b300; do
    awk -F'\t' -v id="$row" '$1 != id' "$OUT.new/$t.tsv" > "$OUT.new/$t.tsv.tmp" && mv "$OUT.new/$t.tsv.tmp" "$OUT.new/$t.tsv"
  done
  sel="$(row_field "$row" select)"; bench="$(row_field "$row" bench)"
  BD="$(bench_dir "$row")"
  for c in $(cache_dirs "$row"); do rm -rf "${BD:?}/$c"; done
  share="" detail=""
  case "$sel" in
    tau=*)
      compile_row "$row" "$W/tmp/$row.exe" "$R/solve.log" "$R/explain" -poseidon-tau="${sel#tau=}" ;;
    match=*)
      want="${sel#match=}"
      case "$want" in *:*) ;; *) want="$(expected_field "$want" pin)" ;; esac
      if [ "$row" = t1-drift-b300 ]; then
        # The B300 Herbie search (the only one): run mode, timed, its peak
        # memory recorded; the answers become the pack's B300 answer pack.
        T0=$(date +%s)
        GENGA_HERBIE_MODE=run COMPILE_WRAP="/usr/bin/time -v -o $R/frontier_solve.time" \
          compile_row "$row" "$W/tmp/$row-b0.exe" "$R/frontier_solve.log" "$R/frontier_explain" -poseidon-cost-budget=0
        echo "herbie_search_wall_s=$(( $(date +%s) - T0 ))" > "$R/herbie_search.txt"
        sed -n 's/^\s*Maximum resident set size (kbytes): /max_rss_kb_of_one_process=/p' "$R/frontier_solve.time" >> "$R/herbie_search.txt"
        [ -r /sys/fs/cgroup/memory.peak ] && echo "container_memory_peak_bytes=$(cat /sys/fs/cgroup/memory.peak)" >> "$R/herbie_search.txt"
        grep -a 'Herbie answers:' "$R/frontier_solve.log" >> "$R/herbie_search.txt"
        A="$OUT.new/herbie_answers_b300/kepler"; rm -rf "$A"; mkdir -p "$A"
        cp "$BD"/cache_b300_kep/cachedHerbieOutput_* "$BD"/cache_b300_kep/cachedHerbieQuestion_* "$A/" 2>/dev/null
        ( cd "$A" && md5sum ./* > ../MD5SUMS.kepler )
      else
        compile_row "$row" "$W/tmp/$row-b0.exe" "$R/frontier_solve.log" "$R/frontier_explain" -poseidon-cost-budget=0
      fi
      "$PY" -I "$TOOLS" frontier "$R/frontier_explain" > "$R/frontier.tsv" 2>&1
      # The drift row's paper cell is a class (expansion + algebraic): when the
      # artifact's exact configuration is not on this frontier, take the most
      # accurate frontier point with the f/g block (PT7) as the artifact's
      # two-component expansion (else any expansion of it) and the Newton step
      # (R13) rewritten by Herbie.
      klass=""
      if [ "$row" = t1-drift-b300 ] && [ -d "$R/frontier_explain" ] &&
         ! "$PY" -I "$TOOLS" match "$R/frontier_explain" "$want" > /dev/null 2>&1; then
        # The artifact's f/g-block candidate first, then any expansion of it.
        klass="$("$PY" -I "$TOOLS" matchclass "$R/frontier_explain" ':PT7=All FP64\(0%\) \+ Expansion2\(100%\)$' ':R13=\(')" ||
          klass="$("$PY" -I "$TOOLS" matchclass "$R/frontier_explain" ':PT7=[^,]*Expansion2' ':R13=')" || klass=""
      fi
      if [ -n "$klass" ]; then
        share="$(echo "$klass" | cut -f1)"; want="$(echo "$klass" | cut -f3)"
        detail="class match: the artifact's RTX 5090 drift configuration is not on this frontier; $(echo "$klass" | cut -f4) frontier points are expansion (PT7) + algebraic (R13); the most accurate one (modelled rel. error $(echo "$klass" | cut -f2)) is taken"
        GENGA_HERBIE_MODE=run compile_row "$row" "$W/tmp/$row.exe" "$R/solve.log" "$R/explain" -poseidon-cost-budget="$share"
      elif share="$("$PY" -I "$TOOLS" match "$R/frontier_explain" "$want")"; then
        GENGA_HERBIE_MODE=$([ "$row" = t1-drift-b300 ] && echo run || echo replay) \
          compile_row "$row" "$W/tmp/$row.exe" "$R/solve.log" "$R/explain" -poseidon-cost-budget="$share"
      elif [ ! -d "$R/frontier_explain" ]; then
        share="" detail="the frontier solve failed: $(grep -a -m1 -E 'fatal error|LLVM ERROR|error:' "$R/frontier_solve.log" | cut -c1-400)"
      else
        share="" detail="the paper's configuration is not on this model's frontier ($(wc -l < "$R/frontier.tsv") points: frontier.tsv)"
      fi ;;
    pin=*)
      # A configuration by name (the paper's, where the solver now picks
      # another): compiled with -poseidon-apply-rewrites, as the replay does.
      compile_row "$row" "$W/tmp/$row.exe" "$R/solve.log" "$R/explain" -poseidon-apply-rewrites="${sel#pin=}" ;;
    *) log "$row: unknown select '$sel'"; continue ;;
  esac
  pin=""; cost="" err="" fresh=0
  if [ -f "$R/solve.log" ]; then
    pin="$(pins_of "$R/solve.log" | paste -sd',' -)"
    fresh="$(herbie_fresh "$R/solve.log")"; fresh="${fresh:-0}"
    if [ -d "$R/explain" ]; then
      res="$("$PY" -I "$TOOLS" result "$R/explain")"
      cost="$(echo "$res" | awk -F'\t' '{print $1 "=" $3}' | paste -sd';' -)"
      err="$(echo "$res" | awk -F'\t' '{print $1 "=" $4}' | paste -sd';' -)"
    fi
  fi
  grep -a -E 'LLVM ERROR|fatal error|posix_spawn|Execution failed' "$R"/*.log > "$R/errors.txt" 2>/dev/null || rm -f "$R/errors.txt"
  [ -n "$pin" ] && printf '%s\n' "$pin" > "$R/apply.txt"
  for c in $(cache_dirs "$row"); do [ -d "$BD/$c" ] && cp -r "$BD/$c" "$R/$c"; done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$row" "$bench" "$(row_field "$row" cost_model)" "$sel" \
    "${share:-}" "$pin" "$cost" "$err" "$fresh" >> "$OUT.new/picks_b300.tsv"

  # ---- the comparison: local pick against the saved run's and the paper's --
  saved="$(row_field "$row" saved_pick)"; paper="$(row_field "$row" paper)"
  case "$sel" in
    match=*)
      if [ -n "$pin" ] && [ "$(pins_of "$R/solve.log")" = "$(echo "$want" | tr ',' '\n' | sed '/^$/d' | sort -u)" ]; then
        if [ -n "$klass" ]; then verdict=same-class; detail="$detail; applied at share $share"
        else verdict=same; detail="the paper's configuration is on the B300 frontier at share $share and is applied there"; fi
      elif [ ! -d "$R/frontier_explain" ]; then
        verdict=blocked
      else
        verdict=different; [ -n "$detail" ] || detail="compiled at share $share, applied: $pin"
        for item in $(echo "$want" | tr ',' '\n' | sed 's/ /\x1f/g'); do
          item="${item//$'\x1f'/ }"
          detail="$detail | wanted $item: $("$PY" -I "$TOOLS" alt "$R/frontier_explain" "$item" | tr '\t' ' ')"
        done
      fi ;;
    tau=*)
      accepted="$(row_field "$row" accepted)"
      if [ "$(echo "$pin" | tr ',' '\n' | sort -u)" = "$(echo "$saved" | tr ',' '\n' | sort -u)" ]; then
        verdict=same; detail=""
      elif [ -n "$accepted" ] && [ "$pin" = "$accepted" ]; then
        verdict=accepted
        a="$("$PY" -I "$TOOLS" alt "$R/explain" "$pin" | cut -f1-4 | tr '\t' ' ')"
        b="$("$PY" -I "$TOOLS" alt "$R/explain" "$saved" | cut -f1-5 | tr '\t' ' ')"
        detail="the pick differs from the paper's and was accepted (b300_rows.tsv: accepted) | local $pin: $a | paper $saved (reference only): $b"
      else
        verdict=different
        a="$("$PY" -I "$TOOLS" alt "$R/explain" "$pin" | cut -f1-4 | tr '\t' ' ')"
        b="$("$PY" -I "$TOOLS" alt "$R/explain" "$saved" | cut -f1-5 | tr '\t' ' ')"
        detail="local $pin: cost_delta/err/domain_err/status $a | saved $saved: $b"
      fi ;;
    pin=*)
      if [ -n "$pin" ] && [ "$(pins_of "$R/solve.log")" = "$(echo "${sel#pin=}" | tr ',' '\n' | sed '/^$/d' | sort -u)" ]; then
        verdict=by-name; detail="compiled by name (-poseidon-apply-rewrites); modelled cost after $cost, error $err"
        for item in $(echo "${sel#pin=}" | tr ',' '\n' | sed 's/ /\x1f/g'); do
          item="${item//$'\x1f'/ }"
          detail="$detail | $item: cost_delta/err/domain_err/status $("$PY" -I "$TOOLS" alt "$R/explain" "$item" | cut -f1-4 | tr '\t' ' ')"
        done
      else
        verdict=different; detail="the compile did not apply the configuration named in b300_rows.tsv"
      fi ;;
  esac
  [ -f "$R/errors.txt" ] && detail="$detail | ERRORS: $(head -3 "$R/errors.txt" | tr '\n' ' ')"
  [ "${fresh:-0}" != 0 ] && detail="$detail | HERBIE SEARCHED $fresh times"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$row" "$paper" "$saved" "$pin" "$verdict" "$detail" >> "$OUT.new/compare_b300.tsv"
  log "$row: $verdict ($pin)"
done

{
  echo "compiler_commit=$COMMIT"
  echo "poseidon_stamp=$(cat "$TC/poseidon.stamp")"
  echo "cuda=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^Cuda compilation tools, //p')"
  echo "artifact_sha256=$(sha256sum "$BUNDLE/artifact.tar.gz" | cut -d' ' -f1)"
  (cd "$CMDIR" && md5sum ./*.csv ./genga_x1e6/* | sed 's/^/cost_model_md5=/')
  echo "rows=$(awk -F'\t' 'NR > 1 {print $1}' "$OUT.new/picks_b300.tsv" | paste -sd' ' -)"
  echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT.new/MANIFEST"
cp "$ROWS" "$OUT.new/b300_rows.tsv"
rm -rf "$OUT" && mv "$OUT.new" "$OUT"
log "wrote $OUT"
cut -f1,4,5 "$OUT/compare_b300.tsv"
