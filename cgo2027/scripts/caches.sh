#!/usr/bin/env bash
# caches.sh: the solve caches shipped with the artifact. Source it (after
# env.sh, which sets PY).
#
# devices/<device>/caches/<bench>/<cache dir>/ holds what the compiles of that
# device's decision rows wrote into the benchmark's -poseidon-cache directory:
# the GENGA DP tables (table.json) and the matrix-product accuracy memo of ozp
# and R-ChFSI (matmulacc_*.json). Every entry stores the key of everything it
# was computed from: for a table the compiler (a digest of the pass sources and
# the LLVM version and revision), the function's IR, the device, the cost model,
# the profile, the solve flags, the Herbie platform and the candidate set; for
# a memo entry the matrix product, its profiled operands and the accuracy
# model's sources. A compile whose key differs in any part prints why ("DP
# cache entry for <f> was solved for a different <part>; re-solving.",
# "accuracy memo ... was computed for a different <part>; recomputing it.") and
# solves itself, so a shipped entry can save time but never change a decision.
#
# Seeding copies only what the benchmark's cache directory lacks: a memo file
# not yet there, a function's table not yet in its table.json. What this
# machine's compiler wrote itself is never replaced.
#
# POSEIDON_SHIPPED_CACHE=0 (set by --from-scratch and --no-shipped-cache) seeds
# nothing. The entries were computed on an x86-64 host; another host
# architecture can round the accuracy model's arithmetic differently, which no
# key part records, so they are not seeded there.

_CACHES_ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

shipped_cache_dir() {   # <device> <bench> <cache dir>
  echo "$_CACHES_ART/devices/$1/caches/$2/$3"
}

seed_cache() {   # <device> <bench dir> <cache dir, relative to the bench dir>
  local dev="$1" bdir="$2" cdir="$3" src
  [ "${POSEIDON_SHIPPED_CACHE:-1}" = 0 ] && return 0
  src="$(shipped_cache_dir "$dev" "$(basename "$bdir")" "$cdir")"
  [ -d "$src" ] || return 0
  if [ "$(uname -m)" != x86_64 ]; then
    echo "[cache] not seeding $cdir: the shipped entries were computed on x86-64, this host is $(uname -m)" >&2
    return 0
  fi
  mkdir -p "$bdir/$cdir"
  local f n=0
  for f in "$src"/matmulacc_*.json; do
    [ -e "$f" ] || continue
    [ -e "$bdir/$cdir/$(basename "$f")" ] && continue
    cp "$f" "$bdir/$cdir/" && n=$((n + 1))
  done
  [ "$n" -gt 0 ] && echo "[cache] seeded $cdir with $n shipped accuracy memo entries ($dev)" >&2
  [ -f "$src/table.json" ] || return 0
  "$PY" - "$src/table.json" "$bdir/$cdir/table.json" "$dev" "$cdir" <<'PY'
import json, os, sys
src, dst, dev, cdir = sys.argv[1:]
shipped = json.load(open(src))
have = {}
if os.path.exists(dst):
    try:
        have = json.load(open(dst))
    except ValueError:
        have = {}
if not isinstance(have.get("functions"), dict):
    have = dict(shipped, functions={})
added = [f for f in shipped["functions"] if f not in have["functions"]]
for f in added:
    have["functions"][f] = shipped["functions"][f]
if added:
    tmp = dst + ".seed.%d" % os.getpid()
    json.dump(have, open(tmp, "w"), indent=2)
    os.replace(tmp, dst)
    sys.stderr.write("[cache] seeded %s/table.json with the shipped table of %s (%s)\n"
                     % (cdir, ", ".join(added), dev))
PY
}

# One line per row: what was seeded (from the row script's output) and what
# the caches did for the row's compile.
cache_summary() {   # <compile log> [row script output]
  local hit miss memo_miss seeded=""
  [ -n "${2:-}" ] && grep -q '^\[cache\] seeded' "$2" && seeded="seeded; "
  hit=$(grep -a -c 'found. Loaded DP tables from cache' "$1")
  miss=$(grep -a -c 'DP cache entry for .* re-solving' "$1")
  memo_miss=$(grep -a -c 'accuracy memo .* recomputing it' "$1")
  local s="${seeded}table hits $hit"
  [ "$miss" -gt 0 ] && s="$s, table MISSES $miss ($(grep -a -m1 -o 'was solved for a different [a-zA-Z -]*' "$1" | sed 's/was solved for a different //'))"
  [ "$memo_miss" -gt 0 ] && s="$s, memo MISSES $memo_miss ($(grep -a -m1 -o 'computed for a different [a-zA-Z -]*\|another cache format' "$1" | sed 's/computed for a different //'))"
  echo "$s"
}
