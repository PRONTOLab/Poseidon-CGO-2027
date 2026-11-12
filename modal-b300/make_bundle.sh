#!/usr/bin/env bash
# make_bundle.sh: build the files poseidon_b300.py uploads to Modal.
#
#   ./make_bundle.sh        # from a clone of this repository with its submodules
#
# Every input has a default inside the repository:
#   --compiler-repo <git checkout of the Enzyme repository>   (default: ../sources/Enzyme)
#   --compiler-commit <rev>                                   (default: HEAD of that checkout)
#   --artifact <cgo2027 artifact directory>                   (default: ../cgo2027)
#   --cumpsgemm-repo <enp1s0/cuMpSGEMM with its submodules>   (default: ../sources/cuMpSGEMM)
#   --cumpsgemm-commit <rev>                                  (default: HEAD of that checkout)
#   --herbie-repo <git clone of Herbie> [--herbie-commit <rev>]
#                           optional: only the full on-device flow searches
#                           with Herbie (default commit: num-taylor-terms-73ba1fe76)
#   --replay-pack <dir>                                       (default: ./replay_pack)
#   --out <dir>                                               (default: ./bundle)
#
# Writes into <out>:
#   poseidon-src.tar.gz  `git archive` of <compiler-commit>, the enzyme/ subtree
#                        only, under poseidon-src/ (read-only git; the worktree's
#                        uncommitted edits are NOT included)
#   herbie-src.tar.gz    (only with --herbie-repo)
#                        `git archive` of <herbie-commit> under herbie/: upstream
#                        Herbie 73ba1fe76 plus the --num-taylor-terms option the
#                        GENGA drift and unified solves search with
#   cumpsgemm-src.tar.gz cuMpSGEMM at <cumpsgemm-commit> with its submodules,
#                        under cuMpSGEMM/ (the error-corrected GEMM backend)
#   artifact.tar.gz      the artifact directory under cgo2027/, without what its
#                        .gitignore files exclude (run outputs, caches, build
#                        products, machine pins, result tarballs) and without a
#                        few compile-only scratch files (list below)
#   replay_pack.tar.gz   the B300 replay pack (--replay-pack, default
#                        ./replay_pack, written by solve_b300_locally.sh)
#   MANIFEST             sha256 and size of each tarball, the two commits, and
#                        the artifact's file count; poseidon_b300.py refuses to
#                        start when a tarball is missing or differs from it
#
# Run it again whenever the compiler commit or the artifact changes; the Modal
# build stage rebuilds Enzyme/Poseidon when poseidon-src.tar.gz changes and
# Herbie when herbie-src.tar.gz changes, never LLVM.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/bundle"
PACK="$HERE/replay_pack"
CUMPS_REPO="$HERE/../sources/cuMpSGEMM" CUMPS_COMMIT="HEAD"
COMPILER_REPO="$HERE/../sources/Enzyme" COMPILER_COMMIT="HEAD" ARTIFACT="$HERE/../cgo2027"
HERBIE_REPO="" HERBIE_COMMIT="num-taylor-terms-73ba1fe76"

usage() { sed -n '2,36p' "$0"; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --compiler-repo)   COMPILER_REPO="$2"; shift 2 ;;
    --compiler-commit) COMPILER_COMMIT="$2"; shift 2 ;;
    --artifact)        ARTIFACT="$2"; shift 2 ;;
    --herbie-repo)     HERBIE_REPO="$2"; shift 2 ;;
    --herbie-commit)   HERBIE_COMMIT="$2"; shift 2 ;;
    --out)             OUT="$2"; shift 2 ;;
    --replay-pack)     PACK="$2"; shift 2 ;;
    --cumpsgemm-repo)  CUMPS_REPO="$2"; shift 2 ;;
    --cumpsgemm-commit) CUMPS_COMMIT="$2"; shift 2 ;;
    -h|--help)         usage ;;
    *) echo "make_bundle: unknown argument $1" >&2; usage ;;
  esac
done
[ -x "$ARTIFACT/run_artifact.sh" ] || { echo "make_bundle: $ARTIFACT has no run_artifact.sh" >&2; exit 1; }
for sub in "$COMPILER_REPO" "$CUMPS_REPO"; do
  [ -e "$sub/.git" ] || { echo "make_bundle: $sub is not checked out: git submodule update --init --recursive" >&2; exit 1; }
done

COMPILER_SHA="$(git -C "$COMPILER_REPO" rev-parse --verify "$COMPILER_COMMIT^{commit}")"
git -C "$COMPILER_REPO" cat-file -e "$COMPILER_SHA:enzyme/Poseidon/CMakeLists.txt" ||
  { echo "make_bundle: $COMPILER_SHA has no enzyme/Poseidon" >&2; exit 1; }
HERBIE_SHA=""
if [ -n "$HERBIE_REPO" ]; then
  HERBIE_SHA="$(git -C "$HERBIE_REPO" rev-parse --verify "$HERBIE_COMMIT^{commit}")"
  git -C "$HERBIE_REPO" cat-file -e "$HERBIE_SHA:src/main.rkt" ||
    { echo "make_bundle: $HERBIE_SHA is not a Herbie tree" >&2; exit 1; }
fi

mkdir -p "$OUT"
TMP="$(mktemp -d "$OUT/.tmp.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# ---- compiler and Herbie: read-only git archive ------------------------------
# --mtime is fixed so that the same commit gives the same bytes (gzip -n).
git -C "$COMPILER_REPO" archive --format=tar --prefix=poseidon-src/ "$COMPILER_SHA" enzyme |
  gzip -n -9 > "$TMP/poseidon-src.tar.gz"
if [ -n "$HERBIE_SHA" ]; then
  git -C "$HERBIE_REPO" archive --format=tar --prefix=herbie/ "$HERBIE_SHA" |
    gzip -n -9 > "$TMP/herbie-src.tar.gz"
fi

# ---- cuMpSGEMM (the TCEC backend the B300 compiler is configured with) ----------
# The repository at CUMPS_COMMIT plus each submodule at the commit that commit
# records, as one tree under cuMpSGEMM/ (read-only git archive of each).
CUMPS_SHA="$(git -C "$CUMPS_REPO" rev-parse --verify "$CUMPS_COMMIT^{commit}")"
mkdir -p "$TMP/cumps"
git -C "$CUMPS_REPO" archive --format=tar --prefix=cuMpSGEMM/ "$CUMPS_SHA" | tar -C "$TMP/cumps" -xf -
git -C "$CUMPS_REPO" ls-tree -r "$CUMPS_SHA" | awk '$2 == "commit" {print $3, $4}' | while read -r sub path; do
  git -C "$CUMPS_REPO/$path" archive --format=tar --prefix="cuMpSGEMM/$path/" "$sub" | tar -C "$TMP/cumps" -xf -
done
tar -C "$TMP/cumps" --owner=0 --group=0 --numeric-owner --sort=name --mtime=@0 -cf - cuMpSGEMM | gzip -n -9 > "$TMP/cumpsgemm-src.tar.gz"
rm -rf "$TMP/cumps"

# ---- artifact: what git would track ------------------------------------------
# The artifact directory is a working copy, not a git repository, so the
# .gitignore files are applied here (root and nested, negation, anchoring,
# directory-only patterns). EXTRA are root-level patterns on top of them: the
# artifact repository's own root .gitignore ignores build*/, and the rest are
# compile-only scratch outputs the port's patch excluded.
EXTRA='build*/
herbie_input_*
matmulacc_*
launches_*
.venv/'
python3 -I - "$ARTIFACT" "$TMP/artifact.list" "$EXTRA" <<'PY'
import os, re, sys

root, out, extra = sys.argv[1], sys.argv[2], sys.argv[3]

def glob_re(pat):
    i, n, r = 0, len(pat), ""
    while i < n:
        c = pat[i]
        if pat.startswith("**/", i):
            r += "(?:.*/)?"; i += 3; continue
        if pat.startswith("/**", i) and i + 3 == n:
            r += "/.*"; i += 3; continue
        if pat.startswith("**", i):
            r += ".*"; i += 2; continue
        if c == "*":
            r += "[^/]*"
        elif c == "?":
            r += "[^/]"
        elif c == "[":
            j = pat.find("]", i + 1)
            if j < 0:
                r += re.escape(c)
            else:
                body = pat[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                r += "[" + body.replace("\\", "\\\\") + "]"
                i = j
        elif c == "\\" and i + 1 < n:
            i += 1; r += re.escape(pat[i])
        else:
            r += re.escape(c)
        i += 1
    return r

def parse(text, base):
    rules = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        if neg:
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        if not line:
            continue
        anchored = "/" in line
        line = line.lstrip("/")
        rx = glob_re(line)
        rx = ("^" + rx + "$") if anchored else ("(?:^|/)" + rx + "$")
        rules.append((base, re.compile(rx), neg, dir_only))
    return rules

def ignored(rel, is_dir, rules):
    verdict = False
    for base, rx, neg, dir_only in rules:
        if dir_only and not is_dir:
            continue
        if base:
            if not rel.startswith(base + "/"):
                continue
            sub = rel[len(base) + 1:]
        else:
            sub = rel
        if rx.search(sub):
            verdict = not neg
    return verdict

def load(d, base):
    p = os.path.join(d, ".gitignore")
    if os.path.isfile(p):
        with open(p, errors="replace") as f:
            return parse(f.read(), base)
    return []

files = []
def walk(d, base, rules):
    rules = rules + load(d, base)
    for name in sorted(os.listdir(d)):
        rel = f"{base}/{name}" if base else name
        full = os.path.join(d, name)
        is_dir = os.path.isdir(full) and not os.path.islink(full)
        if name == ".git" or ignored(rel, is_dir, rules):
            continue
        if is_dir:
            walk(full, rel, rules)
        else:
            files.append(rel)

walk(root, "", parse(extra, ""))
with open(out, "w") as f:
    for rel in files:
        f.write(rel + "\n")
PY

# The files the Modal run cannot do without.
for need in run_artifact.sh env.sh config.mk scripts/detect.sh scripts/bootstrap_cost_model.sh \
            cost_models/cm_sm_120_RTX5090.csv expected/picks.tsv EXPECTED_RESULTS.md \
            benchmarks/genga_nbody/herbie_answers/MD5SUMS benchmarks/ozp/profiles/MD5SUMS \
            benchmarks/chase_eigensolver/profiles/MD5SUMS benchmarks/genga_nbody/profiles/MD5SUMS; do
  grep -qx "$need" "$TMP/artifact.list" || { echo "make_bundle: $need would not be shipped" >&2; exit 1; }
done
for gone in local.env cost_models/target.env; do
  if grep -qx "$gone" "$TMP/artifact.list"; then echo "make_bundle: $gone would be shipped" >&2; exit 1; fi
done

# Reproducible member metadata: fixed owner, sorted list, the files' own mtimes.
tar -C "$ARTIFACT" --transform 's,^,cgo2027/,' --owner=0 --group=0 --numeric-owner \
    --no-recursion -cf - -T "$TMP/artifact.list" | gzip -n -9 > "$TMP/artifact.tar.gz"
ART_FILES="$(wc -l < "$TMP/artifact.list")"

# ---- the B300 replay pack (solve_b300_locally.sh), when there is one ------------
TARBALLS="poseidon-src cumpsgemm-src artifact"
[ -n "$HERBIE_SHA" ] && TARBALLS="$TARBALLS herbie-src"
if [ -f "$PACK/picks_b300.tsv" ]; then
  tar -C "$(dirname "$PACK")" --transform "s,^$(basename "$PACK"),replay_pack," --owner=0 --group=0 \
      --numeric-owner --sort=name -cf - "$(basename "$PACK")" | gzip -n -9 > "$TMP/replay_pack.tar.gz"
  TARBALLS="$TARBALLS replay_pack"
else
  echo "make_bundle: NOTE no replay pack at $PACK (run solve_b300_locally.sh); the replay stage will refuse to run"
fi

# ---- privacy check: no local paths or account names in what ships ------------
# A file naming this machine's mount points or this account is listed (not
# fatal: decide before sending). Files with some /home/ path are only counted:
# upstream Enzyme test IR and Herbie's web pages carry their authors' paths.
PRIV_RE="/mnt[0-9]+/|$(id -un)"
check_privacy() {   # $1 = tarball
  local d="$TMP/scan.$(basename "$1" .tar.gz)"
  mkdir -p "$d" && tar -C "$d" -xzf "$1"
  local hits homes
  hits="$(grep -rl -E "$PRIV_RE" "$d" 2>/dev/null | sed "s,^$d/,," || true)"
  homes="$(grep -rIl '/home/' "$d" 2>/dev/null | wc -l || true)"
  if [ -n "$hits" ]; then
    echo "make_bundle: NOTE $(basename "$1"): files naming a local mount point or this account:"
    echo "$hits" | sed 's/^/    /' | head -40
  fi
  echo "make_bundle: $(basename "$1"): $homes text files mention some /home/ path"
  rm -rf "$d"
}
for t in $TARBALLS; do check_privacy "$TMP/$t.tar.gz"; done

# ---- manifest -----------------------------------------------------------------
{
  echo "# poseidon_b300 bundle manifest: sha256  size  file"
  for t in $TARBALLS; do
    printf '%s  %s  %s\n' "$(sha256sum < "$TMP/$t.tar.gz" | cut -d' ' -f1)" \
      "$(stat -c %s "$TMP/$t.tar.gz")" "$t.tar.gz"
  done
  echo "compiler_commit=$COMPILER_SHA"
  [ -n "$HERBIE_SHA" ] && echo "herbie_commit=$HERBIE_SHA"
  echo "cumpsgemm_commit=$CUMPS_SHA"
  echo "artifact_files=$ART_FILES"
  echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$TMP/MANIFEST"

rm -f "$OUT/replay_pack.tar.gz" "$OUT/herbie-src.tar.gz"
for t in $TARBALLS; do mv -f "$TMP/$t.tar.gz" "$OUT/$t.tar.gz"; done
mv -f "$TMP/MANIFEST" "$OUT/MANIFEST"
cp "$TMP/artifact.list" "$OUT/artifact.files"
echo "make_bundle: wrote $OUT"
( cd "$OUT" && ls -l --block-size=K ./*.tar.gz MANIFEST | awk '{print "  " $5 "\t" $9}' )
echo "  compiler $COMPILER_SHA, herbie ${HERBIE_SHA:-not bundled}, artifact $ART_FILES files"
