#!/usr/bin/env bash
# make_bundle.sh [options]: write bundle/, the files poseidon_b300.py uploads to Modal.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/bundle"
CUMPS_REPO="$HERE/../sources/cuMpSGEMM" CUMPS_COMMIT="HEAD"
CUMPS_PATCHES="$HERE/../sources/cumpsgemm-explicit-gemm-instantiation.patch"
COMPILER_REPO="$HERE/../sources/Enzyme" COMPILER_COMMIT="HEAD" ARTIFACT="$HERE/.."
WITH_HERBIE=0 HERBIE_REPO="$HERE/../sources/herbie" HERBIE_PATCH="$HERE/../sources/herbie-num-taylor-terms.patch"

usage() {
  cat <<'EOF'
usage: ./make_bundle.sh [options]
  --compiler-repo <dir>     Enzyme checkout (default: ../sources/Enzyme)
  --compiler-commit <rev>   (default: HEAD of that checkout)
  --artifact <dir>          artifact root (default: ..)
  --cumpsgemm-repo <dir>    cuMpSGEMM checkout (default: ../sources/cuMpSGEMM)
  --cumpsgemm-commit <rev>  (default: HEAD of that checkout)
  --with-herbie             also bundle Herbie (needed by --stage full only)
  --out <dir>               (default: ./bundle)
EOF
  exit 2
}
while [ $# -gt 0 ]; do
  case "$1" in
    --compiler-repo)   COMPILER_REPO="$2"; shift 2 ;;
    --compiler-commit) COMPILER_COMMIT="$2"; shift 2 ;;
    --artifact)        ARTIFACT="$2"; shift 2 ;;
    --with-herbie)     WITH_HERBIE=1; shift ;;
    --out)             OUT="$2"; shift 2 ;;
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
if [ "$WITH_HERBIE" = 1 ]; then
  [ -e "$HERBIE_REPO/.git" ] ||
    { echo "make_bundle: $HERBIE_REPO is not checked out: git submodule update --init --recursive" >&2; exit 1; }
  [ -f "$HERBIE_PATCH" ] || { echo "make_bundle: no $HERBIE_PATCH" >&2; exit 1; }
  HERBIE_SHA="$(git -C "$HERBIE_REPO" rev-parse --verify "HEAD^{commit}")"
  git -C "$HERBIE_REPO" cat-file -e "$HERBIE_SHA:src/main.rkt" ||
    { echo "make_bundle: $HERBIE_SHA is not a Herbie tree" >&2; exit 1; }
fi

mkdir -p "$OUT"
TMP="$(mktemp -d "$OUT/.tmp.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

# Fixed --mtime and gzip -n: the same commit gives the same bytes.
git -C "$COMPILER_REPO" archive --format=tar --prefix=poseidon-src/ "$COMPILER_SHA" enzyme |
  gzip -n -9 > "$TMP/poseidon-src.tar.gz"
if [ -n "$HERBIE_SHA" ]; then
  mkdir -p "$TMP/herbie-tree"
  git -C "$HERBIE_REPO" archive --format=tar --prefix=herbie/ "$HERBIE_SHA" | tar -C "$TMP/herbie-tree" -xf -
  patch -s -d "$TMP/herbie-tree/herbie" -p1 < "$HERBIE_PATCH"
  tar -C "$TMP/herbie-tree" --owner=0 --group=0 --numeric-owner --sort=name --mtime=@0 -cf - herbie |
    gzip -n -9 > "$TMP/herbie-src.tar.gz"
  rm -rf "$TMP/herbie-tree"
fi

CUMPS_SHA="$(git -C "$CUMPS_REPO" rev-parse --verify "$CUMPS_COMMIT^{commit}")"
mkdir -p "$TMP/cumps"
git -C "$CUMPS_REPO" archive --format=tar --prefix=cuMpSGEMM/ "$CUMPS_SHA" | tar -C "$TMP/cumps" -xf -
git -C "$CUMPS_REPO" ls-tree -r "$CUMPS_SHA" | awk '$2 == "commit" {print $3, $4}' | while read -r sub path; do
  git -C "$CUMPS_REPO/$path" archive --format=tar --prefix="cuMpSGEMM/$path/" "$sub" | tar -C "$TMP/cumps" -xf -
done
for f in $CUMPS_PATCHES; do
  [ -f "$f" ] || { echo "make_bundle: no $f" >&2; exit 1; }
  patch -s -d "$TMP/cumps/cuMpSGEMM" -p1 < "$f"
done
tar -C "$TMP/cumps" --owner=0 --group=0 --numeric-owner --sort=name --mtime=@0 -cf - cuMpSGEMM | gzip -n -9 > "$TMP/cumpsgemm-src.tar.gz"
rm -rf "$TMP/cumps"

# The artifact directory need not be a git repository, so its .gitignore files are applied here.
EXTRA='build*/
herbie_input_*
launches_*
.venv/
/sources/
/modal/
/.git
/.git/
/.gitmodules
/.dockerignore'
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

for need in run_artifact.sh env.sh config.mk scripts/detect.sh scripts/bootstrap_cost_model.sh scripts/measure.sh \
            devices/rtx5090/cost_models/cm_sm_120_RTX5090.csv expected/picks.tsv \
            devices/rtx5090/herbie_answers/MD5SUMS benchmarks/ozp/profiles/MD5SUMS \
            benchmarks/chase_eigensolver/profiles/MD5SUMS benchmarks/genga_nbody/profiles/MD5SUMS; do
  grep -qx "$need" "$TMP/artifact.list" || { echo "make_bundle: $need would not be shipped" >&2; exit 1; }
done
for gone in local.env cost_models/target.env; do
  if grep -qx "$gone" "$TMP/artifact.list"; then echo "make_bundle: $gone would be shipped" >&2; exit 1; fi
done

tar -C "$ARTIFACT" --transform 's,^,artifact/,' --owner=0 --group=0 --numeric-owner \
    --no-recursion -cf - -T "$TMP/artifact.list" | gzip -n -9 > "$TMP/artifact.tar.gz"
ART_FILES="$(wc -l < "$TMP/artifact.list")"

TARBALLS="poseidon-src cumpsgemm-src artifact"
[ -n "$HERBIE_SHA" ] && TARBALLS="$TARBALLS herbie-src"

# Listed, not fatal: upstream Enzyme test IR and Herbie's pages carry their authors' /home/ paths.
PRIV_RE="/mnt[0-9]+/|$(id -un)"
check_privacy() {
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

{
  echo "# poseidon_b300 bundle manifest: sha256  size  file"
  for t in $TARBALLS; do
    printf '%s  %s  %s\n' "$(sha256sum < "$TMP/$t.tar.gz" | cut -d' ' -f1)" \
      "$(stat -c %s "$TMP/$t.tar.gz")" "$t.tar.gz"
  done
  echo "compiler_commit=$COMPILER_SHA"
  [ -n "$HERBIE_SHA" ] && echo "herbie_commit=$HERBIE_SHA+num-taylor-terms.patch"
  echo "cumpsgemm_commit=$CUMPS_SHA$(for f in $CUMPS_PATCHES; do printf '+%s' "$(basename "$f")"; done)"
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
