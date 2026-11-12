#!/usr/bin/env bash
# genga_x1e6.sh <cost model csv> <out csv>: the copy of a cost model GENGA
# prices with, and its Herbie platform (<out stem>.herbie.rkt).
#
# GENGA solves on the device's cost model uniformly scaled x1e6: a uniform
# scale leaves every solver decision unchanged, and the Herbie answers a GENGA
# solve replays were searched against the platform generated from the scaled
# copy (the platform is part of every Herbie question). The dimensionless
# *_rel rows (matrix-product ratios, unused by GENGA) are copied unscaled. The
# first line names the source model and its checksum; a copy whose first line
# matches is left as it is.
#
# This is how cost_models/x1e6/ and devices/<device>/cost_models/x1e6/ were
# made from the models next to them. The platform is written by
# poseidon-calibrate --only herbie-platform from the Herbie the compiler was
# configured with; a Herbie of other module names gives a platform of another
# digest, whose questions the shipped answers do not hold.
set -euo pipefail
case "${1:-}" in -h|--help|"") sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; [ -n "${1:-}" ]; exit $? ;; esac
SRC="$1" OUT="${2:?out csv}"
[ -f "$SRC" ] || { echo "genga_x1e6.sh: no cost model $SRC" >&2; exit 1; }
if [ -z "${POSEIDON_CALIBRATE:-}" ]; then
  POSEIDON_CALIBRATE="$(bash "$(dirname "$0")/detect.sh" poseidon_calibrate)" || exit 1
fi
STAMP="# poseidon-derived-from: $(basename "$SRC") $(cksum < "$SRC")"
if [ ! -s "$OUT" ] || [ "$(head -1 "$OUT")" != "$STAMP" ]; then
  mkdir -p "$(dirname "$OUT")"
  { echo "$STAMP"
    awk -F, 'BEGIN{OFS=","}
      /^#/ || $1 ~ /_rel$/ {print; next}
      NF>=3 {$NF=sprintf("%.6f",$NF*1000000); print; next}
      {print}' "$SRC"; } > "$OUT.tmp.$$"
  mv -f "$OUT.tmp.$$" "$OUT"
  echo "[config] derived $(basename "$OUT") from $(basename "$SRC")" >&2
fi
if [ ! -s "${OUT%.csv}.herbie.rkt" ] || [ "$OUT" -nt "${OUT%.csv}.herbie.rkt" ]; then
  "$POSEIDON_CALIBRATE" --only herbie-platform --out "$OUT" >&2 ||
    { echo "genga_x1e6.sh: could not generate the Herbie platform of $OUT" >&2; exit 1; }
fi
