#!/usr/bin/env bash
# bcshim.sh -- exact drop-in replacements for the two `bc` expressions the
# benchmark timing scripts used to average wall clocks.
#
# WHY THIS EXISTS. The artifact's stated dependencies are nvidia-smi, cmake,
# rsync, python3 and bash. `bc` is NOT among them, and it is absent on stock
# CUDA container images. When it is missing, `$(echo ... | bc -l)` expands to
# the empty string, every avg-of-N wall clock silently becomes `?`, and the
# only trace is `bc: command not found` scrolling past in the log. That cost
# the B300 campaign a full re-measurement pass over every rung of every arm
# (docs/b300-prep-2026-08.md, "Defect: `bc` is not on the box"). The accuracy
# and convergence columns were unaffected (they come from grep over the
# program's own stdout), which is exactly what made it easy to miss.
#
# EXACT EQUIVALENCE, NOT AN APPROXIMATION. These reproduce GNU bc 1.07.1
# digit for digit, so no number recorded with the old code would change:
#
#   num_add A B        ==  echo "A + B" | bc -l
#       Exact decimal addition, arbitrary precision. bc's result scale for
#       `+`/`-` is max(scale(A), scale(B)) and is NOT affected by the `scale`
#       variable, so trailing zeros are preserved exactly as bc preserves them
#       (`604 + 1.2300` -> `605.2300`). Python's Decimal has the identical
#       ideal-exponent rule, so it is used directly.
#
#   num_div A B [S]    ==  echo "scale=S; A/B" | bc -l          (S defaults to 3)
#       Exact decimal division TRUNCATED toward zero to S fractional digits --
#       bc NEVER rounds. This is the trap: bc gives `2/3` -> `.666` and
#       `1.0015/1` -> `1.001`, where Python's `f"{x:.3f}"` gives `0.667` and
#       `1.002`. Computed over Fractions so there is no binary-float error to
#       tip a boundary case either way.
#
#   Output formatting is bc's, not Python's:
#       * magnitudes below 1 print without the leading zero  (`.500`, `-.500`)
#       * a zero result prints as `0`, never `.000`          (`.0005/1` -> `0`)
#       * a division result always carries exactly S fractional digits (`5.000`)
#       * division by zero writes bc's runtime-error text to stderr, prints
#         nothing on stdout, and exits 0 -- as bc does.
#
# Usage: source this file, then use num_add / num_div in place of the pipes.
#   sum=$(num_add "$sum" "$m")            # was: $(echo "$sum + $m" | bc -l)
#   avg=$(num_div "$sum" "$n")            # was: $(echo "scale=3; $sum/$n" | bc -l)
#
# Verified against the system bc over a spread of values including exact
# .0005 half-way boundaries; see the equivalence test in scripts/test_bcshim.sh.

_BCSHIM_PY_PROG='
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 2000   # far beyond any wall-clock sum; keeps addition exact


def bcfmt_decimal(d):
    """Render a Decimal the way bc renders a number."""
    if d == 0:
        return "0"
    s = format(d, "f")
    if s.startswith("-0."):
        return "-" + s[2:]
    if s.startswith("0."):
        return s[1:]
    return s


def bcfmt_scaled(i, scale):
    """Render (i / 10**scale) the way bc renders a division result."""
    if i == 0:
        return "0"
    sign = "-" if i < 0 else ""
    a = str(abs(i))
    if scale == 0:
        return sign + a
    a = a.zfill(scale)
    return sign + a[:-scale] + "." + a[-scale:]


op = sys.argv[1]

if op == "add":
    print(bcfmt_decimal(Decimal(sys.argv[2]) + Decimal(sys.argv[3])))

elif op == "div":
    num, den, scale = Fraction(sys.argv[2]), Fraction(sys.argv[3]), int(sys.argv[4])
    if den == 0:
        # bc: message on stderr, nothing on stdout, exit status 0.
        sys.stderr.write("Runtime error (func=(main), adr=9): Divide by zero\n")
        sys.exit(0)
    q = (num / den) * (10 ** scale)
    t = abs(q.numerator) // q.denominator          # truncate toward zero
    print(bcfmt_scaled(-t if q < 0 else t, scale))

else:
    sys.exit("bcshim: unknown op %r" % op)
'

# Exact equivalent of:  echo "$1 + $2" | bc -l
num_add() { "${PY:-python3}" -c "$_BCSHIM_PY_PROG" add "$1" "$2"; }

# Exact equivalent of:  echo "scale=${3:-3}; $1/$2" | bc -l
num_div() { "${PY:-python3}" -c "$_BCSHIM_PY_PROG" div "$1" "$2" "${3:-3}"; }
