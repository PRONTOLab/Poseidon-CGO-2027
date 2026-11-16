#!/usr/bin/env bash
# bcshim.sh: num_add / num_div, digit-exact replacements for `bc -l` addition and
# scale=S division (truncating), since bc is absent from stock CUDA images.

_BCSHIM_PY_PROG='
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 2000   # far beyond any wall-clock sum; keeps addition exact


def bcfmt_decimal(d):
    if d == 0:
        return "0"
    s = format(d, "f")
    if s.startswith("-0."):
        return "-" + s[2:]
    if s.startswith("0."):
        return s[1:]
    return s


def bcfmt_scaled(i, scale):
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
        sys.stderr.write("Runtime error (func=(main), adr=9): Divide by zero\n")
        sys.exit(0)
    q = (num / den) * (10 ** scale)
    t = abs(q.numerator) // q.denominator          # truncate toward zero
    print(bcfmt_scaled(-t if q < 0 else t, scale))

else:
    sys.exit("bcshim: unknown op %r" % op)
'

num_add() { "${PY:-python3}" -c "$_BCSHIM_PY_PROG" add "$1" "$2"; }

num_div() { "${PY:-python3}" -c "$_BCSHIM_PY_PROG" div "$1" "$2" "${3:-3}"; }
