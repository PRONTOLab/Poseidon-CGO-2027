"""cm_compare.py <fresh.csv> <saved.csv> [threshold]: a fresh cost model
against a saved one. Per row family (the first column: an op such as fadd or
fma, or a family such as ozaki_dispatch_rel) the geometric mean of new/saved
over the rows both have; then every row that moved by more than the threshold
(default 10%), and the rows only one of the two has."""
import math
import sys


def load(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        try:
            rows[(parts[0], ",".join(parts[1:-1]))] = float(parts[-1])
        except ValueError:
            continue
    return rows


def main():
    fresh, saved = load(sys.argv[1]), load(sys.argv[2])
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
    print(f"fresh {sys.argv[1]}\nsaved {sys.argv[2]}\n")
    fams = {}
    moved = []
    for key in sorted(set(fresh) & set(saved)):
        a, b = fresh[key], saved[key]
        if a <= 0 or b <= 0:
            continue
        r = a / b
        fams.setdefault(key[0], []).append(r)
        if abs(r - 1.0) > thr:
            moved.append((key, b, a, r))
    print("family\trows\tgeomean(new/saved)\tmin\tmax")
    for fam in sorted(fams):
        rs = fams[fam]
        g = math.exp(sum(math.log(r) for r in rs) / len(rs))
        print(f"{fam}\t{len(rs)}\t{g:.3f}\t{min(rs):.3f}\t{max(rs):.3f}")
    for key in sorted(k for k in set(fresh) | set(saved) if k[0] == "tcec_dispatch_rel"):
        a, b = fresh.get(key), saved.get(key)
        r = f"{a / b:.3f}" if a and b else "n/a"
        print(f"\ntcec_dispatch_rel,{key[1]} (depends on the TCEC backend): saved {b}  new {a}  ratio {r}")
    print(f"\nrows that moved by more than {thr:.0%}: {len(moved)}")
    for (fam, typ), b, a, r in moved:
        print(f"  {fam},{typ}\tsaved {b:.6g}\tnew {a:.6g}\tratio {r:.3f}")
    only_new = sorted(set(fresh) - set(saved))
    only_saved = sorted(set(saved) - set(fresh))
    print(f"\nrows only in the fresh model: {len(only_new)}")
    for fam, typ in only_new:
        print(f"  {fam},{typ}")
    print(f"rows only in the saved model: {len(only_saved)}")
    for fam, typ in only_saved:
        print(f"  {fam},{typ}")


if __name__ == "__main__":
    main()
