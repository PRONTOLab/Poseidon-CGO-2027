#!/usr/bin/env python3
"""Fig. 6 (figures/rchfsi_residuals.pdf) from traces/ if complete, else recorded/rtx5090/traces (--trace-dir, --out)."""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from figstyle import *  # noqa: F403,E402

RECORDED = os.path.join(HERE, "recorded", "rtx5090", "traces")
FRESH = os.path.join(HERE, "traces")
# (row id, trace file, colour, linestyle, marker, width, label)
GREEN, BLUE, GRAY, RED, ORANGE = COLORS[0], COLORS[2], "#555555", COLORS[3], COLORS[1]
specs = [("fp64", "eigresid_fp64.txt", "0.15", "-", None, 2.2, r"Scalar FP64"),
         ("rchfsi-tf32tcec", "eigresid_rchfsi-tf32tcec.txt", GREEN, "--", "o", 1.0, r"TF32TCEC Filter"),
         ("rchfsi-bf16tcec", "eigresid_rchfsi-bf16tcec.txt", GREEN, "-", "^", 1.0, r"BF16TCEC Filter"),
         ("rchfsi-bf16tcec-nm14", "eigresid_rchfsi-bf16tcec-nm14.txt",
          BLUE, "--", "D", 1.0, r"BF16TCEC Filter $+$ Ozaki-II $\mathit{nm} = 14$"),
         ("rchfsi-bf16-nm12", "eigresid_rchfsi-bf16-nm12.txt",
          BLUE, "-", "s", 1.0, r"BF16 Filter $+$ Ozaki-II $\mathit{nm} = 12$"),
         ("rchfsi-blind", "eigresid_rchfsi-blind.txt", RED, "-", "v", 1.0, r"Sensitivity-Blind")]

def _default_traces():
    if all(os.path.exists(os.path.join(FRESH, s[1])) for s in specs):
        return FRESH
    return RECORDED

_ap = argparse.ArgumentParser()
_ap.add_argument("--trace-dir", default=None,
                 help="trace bundle (default: traces/ if complete, else recorded/rtx5090/traces)")
_ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "rchfsi_residuals.pdf"))
_args = _ap.parse_args()
TRACES = os.path.abspath(_args.trace_dir or _default_traces())
OUT = os.path.abspath(_args.out)

def _fresh_speedups():
    # Speedup of each row over the FP64 baseline, from the tables beside the traces.
    import csv
    base = os.path.dirname(TRACES)
    sel = {}
    sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
    try:
        from expected_results import load_picks
        sel = {r["id"]: r["select"] for r in load_picks()}
    except Exception:
        return {}
    ms = {}
    def rows(name):
        p = os.path.join(base, name)
        return list(csv.DictReader(open(p), delimiter="\t")) if os.path.exists(p) else []
    for r in rows("strategies.tsv"):
        if r["budget"] == "0":
            ms[("sensitivity-guided", 0.0)] = float(r["solve_ms"])
    for r in rows("ablation.tsv"):
        if r["strategy"] == "sensitivity-blind":
            ms[(r["strategy"], float(r["budget"]))] = float(r["solve_ms_avg5"])
    for r in rows("guided.tsv"):
        if float(r["budget"]) != 0.0:
            ms[("sensitivity-guided", float(r["budget"]))] = float(r["solve_ms_avg5"])
    fp64 = ms.get(("sensitivity-guided", 0.0))
    if not fp64:
        return {}
    out = {"fp64": 1.0}
    for row, kind in (("rchfsi-tf32tcec", "sensitivity-guided"), ("rchfsi-bf16tcec", "sensitivity-guided"),
                     ("rchfsi-bf16tcec-nm14", "sensitivity-guided"), ("rchfsi-bf16-nm12", "sensitivity-guided"),
                     ("rchfsi-blind", "sensitivity-blind")):
        try:
            b = float(sel[row].split("=", 1)[1])
            out[row] = fp64 / ms[(kind, b)]
        except (KeyError, ValueError, IndexError, ZeroDivisionError):
            pass
    return out

SPEEDUP = _fresh_speedups()

fig, axM = plt.subplots(figsize=(WIDTH_COL, 1.95))

TOL = 1e-10

def load_spectrum(path):
    with open(os.path.join(TRACES, path)) as f:
        for line in f:
            if "OPT EIGRESID =" in line or "GOLD EIGRESID =" in line:
                return [float(t) for t in line.split("=")[1].split()]
    raise SystemExit(f"no EIGRESID in {path}")

idx = list(range(1, 65))
# The TCEC curves coincide with FP64: FP64 is a translucent band, markers are staggered.
STYLE = {"fp64":                 dict(lw=3.2, alpha=0.30, zorder=2, markevery=5),
         "rchfsi-tf32tcec":      dict(lw=1.0, alpha=1.0,  zorder=4, markevery=(0, 5)),
         "rchfsi-bf16tcec":      dict(lw=1.0, alpha=1.0,  zorder=4, markevery=(2, 5)),
         "rchfsi-bf16tcec-nm14": dict(lw=1.0, alpha=0.9,  zorder=5, markevery=(4, 5))}
for row, trace, col, ls, mk, lw, name in specs:
    r = load_spectrum(trace)
    x = SPEEDUP.get(row)
    lab = name + ("" if x is None else r" ($1\times$)" if row == "fp64" else r" ($%.2f\times$)" % x)
    st = STYLE.get(row, dict(lw=lw, alpha=1.0, zorder=3, markevery=5))
    axM.plot(idx, r, color=col, ls=ls, lw=st["lw"], alpha=st["alpha"], marker=mk,
             ms=2.6, mec="k", mew=0.25, markevery=st["markevery"],
             zorder=st["zorder"], label=lab)

axM.axhline(TOL, color=RED, ls=":", lw=1.0, zorder=2)
axM.text(63.5, TOL * 1.6, r"locking tolerance", fontsize=5.8, color=RED,
         ha="right")
axM.set_yscale("log")
axM.set_xlim(0, 65)
axM.set_ylim(1e-14, 1e-4)
axM.set_xticks([1, 16, 32, 48, 64])
axM.set_xlabel(r"Eigenpair Index")
axM.set_ylabel(r"Residual Norm at Exit")
axM.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
axM.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=5.0, ncol=2,
           columnspacing=0.8, handlelength=1.8, framealpha=0.92,
           borderaxespad=0.3, labelspacing=0.3)

fig.tight_layout(pad=0.35)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
savepdf(fig, OUT)
print("wrote %s  (traces %s)" % (OUT, TRACES))
