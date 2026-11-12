#!/usr/bin/env python3
"""R-ChFSI eigensolver paper figure (Fig. 6, camera-quality, vector PDF), single panel.

Per-eigenpair residual spectrum: the residual norm of each of the 64 wanted
eigenpairs at solver exit (the locking criterion), one curve per configuration,
against the locking tolerance.  Every rung of the solver's frontier locks 64/64
below the tolerance; the two filter-only rungs are indistinguishable from scalar
FP64.  Data: <trace-dir>/eigresid_*.txt.  B&W-safe (markers + linestyle).

Trace resolution. With no --trace-dir, a fresh traces/ directory wins when it
holds every trace the figure needs, else golden/rtx5090/traces (the RTX 5090
run EXPECTED_RESULTS.md cites). The legend's speedups come from the ladder
tables beside the traces. --out defaults to ../../figures/rchfsi_residuals.pdf.

  python plot_chase.py --trace-dir golden/rtx5090/traces --out /tmp/rchfsi_residuals.pdf
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))  # the one figstyle.py
from figstyle import *  # noqa: F403,E402

GOLDEN = os.path.join(HERE, "golden", "rtx5090", "traces")
FRESH = os.path.join(HERE, "traces")
# One curve per configuration: (row id of EXPECTED_RESULTS.md, the trace
# scripts/run_ladder.sh writes, colour, linestyle, marker, width, label).
GREEN, BLUE, GRAY, RED, ORANGE = COLORS[0], COLORS[2], "#555555", COLORS[3], COLORS[1]
specs = [("fp64", "eigresid_fp64.txt", "0.15", "-", None, 2.2, r"Scalar FP64"),
         ("rchfsi-tf32tcec", "eigresid_rchfsi-tf32tcec.txt", GREEN, "--", "o", 1.0, r"TF32TCEC Filter"),
         ("rchfsi-bf16tcec", "eigresid_rchfsi-bf16tcec.txt", GREEN, "-", "^", 1.0, r"BF16TCEC Filter"),
         ("rchfsi-bf16tcec-nm14", "eigresid_rchfsi-bf16tcec-nm14.txt",
          BLUE, "--", "D", 1.0, r"BF16TCEC Filter $+$ Ozaki-II $\mathit{nm} = 14$"),
         ("rchfsi-knee", "eigresid_rchfsi-knee.txt",
          BLUE, "-", "s", 1.0, r"BF16 Filter $+$ Ozaki-II $\mathit{nm} = 12$"),
         ("rchfsi-blind", "eigresid_rchfsi-blind.txt", RED, "-", "v", 1.0, r"Sensitivity-Blind")]

def _default_traces():
    if all(os.path.exists(os.path.join(FRESH, s[1])) for s in specs):
        return FRESH
    return GOLDEN

_ap = argparse.ArgumentParser()
_ap.add_argument("--trace-dir", default=None,
                 help="trace bundle (default: traces/ if complete, else golden/rtx5090/traces)")
_ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "rchfsi_residuals.pdf"))
_args = _ap.parse_args()
TRACES = os.path.abspath(_args.trace_dir or _default_traces())
OUT = os.path.abspath(_args.out)

def _fresh_speedups():
    """Speedup of each expected row over the run's FP64 baseline, from the
    ladder tables beside the traces; {} when they are missing."""
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
    for name, key in (("chase_money_data.tsv", "solve_ms"), ("ablation_knee.tsv", "solve_ms_avg5")):
        p = os.path.join(base, name)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p), delimiter="\t"):
            arm = r.get("arm", "sensitivity-guided")
            try:
                ms[(arm, float(r["budget"]))] = float(r[key])
            except (ValueError, KeyError):
                pass
    fp64 = ms.get(("sensitivity-guided", 0.0))
    if not fp64:
        return {}
    out = {"fp64": 1.0}
    for row, arm in (("rchfsi-tf32tcec", "sensitivity-guided"), ("rchfsi-bf16tcec", "sensitivity-guided"),
                     ("rchfsi-bf16tcec-nm14", "sensitivity-guided"), ("rchfsi-knee", "sensitivity-guided"),
                     ("rchfsi-blind", "sensitivity-blind")):
        try:
            b = float(sel[row].split("=", 1)[1])
            out[row] = fp64 / ms[(arm, b)]
        except (KeyError, ValueError, IndexError, ZeroDivisionError):
            pass
    return out

SPEEDUP = _fresh_speedups()

fig, axM = plt.subplots(figsize=(WIDTH_COL, 1.95))

# ------------- Per-eigenpair residual spectrum ---------------------------------
# The solid accuracy measure with a populated axis: the residual norm of each of
# the 64 wanted eigenpairs at solver exit (the locking criterion; for a
# Hermitian operator it also bounds the eigenvalue error), one curve per
# configuration, against the locking tolerance. Every rung of the solver's
# frontier sits below the tolerance for every eigenpair (all locked, 64/64); the
# two filter-only rungs are indistinguishable from scalar FP64.
TOL = 1e-10

def load_spectrum(path):
    with open(os.path.join(TRACES, path)) as f:
        for line in f:
            if "OPT EIGRESID =" in line or "GOLD EIGRESID =" in line:
                return [float(t) for t in line.split("=")[1].split()]
    raise SystemExit(f"no EIGRESID in {path}")

idx = list(range(1, 65))
# The two TCEC filter-only curves coincide with the FP64 curve (all three are
# FP64-class), so FP64 is drawn as a wide translucent band underneath and the
# TCEC curves on top with staggered markers.
STYLE = {"fp64":                 dict(lw=3.2, alpha=0.30, zorder=2, markevery=5),
         "rchfsi-tf32tcec":      dict(lw=1.0, alpha=1.0,  zorder=4, markevery=(0, 5)),
         "rchfsi-bf16tcec":      dict(lw=1.0, alpha=1.0,  zorder=4, markevery=(2, 5)),
         # nm = 14 joins the FP64-class curves for the last third of the
         # spectrum; draw it on top with its own marker phase.
         "rchfsi-bf16tcec-nm14": dict(lw=1.0, alpha=0.9,  zorder=5, markevery=(4, 5))}
for row, trace, col, ls, mk, lw, name in specs:
    r = load_spectrum(trace)
    x = SPEEDUP.get(row)
    lab = name + ("" if x is None else r" ($%.2f\times$)" % x)
    st = STYLE.get(row, dict(lw=lw, alpha=1.0, zorder=3, markevery=5))
    axM.plot(idx, r, color=col, ls=ls, lw=st["lw"], alpha=st["alpha"], marker=mk,
             ms=2.6, mec="k", mew=0.25, markevery=st["markevery"],
             zorder=st["zorder"], label=lab)

axM.axhline(TOL, color=RED, ls=":", lw=1.0, zorder=2)
axM.text(63.5, TOL * 1.6, r"locking tolerance", fontsize=5.8, color=RED,
         ha="right")
axM.set_yscale("log")
axM.set_xlim(0, 65)
axM.set_ylim(1e-14, 1e-5)
axM.set_xticks([1, 16, 32, 48, 64])
axM.set_xlabel(r"Eigenpair Index")
axM.set_ylabel(r"Residual Norm at Exit")
axM.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
axM.legend(loc="upper left", bbox_to_anchor=(0.0, 0.79), fontsize=5.0, ncol=2,
           columnspacing=0.8, handlelength=1.8, framealpha=0.92,
           borderaxespad=0.3, labelspacing=0.3)

fig.tight_layout(pad=0.35)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
savepdf(fig, OUT)
print("wrote %s  (traces %s)" % (OUT, TRACES))
