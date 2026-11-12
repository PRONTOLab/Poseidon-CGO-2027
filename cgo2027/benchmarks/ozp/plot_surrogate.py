#!/usr/bin/env python3
"""ozp_surrogate.pdf — surrogate-profile generality (eval §5.2, SCVH).
(a) the Ozaki-II rung tau = 1e-7 picks from the N=512 profile (nm8) clears the
    chemistry bar at every deploy scale; nm9 and nm10 are the next rungs up.
(b) the win grows with scale, over the NAIVE scalar-FP64 baseline shared with the
    money plot (cuBLAS DGEMM shown as a per-N reference). Panel titles live in the
    LaTeX caption. Reads surrogate.tsv (N, method, rmsd, time_ms, speedup_naive);
    falls back to golden/rtx5090/surrogate.tsv if no fresh sweep is present. Writes into
    ../../figures/. B&W-safe: distinct by marker + linestyle, not color alone."""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))  # the one figstyle.py
from figstyle import *  # noqa: F403,E402

FIGDIR = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))
CHEM = 1e-7
# PICK = the rung tau=1e-7 selects from the N=512 profile; MID and TOP the
# next rungs up.
PICK, MID, TOP = "nm8", "nm9", "nm10"
PICK_LBL = rf"$\mathit{{nm}} = {PICK[2:]}$ ($\tau = 10^{{-7}}$)"
MID_LBL  = rf"$\mathit{{nm}} = {MID[2:]}$"
TOP_LBL  = rf"$\mathit{{nm}} = {TOP[2:]}$"

_p = os.path.join(HERE, "surrogate.tsv")
if not os.path.exists(_p):
    _p = os.path.join(HERE, "golden", "rtx5090", "surrogate.tsv")
rows = list(csv.DictReader(open(_p), delimiter="\t"))


def series(method, key):
    rs = [r for r in rows if r["method"] == method]
    return [int(r["N"]) for r in rs], [float(r[key]) for r in rs]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.7))

# ---- panel (a): accuracy transfer (RMSD vs N) ----
ax1.axhline(CHEM, color="#9467bd", ls="-", lw=1.0)
ax1.text(8192, 4.2e-8, r"Scientifically Meaningful", color="#9467bd", fontsize=6.5, ha="right", va="top")
ax1.axhspan(1e-16, CHEM, color="#9467bd", alpha=0.07)
for m, mk, ls, lbl, col in [(TOP, "^", "-", TOP_LBL, "#1f77b4"),
                            (MID, "s", "--", MID_LBL, "#ff7f0e"),
                            (PICK, "o", "-", PICK_LBL, "#2ca02c")]:
    x, y = series(m, "rmsd")
    ax1.plot(x, y, color=col, marker=mk, ls=ls,
             lw=1.3, ms=5, mec="k", mew=0.4, label=lbl)
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xticks([1024, 2048, 4096, 8192])
ax1.set_xticklabels(["1024", "2048", "4096", "8192"])
ax1.set_xlabel(r"Deploy Size $N$")
ax1.set_ylabel(r"Density-Matrix RMSD")
ax1.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
ax1.legend(fontsize=6.6, loc="lower right", framealpha=0.92, handletextpad=0.4,
           borderpad=0.3, labelspacing=0.25)

# ---- panel (b): speedup vs N, over the NAIVE scalar-FP64 baseline ----
ax2.axhline(1.0, color="#222222", ls="--", lw=1.0, label=r"Scalar FP64")
xc, yc = series("cublas", "speedup_naive")   # cuBLAS over naive: a per-N reference curve
ax2.plot(xc, yc, color="#555555", ls="-.", marker="D", ms=4.5, lw=1.1, mec="k", mew=0.4,
         label=r"cuBLAS DGEMM (Reference)")
x, y = series(TOP, "speedup_naive")
ax2.plot(x, y, color="#1f77b4", marker="^", ls="-",
         lw=1.3, ms=5, mec="k", mew=0.4, label=TOP_LBL)
x, y = series(PICK, "speedup_naive")
ax2.plot(x, y, color="#2ca02c", marker="o", ls="-",
         lw=1.3, ms=5, mec="k", mew=0.4, label=PICK_LBL)
ax2.set_xscale("log", base=2)
ax2.set_xticks([1024, 2048, 4096, 8192])
ax2.set_xticklabels(["1024", "2048", "4096", "8192"])
ax2.set_xlabel(r"Deploy Size $N$")
ax2.set_ylabel(r"Speedup")
xn, yn = series(PICK, "speedup_naive")
ax2.annotate(rf"${yn[-1]:.1f}\times$", xy=(xn[-1], yn[-1]), xytext=(xn[-1] * 0.80, yn[-1]),
             fontsize=8, color="#2ca02c", ha="right", va="center")
ax2.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
ax2.legend(fontsize=6.6, loc="upper left",
           framealpha=0.92, handletextpad=0.4, borderpad=0.3, labelspacing=0.25)

os.makedirs(FIGDIR, exist_ok=True)
savepdf(fig, os.path.join(FIGDIR, "ozp_surrogate.pdf"))
print("wrote", os.path.join(FIGDIR, "ozp_surrogate.pdf"))
