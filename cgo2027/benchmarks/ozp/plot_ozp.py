#!/usr/bin/env python3
"""ozp merged frontier figure (paper Fig. 5, camera-quality, vector PDF), scored
in the DOMAIN metric (Dawson JCTC'24): density-matrix RMSD with TWO thresholds
(chemically sufficient ~1e-7; near-FP64 ~2^-37). Titles live in the caption.
  ozp_convergence.pdf -- one figure, two horizontal panels:
    Left  : per-iteration RMSD (static TC flatlines above the chemistry bar).
    Right : speed/accuracy frontier; y = speedup over the NAIVE scalar-FP64 GEMM,
            cuBLAS DGEMM as an FP64 reference, auto-pick circled.
Reads pareto.tsv + dispatch_rungs.tsv + surrogate.tsv + direct_tc_lib.tsv +
autopick.tsv + traj_*.tsv. B&W-safe: every series is distinct by marker/linestyle.

Data resolution. With no --data-dir, each file is taken from this directory
when a fresh run left it here and from golden/rtx5090/ otherwise (the RTX 5090
run EXPECTED_RESULTS.md cites), so the figure regenerates from a fresh unpack
without a hardware run. An explicit
--data-dir plots that bundle alone. --out defaults to ../../figures/.

  python plot_ozp.py --data-dir golden/rtx5090 --out /tmp/ozp_convergence.pdf
"""
import argparse, csv, os, math, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))  # the one figstyle.py
from figstyle import *  # noqa: F403,E402

GOLDEN = os.path.join(HERE, "golden", "rtx5090")
_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir", default=None,
                 help="plot this bundle only (default: fresh files here, else golden/rtx5090)")
_ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "ozp_convergence.pdf"))
_args = _ap.parse_args()
OUT = os.path.abspath(_args.out)
CHEM = 1e-7              # chemically-sufficient RMSD (Dawson JCTC'24)
INDIST = 2.0 ** -37      # ~7.28e-12: indistinguishable from FP64

def _resolve(name):
    if _args.data_dir:
        return os.path.join(os.path.abspath(_args.data_dir), name)
    p = os.path.join(HERE, name)
    return p if os.path.exists(p) else os.path.join(GOLDEN, name)

def load_tsv(name):
    with open(_resolve(name)) as f: return list(csv.DictReader(f, delimiter="\t"))
def fnum(x):
    try: return float(x)
    except: return float("nan")

pareto = load_tsv("pareto.tsv")
fp64 = next(r for r in pareto if r["tag"] == "FP64")   # the naive scalar-FP64 baseline row
NAIVE_T = fnum(fp64["time_ms"])   # naive scalar-FP64 GEMM @ N=2048 (= 1x baseline)
def cublas_time():   # cuBLAS DGEMM @ N=2048, read from the sweep that measured it
    for r in load_tsv("surrogate.tsv"):
        if r["method"] == "cublas" and int(r["N"]) == 2048: return fnum(r["time_ms"])
    raise SystemExit("plot_ozp: no cuBLAS N=2048 row in surrogate.tsv")
CUBLAS_T = cublas_time()
CUBLAS_RMSD = fnum(fp64["RMSD"])  # FP64-grade, same accuracy class as naive
L2_TO_RMSD = fnum(fp64["RMSD"]) / fnum(fp64["L2_vs_Kref"])   # constant 1/sqrt(2N)

def categorize(c):
    c = c.lower()
    if "ozaki-ii" in c or "ozii" in c:    return "ozii"
    if "tcec" in c or "ozaki-i_n" in c:   return "tcec"
    if "ozaki-i-ef" in c:                 return "ef"
    if c.startswith("wmma") or "f16/f32" in c or "bf16" in c or "tf32" in c: return "static"
    return "other"

pts = []  # (rmsd, speedup_over_naive, cat, label, time_ms)
# pareto.tsv's other rows are in-kernel WMMA materializations (direct raises and
# FP16TCEC inside ozp's own loop nest); the figure shows every product in its
# library form, so only the scalar-FP64 baseline row is used from it.
pts.append((fnum(fp64["RMSD"]), 1.0, "naive", "naive FP64", NAIVE_T))

pts.append((CUBLAS_RMSD, NAIVE_T / CUBLAS_T, "cublas", "cuBLAS", CUBLAS_T))

# Direct raises as plain cuBLAS GEMMs (the reference arm,
# scripts/run_direct_tc_lib.sh; FP16 / BF16 / TF32 inputs, FP32 accumulation). They stand in for the direct-raise family in both
# frontiers.
libpath = _resolve("direct_tc_lib.tsv")
if os.path.exists(libpath):
    for r in load_tsv("direct_tc_lib.tsv"):
        if not r.get("tag") or not r["tag"].startswith("LIB_"): continue
        rm, t = fnum(r["RMSD"]), fnum(r["time_ms"])
        if math.isnan(rm) or math.isnan(t): continue
        pts.append((rm, NAIVE_T / t, "lib", r["tag"], t))

drpath = _resolve("dispatch_rungs.tsv")
if os.path.exists(drpath):
    for r in load_tsv("dispatch_rungs.tsv"):
        if not r.get("budget") or r["budget"].startswith("BASELINE"): continue
        rm, t = fnum(r["RMSD"]), fnum(r["time_ms"])
        if math.isnan(rm) or math.isnan(t): continue
        # Every budget-swept rung on the solver's frontier, not the Ozaki-II
        # family alone: under the current cost model the cheapest rung is the
        # host-dispatched FP16TCEC, which categorize() puts in the "tcec" class,
        # the same marker as the in-kernel FP16TCEC points from pareto.tsv.
        pick = r.get("pick") or ""
        lbl = ("nm=" + pick.split("nm=")[-1]) if "nm=" in pick else pick
        pts.append((rm, NAIVE_T / t, categorize(pick), lbl, t))

STYLE = {  # marker carries the distinction in B&W
    "naive":  dict(c="#222222", m="*", s=120, label=r"Scalar FP64"),
    "cublas": dict(c="#555555", m="D", s=42,  label=r"cuBLAS DGEMM"),
    "lib":    dict(c="#d62728", m="X", s=42,  label=r"Direct TC"),
    "tcec":   dict(c="#ff7f0e", m="s", s=34,  label=r"FP16TCEC"),
    "ozii":   dict(c="#2ca02c", m="o", s=52,  label=r"Ozaki-II (Poseidon)"),
}

# ===================== merged figure: convergence | frontier =====================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.3))

# ---- Left: convergence trajectory (per-iteration RMSD) ----
def plot_traj(ax, tag, color, ls, label, lw=1.2, marker=None, me=6, alpha=1.0):
    p = _resolve(f"traj_{tag}.tsv")
    if not os.path.exists(p): return False
    it, er = [], []
    for row in open(p):
        a = row.split()
        if len(a) >= 2:
            it.append(int(a[0])); er.append(float(a[1]) * L2_TO_RMSD)
    ax.plot(it, er, color=color, lw=lw, ls=ls, label=label, zorder=3,
            marker=marker, markevery=me, markersize=3.2, mec="k", mew=0.3,
            alpha=alpha)
    return True

plot_traj(axL, "FP64",  "#222222", "-",  r"Scalar FP64", lw=1.3)
plot_traj(axL, "LIB_f16",  "#d62728", "--", r"Direct FP16/FP32 TC")
plot_traj(axL, "LIB_tf32", "#e377c2", ":",  r"Direct TF32/FP32 TC")
# The dispatched FP16TCEC product's own trajectory (forced build, --trace);
# the in-kernel FP16TCEC trace is the fallback for older bundles.
if not plot_traj(axL, "TCEC_dispatch", "#ff7f0e", "-.", r"FP16TCEC"):
    plot_traj(axL, "M0_4", "#ff7f0e", "-.", r"FP16TCEC")
# Three rungs out of the nm = 8..14 family: the tau=1e-7 pick, the tau=1e-12
# pick (also the cheapest rung that clears 2^-37), and the top of the family.
# All seven would overplot.
plot_traj(axL, "OZII_nm9", "#2ca02c", "-", r"Ozaki-II (Poseidon)", lw=1.4,
          marker="o", me=6, alpha=0.65)
plot_traj(axL, "OZII_nm13", "#2ca02c", "-", "_nolegend_", lw=1.4, marker="s", me=(2, 6),
          alpha=0.65)
plot_traj(axL, "OZII_nm14", "#2ca02c", "-", "_nolegend_", lw=1.4, marker="D", me=(4, 6),
          alpha=0.65)
axL.text(20.5, 9.0e-9,  r"$\mathit{nm} = 9$",  color="#2ca02c", fontsize=6)
axL.text(20.5, 1.5e-12, r"$\mathit{nm} = 13$", color="#2ca02c", fontsize=6)
axL.text(20.5, 1.6e-14, r"$\mathit{nm} = 14$", color="#2ca02c", fontsize=6)
axL.axhline(CHEM,   color="#9467bd", ls="-",  lw=0.9)
axL.axhline(INDIST, color="#1f77b4", ls="--", lw=0.8)
# The chemistry bar's label moves to the empty left half: the tau=1e-7 pick
# (nm = 9) flatlines at RMSD 3.0e-8, right under the bar, and would collide with
# a right-aligned label there.
axL.text(1.3, 3.0e-6, r"Scientifically", color="#9467bd", fontsize=6.5, ha="left", va="center")
axL.text(1.3, 5.0e-7, r"Meaningful",     color="#9467bd", fontsize=6.5, ha="left", va="center")
axL.text(30, 4.0e-11, r"Near FP64",  color="#1f77b4", fontsize=6.5, ha="right", va="center")
axL.set_yscale("log")
axL.set_xlabel(r"Purification Iteration")
axL.set_ylabel(r"Density-Matrix RMSD")
axL.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
axL.legend(fontsize=6.2, loc="lower left", framealpha=0.92, handletextpad=0.4,
           borderpad=0.3, labelspacing=0.25)

# ---- Right: speed/accuracy frontier ----
# Speedups span 1x (scalar FP64) to 73x (cuBLAS direct TC): log y axis.
XLO, XHI, YLO, YHI = 4e-16, 1e-2, 0.7, 130.0
axR.axvspan(XLO, INDIST, color="#1f77b4", alpha=0.08, zorder=0)
axR.axvspan(INDIST, CHEM, color="#9467bd", alpha=0.08, zorder=0)
axR.axvline(CHEM,   color="#9467bd", ls="-",  lw=1.0, zorder=1)
axR.axvline(INDIST, color="#1f77b4", ls="--", lw=0.9, zorder=1)

def dominated(p, ps):
    return any((q[0] <= p[0] and q[1] >= p[1] and q is not p and
                (q[0] < p[0] or q[1] > p[1])) for q in ps)
def pfront(ps):
    return sorted([p for p in ps if not dominated(p, ps)], key=lambda p: p[0])
# ON = full candidate set (keeping scalar FP64 is always available); OFF = the
# ablation (FP16TCEC + Ozaki-II disabled -> direct downcasts + scalar FP64 only).
# Solid = ON on top; dashed = OFF underneath (GENGA ablation convention).
front     = pfront([p for p in pts if p[2] in ("lib", "tcec", "ozii", "naive")])
front_off = pfront([p for p in pts if p[2] in ("lib", "naive")])
axR.plot([p[0] for p in front], [p[1] for p in front], color="#999999",
         ls="-", lw=1.2, alpha=0.85, drawstyle="steps-post", zorder=2.2,
         label=r"Frontier, Emulation ON")
axR.plot([p[0] for p in front_off], [p[1] for p in front_off], color="#666666",
         ls="--", lw=0.9, drawstyle="steps-post", zorder=2,
         label=r"Frontier, Emulation OFF")
for cat in ["lib", "tcec", "ozii", "cublas", "naive"]:
    xs = [p[0] for p in pts if p[2] == cat]; ys = [p[1] for p in pts if p[2] == cat]
    if not xs: continue
    st = STYLE[cat]
    axR.scatter(xs, ys, c=st["c"], marker=st["m"], s=st["s"], edgecolors="k",
                linewidths=0.5, zorder=3, label=st["label"])
# The compile-time picks of the call-site accuracy budget: one tau per circle.
# Tighter budgets walk the frontier toward higher precision; this is the
# automation the figure must show, not a single operating point.
# The SOLVER's picks (scripts/run_autopick.sh writes autopick.tsv), not the
# cheapest point under tau: the solver compares tau against its MODELLED domain
# error, so a circled rung can sit on either side of its own tau.
# Each circle is drawn on that rung's own dispatch_rungs.tsv point, so the
# circled coordinates are the frontier sweep's measurement, not the autopick
# session's. autopick.tsv, when present, names the rung the solver picked at
# each tau, so the circles follow the solve instead of being restated here;
# EXPECTED_RESULTS.md (rows ozp-tau*) states the picks. The fallback list below
# is used only for a bundle that carries no autopick.tsv.
TAU_TEXT = {
    "1e-6":  (r"$\tau = 10^{-6}$",  (1.5e-6,  9.0), "left"),
    "1e-7":  (r"$\tau = 10^{-7}$",  (2.0e-10, 9.0), "left"),
    "1e-12": (r"$\tau = 10^{-12}$", (6.5e-13, 6.3), "center"),
}
_apick = _resolve("autopick.tsv")
if os.path.exists(_apick):
    PICK_BY_TAU = [(r["rung_label"].strip(),) + TAU_TEXT[r["tau"].strip()]
                   for r in load_tsv("autopick.tsv") if r["tau"].strip() in TAU_TEXT]
else:
    PICK_BY_TAU = [("nm=8",) + TAU_TEXT["1e-6"],
                   ("nm=9",) + TAU_TEXT["1e-7"],
                   ("nm=13",) + TAU_TEXT["1e-12"]]
def by_label(lbl): return next((p for p in pts if p[3] == lbl), None)
PICKS = [(by_label(l), t, xy, ha) for l, t, xy, ha in PICK_BY_TAU]
for p, lab, txt, ha in PICKS:
    if p is None: continue
    # The ring marks "selected"; it takes the picked point's own family colour so
    # a pick that is not an Ozaki-II rung is not miscoloured as one.
    ring = STYLE[p[2]]["c"]
    axR.scatter([p[0]], [p[1]], s=170, facecolors="none", edgecolors=ring,
                linewidths=1.6, zorder=4)
    # shrinkB pulls the arrowhead back from the point's centre to the ring's
    # outer edge (s=170 -> radius 6.5 pt, plus half the 1.6 pt edge).
    axR.annotate(lab, xy=(p[0], p[1]), xytext=txt, fontsize=6.5, color=ring,
                 ha=ha, arrowprops=dict(arrowstyle="->", color=ring, lw=1.0,
                                        shrinkA=1, shrinkB=7.5))
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlim(XLO, XHI); axR.set_ylim(YLO, YHI)
axR.yaxis.set_major_locator(mpl.ticker.FixedLocator([1, 2, 5, 10, 20, 50, 100]))
axR.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: "%g" % v))
axR.yaxis.set_minor_locator(mpl.ticker.NullLocator())
axR.set_xlabel(r"Density-Matrix RMSD")
axR.set_ylabel(r"Speedup")
# Threshold labels share one baseline band under the tau labels; the legend
# sits in the empty top-left corner (two columns, so it clears the 17x-27x
# Ozaki-II points below it and the direct-TC points to its right).
# Labels sit just right of their lines (the near-FP64 band is too narrow to
# hold "Near FP64" without the frontier's vertical step striking through it).
axR.text(CHEM*1.5, 3.8, r"Scientifically", color="#9467bd", fontsize=6.5, ha="left", va="center")
axR.text(CHEM*1.5, 2.75, r"Meaningful", color="#9467bd", fontsize=6.5, ha="left", va="center")
axR.text(INDIST*1.5, 3.8, r"Near FP64", color="#1f77b4", fontsize=6.5, ha="left", va="center")
axR.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
axR.legend(fontsize=5.6, loc="upper left", ncol=2, framealpha=0.92, handletextpad=0.3,
           borderpad=0.25, labelspacing=0.2, columnspacing=0.6, handlelength=1.4,
           markerscale=0.85)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
savepdf(fig, OUT)
print("wrote %s  (data-dir %s)" % (OUT, os.path.abspath(_args.data_dir) if _args.data_dir
                                   else "%s, falling back to %s" % (HERE, GOLDEN)))
