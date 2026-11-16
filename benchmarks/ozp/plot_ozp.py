#!/usr/bin/env python3
"""Fig. 5 (figures/ozp_convergence.pdf) from the ozp TSVs here, else from recorded/rtx5090/ (--data-dir, --out)."""
import argparse, csv, os, math, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from figstyle import *  # noqa: F403,E402

RECORDED = os.path.join(HERE, "recorded", "rtx5090")
_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir", default=None,
                 help="plot this bundle only (default: fresh files here, else recorded/rtx5090)")
_ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "figures", "ozp_convergence.pdf"))
_args = _ap.parse_args()
OUT = os.path.abspath(_args.out)
CHEM = 1e-7              # scientifically meaningful RMSD
INDIST = 2.0 ** -37      # indistinguishable from FP64

def _resolve(name):
    if _args.data_dir:
        return os.path.join(os.path.abspath(_args.data_dir), name)
    p = os.path.join(HERE, name)
    return p if os.path.exists(p) else os.path.join(RECORDED, name)

def load_tsv(name):
    with open(_resolve(name)) as f: return list(csv.DictReader(f, delimiter="\t"))
def fnum(x):
    try: return float(x)
    except: return float("nan")

pareto = load_tsv("pareto.tsv")
fp64 = next(r for r in pareto if r["tag"] == "FP64")
NAIVE_T = fnum(fp64["time_ms"])
def cublas_time():
    for r in load_tsv("surrogate.tsv"):
        if r["method"] == "cublas" and int(r["N"]) == 2048: return fnum(r["time_ms"])
    raise SystemExit("plot_ozp: no cuBLAS N=2048 row in surrogate.tsv")
CUBLAS_T = cublas_time()
CUBLAS_RMSD = fnum(fp64["RMSD"])
L2_TO_RMSD = fnum(fp64["RMSD"]) / fnum(fp64["L2_vs_Kref"])

def categorize(c):
    c = c.lower()
    if "ozaki-ii" in c or "ozii" in c:    return "ozii"
    if "tcec" in c or "ozaki-i_n" in c:   return "tcec"
    if "ozaki-i-ef" in c:                 return "ef"
    if c.startswith("wmma") or "f16/f32" in c or "bf16" in c or "tf32" in c: return "static"
    return "other"

pts = []  # (rmsd, speedup, category, label, time_ms)
# Only pareto.tsv's baseline row is drawn: the figure shows every product in library form.
pts.append((fnum(fp64["RMSD"]), 1.0, "naive", "naive FP64", NAIVE_T))

pts.append((CUBLAS_RMSD, NAIVE_T / CUBLAS_T, "cublas", "cuBLAS", CUBLAS_T))

libpath = _resolve("direct_tc_lib.tsv")
if os.path.exists(libpath):
    for r in load_tsv("direct_tc_lib.tsv"):
        if not r.get("tag") or not r["tag"].startswith("LIB_"): continue
        rm, t = fnum(r["RMSD"]), fnum(r["time_ms"])
        if math.isnan(rm) or math.isnan(t): continue
        pts.append((rm, NAIVE_T / t, "lib", r["tag"], t))

drpath = _resolve("dispatch.tsv")
if os.path.exists(drpath):
    for r in load_tsv("dispatch.tsv"):
        if not r.get("budget") or r["budget"].startswith("BASELINE"): continue
        rm, t = fnum(r["RMSD"]), fnum(r["time_ms"])
        if math.isnan(rm) or math.isnan(t): continue
        pick = r.get("pick") or ""
        lbl = ("nm=" + pick.split("nm=")[-1]) if "nm=" in pick else pick
        pts.append((rm, NAIVE_T / t, categorize(pick), lbl, t))

STYLE = {
    "naive":  dict(c="#222222", m="*", s=120, label=r"Scalar FP64"),
    "cublas": dict(c="#555555", m="D", s=42,  label=r"cuBLAS DGEMM"),
    "lib":    dict(c="#d62728", m="X", s=42,  label=r"Direct TC"),
    "tcec":   dict(c="#ff7f0e", m="s", s=34,  label=r"FP16TCEC"),
    "ozii":   dict(c="#2ca02c", m="o", s=52,  label=r"Ozaki-II (Poseidon)"),
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.3))

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
if not plot_traj(axL, "TCEC_dispatch", "#ff7f0e", "-.", r"FP16TCEC"):
    plot_traj(axL, "M0_4", "#ff7f0e", "-.", r"FP16TCEC")
# The nm selected at tau = 1e-7 and 1e-12 (autopick.tsv), and nm = 14.
def selected_nm(tau, default):
    if os.path.exists(_resolve("autopick.tsv")):
        for r in load_tsv("autopick.tsv"):
            if r["tau"].strip() == tau and r["label"].strip().startswith("nm="):
                return int(r["label"].strip()[3:])
    return default
def rmsd_at(tag, it):   # RMSD at iteration it, or the last one
    rows = [l.split() for l in open(_resolve(f"traj_{tag}.tsv")) if len(l.split()) >= 2]
    return float(next((a for a in rows if int(a[0]) == it), rows[-1])[1]) * L2_TO_RMSD
for nm, label, marker, me in ((selected_nm("1e-7", 8), r"Ozaki-II (Poseidon)", "o", 6),
                              (selected_nm("1e-12", 12), "_nolegend_", "s", (2, 6)),
                              (14, "_nolegend_", "D", (4, 6))):
    if plot_traj(axL, "OZII_nm%d" % nm, "#2ca02c", "-", label, lw=1.4, marker=marker,
                 me=me, alpha=0.65):
        axL.text(20.5, (1.6 if nm == 8 else 2.5) * rmsd_at("OZII_nm%d" % nm, 20), r"$\mathit{nm} = %d$" % nm,
                 color="#2ca02c", fontsize=6)
axL.axhline(CHEM,   color="#9467bd", ls="-",  lw=0.9)
axL.axhline(INDIST, color="#1f77b4", ls="--", lw=0.8)
axL.text(1.3, 3.0e-6, r"Scientifically", color="#9467bd", fontsize=6.5, ha="left", va="center")
axL.text(1.3, 5.0e-7, r"Meaningful",     color="#9467bd", fontsize=6.5, ha="left", va="center")
axL.text(30, 4.0e-11, r"Near FP64",  color="#1f77b4", fontsize=6.5, ha="right", va="center")
axL.set_yscale("log")
axL.set_xlabel(r"Purification Iteration")
axL.set_ylabel(r"Density-Matrix RMSD")
axL.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
axL.legend(fontsize=6.2, loc="lower left", framealpha=0.92, handletextpad=0.4,
           borderpad=0.3, labelspacing=0.25)

XLO, XHI, YLO, YHI = 5e-17, 1e-2, 0.55, 300.0
axR.axvspan(XLO, INDIST, color="#1f77b4", alpha=0.08, zorder=0)
axR.axvspan(INDIST, CHEM, color="#9467bd", alpha=0.08, zorder=0)
axR.axvline(CHEM,   color="#9467bd", ls="-",  lw=1.0, zorder=1)
axR.axvline(INDIST, color="#1f77b4", ls="--", lw=0.9, zorder=1)

def dominated(p, ps):
    return any((q[0] <= p[0] and q[1] >= p[1] and q is not p and
                (q[0] < p[0] or q[1] > p[1])) for q in ps)
def pfront(ps):
    return sorted([p for p in ps if not dominated(p, ps)], key=lambda p: p[0])
# Emulation OFF: direct downcasts and scalar FP64 only.
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
# One circle per tau, on the dispatch.tsv point of the configuration autopick.tsv names.
TAU_TEXT = {
    "1e-6":  (r"$\tau = 10^{-6}$",  (1.5e-6,  9.0), "left"),
    "1e-7":  (r"$\tau = 10^{-7}$",  (2.0e-10, 9.0), "left"),
    "1e-12": (r"$\tau = 10^{-12}$", (6.5e-13, 6.3), "center"),
}
_apick = _resolve("autopick.tsv")
if os.path.exists(_apick):
    PICK_BY_TAU = [(r["label"].strip(),) + TAU_TEXT[r["tau"].strip()]
                   for r in load_tsv("autopick.tsv") if r["tau"].strip() in TAU_TEXT]
else:
    PICK_BY_TAU = [("tcec-dispatch",) + TAU_TEXT["1e-6"],
                   ("nm=8",) + TAU_TEXT["1e-7"],
                   ("nm=12",) + TAU_TEXT["1e-12"]]
def by_label(lbl): return next((p for p in pts if p[3] == lbl or p[3].startswith(lbl + "_")), None)
PICKS = [(by_label(l), t, xy, ha) for l, t, xy, ha in PICK_BY_TAU]
for p, lab, txt, ha in PICKS:
    if p is None: continue
    ring = STYLE[p[2]]["c"]
    axR.scatter([p[0]], [p[1]], s=170, facecolors="none", edgecolors=ring,
                linewidths=1.6, zorder=4)
    # shrinkB ends the arrow at the ring's outer edge.
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
                                   else "%s, falling back to %s" % (HERE, RECORDED)))
