#!/usr/bin/env python3
"""GENGA paper figures (camera-quality, vector PDF); numbers unchanged from the
measured TSVs.
  genga_force_drift.pdf      Fig. 3: force kernel (expansion) + drift kernel
                             (algebraic rewrites), each with its ablation.
  genga_drift_timeseries.pdf Fig. 4: silent-failure energy-drift trajectory.
Reads force_full_g0.tsv, force_mfoff.tsv, kepler_full_g0.tsv,
kepler_herbieoff_g0.tsv, kepler_timeseries.tsv. Titles live in the LaTeX
captions; B&W-safe (series distinct by marker + linestyle, not color alone).

Data resolution. With no --data-dir, each TSV is taken from this directory
when a fresh run left it here and from golden/rtx5090/ otherwise (the RTX 5090
run EXPECTED_RESULTS.md cites). --out-dir defaults to ../../figures.

  python genga_figures.py --data-dir golden/rtx5090 --out-dir /tmp/figs
"""
import argparse, os, sys, csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))  # the one figstyle.py
from figstyle import *  # noqa: F403,E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--data-dir", default=None,
                 help="read every TSV from this dir (default: fresh files here, else golden/rtx5090)")
_ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "..", "figures"))
_args = _ap.parse_args()
OUT_DIR = os.path.abspath(_args.out_dir)
ACC_FLOOR = 3e-16

def _resolve(tsv):
    if _args.data_dir:
        return os.path.join(os.path.abspath(_args.data_dir), tsv)
    p = os.path.join(HERE, tsv)
    return p if os.path.exists(p) else os.path.join(HERE, "golden", "rtx5090", tsv)

def load(tsv, errcol):
    pts = []
    p = _resolve(tsv)
    if not os.path.exists(p): return pts
    for r in csv.DictReader(open(p), delimiter="\t"):
        try:
            e = float(r[errcol]); s = float(r["speedup"])
            if e == e and s == s: pts.append((max(e, ACC_FLOOR), s))
        except (ValueError, KeyError): pass
    return pts

def pareto(pts):
    """non-dominated on (error down, speedup up); sorted by error for a line."""
    nd = [p for p in pts if not any(q is not p and q[0] <= p[0] and q[1] >= p[1]
                                    and (q[0] < p[0] or q[1] > p[1]) for q in pts)]
    return sorted(set(nd), key=lambda p: p[0])

def front_plot(ax, pts, sty, label, lw=1.3, z=3, alpha=1.0):
    if not pts: return
    x, y = zip(*pts)
    ax.plot(x, y, color=sty["color"], marker=sty["marker"], ls=sty["ls"], lw=lw,
            ms=4, mec="k", mew=0.3, zorder=z, alpha=alpha, label=label,
            drawstyle="steps-post")

def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    savepdf(fig, os.path.join(OUT_DIR, name))

# ───────── merged force+drift ablation panel (one figure*, 1x2) ─────────
def fig_force_drift():
    full_f = pareto(load("force_full_g0.tsv", "L2"))
    mfoff  = pareto(load("force_mfoff.tsv", "L2"))
    full_d = pareto(load("kepler_full_g0.tsv", "Edrift"))
    hoff   = pareto(load("kepler_herbieoff_g0.tsv", "Edrift"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDTH_FULL, 2.1))

    # left: force kernel (precision axis); log error axis, log speedup
    front_plot(axL, full_f, series_style(0), r"Expansion Enabled", z=4)
    front_plot(axL, mfoff,  series_style(1), r"Expansion Disabled")
    axL.scatter([1.71e-13], [4.73], marker="X", s=55, color=COLORS[4],
                edgecolors="k", linewidths=0.4, zorder=5, label=r"Reference")
    axL.set_xscale("log"); axL.set_yscale("log"); axL.set_xlim(5e-16, 2e-5)
    axL.set_xlabel(r"Relative $\ell_2$ Error")
    axL.set_ylabel(r"Speedup")
    axL.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    axL.legend(fontsize=6.2, loc="upper left", framealpha=0.92,
               handletextpad=0.4, borderpad=0.3, labelspacing=0.25)

    # right: Kepler drift (algebra axis)
    front_plot(axR, full_d, series_style(0), r"Algebraic Rewrites Enabled", z=4, alpha=0.7)
    front_plot(axR, hoff,   series_style(1), r"Algebraic Rewrites Disabled")
    axR.set_xscale("log")
    axR.set_xlabel(r"Orbital-Energy Drift")
    axR.set_ylabel(r"Speedup")
    axR.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    axR.legend(fontsize=6.2, loc="upper left", framealpha=0.92,
               handletextpad=0.4, borderpad=0.3, labelspacing=0.25)

    fig.tight_layout(pad=0.4)
    savefig(fig, "genga_force_drift.pdf")
    print("wrote genga_force_drift.pdf (force full %d/mfoff %d, drift full %d/hoff %d)"
          % (len(full_f), len(mfoff), len(full_d), len(hoff)))

# ───────── energy-drift time series (silent-failure money plot) ─────────
def fig_drift_timeseries():
    rows = list(csv.DictReader(open(_resolve("kepler_timeseries.tsv")), delimiter="\t"))

    # Tolerate missing cells (a drift run that printed nothing leaves an empty
    # field); each series keeps its own step axis.
    def col(key):
        pts = [(float(r["steps"]), float(r[key])) for r in rows
               if (r.get(key) or "").strip()]
        return [s for s, _ in pts], [v for _, v in pts]

    steps_f, fp64 = col("fp64")
    steps_n, naive = col("naive_fp32")
    steps_p, pos = col("poseidon")
    fig, ax = plt.subplots(figsize=(0.84 * WIDTH_COL, 1.75))
    # naive FP32 = secular (slope ~1); FP64 and Poseidon = random walk (slope ~1/2)
    if naive:
        ax.plot(steps_n, naive, color=COLORS[3], marker="s", ls="--", lw=1.3, ms=4,
                mec="k", mew=0.3, zorder=4, label=r"Naive FP32")
    if pos:
        ax.plot(steps_p, pos, color=COLORS[0], marker="o", ls="-", lw=1.3, ms=4,
                mec="k", mew=0.3, zorder=5, label=r"Poseidon")
    if fp64:
        ax.plot(steps_f, fp64, color=COLORS[6], marker="^", ls=":", lw=1.3, ms=4,
                mec="k", mew=0.3, zorder=3, label=r"FP64")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Integration Steps")
    ax.set_ylabel(r"Relative Energy Drift")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.92,
              handletextpad=0.4, borderpad=0.3, labelspacing=0.25)
    savefig(fig, "genga_drift_timeseries.pdf")
    print("wrote genga_drift_timeseries.pdf (%d/%d/%d pts naive/pos/fp64)"
          % (len(naive), len(pos), len(fp64)))

if __name__ == "__main__":
    fig_force_drift(); fig_drift_timeseries()
    print("figures in", OUT_DIR)
