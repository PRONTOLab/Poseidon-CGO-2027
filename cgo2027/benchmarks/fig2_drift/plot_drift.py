#!/usr/bin/env python3
"""Regenerate the intro drift figure (paper img/intro_drift.pdf).

Reads drift.csv (produced by ./fig2_drift.exe), falling back to the shipped
drift.csv.reference when no run has happened here, and writes
../../figures/intro_drift.pdf. This is the standalone artifact port of
paper/img/intro_figures.py::fig_drift() and uses the same figstyle so the
output is visually identical to the paper figure.

Run with any Python that has numpy + matplotlib (the artifact runs it as "$PY"):
    python plot_drift.py
"""
import os, sys, csv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))  # the one figstyle.py
from figstyle import *  # applies rcParams; provides plt, series_style, savepdf

def _resolve_csv():
    """Prefer a freshly measured drift.csv; fall back to the shipped reference."""
    p = os.path.join(HERE, "drift.csv")
    return p if os.path.exists(p) else os.path.join(HERE, "drift.csv.reference")


CSV = _resolve_csv()
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "intro_drift.pdf"))


def fig_drift():
    """Rounding drift of a long leapfrog integration (intro figure):
    FP64 vs naive FP32 vs 3xFP32 triple-float emulation, against a 2xFP64
    reference with the SAME dt (identical truncation => pure rounding).
    Data: drift.csv / drift.csv.reference (mean |x - ref| over 4096
    oscillators on the RTX 5090)."""
    n, e32, e64, etf = [], [], [], []
    floor = 1e-21
    with open(CSV) as f:
        for row in csv.DictReader(f):
            n.append(int(row["steps"]))
            e32.append(max(float(row["fp32"]), floor))
            e64.append(max(float(row["fp64"]), floor))
            etf.append(max(float(row["tf32x3"]), floor))

    # true printed width: 0.70\textwidth panel inside a figure* (~4.9 in);
    # height matched to the three stacked code panels beside it
    fig, ax = plt.subplots(figsize=(4.9, 2.05))
    ax.plot(n, e32, label="Naive FP32",
            **{**series_style(3), "marker": "None"})
    ax.plot(n, e64, label="Original FP64",
            **{**series_style(2), "marker": "None"})
    ax.plot(n, etf, label="3xFP32 MultiFloat",
            **{**series_style(0), "marker": "None"}, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Integration Steps")
    ax.set_ylabel("Mean Drift from Truth")
    ax.set_xlim(1, 1.1e6); ax.set_ylim(1e-23, 1e-3)
    ax.legend(loc="center left", frameon=False, handlelength=1.8,
              labelspacing=0.2, borderpad=0.2, fontsize=7.0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    savepdf(fig, OUT)
    print("wrote", OUT)
    print(f"  final drift @ {n[-1]:>8d} steps: "
          f"fp32={e32[-1]:.2e}  fp64={e64[-1]:.2e}  3xfp32={etf[-1]:.2e}")


if __name__ == "__main__":
    fig_drift()
