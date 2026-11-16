"""Shared matplotlib style of the paper figures; importing it applies the rcParams."""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# Printed widths in inches.
WIDTH_COL  = 3.3   # single-column figure
WIDTH_SUB  = 3.3   # half-width subfigure (0.49\textwidth inside a figure*)
WIDTH_FULL = 7.0   # full-width figure* (a 2-panel figure* is ~7.0 wide)

_BASE = {
    "font.family": "serif",
    "pgf.rcfonts": False,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
}
_USETEX = {
    "text.usetex": True,
    "font.serif": ["Linux Libertine O", "Libertinus Serif", "Times New Roman"],
    "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
}
_FALLBACK = {  # still serif + vector PDF, just not the exact Libertine face
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
}

def _usetex_ok():
    # usetex failures surface only at draw time, hence a real save.
    if os.environ.get("FIGSTYLE_NO_USETEX"):
        return False
    try:
        import io
        mpl.rcParams.update({**_BASE, **_USETEX})
        fig = plt.figure(figsize=(0.6, 0.6))
        fig.text(0.5, 0.5, r"\texttt{-{}-x} $\times\,2^{-37}$")  # the tricky cases
        fig.savefig(io.BytesIO(), format="pdf")
        plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False

USETEX = _usetex_ok()
mpl.rcParams.update({**_BASE, **(_USETEX if USETEX else _FALLBACK)})

# Series differ by marker and linestyle, not color alone, for B&W print.
MARKERS    = ["o", "s", "D", "^", "v", "X", "P", "*"]
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
COLORS     = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd", "#8c564b", "#555555"]

def series_style(i):
    return dict(color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                ls=LINESTYLES[i % len(LINESTYLES)])

def tt(s):
    """Flag text safe under usetex (no '--' en-dash ligature, escaped underscores)."""
    if not USETEX:
        return s
    s = s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("--", "-{}-")
    return r"\texttt{" + s + "}"

def savepdf(fig, path):
    fig.savefig(path)
    plt.close(fig)

__all__ = ["mpl", "plt", "WIDTH_COL", "WIDTH_SUB", "WIDTH_FULL",
           "MARKERS", "LINESTYLES", "COLORS", "series_style", "tt", "savepdf", "USETEX"]
