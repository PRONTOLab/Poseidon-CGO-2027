"""Shared figure style for the Poseidon-GPU paper (camera-quality, vector, B&W-safe).

One source of truth so every benchmark's figures look identical. Importing this
module applies the rcParams (acmart's Libertine via usetex when available; a serif
STIX vector fallback otherwise) and exposes width constants + B&W helpers.

This is the artifact's single copy (byte-identical in code to paper/img/figstyle.py).
Use from any benchmark plot script under benchmarks/<bench>/:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
    from figstyle import *            # applies rcParams; provides plt, mpl, WIDTH_*, helpers

    fig, ax = plt.subplots(figsize=(WIDTH_SUB, 2.75))   # true on-page width
    ...
    savepdf(fig, "img/<benchmark>_<name>.pdf")          # vector PDF, embedded fonts

Then reference it from LaTeX extension-less so pdflatex picks the PDF:
    \includegraphics[width=\linewidth]{img/<benchmark>_<name>}

Rules this module bakes in (see the migration prompt): vector PDF + embedded fonts,
8pt labels / 7pt ticks sized to the printed width, NO in-figure titles (move them to
the LaTeX caption), and every series distinct by marker + linestyle (not color alone,
since CGO prints B&W). Set env FIGSTYLE_NO_USETEX=1 to force the fallback.
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ---- on-page widths (inches): set figsize WIDTH to the TRUE printed width ----
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
    """True iff matplotlib can actually render AND save with usetex+libertine here.
    usetex failures surface only at draw time, so we probe with a tiny real save."""
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

# ---- B&W-safe series styles: distinct by marker + linestyle, NOT color alone ----
# Index your series consistently across panels; color is kept for screen only.
MARKERS    = ["o", "s", "D", "^", "v", "X", "P", "*"]
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
COLORS     = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd", "#8c564b", "#555555"]

def series_style(i):
    """dict(color=, marker=, ls=) for series index i — distinguishable in B&W."""
    return dict(color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                ls=LINESTYLES[i % len(LINESTYLES)])

def tt(s):
    """Literal CLI/flag text, usetex-safe: avoids the '--'->en-dash ligature and
    escapes underscores. No-op when usetex is off. e.g. tt('--poseidon-tau')."""
    if not USETEX:
        return s
    s = s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("--", "-{}-")
    return r"\texttt{" + s + "}"

def savepdf(fig, path):
    """Save a vector PDF (embedded fonts, tight bbox) and close the figure."""
    fig.savefig(path)
    plt.close(fig)

__all__ = ["mpl", "plt", "WIDTH_COL", "WIDTH_SUB", "WIDTH_FULL",
           "MARKERS", "LINESTYLES", "COLORS", "series_style", "tt", "savepdf", "USETEX"]
