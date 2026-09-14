"""
Graphical abstract for the JSS submission.

Elsevier's specification: at least 531 x 1328 pixels (h x w), and legible when
printed at 5 x 13 cm. The figure is therefore laid out at its PRINTED size --
5.2 x 13.3 cm -- so that a 7pt label really is 7pt on the page, and rendered at
300 dpi, which clears the pixel minimum with room to spare.

Two panels, because the paper has two things to say and a graphical abstract
read at postcard size cannot carry more:

  left   the failure itself. The exact solution of the stated initial-value
         problem is the flat line at zero. Every implementation, and the
         independent reference, returns the rising curve instead, and reports
         success while doing it.
  right  why adding implementations cannot help. Divergence time grows linearly
         with working precision and the failure never disappears, so it is a
         property of floating-point evaluation rather than of any one engine.

Data comes from out/precision_equilibrium.json for the right panel; the left
panel integrates the same linearised system the paper uses in Section 7.3.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "docs"))
DATA = os.path.join(HERE, "out")

G, L = 9.81, 1.0
EPS64 = 1.2246467991473532e-16      # sin(pi) in binary64

C_CURVE = "#1b4965"
C_EXACT = "#bc4749"
C_GREY = "#6b6b6b"
C_GRID = "#dedede"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#555555",
    "axes.labelsize": 7,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 6.2,
    "legend.frameon": False,
})


def growth(eps, t_end=14.0, n=1400):
    """phi(t) = eps cosh(lambda t), the linearised departure from equilibrium."""
    lam = math.sqrt(G / L)
    t = np.linspace(0.0, t_end, n)
    return t, eps * np.cosh(lam * t)


def main() -> int:
    rows = [r for r in json.load(open(os.path.join(
        DATA, "precision_equilibrium.json"), encoding="utf-8")) if r["diverged"]]

    # 13.3 x 5.2 cm, expressed in inches so type is sized for the printed page.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.3 / 2.54, 5.2 / 2.54),
        gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.38})

    # ---------------------------------------------------------------- left
    t, phi = growth(EPS64)
    axL.plot(t, phi, color=C_CURVE, linewidth=1.5,
             label="every engine, and the reference")
    axL.axhline(0.0, color=C_EXACT, linewidth=1.5, linestyle=(0, (3.5, 2.5)),
                label="exact solution: no motion")
    axL.set_xlim(0, 14)
    axL.set_ylim(-0.35, 3.4)
    axL.set_xlabel("time (s)")
    axL.set_ylabel("departure from equilibrium (rad)")
    axL.set_title("All implementations fail identically", fontsize=7.4,
                  pad=3.5)
    axL.legend(loc="upper left", handlelength=1.6)
    axL.grid(True, color=C_GRID, linewidth=0.5)
    axL.set_axisbelow(True)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    axL.text(6.4, 1.30, "all report success;\nnone warns", fontsize=6.4,
             color=C_GREY, style="italic", ha="center")

    # --------------------------------------------------------------- right
    d = [r["digits"] for r in rows]
    meas = [r["t_divergence"] for r in rows]
    pred = [r["t_predicted"] for r in rows]
    lam = rows[0]["lambda"]

    axR.plot(d, pred, "-", color=C_GREY, linewidth=1.9, alpha=0.8,
             label="predicted")
    axR.plot(d, meas, "o", color=C_CURVE, markersize=3.4, markeredgewidth=0,
             label="measured")
    axR.set_xlim(0, 162)
    axR.set_ylim(0, 132)
    axR.set_xlabel("working precision (decimal digits)")
    axR.set_ylabel("time to diverge (s)")
    axR.set_title("Precision delays it, never removes it", fontsize=7.4,
                  pad=3.5)
    axR.legend(loc="upper left", handlelength=1.6)
    axR.grid(True, color=C_GRID, linewidth=0.5)
    axR.set_axisbelow(True)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.text(58, 26, f"slope $\\ln 10/\\lambda$\n= {math.log(10)/lam:.2f} s per digit",
             fontsize=6.3, color=C_GREY, style="italic")

    fig.text(0.5, 0.007,
             "Differential testing cannot detect a failure that every "
             "implementation shares.",
             ha="center", fontsize=7.2, color="#222222")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.235)

    for ext, kw in (("pdf", {}), ("tif", {"dpi": 300}), ("png", {"dpi": 300})):
        path = os.path.join(OUT, "Graphical_Abstract." + ext)
        fig.savefig(path, **kw)
        print(f"  {os.path.basename(path)}")
    plt.close(fig)

    w_px = int(round(13.3 / 2.54 * 300))
    h_px = int(round(5.2 / 2.54 * 300))
    print(f"\n  pixels: {h_px} x {w_px} (h x w); minimum is 531 x 1328")
    print(f"  print size: 5.2 x 13.3 cm; target is 5 x 13 cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
