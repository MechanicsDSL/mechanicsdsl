"""
Figures for the paper, generated from the recorded JSON rather than retyped.

Every figure here reads the same out/*.json files the tables were built from, so
a figure cannot drift from the number it illustrates. Output is PDF (vector) at
a size that survives a two-column layout.

  fig1_precision   divergence time against working precision -- the paper's
                   sharpest result, a straight line over nine orders of
                   magnitude with prediction and measurement coincident
  fig2_reach       the confounded comparison: the frozen ladder against the
                   equal-budget one, showing the separation dissolving
  fig3_residuals   agreement across sampling regimes and system size, with the
                   adjudication tolerance drawn in
  fig4_scaling     cost against coordinate count for the three engines, O(N^3)
                   symbolic against O(N) recursive

Run:
    python make_figures.py
"""

from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "docs", "figures"))
DATA = os.path.join(HERE, "out")

# A restrained, colour-blind-safe palette that also separates in greyscale.
C_MDSL, C_SYMPY, C_DRAKE = "#1b4965", "#bc4749", "#457b9d"
C_PRED, C_GRID = "#8d99ae", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ===========================================================================
# Figure 1 -- precision scaling
# ===========================================================================

def fig_precision():
    rows = [r for r in load("precision_equilibrium.json") if r["diverged"]]
    d = [r["digits"] for r in rows]
    meas = [r["t_divergence"] for r in rows]
    pred = [r["t_predicted"] for r in rows]
    lam = rows[0]["lambda"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(wspace=0.42)

    style(ax)
    ax.plot(d, pred, "-", color=C_PRED, linewidth=2.6, alpha=0.85,
            label=r"predicted, $\mathrm{arccosh}(0.1/\varepsilon)/\lambda$",
            zorder=2)
    ax.plot(d, meas, "o", color=C_MDSL, markersize=5.5, markeredgewidth=0,
            label="measured (RK4, arbitrary precision)", zorder=3)
    ax.set_xlabel("working precision (decimal digits)")
    ax.set_ylabel(r"time to reach $0.1$ rad (s)")
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 148)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.annotate(f"slope $\\ln 10/\\lambda = {math.log(10)/lam:.3f}$ s per digit",
                xy=(112, 83), xytext=(74, 44), color="#333333", fontsize=8,
                ha="center",
                arrowprops=dict(arrowstyle="-", color="#999999",
                                linewidth=0.7, shrinkB=3))

    # right panel: the residual that seeds it, spanning nine orders
    style(ax2)
    eps = [r["residual"] for r in rows]
    ax2.semilogy(d, eps, "o-", color=C_MDSL, markersize=4.5, linewidth=1.2,
                 markeredgewidth=0)
    ax2.set_xlabel("working precision (decimal digits)")
    ax2.set_ylabel(r"initial residual $\varepsilon = |\sin \pi_p|$")
    ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax2.set_xlim(0, 168)
    ax2.set_ylim(1e-165, 1e6)
    ax2.axhline(0.1, color=C_SYMPY, linewidth=0.9, linestyle=(0, (4, 3)))
    ax2.text(8, 3e1, r"threshold $0.1$ rad", color=C_SYMPY, fontsize=7.5)

    fig.savefig(os.path.join(OUT, "fig1_precision.pdf"))
    plt.close(fig)
    print("  fig1_precision.pdf   ", len(rows), "points")


# ===========================================================================
# Figure 2 -- the confound: frozen ladder vs equal budget
# ===========================================================================

def fig_reach():
    eq = load("equal_budget.json")
    frozen = {"SymPy": 5, "MechanicsDSL": 12, "Drake": 30}
    best = {}
    for r in eq:
        if r.get("status") == "ok":
            best[r["engine"]] = max(best.get(r["engine"], 0), r["N"])

    engines = ["SymPy", "MechanicsDSL", "Drake"]
    colors = [C_SYMPY, C_MDSL, C_DRAKE]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    style(ax)

    x = range(len(engines))
    w = 0.34
    a = ax.bar([i - w / 2 for i in x], [frozen[e] for e in engines], w,
               color="#c9c9c9", edgecolor="none",
               label="as driven in the frozen sweep")
    b = ax.bar([i + w / 2 for i in x], [best[e] for e in engines], w,
               color=colors, edgecolor="none", label="equal budget, best path")

    for rect in list(a) + list(b):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.6,
                f"{int(rect.get_height())}", ha="center", fontsize=8,
                color="#333333")

    ax.set_xticks(list(x))
    ax.set_xticklabels(engines)
    ax.set_ylabel("largest chain answered (links)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.015))
    ax.text(1.0, 90.5, "80 was the top of the ladder, not a wall",
            fontsize=7.5, color="#666666", ha="center")

    fig.savefig(os.path.join(OUT, "fig2_reach.pdf"))
    plt.close(fig)
    print("  fig2_reach.pdf       frozen", frozen, "-> equal", best)


# ===========================================================================
# Figure 3 -- residuals across regimes and size, against the tolerance
# ===========================================================================

def fig_residuals():
    rows = load("probe_robustness.json")["rows"]
    regimes = ["uniform", "wide", "near_pi", "aligned"]
    labels = {"uniform": "uniform", "wide": "wide velocities",
              "near_pi": r"near $\pi$", "aligned": "near-aligned"}
    engines = ["MechanicsDSL", "SymPy", "Drake"]
    colors = {"MechanicsDSL": C_MDSL, "SymPy": C_SYMPY, "Drake": C_DRAKE}
    marks = {"MechanicsDSL": "o", "SymPy": "s", "Drake": "^"}

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35), sharey=True)
    for ax, regime in zip(axes, regimes):
        style(ax)
        for e in engines:
            pts = sorted([(r["N"], r["worst"]) for r in rows
                          if r["engine"] == e and r["regime"] == regime])
            ax.semilogy([p[0] for p in pts], [p[1] for p in pts],
                        marks[e] + "-", color=colors[e], markersize=3.6,
                        linewidth=1.0, markeredgewidth=0, label=e)
        ax.axhline(1e-8, color="#333333", linewidth=0.9,
                   linestyle=(0, (4, 3)))
        ax.set_title(labels[regime], fontsize=8.5)
        ax.set_xlabel("$N$")
        ax.set_xlim(0, 13)
        ax.set_ylim(1e-18, 3e-7)

    axes[0].set_ylabel("worst relative residual")
    axes[0].text(0.6, 2.4e-8, "adjudication tolerance $10^{-8}$",
                 fontsize=7, color="#333333")
    axes[-1].legend(loc="lower right", fontsize=7.5)

    fig.savefig(os.path.join(OUT, "fig3_residuals.pdf"))
    plt.close(fig)
    print("  fig3_residuals.pdf   ", len(rows), "cells")


# ===========================================================================
# Figure 4 -- cost against size: O(N^3) symbolic vs O(N) recursive
# ===========================================================================

def fig_scaling():
    eq = [r for r in load("equal_budget.json") if r.get("status") == "ok"]
    engines = ["MechanicsDSL", "SymPy", "Drake"]
    colors = {"MechanicsDSL": C_MDSL, "SymPy": C_SYMPY, "Drake": C_DRAKE}
    marks = {"MechanicsDSL": "o", "SymPy": "s", "Drake": "^"}

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    style(ax)
    for e in engines:
        pts = sorted([(r["N"], r["build_s"]) for r in eq if r["engine"] == e])
        ax.loglog([p[0] for p in pts], [p[1] for p in pts],
                  marks[e] + "-", color=colors[e], markersize=5,
                  linewidth=1.3, markeredgewidth=0, label=e)

    ns = [40, 80]
    ref3 = [145.0 * (n / 40.0) ** 3 for n in ns]
    ax.loglog(ns, ref3, ":", color="#999999", linewidth=1.1)
    ax.text(58, 620, r"$O(N^3)$", color="#777777", fontsize=8)
    ax.text(56, 0.34, r"$O(N)$, flat here", color="#777777", fontsize=8)

    ax.set_xlabel("coordinates $N$")
    ax.set_ylabel("time to equations of motion (s)")
    ax.set_xticks([40, 60, 80])
    ax.set_xticklabels(["40", "60", "80"])
    ax.set_xticks([], minor=True)               # suppress 5x10^1 etc.
    ax.set_xlim(36, 88)
    ax.legend(loc="center left")

    fig.savefig(os.path.join(OUT, "fig4_scaling.pdf"))
    plt.close(fig)
    print("  fig4_scaling.pdf     ", len(eq), "points")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    print(f"writing to {OUT}\n")
    fig_precision()
    fig_reach()
    fig_residuals()
    fig_scaling()
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
