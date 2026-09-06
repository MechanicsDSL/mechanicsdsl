"""
Is MechanicsDSL's accuracy advantage real, or an artefact of one family?

WHY THIS EXISTS
---------------
The equal-budget ladder (TR-2026-06) showed MechanicsDSL returning residuals
roughly half SymPy's and some three orders below Drake's at every rung. That was
measured on ONE mechanism family, the planar chain, and a difference seen on one
family is a difference seen on one family.

This test is designed so that it can fail. If the advantage is a property of the
engine's formulation -- a direct symbolic mass matrix, solved numerically --
then it should persist on mechanisms with different structure. If it is a
property of the chain, it should vanish or reverse on the slider-crank and the
cart-pole, whose degeneracies arise from unrelated mechanisms: a vanishing
transmission ratio in one, a collapsing mass-matrix determinant in the other.

No outcome here is a good outcome in advance. A reversal would mean the
advantage should not be claimed at all, which is worth knowing before it is
claimed.

Run in WSL, where all three engines import in one process:

    PYTHONPATH=../src ~/drake-venv/bin/python accuracy_across_families.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

SEEDS = (20260905, 1, 7919)
PROBES = 48
TOL = 1e-8


# ===========================================================================
# Cart-pole: all three engines participate on equal terms (a tree, no loop)
# ===========================================================================

def cartpole_cases():
    import reference_cartpole as RCP
    for mr in (10.0, 1.0, 1e-2, 1e-4):
        for th0 in (0.3, 1.0, 2.0):
            yield RCP.CartPole(mass_ratio=mr), th0


def cartpole_engines(cp, th0):
    import sweep_families as SF
    out = {}
    for name, factory in SF.CARTPOLE_ENGINES:
        try:
            fn = factory(cp, th0)
            out[name] = fn[0] if isinstance(fn, tuple) else fn
        except Exception as e:
            out[name] = ("error", f"{type(e).__name__}: {str(e)[:50]}")
    return out


def probe_cartpole():
    print("Cart-pole: mixed joints, coupled mass matrix, all three engines\n")
    stats = {}
    for cp, th0 in cartpole_cases():
        engines = cartpole_engines(cp, th0)
        for name, fn in engines.items():
            if isinstance(fn, tuple):
                continue
            worst = 0.0
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                for _ in range(PROBES):
                    st = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1),
                                   th0 + rng.normal(0, 0.3), rng.uniform(-1, 1)])
                    a = np.asarray(fn(st), float).reshape(2)
                    r = np.asarray(cp.accel(st), float).reshape(2)
                    worst = max(worst, float(np.max(
                        np.abs(a - r) / np.maximum(np.abs(r), 1.0))))
            stats.setdefault(name, []).append(worst)
    return stats


# ===========================================================================
# Slider-crank: a closed loop. Drake must run discrete here and is excluded
# from the residual comparison, as the paper already reports.
# ===========================================================================

def slidercrank_engines(sc):
    import sweep_slidercrank as SS
    out = {}
    for name, factory in SS.ENGINES:
        if name == "Drake":
            continue                      # discrete mode; not comparable here
        try:
            fn = factory(sc)
            out[name] = fn[0] if isinstance(fn, tuple) else fn
        except Exception as e:
            out[name] = ("error", f"{type(e).__name__}: {str(e)[:50]}")
    return out


def probe_slidercrank():
    print("\nSlider-crank: closed loop, prismatic joint (Drake runs discrete, "
          "excluded)\n")
    import reference_slidercrank as RSC
    stats = {}
    for ratio in (1.05, 1.5, 3.0):
        for mr in (1.0, 1e2, 1e4):
            sc = RSC.SliderCrank(ratio=ratio, mass_ratio=mr) \
                if hasattr(RSC, "SliderCrank") else None
            if sc is None:
                print("  (no SliderCrank constructor found; skipped)")
                return stats
            for name, fn in slidercrank_engines(sc).items():
                if isinstance(fn, tuple):
                    continue
                worst = 0.0
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    y0 = sc.initial_state()
                    for _ in range(PROBES):
                        st = y0 + rng.uniform(-0.05, 0.05, size=len(y0))
                        a = np.asarray(fn(st), float).reshape(2)
                        r = np.asarray(sc.accel(st), float).reshape(2)
                        worst = max(worst, float(np.max(
                            np.abs(a - r) / np.maximum(np.abs(r), 1.0))))
                stats.setdefault(name, []).append(worst)
    return stats


def report(title, stats):
    if not stats:
        return {}
    print(f"{'engine':<16}{'worst':>13}{'median':>13}{'cases':>8}")
    print("-" * 50)
    summary = {}
    for name, vals in stats.items():
        v = sorted(vals)
        med = v[len(v) // 2]
        summary[name] = {"worst": max(v), "median": med, "cases": len(v)}
        print(f"{name:<16}{max(v):>13.2e}{med:>13.2e}{len(v):>8}")
    best = min(summary, key=lambda k: summary[k]["median"])
    print(f"\n  lowest median residual: {best}")
    return summary


def main() -> int:
    print("Does the accuracy advantage survive a change of mechanism?\n")
    cp = report("cart-pole", probe_cartpole())
    sc = report("slider-crank", probe_slidercrank())

    out = os.path.join(HERE, "out", "accuracy_across_families.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"seeds": list(SEEDS), "probes": PROBES,
                   "cartpole": cp, "slidercrank": sc}, fh, indent=1)
    print(f"\n  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
