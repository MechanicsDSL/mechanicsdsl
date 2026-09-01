"""
Two-engine comparison on the portable axes, adjudicated by the reference.

This is the Week 2 sweep in miniature, run with the two columns that exist.
For every portable system it asks each engine for the equations of motion,
compares both against the library-independent reference, and records which
engines answered, which refused, and whether any answer was wrong.

The reference is the referee. Neither engine adjudicates the other, and no
engine adjudicates itself.

Run:
    python compare_engines.py [--tool lagrangian|hamiltonian]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)

import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import adapter_sympy as A
import reference
import systems
import worker

K_PROBE = 16
TOL = 1e-8
SEED = 20260822
MODE = "idiomatic"          # study decision, 25 August 2026


def probe_states(n, rng):
    return [rng.uniform(-0.5, 0.5, size=2 * n) for _ in range(K_PROBE)]


def worst_vs_reference(fn, ref, states):
    worst = 0.0
    for st in states:
        a = np.asarray(fn(st), dtype=float)
        r = ref.accel(st)
        if not np.all(np.isfinite(a)):
            return float("inf")
        worst = max(worst, float(np.max(np.abs(a - r)
                                        / np.maximum(np.abs(r), 1.0))))
    return worst


def run_mechanicsdsl(case, tool, ref, states):
    """Returns (status, worst, seconds, note)."""
    from mechanics_dsl import PhysicsCompiler
    t0 = time.time()
    try:
        c = PhysicsCompiler()
        res = c.compile_dsl(case["dsl"], use_hamiltonian=(tool == "hamiltonian"),
                            use_constraints=False)
        if not res.get("success"):
            return "refused", None, time.time() - t0, "compile_success=False"
        fn, route = worker._engine_accel_fn(c, case["coords"])
        if fn is None:
            return "no_route", None, time.time() - t0, route
        w = worst_vs_reference(fn, ref, states)
        return ("pass" if w <= TOL else "WRONG"), w, time.time() - t0, route
    except Exception as e:
        return "refused", None, time.time() - t0, f"{type(e).__name__}"


def run_sympy(case, tool, ref, states):
    t0 = time.time()
    try:
        spec = A.build_system(case["axis"], case["knob"])
        eng = A.SymPyEngine(spec, mode=MODE)
        out = eng.compile()
        if not out.compiled:
            return "refused", None, out.compile_seconds, (out.error or "")[:34]
        w = worst_vs_reference(eng.accel, ref, states)
        return ("pass" if w <= TOL else "WRONG"), w, time.time() - t0, out.route
    except Exception as e:
        return "refused", None, time.time() - t0, f"{type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="lagrangian",
                    choices=["lagrangian", "hamiltonian"])
    args = ap.parse_args()

    import mechanics_dsl
    if not mechanics_dsl.__file__.startswith(SRC):
        print("!! engine not loaded from repo src/ -- aborting")
        return 2

    print(f"Two-engine comparison -- {args.tool} pathway, "
          f"SymPy mode '{MODE}'")
    print(f"  referee : reference.py (numpy closed form, no shared library)")
    print(f"  engines : MechanicsDSL {mechanics_dsl.__version__}, "
          f"sympy.physics.mechanics")
    print(f"  probes  : {K_PROBE} states/case, tolerance {TOL:g} relative\n")

    cases = [c for c in systems.all_cases()
             if c["axis"] in ("dof", "near_singular", "mass_ratio")
             and args.tool in c["tools"]]

    hdr = (f"{'system':<18}{'MechanicsDSL':<22}{'SymPy':<22}{'agree?':<8}")
    print(hdr)
    print("-" * len(hdr))

    tally = {"both_pass": 0, "both_refuse": 0, "split": 0, "wrong": 0}
    for case in cases:
        case = dict(case)
        n = case["dsl"].count("\\defvar")
        case["coords"] = ([f"theta{i}" for i in range(int(case["knob"]))]
                          if case["axis"] == "dof" else ["x", "y"])
        ref = reference.reference_for_case(case)
        if ref is None:
            continue
        rng = np.random.default_rng(SEED)
        states = probe_states(len(case["coords"]), rng)

        m_st, m_w, m_s, m_note = run_mechanicsdsl(case, args.tool, ref, states)
        s_st, s_w, s_s, s_note = run_sympy(case, args.tool, ref, states)

        def cell(st, w, s):
            v = "   --  " if w is None else f"{w:.1e}"
            return f"{st:<8}{v:>9}{s:>6.1f}s "

        if m_st == "WRONG" or s_st == "WRONG":
            agree = "WRONG"
            tally["wrong"] += 1
        elif m_st == "pass" and s_st == "pass":
            agree = "yes"
            tally["both_pass"] += 1
        elif m_st == "refused" and s_st == "refused":
            agree = "both no"
            tally["both_refuse"] += 1
        else:
            agree = "SPLIT"
            tally["split"] += 1

        print(f"{case['name']:<18}{cell(m_st, m_w, m_s):<22}"
              f"{cell(s_st, s_w, s_s):<22}{agree:<8}")

    print()
    print(f"  both answered correctly : {tally['both_pass']}")
    print(f"  both refused            : {tally['both_refuse']}")
    print(f"  disagreed (one answered): {tally['split']}")
    print(f"  wrong answer given      : {tally['wrong']}")
    if tally["wrong"] == 0:
        print("\n  No engine returned an answer that disagreed with the reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
