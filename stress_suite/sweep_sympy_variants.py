"""
Does the SymPy wall at N=5 measure the library, or the way it was driven?

WHY THIS EXISTS
---------------
The frozen scaling sweep reports that `sympy.physics.mechanics`, driven as its
documentation presents it, answers the planar chain up to N=5 and walls at
N=8 under a 120 s clock. That result is honest but incomplete as an account of
the library, because the idiomatic path routes through `rhs()`, which performs
a SYMBOLIC linear solve (`mass_matrix_full.LUsolve(forcing_full)`) whose cost
grows explosively in the number of coordinates.

Two standard remedies exist, and neither was applied in the frozen run:

  * Kane's method, which assembles the equations from partial velocities and
    avoids differentiating a single large scalar Lagrangian;
  * common subexpression elimination, applied when the symbolic mass matrix
    and forcing vector are turned into numeric callables, so that repeated
    trigonometric subterms are computed once.

This sweep measures four ways of driving the SAME library on the SAME chain,
against the SAME closed-form reference, under the SAME wall clock.

  idiomatic     LagrangesMethod -> rhs()          symbolic solve   (frozen path)
  lagrange_cse  LagrangesMethod -> M, F lambdified with cse, solved numerically
  kane          KanesMethod     -> rhs()          symbolic solve
  kane_cse      KanesMethod     -> M, F lambdified with cse, solved numerically

GOVERNANCE
----------
This is harness growth, which the freeze permits: no case is removed and no
frozen number is altered. The frozen matrix and the figures already reported
stand exactly as they are. What this can change is the INTERPRETATION of the
reach comparison -- specifically whether "SymPy reached 5" is a statement about
the library or about the documented path through it.

CORRECTNESS IS THE PRECONDITION
-------------------------------
A faster wrong answer is worth nothing, and three of the four faults recorded
in this study's own apparatus were convention mismatches that produced clean
constant error ratios. Every variant here is therefore validated against the
independent closed-form reference at 16 probe states before its timing is
allowed to count. A variant that is fast and disagrees is reported as
DISAGREE, not as a success.

Run:
    python sweep_sympy_variants.py                 # full ladder
    python sweep_sympy_variants.py --wall 60       # shorter clock
    python sweep_sympy_variants.py --child kane 5  # one measurement
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VARIANTS = ("idiomatic", "lagrange_cse", "kane", "kane_cse")
LADDER = (1, 2, 3, 5, 6, 7, 8, 10, 12, 16, 20)
PROBES = 16
SEED = 20260905
TOL = 1e-8


# ===========================================================================
# System construction
# ===========================================================================

def _lagrangian_chain(N, m=1.0, l=1.0, g=9.81):
    """The chain exactly as adapter_sympy.build_chain states it."""
    import sympy as sp
    from sympy.physics.mechanics import dynamicsymbols
    t = dynamicsymbols._t
    q = [dynamicsymbols(f"theta{i}") for i in range(N)]
    w = [qi.diff(t) for qi in q]
    T = sp.S.Zero
    for j in range(N):
        for k in range(N):
            coeff = N - max(j, k)
            if coeff:
                T += (sp.Rational(1, 2) * coeff * m * l ** 2
                      * sp.cos(q[j] - q[k]) * w[j] * w[k])
    V = sum((N - j) * m * g * l * (1 - sp.cos(q[j])) for j in range(N))
    return q, sp.simplify(T - V)


def _kane_chain(N, m=1.0, l=1.0, g=9.81):
    """The same chain built from partial velocities.

    Convention note, because this is exactly where this study has been bitten
    before: the reference places mass i at sum_{j<=i} l (sin q_j, -cos q_j),
    i.e. angles are measured from the DOWNWARD vertical. A frame obtained by
    rotating N by q about z has -F.y = (sin q, -cos q), so the link vector is
    -l * F.y and gravity acts along -N.y. Getting this wrong yields a clean
    constant error ratio against the reference rather than a plausible one.
    """
    import sympy as sp
    from sympy.physics.mechanics import (KanesMethod, Particle, Point,
                                         ReferenceFrame, dynamicsymbols)
    q = dynamicsymbols(f"q:{N}")
    u = dynamicsymbols(f"u:{N}")
    Nf = ReferenceFrame("Nf")
    O = Point("O")
    O.set_vel(Nf, 0)

    bodies, loads, prev = [], [], O
    for i in range(N):
        Fi = Nf.orientnew(f"F{i}", "Axis", [q[i], Nf.z])
        Fi.set_ang_vel(Nf, u[i] * Nf.z)
        Pi = prev.locatenew(f"P{i}", -l * Fi.y)
        Pi.v2pt_theory(prev, Nf, Fi)
        bodies.append(Particle(f"Pa{i}", Pi, m))
        loads.append((Pi, -m * g * Nf.y))
        prev = Pi

    kd = [u[i] - q[i].diff() for i in range(N)]
    KM = KanesMethod(Nf, q_ind=list(q), u_ind=list(u), kd_eqs=kd)
    KM.kanes_equations(bodies, loads)
    return KM, list(q), list(u)


# ===========================================================================
# The four ways of driving it
# ===========================================================================

def build_idiomatic(N):
    """LagrangesMethod -> rhs(). The frozen path: a symbolic LUsolve."""
    import numpy as np
    import sympy as sp
    from sympy.physics.mechanics import LagrangesMethod
    q, L = _lagrangian_chain(N)
    lm = LagrangesMethod(L, q)
    lm.form_lagranges_equations()
    rhs = lm.rhs()                                    # symbolic solve happens here
    syms = list(q) + [qi.diff() for qi in q]
    f = sp.lambdify([syms], rhs, "numpy")

    def accel(state):
        s = np.asarray(state, float)
        args = list(s[0::2]) + list(s[1::2])
        return np.asarray(f(args), float).reshape(-1)[N:]
    return accel


def build_lagrange_cse(N):
    """LagrangesMethod, but solve M a = F numerically, lambdified with CSE."""
    import numpy as np
    import sympy as sp
    from sympy.physics.mechanics import LagrangesMethod
    q, L = _lagrangian_chain(N)
    lm = LagrangesMethod(L, q)
    lm.form_lagranges_equations()
    M, F = lm.mass_matrix, lm.forcing            # no symbolic solve
    syms = list(q) + [qi.diff() for qi in q]
    fM = sp.lambdify([syms], M, "numpy", cse=True)
    fF = sp.lambdify([syms], F, "numpy", cse=True)

    def accel(state):
        s = np.asarray(state, float)
        args = list(s[0::2]) + list(s[1::2])
        return np.linalg.solve(np.asarray(fM(args), float).reshape(N, N),
                               np.asarray(fF(args), float).reshape(N))
    return accel


def build_kane(N):
    """KanesMethod -> rhs(). Kane's formulation, still a symbolic solve."""
    import numpy as np
    import sympy as sp
    KM, q, u = _kane_chain(N)
    rhs = KM.rhs()                                    # symbolic solve happens here
    f = sp.lambdify([q + u], rhs, "numpy")

    def accel(state):
        s = np.asarray(state, float)
        args = list(s[0::2]) + list(s[1::2])
        return np.asarray(f(args), float).reshape(-1)[N:]
    return accel


def build_kane_cse(N):
    """KanesMethod with a numeric solve and CSE. Both remedies together."""
    import numpy as np
    import sympy as sp
    KM, q, u = _kane_chain(N)
    M, F = KM.mass_matrix, KM.forcing            # no symbolic solve
    fM = sp.lambdify([q + u], M, "numpy", cse=True)
    fF = sp.lambdify([q + u], F, "numpy", cse=True)

    def accel(state):
        s = np.asarray(state, float)
        args = list(s[0::2]) + list(s[1::2])
        return np.linalg.solve(np.asarray(fM(args), float).reshape(N, N),
                               np.asarray(fF(args), float).reshape(N))
    return accel


BUILDERS = {"idiomatic": build_idiomatic, "lagrange_cse": build_lagrange_cse,
            "kane": build_kane, "kane_cse": build_kane_cse}


# ===========================================================================
# One measurement
# ===========================================================================

def measure(variant, N):
    import numpy as np
    import reference

    out = {"variant": variant, "N": N}
    t0 = time.time()
    accel = BUILDERS[variant](N)
    build_s = time.time() - t0

    ref = reference.NLinkChain(N)
    rng = np.random.default_rng(SEED)
    worst, t1 = 0.0, time.time()
    for _ in range(PROBES):
        st = np.zeros(2 * N)
        st[0::2] = rng.uniform(-1.0, 1.0, N)
        st[1::2] = rng.uniform(-0.8, 0.8, N)
        a = np.asarray(accel(st), float).reshape(N)
        r = np.asarray(ref.accel(st), float).reshape(N)
        worst = max(worst, float(np.max(np.abs(a - r)
                                        / np.maximum(np.abs(r), 1.0))))
    out.update(status="ok" if worst <= TOL else "DISAGREE",
               build_s=build_s, eval_s=(time.time() - t1) / PROBES,
               err=worst, total_s=time.time() - t0)
    return out


# ===========================================================================
# Driver
# ===========================================================================

def run_one(variant, N, wall):
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", variant,
             str(N)],
            capture_output=True, text=True, timeout=wall, cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"variant": variant, "N": N, "status": "TIMEOUT",
                "total_s": time.time() - t0}
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {"variant": variant, "N": N, "status": "error",
            "total_s": time.time() - t0,
            "error": ((p.stderr or "no output").strip()[-160:])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("VARIANT", "N"))
    ap.add_argument("--wall", type=int, default=120)
    ap.add_argument("--json", default="out/sympy_variants.json")
    ap.add_argument("--ladder", default="",
                    help="comma-separated N values, overriding the default")
    ap.add_argument("--only", default="",
                    help="comma-separated variants, overriding the default")
    args = ap.parse_args()

    ladder = (tuple(int(x) for x in args.ladder.split(","))
              if args.ladder else LADDER)
    variants = (tuple(x for x in args.only.split(","))
                if args.only else VARIANTS)

    if args.child:
        variant, N = args.child[0], int(args.child[1])
        try:
            print(json.dumps(measure(variant, N)))
        except Exception as e:
            print(json.dumps({"variant": variant, "N": N, "status": "error",
                              "error": (type(e).__name__ + ": "
                                        + str(e))[:160]}))
        return 0

    print("Does the SymPy wall measure the library or the driving?")
    print(f"  wall clock : {args.wall}s per (variant, N)")
    print(f"  validation : {PROBES} probes against reference.NLinkChain, "
          f"tolerance {TOL:g}\n")
    print(f"{'N':>3}  " + "".join(f"{v:>16}" for v in variants))
    print("-" * (5 + 16 * len(variants)))

    rows, walled = [], set()
    for N in ladder:
        cells = []
        for v in variants:
            if v in walled:
                cells.append(f"{'(walled)':>16}")
                continue
            r = run_one(v, N, args.wall)
            rows.append(r)
            st = r["status"]
            if st == "ok":
                cells.append(f"{r['build_s']:>13.2f}s ")
            elif st == "TIMEOUT":
                cells.append(f"{'TIMEOUT':>16}")
                walled.add(v)
            elif st == "DISAGREE":
                cells.append(f"{'DISAGREE':>16}")
                walled.add(v)
            else:
                cells.append(f"{'error':>16}")
                walled.add(v)
        print(f"{N:>3}  " + "".join(cells), flush=True)
        if len(walled) == len(variants):
            break

    path = os.path.join(HERE, args.json)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    print("\nLargest N answered correctly, by variant:")
    for v in variants:
        ok = [r["N"] for r in rows if r["variant"] == v and r["status"] == "ok"]
        print(f"  {v:<14} {max(ok) if ok else 0}")
    bad = [r for r in rows if r["status"] == "DISAGREE"]
    print(f"\n  variants disagreeing with the reference: {len(bad)}")
    print(f"  written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
