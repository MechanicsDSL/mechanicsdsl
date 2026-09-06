"""
The symmetric question: does MechanicsDSL's wall at N=12 measure the engine,
or the way it was driven?

WHY THIS EXISTS
---------------
`sweep_sympy_variants.py` found that SymPy's frozen wall at N=5 was an artefact
of the documented path: driven with Kane's method and CSE it reaches N=30. That
result is only usable if the same search is made for MechanicsDSL. Reporting an
optimisation found for one contestant and not attempted for the other would be
a methodological asymmetry favouring the author's own engine, which is the one
bias this study is least able to afford.

WHAT THE ENGINE ALREADY DOES
----------------------------
MechanicsDSL is NOT limited by a symbolic mass-matrix inversion. At
`NUMERIC_MASS_MATRIX_MIN_COORDS = 4` (compiler.py:69) it already returns
`{"__mass_matrix__": (M, F)}` and defers the M-inverse solve to numeric
evaluation time. The remedy that carried SymPy from 8 to 30 -- avoiding the
symbolic solve -- is therefore already in place here, and cannot be the lever.

THE REMAINING LEVER
-------------------
`solver/core.py:255` lambdifies the symbolic M and F with

    sp.lambdify(syms, M_sub, modules=["numpy", "math"])

and no `cse=True`. The N-link chain's mass matrix has O(N^2) entries, each a
cosine of a difference of angles, and those subexpressions repeat heavily
across entries. This is exactly the structure CSE exists to exploit, and it is
the second of the two remedies that moved SymPy.

HOW IT IS TESTED, WITHOUT BREAKING THE FREEZE
---------------------------------------------
The engine is pinned at a8dc2b2 and its source is NOT edited. The variant is
applied the same way SymPy's was -- from outside, by driving the library
differently. `sympy.lambdify` is wrapped for the duration of the compile so
that the engine's own call sites request CSE. The engine's derivation, its
mass-matrix decision, its solve and its singularity accounting are untouched.

  frozen   the compile path exactly as the frozen sweep drove it
  cse      the same path, with CSE requested at lambdify time

CORRECTNESS IS THE PRECONDITION
-------------------------------
Every variant is validated against the independent closed-form reference at 16
probe states before its timing counts. A fast variant that disagrees is
reported as DISAGREE, not as a success.

Run:
    python sweep_mdsl_variants.py
    python sweep_mdsl_variants.py --child cse 12
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
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

VARIANTS = ("frozen", "cse", "fastsimp", "nosimplify", "both")

# `fastsimp` lowers the engine's OWN simplification timeout. Note the guard in
# symbolic.py:435 -- `if config.simplification_timeout > 0` -- so 0 does not
# disable simplification, it removes the timeout around it and makes matters
# worse. A small positive value makes each sp.simplify bail almost at once, and
# the engine then proceeds with the unsimplified equation, which is
# mathematically identical and merely bulkier.
FAST_SIMPLIFY_TIMEOUT = 0.05
LADDER = (8, 10, 12, 14, 16, 20, 24, 30)
PROBES = 16
SEED = 20260905
TOL = 1e-8


def _install_cse_lambdify():
    """Wrap sympy.lambdify so the engine's own call sites request CSE.

    Returns a restore callable. Only kwargs are added; no engine source is
    edited and no expression is rewritten.
    """
    import sympy as sp
    original = sp.lambdify

    def wrapper(args, expr, *a, **kw):
        kw.setdefault("cse", True)
        return original(args, expr, *a, **kw)

    sp.lambdify = wrapper
    import mechanics_dsl.solver.core as core
    core_sp_original = core.sp.lambdify
    core.sp.lambdify = wrapper

    def restore():
        sp.lambdify = original
        core.sp.lambdify = core_sp_original
    return restore


def _install_identity_simplify():
    """Make sp.simplify a no-op for the duration of the compile.

    This is not a gratuitous mutilation of the engine. The engine ALREADY
    intends to proceed without simplification when it is too costly: on
    TimeoutError it logs "using unsimplified equation" and carries on
    (symbolic.py:440). What the profile shows is that the timeout cannot
    deliver that intent -- the watchdog raises from a separate thread and
    cannot interrupt SymPy's C-level polynomial factorisation, so the cost is
    paid in full and the exception only surfaces afterwards. This variant
    realises the fallback the engine already specifies.

    Correctness is unaffected in principle: simplification rewrites an
    expression into an equivalent one, so skipping it changes the size of the
    expression, not its value. That is checked, not assumed, by the same
    16-probe comparison against the closed-form reference.
    """
    import sympy as sp
    original = sp.simplify

    def identity(expr, *a, **kw):
        return expr

    sp.simplify = identity
    import mechanics_dsl.symbolic as msym
    msym_original = msym.sp.simplify
    msym.sp.simplify = identity
    import mechanics_dsl.solver.core as core
    core_original = core.sp.simplify
    core.sp.simplify = identity

    def restore():
        sp.simplify = original
        msym.sp.simplify = msym_original
        core.sp.simplify = core_original
    return restore


def measure(variant, N):
    import numpy as np
    import reference
    import systems
    import worker
    from mechanics_dsl import PhysicsCompiler

    out = {"variant": variant, "N": N, "engine": "MechanicsDSL"}

    undo = []
    if variant in ("cse", "both"):
        undo.append(_install_cse_lambdify())
    if variant in ("nosimplify", "both"):
        undo.append(_install_identity_simplify())
    if variant in ("fastsimp", "both"):
        from mechanics_dsl.utils import config
        prior = config.simplification_timeout
        config.simplification_timeout = FAST_SIMPLIFY_TIMEOUT
        out["simplification_timeout"] = FAST_SIMPLIFY_TIMEOUT

        def _restore_cfg(_p=prior):
            config.simplification_timeout = _p
        undo.append(_restore_cfg)

    def restore():
        for f in reversed(undo):
            f()

    try:
        t0 = time.time()
        c = PhysicsCompiler()
        r = c.compile_dsl(systems.n_pendulum_dsl(N), use_hamiltonian=False,
                          use_constraints=False)
        if not r.get("success"):
            raise RuntimeError("compile_success=False")
        fn, route = worker._engine_accel_fn(c, [f"theta{i}" for i in range(N)])
        if fn is None:
            raise RuntimeError("no_accel_route:" + str(route))
        build_s = time.time() - t0
    finally:
        restore()

    ref = reference.NLinkChain(N)
    rng = np.random.default_rng(SEED)
    worst, t1 = 0.0, time.time()
    for _ in range(PROBES):
        st = np.zeros(2 * N)
        st[0::2] = rng.uniform(-1.0, 1.0, N)
        st[1::2] = rng.uniform(-0.8, 0.8, N)
        a = np.asarray(fn(st), float).reshape(N)
        rr = np.asarray(ref.accel(st), float).reshape(N)
        worst = max(worst, float(np.max(np.abs(a - rr)
                                        / np.maximum(np.abs(rr), 1.0))))
    out.update(status="ok" if worst <= TOL else "DISAGREE", route=route,
               build_s=build_s, eval_s=(time.time() - t1) / PROBES,
               err=worst, total_s=time.time() - t0)
    return out


def run_one(variant, N, wall):
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", variant,
             str(N)],
            capture_output=True, text=True, timeout=wall, cwd=HERE, env=env)
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
            "error": ((p.stderr or "no output").strip()[-200:])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("VARIANT", "N"))
    ap.add_argument("--wall", type=int, default=120)
    ap.add_argument("--json", default="out/mdsl_variants.json")
    ap.add_argument("--ladder", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    if args.child:
        variant, N = args.child[0], int(args.child[1])
        try:
            print(json.dumps(measure(variant, N)))
        except Exception as e:
            print(json.dumps({"variant": variant, "N": N, "status": "error",
                              "error": (type(e).__name__ + ": "
                                        + str(e))[:200]}))
        return 0

    ladder = (tuple(int(x) for x in args.ladder.split(","))
              if args.ladder else LADDER)
    variants = (tuple(x for x in args.only.split(","))
                if args.only else VARIANTS)

    print("Does the MechanicsDSL wall measure the engine or the driving?")
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
            else:
                cells.append(f"{st:>16}")
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
        print(f"  {v:<10} {max(ok) if ok else 0}")
    print(f"\n  disagreements: "
          f"{sum(1 for r in rows if r['status'] == 'DISAGREE')}")
    print(f"  written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
