"""
Three-engine INTEGRATION comparison on the N-link chain.

Everything measured so far compares derivations: each engine is asked for its
equations of motion and checked against the reference. Nothing has been
integrated. Every timeout in the frozen baseline occurs during simulation, so
the half of the pipeline where the known failures live is untested across
engines. This closes that gap.

TWO DESIGN DECISIONS, BOTH LOAD-BEARING
---------------------------------------

1. THE INTEGRATOR IS PINNED.

   Each engine supplies only its right-hand side; the integration is performed
   by one scipy DOP853 call with identical tolerances for all of them, and for
   the reference. Drake's native integrator is not scipy's, and MechanicsDSL's
   is not either. Letting each engine integrate its own way would measure the
   integrator choice and report it as engine disagreement.

   This is the controlled experiment: identical initial conditions, identical
   integrator, identical tolerances, and only the equations differ.

2. TRAJECTORY DIVERGENCE IS NOT A CORRECTNESS TEST HERE.

   The N-link chain is chaotic for N >= 2. Two right-hand sides agreeing to
   1e-14 will still produce visibly different trajectories after enough time,
   because the system amplifies any difference exponentially. Reporting "the
   engines disagree at t = 8 s" would therefore be reporting the Lyapunov
   exponent, not an engine defect.

   So the primary long-time oracle is ENERGY DRIFT, which is conserved
   regardless of chaos and does not require two trajectories to stay close.
   Trajectory divergence is still reported, but paired with an estimate of the
   chaotic doubling time, so it can be read against the timescale on which any
   difference would grow anyway.

Run inside WSL, where all three engines import in one process:

    wsl -d Ubuntu -e bash -lc "cd .../stress_suite && \\
      PYTHONPATH=.../src ~/drake-venv/bin/python compare_trajectories.py"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import solve_ivp

import reference
import systems

RTOL, ATOL = 1e-10, 1e-12
METHOD = "DOP853"
T_END = 10.0
N_POINTS = 1500


# ---------------------------------------------------------------------------
# Right-hand sides, one per engine
# ---------------------------------------------------------------------------

def rhs_reference(N):
    ref = reference.NLinkChain(N)
    return ref.rhs, ref


def rhs_mechanicsdsl(N):
    import worker
    from mechanics_dsl import PhysicsCompiler
    c = PhysicsCompiler()
    res = c.compile_dsl(systems.n_pendulum_dsl(N), use_hamiltonian=False,
                        use_constraints=False)
    if not res.get("success"):
        raise RuntimeError("compile_success=False")
    fn, route = worker._engine_accel_fn(c, [f"theta{i}" for i in range(N)])
    if fn is None:
        raise RuntimeError(f"no accel route: {route}")

    def rhs(_t, y):
        out = np.empty_like(y)
        out[0::2] = y[1::2]
        out[1::2] = fn(y)
        return out
    return rhs, None


def rhs_sympy(N):
    import adapter_sympy as A
    eng = A.SymPyEngine(A.build_chain(N), mode="idiomatic")
    out = eng.compile()
    if not out.compiled:
        raise RuntimeError(out.error or "compile failed")
    return eng.rhs, None


def rhs_drake(N):
    import adapter_drake as D
    d = D.DrakeChain(N)
    return d.rhs, None


ENGINES = [
    ("reference", rhs_reference),
    ("MechanicsDSL", rhs_mechanicsdsl),
    ("SymPy", rhs_sympy),
    ("Drake", rhs_drake),
]


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def integrate(rhs, y0, t_eval):
    t0 = time.time()
    sol = solve_ivp(rhs, (0.0, T_END), y0, t_eval=t_eval,
                    method=METHOD, rtol=RTOL, atol=ATOL)
    return sol, time.time() - t0


def energy_drift(ref_obj, y):
    """Relative energy drift along a trajectory, using the reference's energy.

    Valid for every engine because they share coordinates; the energy function
    is a property of the system, not of whoever integrated it.
    """
    E = np.array([ref_obj.energy(y[:, i]) for i in range(y.shape[1])])
    E0 = E[0]
    return float(np.max(np.abs(E - E0)) / max(abs(E0), 1e-30))


def separation_growth(ref_obj, y0, t_eval):
    """Total growth of an initial perturbation over the horizon.

    Integrates the REFERENCE twice from states differing by 1e-10 and returns
    the ratio of final to initial separation. This distinguishes the two
    regimes the chain can be in, which matters because it decides whether a
    trajectory comparison means anything:

        growth ~ 10^0 - 10^2   regular, quasi-periodic. Perturbations grow
                               polynomially. Trajectories stay comparable, and
                               engine agreement is a weak result because the
                               system is not amplifying anything.

        growth >> 10^3         chaotic. Perturbations grow exponentially, so
                               trajectories separate regardless of engine
                               correctness and comparing them measures the
                               Lyapunov exponent rather than the engine.

    An earlier version of this function reported the time at which separation
    first doubled. That was misleading: separation doubles quickly even in
    regular motion, so it reported a short "doubling time" for the integrable
    N=1 pendulum and implied chaos where there was none.
    """
    d0 = 1e-10
    y0b = np.array(y0, dtype=float)
    y0b[0] += d0
    sa, _ = integrate(ref_obj.rhs, y0, t_eval)
    sb, _ = integrate(ref_obj.rhs, y0b, t_eval)
    if not (sa.success and sb.success):
        return None
    sep = np.linalg.norm(sa.y - sb.y, axis=0)
    return float(sep[-1] / d0)


def regime(growth):
    if growth is None:
        return "unknown"
    return "REGULAR" if growth < 1e3 else "CHAOTIC"


def first_divergence(ya, yb, t_eval, thresh):
    d = np.linalg.norm(ya - yb, axis=0)
    idx = np.argmax(d > thresh)
    if d[-1] <= thresh:
        return None
    return float(t_eval[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=5)
    ap.add_argument("--amp", type=float, default=None,
                    help="override initial angles (rad) for every link; the "
                         "suite default is 0.3 for link 0 and 0.15 elsewhere, "
                         "which is the REGULAR regime. Try 3.0 for chaos.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t_eval = np.linspace(0.0, T_END, N_POINTS)
    print("Three-engine INTEGRATION comparison -- N-link chain")
    print(f"  integrator : {METHOD} PINNED across all engines, "
          f"rtol={RTOL:g} atol={ATOL:g}")
    print(f"  horizon    : {T_END:g}s, {N_POINTS} samples")
    print("  oracle     : energy drift (chaos-immune); trajectory divergence "
          "reported against the doubling time\n")

    all_rows = []
    for N in range(1, args.max_n + 1):
        ref_obj = reference.NLinkChain(N)
        if args.amp is None:
            y0 = ref_obj.default_initial_state()
            amp_label = "suite default (0.3/0.15 rad)"
        else:
            y0 = np.zeros(2 * N)
            y0[0::2] = args.amp
            amp_label = f"all links at {args.amp:g} rad"
        growth = separation_growth(ref_obj, y0, t_eval)

        print(f"=== N = {N} " + "=" * 52)
        print(f"  initial condition    : {amp_label}")
        if growth is not None:
            print(f"  perturbation growth  : {growth:.2e}x over {T_END:g}s "
                  f"-> {regime(growth)}")

        print(f"  {'engine':<14}{'ok':<5}{'energy drift':>14}"
              f"{'diverge>1e-6':>14}{'seconds':>9}")
        print("  " + "-" * 56)

        traj = {}
        for name, factory in ENGINES:
            row = {"N": N, "engine": name}
            try:
                rhs, _ = factory(N)
                sol, secs = integrate(rhs, y0, t_eval)
                if not sol.success:
                    raise RuntimeError(sol.message[:40])
                traj[name] = sol.y
                drift = energy_drift(ref_obj, sol.y)
                dv = (None if name == "reference"
                      else first_divergence(traj["reference"], sol.y,
                                            t_eval, 1e-6))
                row.update(ok=True, drift=drift, diverge=dv, seconds=secs)
                dvs = "  --  " if dv is None else f"{dv:.2f}s"
                print(f"  {name:<14}{'yes':<5}{drift:14.3e}{dvs:>14}{secs:9.2f}")
            except Exception as e:
                row.update(ok=False, error=f"{type(e).__name__}: {e}"[:70])
                print(f"  {name:<14}{'NO':<5}   {row['error']}")
            all_rows.append(row)
        print()

    # -- summary -------------------------------------------------------------
    print("=" * 62)
    drifts = {}
    for r in all_rows:
        if r.get("ok") and r["engine"] != "reference":
            drifts.setdefault(r["engine"], []).append(r["drift"])
    print("\nWorst energy drift by engine, across all N:")
    for eng, ds in drifts.items():
        print(f"  {eng:<14}{max(ds):.3e}")
    failed = [r for r in all_rows if not r.get("ok")]
    print(f"\nEngines that failed to integrate: {len(failed)}")
    for r in failed:
        print(f"  N={r['N']} {r['engine']}: {r.get('error')}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
