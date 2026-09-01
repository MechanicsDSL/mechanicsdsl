"""
Three-engine sweep of the slider-crank -- the second problem family.

Dials two knobs and checks every engine against the library-independent
reference in `reference_slidercrank.py`:

  * rod-to-crank ratio l/r, from benign (3.0) toward the folding
    configuration (1.01);
  * slider-to-crank mass ratio m_s/m_c, which controls how violently the
    effective inertia collapses at dead centre.

Run inside WSL, where all three engines import in one process:

    PYTHONPATH=<repo>/src ~/drake-venv/bin/python sweep_slidercrank.py
"""

from __future__ import annotations

import argparse
import json
import math
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

import reference_slidercrank as RSC

K_PROBE = 16
TOL = 1e-8
SEED = 20260825
RTOL, ATOL = 1e-10, 1e-12
METHOD = "DOP853"
T_END = 5.0


# ---------------------------------------------------------------------------
# Engine models
# ---------------------------------------------------------------------------

def dsl_for(sc: RSC.SliderCrank) -> str:
    y0 = sc.initial_state()
    return "\n".join([
        r"\system{slidercrank}",
        r"\defvar{theta}{Angle}{rad}",
        r"\defvar{x}{Position}{m}",
        r"\parameter{mc}{%s}{kg}" % repr(sc.m_c),
        r"\parameter{ms}{%s}{kg}" % repr(sc.m_s),
        r"\parameter{r}{%s}{m}" % repr(sc.r),
        r"\parameter{l}{%s}{m}" % repr(sc.l),
        r"\parameter{g}{%s}{m/s^2}" % repr(sc.g),
        r"\lagrangian{0.5*mc*r^2*\dot{theta}^2 + 0.5*ms*\dot{x}^2 "
        r"- mc*g*r*\sin{theta}}",
        r"\constraint{x^2 - 2*r*x*\cos{theta} + r^2 - l^2}",
        r"\initial{theta=%s, theta_dot=%s, x=%s, x_dot=%s}"
        % (repr(float(y0[0])), repr(float(y0[1])),
           repr(float(y0[2])), repr(float(y0[3]))),
    ])


def accel_mechanicsdsl(sc):
    import worker
    from mechanics_dsl import PhysicsCompiler
    c = PhysicsCompiler()
    res = c.compile_dsl(dsl_for(sc), use_hamiltonian=False, use_constraints=True)
    if not res.get("success"):
        raise RuntimeError("compile_success=False")
    fn, route = worker._engine_accel_fn(c, ["theta", "x"])
    if fn is None:
        raise RuntimeError(f"no_accel_route:{route}")
    return fn, route


def accel_sympy(sc):
    import sympy as sp
    from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols
    t = dynamicsymbols._t
    th, x = dynamicsymbols("th x")
    thd, xd = th.diff(t), x.diff(t)
    L = (sp.Rational(1, 2) * sc.m_c * sc.r ** 2 * thd ** 2
         + sp.Rational(1, 2) * sc.m_s * xd ** 2
         - sc.m_c * sc.g * sc.r * sp.sin(th))
    g = x ** 2 - 2 * sc.r * x * sp.cos(th) + sc.r ** 2 - sc.l ** 2
    lm = LagrangesMethod(L, [th, x], hol_coneqs=[g])
    lm.form_lagranges_equations()
    M = lm.mass_matrix_full
    F = lm.forcing_full
    Mf = sp.lambdify([[th, x], [thd, xd]], M, "numpy")
    Ff = sp.lambdify([[th, x], [thd, xd]], F, "numpy")
    dim = M.shape[0]

    def fn(state):
        s = np.asarray(state, dtype=float)
        q, v = list(s[0::2]), list(s[1::2])
        Mn = np.array(Mf(q, v), dtype=float).reshape(dim, dim)
        Fn = np.array(Ff(q, v), dtype=float).reshape(dim)
        try:
            sol = np.linalg.solve(Mn, Fn)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(Mn, Fn, rcond=None)
        return sol[2:4]
    return fn, "lagranges_hol_coneqs"


def accel_drake(sc):
    """Drake via MultibodyPlant with a distance constraint for the rod.

    Drake models closed loops with AddDistanceConstraint, which is only
    supported by the discrete (SAP) solver -- the continuous plant used for
    the open chain in adapter_drake.py cannot express it. If the installed
    version refuses, that is reported rather than worked around: an engine
    declining to model a system is a data point in this study.
    """
    from pydrake.all import (MultibodyPlant, SpatialInertia, RotationalInertia,
                             RevoluteJoint, PrismaticJoint, RigidTransform,
                             FixedOffsetFrame)
    plant = MultibodyPlant(0.0)
    zero_I = RotationalInertia(0.0, 0.0, 0.0)

    crank = plant.AddRigidBody("crank", SpatialInertia.MakeFromCentralInertia(
        sc.m_c, np.array([sc.r, 0.0, 0.0]), zero_I))
    slider = plant.AddRigidBody("slider", SpatialInertia.MakeFromCentralInertia(
        sc.m_s, np.zeros(3), zero_I))
    plant.AddJoint(RevoluteJoint("j_crank", plant.world_frame(),
                                 crank.body_frame(),
                                 np.array([0.0, 0.0, 1.0])))
    plant.AddJoint(PrismaticJoint("j_slider", plant.world_frame(),
                                  slider.body_frame(),
                                  np.array([1.0, 0.0, 0.0])))
    plant.AddDistanceConstraint(crank, np.array([sc.r, 0.0, 0.0]),
                                slider, np.zeros(3), sc.l)
    plant.mutable_gravity_field().set_gravity_vector([0.0, -sc.g, 0.0])
    plant.Finalize()
    ctx = plant.CreateDefaultContext()

    def fn(state):
        s = np.asarray(state, dtype=float)
        plant.SetPositions(ctx, s[0::2])
        plant.SetVelocities(ctx, s[1::2])
        M = plant.CalcMassMatrix(ctx)
        Cv = plant.CalcBiasTerm(ctx)
        tau = plant.CalcGravityGeneralizedForces(ctx)
        return np.linalg.solve(M, tau - Cv)
    fn(sc.initial_state())          # probe now, so failure is attributable
    return fn, "distance_constraint"


ENGINES = [("MechanicsDSL", accel_mechanicsdsl),
           ("SymPy", accel_sympy),
           ("Drake", accel_drake)]


# ---------------------------------------------------------------------------

def evaluate(sc, fn):
    """Worst relative acceleration error over probe states on the manifold."""
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(K_PROBE):
        th = rng.uniform(-math.pi, math.pi)
        thd = rng.uniform(-4.0, 4.0)
        st = np.array([th, thd, sc.slider_position(th),
                       sc.slider_velocity(th, thd)])
        a = np.asarray(fn(st), dtype=float)[:2]
        r = sc.accel(st)
        worst = max(worst, float(np.max(np.abs(a - r)
                                        / np.maximum(np.abs(r), 1.0))))
    return worst


def dead_centre_error(sc, fn):
    """Error evaluated AT dead centre, where the effective inertia collapses."""
    worst = 0.0
    for th in (0.0, math.pi):
        for thd in (1.0, 4.0):
            st = np.array([th, thd, sc.slider_position(th),
                           sc.slider_velocity(th, thd)])
            a = np.asarray(fn(st), dtype=float)[:2]
            r = sc.accel(st)
            worst = max(worst, float(np.max(np.abs(a - r)
                                            / np.maximum(np.abs(r), 1.0))))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/slidercrank.json")
    args = ap.parse_args()

    print("Slider-crank sweep -- second problem family")
    print(f"  referee : reference_slidercrank.py (numpy closed form)")
    print(f"  probes  : {K_PROBE} states on the manifold, plus both dead centres")
    print(f"  tol     : {TOL:g} relative\n")

    rows = []
    cases = ([("l/r", ratio, 1e2) for ratio in RSC.RATIOS]
             + [("m_s/m_c", 3.0, mr) for mr in RSC.MASS_RATIOS])

    hdr = (f"{'knob':<9}{'l/r':>6}{'m_s/m_c':>9}{'collapse':>10}  "
           f"{'engine':<14}{'probe err':>11}{'dead-ctr':>11}  status")
    print(hdr)
    print("-" * len(hdr))

    for knob, ratio, mr in cases:
        sc = RSC.SliderCrank(ratio, mass_ratio=mr)
        collapse = sc.inertia_collapse()
        first = True
        for name, factory in ENGINES:
            row = {"knob": knob, "ratio": ratio, "mass_ratio": mr,
                   "collapse": collapse, "engine": name}
            pre = (f"{knob:<9}{ratio:>6g}{mr:>9g}{collapse:>10.2e}  "
                   if first else " " * 34)
            try:
                fn, route = factory(sc)
                e_probe = evaluate(sc, fn)
                e_dead = dead_centre_error(sc, fn)
                ok = max(e_probe, e_dead) <= TOL
                row.update(ok=True, probe=e_probe, dead=e_dead,
                           status="agree" if ok else "DISAGREE", route=route)
                print(f"{pre}{name:<14}{e_probe:11.2e}{e_dead:11.2e}  "
                      f"{row['status']}")
            except Exception as e:
                row.update(ok=False, status="refused",
                           error=f"{type(e).__name__}: {e}"[:80])
                print(f"{pre}{name:<14}{'--':>11}{'--':>11}  refused: "
                      f"{type(e).__name__}")
            rows.append(row)
            first = False
        print()

    print("=" * len(hdr))
    dis = [r for r in rows if r.get("status") == "DISAGREE"]
    ref = [r for r in rows if r.get("status") == "refused"]
    print(f"\n  engine-case pairs : {len(rows)}")
    print(f"  disagreements     : {len(dis)}")
    print(f"  refusals          : {len(ref)}")
    for r in dis:
        print(f"    l/r={r['ratio']:g} m={r['mass_ratio']:g} {r['engine']}: "
              f"probe {r['probe']:.2e}, dead centre {r['dead']:.2e}")
    for r in ref:
        print(f"    l/r={r['ratio']:g} m={r['mass_ratio']:g} {r['engine']}: "
              f"{r.get('error')}")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
