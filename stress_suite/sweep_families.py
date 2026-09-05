"""
Full treatment of families 2 and 3: derivation AND integration, three engines.

Family 2, slider-crank, was previously checked on derivation only. Family 3,
cart-pole, is new. Both get what the pendulum chain got: equations checked
against a library-independent reference at probe states, then integration under
ONE pinned integrator with energy drift compared.

DRAKE'S PARTICIPATION DIFFERS BY FAMILY, AND THE REASON IS RECORDED
------------------------------------------------------------------
  cart-pole      A tree: prismatic cart, revolute pole, no constraint. Drake
                 handles it in continuous mode like any other mechanism, so all
                 three engines are compared on equal terms under one integrator.

  slider-crank   Requires a loop closure. Constrained dynamics in Drake are
                 enforced inside its discrete solver and are not available
                 through the continuous-time interface used for the other two
                 engines, so Drake is run in DISCRETE mode here. A discrete
                 plant integrates itself and cannot share the pinned
                 integrator, so it is scored on constraint satisfaction and
                 energy rather than against the same trajectory. That asymmetry
                 follows from the engine's supported interfaces rather than
                 from a choice, and is reported rather than smoothed over.

Run inside WSL:
    PYTHONPATH=<repo>/src ~/drake-venv/bin/python sweep_families.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
import reference_cartpole as RCP

K_PROBE = 16
TOL = 1e-8
SEED = 20260825
RTOL, ATOL = 1e-10, 1e-12
METHOD = "DOP853"


# ==========================================================================
# Family 3: cart-pole. Full three-engine comparison, one pinned integrator.
# ==========================================================================

def cartpole_dsl(cp, th0):
    return "\n".join([
        r"\system{cartpole}",
        r"\defvar{x}{Position}{m}",
        r"\defvar{theta}{Angle}{rad}",
        r"\parameter{Mc}{%s}{kg}" % repr(cp.M),
        r"\parameter{mp}{%s}{kg}" % repr(cp.m),
        r"\parameter{l}{%s}{m}" % repr(cp.l),
        r"\parameter{g}{%s}{m/s^2}" % repr(cp.g),
        r"\lagrangian{0.5*(Mc + mp)*\dot{x}^2 "
        r"+ mp*l*\dot{x}*\dot{theta}*\cos{theta} "
        r"+ 0.5*mp*l^2*\dot{theta}^2 + mp*g*l*\cos{theta}}",
        r"\initial{x=0.0, x_dot=0.0, theta=%s, theta_dot=0.0}" % repr(th0),
    ])


def cartpole_mechanicsdsl(cp, th0):
    import worker
    from mechanics_dsl import PhysicsCompiler
    c = PhysicsCompiler()
    r = c.compile_dsl(cartpole_dsl(cp, th0), use_hamiltonian=False,
                      use_constraints=False)
    if not r.get("success"):
        raise RuntimeError("compile_success=False")
    fn, route = worker._engine_accel_fn(c, ["x", "theta"])
    if fn is None:
        raise RuntimeError(f"no_accel_route:{route}")
    return fn


def cartpole_sympy(cp, th0):
    import sympy as sp
    from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols
    t = dynamicsymbols._t
    x, th = dynamicsymbols("x th")
    xd, thd = x.diff(t), th.diff(t)
    L = (sp.Rational(1, 2) * (cp.M + cp.m) * xd ** 2
         + cp.m * cp.l * xd * thd * sp.cos(th)
         + sp.Rational(1, 2) * cp.m * cp.l ** 2 * thd ** 2
         + cp.m * cp.g * cp.l * sp.cos(th))
    lm = LagrangesMethod(L, [x, th])
    lm.form_lagranges_equations()
    M, F = lm.mass_matrix_full, lm.forcing_full
    Mf = sp.lambdify([[x, th], [xd, thd]], M, "numpy")
    Ff = sp.lambdify([[x, th], [xd, thd]], F, "numpy")
    dim = M.shape[0]

    def fn(state):
        s = np.asarray(state, dtype=float)
        Mn = np.array(Mf(list(s[0::2]), list(s[1::2])), float).reshape(dim, dim)
        Fn = np.array(Ff(list(s[0::2]), list(s[1::2])), float).reshape(dim)
        return np.linalg.solve(Mn, Fn)[2:4]
    return fn


def cartpole_drake(cp, th0):
    from pydrake.all import (MultibodyPlant, SpatialInertia, RotationalInertia,
                             RevoluteJoint, PrismaticJoint, RigidTransform,
                             FixedOffsetFrame)
    p = MultibodyPlant(0.0)
    zI = RotationalInertia(0.0, 0.0, 0.0)
    cart = p.AddRigidBody("cart", SpatialInertia.MakeFromCentralInertia(
        cp.M, np.zeros(3), zI))
    # Pole point mass at (0,0,-l) in the pole frame; pole frame origin at pivot.
    pole = p.AddRigidBody("pole", SpatialInertia.MakeFromCentralInertia(
        cp.m, np.array([0.0, 0.0, -cp.l]), zI))
    p.AddJoint(PrismaticJoint("jx", p.world_frame(), cart.body_frame(),
                              np.array([1.0, 0.0, 0.0])))
    # Axis is -y, not +y. Rotating the pole's mass offset (0,0,-l) about +y
    # places the mass at x - l sin(th); the reference uses x + l sin(th).
    # Using +y here mirrors the mechanism horizontally, which leaves the pole
    # equation unchanged but flips the sign of the cart's acceleration and of
    # the mass matrix's off-diagonal coupling. The resulting relative error is
    # exactly 2.0 wherever |a_ref| > 1 -- the same "suspiciously exact number"
    # signature that flagged the (q,p) and relative-angle errors earlier.
    p.AddJoint(RevoluteJoint("jth", cart.body_frame(), pole.body_frame(),
                             np.array([0.0, -1.0, 0.0])))
    p.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -cp.g])
    p.Finalize()
    ctx = p.CreateDefaultContext()

    def fn(state):
        s = np.asarray(state, dtype=float)
        p.SetPositions(ctx, s[0::2])
        p.SetVelocities(ctx, s[1::2])
        M = p.CalcMassMatrix(ctx)
        return np.linalg.solve(
            M, p.CalcGravityGeneralizedForces(ctx) - p.CalcBiasTerm(ctx))
    return fn


CARTPOLE_ENGINES = [("MechanicsDSL", cartpole_mechanicsdsl),
                    ("SymPy", cartpole_sympy),
                    ("Drake", cartpole_drake)]


def run_cartpole(rows):
    print("=" * 78)
    print("FAMILY 3: CART-POLE   (tree; all three engines, one pinned integrator)")
    print("=" * 78)
    hdr = (f"{'M/m':>8}{'th0':>6}{'detM range':>12}  {'engine':<14}"
           f"{'EOM vs ref':>12}{'energy drift':>14}  status")
    print(hdr)
    print("-" * len(hdr))

    for mr in RCP.MASS_RATIOS:
        for th0 in RCP.ANGLES:
            cp = RCP.CartPole(mass_ratio=mr)
            deg = cp.degeneracy()
            y0 = cp.initial_state(th0)
            first = True
            for name, factory in CARTPOLE_ENGINES:
                row = {"family": "cartpole", "mass_ratio": mr, "th0": th0,
                       "degeneracy": deg, "engine": name}
                pre = (f"{mr:>8g}{th0:>6.1f}{deg:>12.1e}  " if first
                       else " " * 28)
                try:
                    fn = factory(cp, th0)
                    rng = np.random.default_rng(SEED)
                    worst = 0.0
                    for _ in range(K_PROBE):
                        st = np.array([rng.uniform(-1, 1), rng.uniform(-2, 2),
                                       rng.uniform(-math.pi, math.pi),
                                       rng.uniform(-3, 3)])
                        a = np.asarray(fn(st), float)[:2]
                        r = cp.accel(st)
                        worst = max(worst, float(np.max(
                            np.abs(a - r) / np.maximum(np.abs(r), 1.0))))

                    def rhs(_t, y):
                        o = np.empty_like(y)
                        o[0::2] = y[1::2]
                        o[1::2] = np.asarray(fn(y), float)[:2]
                        return o
                    sol = solve_ivp(rhs, (0.0, 10.0), y0, method=METHOD,
                                    rtol=RTOL, atol=ATOL)
                    if not sol.success:
                        raise RuntimeError(sol.message[:30])
                    E0 = cp.energy(sol.y[:, 0])
                    drift = max(abs(cp.energy(sol.y[:, i]) - E0)
                                for i in range(sol.y.shape[1])) / max(abs(E0), 1e-12)
                    st_ = "agree" if worst <= TOL else "DISAGREE"
                    row.update(ok=True, eom=worst, drift=drift, status=st_)
                    print(f"{pre}{name:<14}{worst:12.2e}{drift:14.3e}  {st_}")
                except Exception as e:
                    row.update(ok=False, status="refused",
                               error=f"{type(e).__name__}: {e}"[:60])
                    print(f"{pre}{name:<14}{'--':>12}{'--':>14}  refused: "
                          f"{type(e).__name__}")
                rows.append(row)
                first = False
        print()


# ==========================================================================
# Family 2: slider-crank. Integration added; Drake discrete, scored apart.
# ==========================================================================

def run_slidercrank(rows):
    import sweep_slidercrank as SSC
    print("=" * 78)
    print("FAMILY 2: SLIDER-CRANK   (loop; MechanicsDSL + SymPy pinned, "
          "Drake discrete)")
    print("=" * 78)
    hdr = (f"{'l/r':>6}{'m_s/m_c':>9}  {'engine':<14}{'EOM vs ref':>12}"
           f"{'energy drift':>14}{'constraint':>12}  status")
    print(hdr)
    print("-" * len(hdr))

    for ratio, mr in [(3.0, 1e2), (1.5, 1e2), (1.05, 1e2), (3.0, 1e4)]:
        sc = RSC.SliderCrank(ratio, mass_ratio=mr)
        y0 = sc.initial_state()
        first = True
        for name, factory in [("MechanicsDSL", SSC.accel_mechanicsdsl),
                              ("SymPy", SSC.accel_sympy)]:
            row = {"family": "slidercrank", "ratio": ratio, "mass_ratio": mr,
                   "engine": name}
            pre = f"{ratio:>6g}{mr:>9g}  " if first else " " * 17
            try:
                fn, _route = factory(sc)
                worst = SSC.evaluate(sc, fn)

                def rhs(_t, y):
                    o = np.empty_like(y)
                    o[0::2] = y[1::2]
                    o[1::2] = np.asarray(fn(y), float)[:2]
                    return o
                sol = solve_ivp(rhs, (0.0, 5.0), y0, method=METHOD,
                                rtol=RTOL, atol=ATOL)
                if not sol.success:
                    raise RuntimeError(sol.message[:30])
                E0 = sc.energy(sol.y[:, 0])
                drift = max(abs(sc.energy(sol.y[:, i]) - E0)
                            for i in range(sol.y.shape[1])) / max(abs(E0), 1e-12)
                cons = max(sc.constraint_residual(sol.y[0::2, i])
                           for i in range(sol.y.shape[1]))
                st_ = "agree" if worst <= TOL else "DISAGREE"
                row.update(ok=True, eom=worst, drift=drift, constraint=cons,
                           status=st_)
                print(f"{pre}{name:<14}{worst:12.2e}{drift:14.3e}"
                      f"{cons:12.2e}  {st_}")
            except Exception as e:
                row.update(ok=False, status="refused",
                           error=f"{type(e).__name__}: {e}"[:60])
                print(f"{pre}{name:<14}{'--':>12}{'--':>14}{'--':>12}  "
                      f"refused: {type(e).__name__}")
            rows.append(row)
            first = False

        # Drake, discrete, scored on constraint satisfaction only.
        try:
            from pydrake.all import (MultibodyPlant, SpatialInertia,
                                     RotationalInertia, RevoluteJoint,
                                     PrismaticJoint, Simulator)
            p = MultibodyPlant(1e-4)
            zI = RotationalInertia(0.0, 0.0, 0.0)
            cr = p.AddRigidBody("crank", SpatialInertia.MakeFromCentralInertia(
                sc.m_c, np.array([sc.r, 0.0, 0.0]), zI))
            sl = p.AddRigidBody("slider", SpatialInertia.MakeFromCentralInertia(
                sc.m_s, np.zeros(3), zI))
            p.AddJoint(RevoluteJoint("jc", p.world_frame(), cr.body_frame(),
                                     np.array([0.0, 0.0, 1.0])))
            p.AddJoint(PrismaticJoint("js", p.world_frame(), sl.body_frame(),
                                      np.array([1.0, 0.0, 0.0])))
            p.AddDistanceConstraint(cr, np.array([sc.r, 0.0, 0.0]),
                                    sl, np.zeros(3), sc.l)
            p.mutable_gravity_field().set_gravity_vector([0.0, -sc.g, 0.0])
            p.Finalize()
            ctx = p.CreateDefaultContext()
            p.SetPositions(ctx, y0[0::2])
            p.SetVelocities(ctx, y0[1::2])
            sim = Simulator(p, ctx)
            sim.Initialize()
            worstc = 0.0
            for t in (0.0, 1.25, 2.5, 5.0):
                sim.AdvanceTo(t)
                q = p.GetPositions(sim.get_context())
                worstc = max(worstc, abs(
                    math.hypot(q[1] - sc.r * math.cos(q[0]),
                               sc.r * math.sin(q[0])) - sc.l))
            print(f"{' ' * 17}{'Drake (disc.)':<14}{'n/a':>12}{'n/a':>14}"
                  f"{worstc:12.2e}  rod held")
            rows.append({"family": "slidercrank", "ratio": ratio,
                         "mass_ratio": mr, "engine": "Drake-discrete",
                         "ok": True, "constraint": worstc,
                         "status": "rod held"})
        except Exception as e:
            print(f"{' ' * 17}{'Drake (disc.)':<14}refused: {type(e).__name__}")
            rows.append({"family": "slidercrank", "ratio": ratio,
                         "mass_ratio": mr, "engine": "Drake-discrete",
                         "ok": False, "status": "refused",
                         "error": str(e)[:60]})
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="out/families.json")
    args = ap.parse_args()

    rows = []
    run_cartpole(rows)
    run_slidercrank(rows)

    print("=" * 78)
    dis = [r for r in rows if r.get("status") == "DISAGREE"]
    ref = [r for r in rows if r.get("status") == "refused"]
    print(f"\n  engine-case pairs : {len(rows)}")
    print(f"  disagreements     : {len(dis)}")
    print(f"  refusals          : {len(ref)}")
    for r in dis + ref:
        print(f"    {r['family']} {r['engine']}: {r.get('status')} "
              f"{r.get('error','')}")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
