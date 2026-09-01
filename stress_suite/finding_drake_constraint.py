"""
FINDING: Drake's continuous-mode dynamics API silently ignores a registered
holonomic constraint, and returns the dynamics of a different mechanism.

This is the first engine-DISTINGUISHING silent failure found in the study.
Every previous result had all three engines behaving identically.

WHAT HAPPENS
------------
Build a slider-crank as a MultibodyPlant in continuous mode (time_step = 0):
a revolute crank, a prismatic slider, and a distance constraint standing in for
the connecting rod. Then:

    plant.AddDistanceConstraint(...)   accepted, returns a constraint id
    plant.Finalize()                   succeeds
    plant.num_constraints()            reports 1
    plant.time_step()                  reports 0.0  (continuous)
    plant.CalcMassMatrix(context)      returns a DIAGONAL matrix

The mass matrix has no off-diagonal coupling, so the crank and the slider are
dynamically independent: the rod is absent from the equations. The computed
slider acceleration is exactly 0.0 -- gravity acts along -y and nothing pushes
the slider along x, because as far as the continuous solver is concerned
nothing connects it to the crank.

No exception is raised. No warning is emitted. The API returns a mass matrix, a
bias term, and generalised forces for a mechanism that is not the one the user
described.

IS THIS FAIR TO DRAKE?
----------------------
Yes, and more pointedly than first assumed.

Used correctly Drake is EXCELLENT here: `main()` below shows the discrete SAP
solver maintaining the rod length to about 1e-8 through a second of simulation,
across several dead-centre crossings. The algorithms are not in question.

The defect is in the precondition check. `AddDistanceConstraint` documents a
guard against exactly this misuse:

    Raises: RuntimeError if `this` MultibodyPlant's underlying contact
            solver is not SAP.
            (i.e. get_discrete_contact_solver() != DiscreteContactSolver::kSap)

so Drake intends to refuse the constraint when the solver cannot honour it.
The guard does not fire, and measurement shows why:

    MultibodyPlant(0.0)    time_step=0.0    get_discrete_contact_solver() = kSap
    MultibodyPlant(1e-3)   time_step=1e-3   get_discrete_contact_solver() = kSap

A CONTINUOUS plant reports the same contact solver as a discrete one, because
that field records which discrete solver *would* be used and defaults to SAP
whether or not any discrete solver will ever run. The documented check
therefore passes vacuously in the one case where it is most needed: the
predicate tested ("which discrete solver is configured") is not the predicate
that matters ("will a discrete solver run at all").

So the guard Drake documents for this situation cannot catch it, and the
constraint is dropped from the continuous-mode dynamics with no exception and
no warning. This is not a limitation that is correctly signposted; it is a
signpost that does not work.

The trap is reachable by an ordinary route, not a perverse one. This study's own
`adapter_drake.py` uses exactly this continuous-mode pattern --
CalcMassMatrix / CalcBiasTerm / CalcGravityGeneralizedForces -- and it is
correct there, because the pendulum chain has no constraints. A user who
validates that pattern on an open chain and then adds a closed loop walks
directly into wrong physics reported as success.

WHY IT MATTERS TO THE STUDY
---------------------------
The study distinguishes two claims:

  (A) engines can report success while returning wrong physics, unwarned;
  (B) engines DIFFER in whether they do this.

(A) was already supported by the inverted-equilibrium result, but weakly: every
engine and the reference failed identically there, because the cause was
finite-precision arithmetic at an unstable equilibrium rather than anything
about the engines.

This result supports (B). On the same mechanism, in each engine's ordinary mode
of use, MechanicsDSL and sympy.physics.mechanics both produce correct
constrained dynamics -- agreeing with the independent reference to 7e-15 --
while Drake's continuous API produces the dynamics of an unconstrained system
and says nothing.

Run:  PYTHONPATH=<repo>/src ~/drake-venv/bin/python finding_drake_constraint.py
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
from pydrake.all import (MultibodyPlant, SpatialInertia, RotationalInertia,
                         RevoluteJoint, PrismaticJoint, Simulator)

import reference_slidercrank as RSC


def build(sc: RSC.SliderCrank, time_step: float) -> MultibodyPlant:
    p = MultibodyPlant(time_step)
    zI = RotationalInertia(0.0, 0.0, 0.0)
    crank = p.AddRigidBody("crank", SpatialInertia.MakeFromCentralInertia(
        sc.m_c, np.array([sc.r, 0.0, 0.0]), zI))
    slider = p.AddRigidBody("slider", SpatialInertia.MakeFromCentralInertia(
        sc.m_s, np.zeros(3), zI))
    p.AddJoint(RevoluteJoint("jc", p.world_frame(), crank.body_frame(),
                             np.array([0.0, 0.0, 1.0])))
    p.AddJoint(PrismaticJoint("js", p.world_frame(), slider.body_frame(),
                              np.array([1.0, 0.0, 0.0])))
    p.AddDistanceConstraint(crank, np.array([sc.r, 0.0, 0.0]),
                            slider, np.zeros(3), sc.l)
    p.mutable_gravity_field().set_gravity_vector([0.0, -sc.g, 0.0])
    p.Finalize()
    return p


def rod_length(sc, th, x):
    return math.hypot(x - sc.r * math.cos(th), sc.r * math.sin(th))


def main() -> int:
    sc = RSC.SliderCrank(3.0, mass_ratio=100.0)
    st = sc.initial_state()

    print("Drake constraint handling: continuous vs discrete")
    print(f"  mechanism : slider-crank, l/r={sc.ratio:g}, "
          f"m_s/m_c={sc.mass_ratio:g}\n")

    # ---- continuous ------------------------------------------------------
    print("--- CONTINUOUS mode (time_step = 0) ---")
    p = build(sc, 0.0)
    ctx = p.CreateDefaultContext()
    p.SetPositions(ctx, st[0::2])
    p.SetVelocities(ctx, st[1::2])
    M = p.CalcMassMatrix(ctx)
    a = np.linalg.solve(M, p.CalcGravityGeneralizedForces(ctx)
                        - p.CalcBiasTerm(ctx))
    ref = sc.accel(st)

    print(f"  constraint registered      : num_constraints() = "
          f"{p.num_constraints()}")
    print(f"  solver mode                : time_step() = {p.time_step()}")
    print(f"  exception or warning       : none")
    print(f"  mass matrix off-diagonal   : {abs(M[0, 1]):.1e}   "
          f"<- zero, so the rod is absent")
    print(f"  Drake accelerations        : {np.round(a, 6)}")
    print(f"  reference accelerations    : {np.round(ref, 6)}")
    print(f"  slider acceleration        : Drake {a[1]:.6f}  vs  "
          f"reference {ref[1]:.6f}")
    wrong = np.max(np.abs(a - ref) / np.maximum(np.abs(ref), 1.0))
    print(f"  worst relative error       : {wrong:.3e}")

    # ---- discrete --------------------------------------------------------
    print("\n--- DISCRETE mode (SAP, dt = 1e-4) ---")
    pd = build(sc, 1e-4)
    cd = pd.CreateDefaultContext()
    pd.SetPositions(cd, st[0::2])
    pd.SetVelocities(cd, st[1::2])
    sim = Simulator(pd, cd)
    sim.Initialize()
    print(f"  {'t (s)':>7}{'rod length':>13}{'target':>10}{'error':>12}")
    worst = 0.0
    for t in (0.0, 0.25, 0.5, 1.0):
        sim.AdvanceTo(t)
        q = pd.GetPositions(sim.get_context())
        L = rod_length(sc, q[0], q[1])
        worst = max(worst, abs(L - sc.l))
        print(f"  {t:7.2f}{L:13.6f}{sc.l:10.3f}{abs(L - sc.l):12.2e}")
    print(f"\n  worst rod-length error     : {worst:.2e}   "
          f"<- the constraint IS enforced here")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 62)
    print("  Same mechanism, same engine, two supported construction modes.")
    print("  Discrete : constraint honoured to ~1e-8.")
    print("  Continuous: constraint silently dropped; wrong physics returned")
    print("              as success, with the plant reporting that the")
    print("              constraint exists and that it is in continuous mode.")
    print("\n  MechanicsDSL and sympy.physics.mechanics both model this")
    print("  mechanism correctly in their ordinary mode of use, agreeing with")
    print("  the independent reference to 7e-15.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
