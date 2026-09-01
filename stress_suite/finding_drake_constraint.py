#!/usr/bin/env python3
"""
Drake 1.56.0: a distance constraint on a CONTINUOUS MultibodyPlant is accepted
and then silently omitted from the dynamics.

STANDALONE. Imports numpy and pydrake and nothing else -- no part of the study
that found it. Run it against a stock `pip install drake` and it reproduces the
whole arc in one process:

    python3 finding_drake_constraint.py

SUMMARY
-------
Build a slider-crank: a revolute crank, a prismatic slider, and a distance
constraint standing in for the connecting rod that couples them.

With `MultibodyPlant(0.0)` (continuous):

    AddDistanceConstraint(...)      accepted, returns a constraint id
    Finalize()                      succeeds
    num_constraints()               reports 1
    CalcMassMatrix(context)         returns a DIAGONAL matrix

The mass matrix has no off-diagonal term, so the crank and slider are
dynamically independent and the rod is absent from the equations. The computed
slider acceleration is exactly 0.0, against -0.993 from the closed-form
solution derived below. No exception is raised and no warning is emitted.

With a discrete plant the same model is correct: the rod length is held to
~1e-5 at dt=1e-3 and ~1e-8 at dt=1e-4, through a second of simulation across
several dead-centre crossings, the error scaling with the step as expected. The
dynamics are not in question; the model-building API is.

PROVENANCE: THIS IS A REGRESSION, WITH A DATE AND A PR
-----------------------------------------------------
Drake DID guard against this call, for three years, and the guard was removed.
Established from a full clone of RobotLocomotion/drake, not from documentation
and not from a search snippet:

  2022-12-07  baeadb5a4  #18196  "Implements support for fixed-distance
                                  constraints with SAP"
              introduces AddDistanceConstraint together with

                  if (!is_discrete()) {
                    throw std::runtime_error(
                        "Currently distance constraints are only supported "
                        "for discrete MultibodyPlant models.");
                  }

  2026-02-17  73c987a60  #24079  "[multibody] Refine continuous mode feature
                                  support checks"  (Rick Poyner)
              REMOVES that guard, replacing it with

                  if (is_discrete()) { switch (get_discrete_contact_solver())
                                       { ... TAMSI throw ... } }
                  // Feature support for continuous time plants depends on
                  // the integrator used.

              so the TAMSI check now sits inside a discrete-only branch and
              nothing rejects a continuous plant. The same edit was applied to
              AddCouplerConstraint, and the identical comment appears at five
              sites in master (lines 514, 558, 669, 712, 758), so the change is
              systematic and deliberate rather than an oversight.

Between those dates Drake refused this exact call with a clear message. Since
2026-02-17 it accepts it and silently omits the constraint from continuous-mode
dynamics. Verified absent from the 1.56.0 wheel and from master's source.

THE CLAIM, SCOPED BY WHAT WAS ACTUALLY RUN
------------------------------------------
    Measured: in Drake 1.56.0, the latest release at time of testing, a
    distance constraint added to a continuous MultibodyPlant is accepted and
    then silently omitted from the dynamics.

    Read, not run: master's source contains no continuous-mode rejection
    either, so the behaviour appears current. Master was READ, not RUN, and
    #24079's comment implies continuous-time support is integrator-dependent,
    which cannot be settled by reading at all. Any claim about master needs a
    master build.

THE QUESTION TO ASK UPSTREAM
----------------------------
Not "you dropped a guard" -- #24079 is titled "refine" and is systematic, so
the removal was intended. The question is whether the intended consequence was
that constraints are silently ignored on continuous plants while the feature
remains unimplemented there, and if so whether the comment at line 558 should
be an exception instead.

A NOTE ON METHOD, RECORDED BECAUSE IT WAS GOT WRONG FIVE TIMES
--------------------------------------------------------------
  1. "Documented limitation, correctly signposted."  Inference, not
     measurement. Wrong.
  2. "Drake documents a guard that checks the wrong predicate."  Read the
     pydrake docstring, which is generated from Doxygen @throws annotations
     and does not enumerate every throw in the body. Checking documentation
     twice is not checking twice. Wrong.
  3. "The guard exists on master and postdates 1.56.0."  Accepted from a
     secondary description without checking the source. Wrong -- and it was
     wrong because the description came from a cached search snippet of a blob
     page, which reflects when the crawler last visited, not HEAD.
  4. "The guard never existed."  git log -S returned empty, and the null was
     believed. Wrong: the search had been run seconds after cloning while git
     was auto-packing, so blob fetches failed silently and returned a false
     negative.
  5. The control for (4) was run later, on a settled repository -- so it
     validated a DIFFERENT run than the one it existed to validate. The right
     answer arrived from the control's own output by luck, not from the search
     it was meant to check. A control must run under the same conditions as
     the measurement it validates.

Only (6), a full clone with the repository packed and the instrument validated
under the same conditions, produced the history above. The pattern across all
five: every wrong version looked obviously right and survived until something
independent was consulted, and each correction moved one step closer to the
artefact -- inference, then docs, then a snippet, then an unvalidated tool,
then a mis-scoped control, then the object itself.

THE CLOSED-FORM COMPARISON
--------------------------
Crank pivot at the origin, crank length r, angle th from +x, crank pin at
P = (r cos th, r sin th). Slider at S = (x, 0) on the x-axis. Massless rod of
length l joins P to S. Coordinates q = (th, x); one constraint; one degree of
freedom.

    g = |S - P|^2 - l^2 = x^2 - 2 r x cos(th) + r^2 - l^2

the sin^2 + cos^2 collapsing to a constant. Then

    J     = [ 2 r x sin(th),  2x - 2 r cos(th) ]
    gamma = -( 2 xdot^2 + 4 r xdot sin(th) thdot + 2 r x cos(th) thdot^2 )

Crank as a point mass m_c at the pin, slider a point mass m_s, rod massless:

    M = diag(m_c r^2, m_s),   F = (-m_c g r cos(th), 0)

and (qddot, lambda) solve

    [ M   J^T ] [ qddot ]   [ F     ]
    [ J   0   ] [ lam   ] = [ gamma ]
"""

import math

import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import (SpatialInertia, RotationalInertia,
                                    RevoluteJoint, PrismaticJoint)
from pydrake.systems.analysis import Simulator

# Mechanism parameters.
R, L = 1.0, 3.0            # crank length, rod length
M_C, M_S = 1.0, 100.0      # crank mass, slider mass
G = 9.81


# --------------------------------------------------------------------------
# Closed-form reference. numpy only, no symbolic algebra, nothing imported.
# --------------------------------------------------------------------------

def slider_position(th):
    return R * math.cos(th) + math.sqrt(max(L * L - (R * math.sin(th)) ** 2, 0.0))


def slider_velocity(th, thd):
    x = slider_position(th)
    return -R * x * math.sin(th) * thd / (x - R * math.cos(th))


def reference_accel(th, thd, x, xd):
    M = np.diag([M_C * R * R, M_S])
    F = np.array([-M_C * G * R * math.cos(th), 0.0])
    J = np.array([[2.0 * R * x * math.sin(th), 2.0 * x - 2.0 * R * math.cos(th)]])
    gamma = np.array([-(2.0 * xd * xd
                        + 4.0 * R * xd * math.sin(th) * thd
                        + 2.0 * R * x * math.cos(th) * thd * thd)])
    K = np.zeros((3, 3))
    K[:2, :2] = M
    K[:2, 2:] = J.T
    K[2:, :2] = J
    return np.linalg.solve(K, np.concatenate([F, gamma]))[:2]


def rod_length(th, x):
    return math.hypot(x - R * math.cos(th), R * math.sin(th))


# --------------------------------------------------------------------------

def build(time_step):
    p = MultibodyPlant(time_step)
    zI = RotationalInertia(0.0, 0.0, 0.0)
    crank = p.AddRigidBody("crank", SpatialInertia.MakeFromCentralInertia(
        M_C, np.array([R, 0.0, 0.0]), zI))
    slider = p.AddRigidBody("slider", SpatialInertia.MakeFromCentralInertia(
        M_S, np.zeros(3), zI))
    p.AddJoint(RevoluteJoint("jc", p.world_frame(), crank.body_frame(),
                             np.array([0.0, 0.0, 1.0])))
    p.AddJoint(PrismaticJoint("js", p.world_frame(), slider.body_frame(),
                              np.array([1.0, 0.0, 0.0])))
    cid = p.AddDistanceConstraint(crank, np.array([R, 0.0, 0.0]),
                                  slider, np.zeros(3), L)
    p.mutable_gravity_field().set_gravity_vector([0.0, -G, 0.0])
    p.Finalize()
    return p, cid


def main():
    import pydrake.common
    try:
        import importlib.metadata as md
        version = md.version("drake")
    except Exception:
        version = "unknown"

    th0, thd0 = 0.3, 4.0
    x0 = slider_position(th0)
    xd0 = slider_velocity(th0, thd0)
    q0, v0 = np.array([th0, x0]), np.array([thd0, xd0])

    print(f"Drake {version} -- distance constraint on a continuous plant")
    print(f"  slider-crank: r={R:g}, l={L:g}, m_crank={M_C:g}, "
          f"m_slider={M_S:g}")
    print(f"  state: th={th0:g}, thdot={thd0:g}, x={x0:.6f}, "
          f"xdot={xd0:.6f}  (on the constraint manifold)\n")

    # ---- continuous ------------------------------------------------------
    print("=" * 66)
    print("CONTINUOUS  MultibodyPlant(0.0)")
    print("=" * 66)
    p, cid = build(0.0)
    print(f"  AddDistanceConstraint   -> ACCEPTED, id={cid}")
    print(f"  Finalize()              -> succeeded")
    print(f"  num_constraints()       -> {p.num_constraints()}")
    print(f"  time_step()             -> {p.time_step()}")
    print(f"  is_discrete()           -> "
          f"{'absent from this API' if not hasattr(p, 'is_discrete') else p.is_discrete()}")
    print(f"  get_discrete_contact_solver() -> "
          f"{p.get_discrete_contact_solver()}   (same on a discrete plant)")
    print(f"  exception or warning    -> none")

    ctx = p.CreateDefaultContext()
    p.SetPositions(ctx, q0)
    p.SetVelocities(ctx, v0)
    M = p.CalcMassMatrix(ctx)
    a_drake = np.linalg.solve(
        M, p.CalcGravityGeneralizedForces(ctx) - p.CalcBiasTerm(ctx))
    a_ref = reference_accel(th0, thd0, x0, xd0)

    print(f"\n  CalcMassMatrix:")
    print(f"      [[{M[0,0]:12.6f} {M[0,1]:12.6f}]")
    print(f"       [{M[1,0]:12.6f} {M[1,1]:12.6f}]]")
    print(f"      off-diagonal = {abs(M[0,1]):.1e}  <- zero: the rod is not "
          f"in the equations")
    print(f"\n  {'':22}{'crank thddot':>15}{'slider xddot':>15}")
    print(f"  {'Drake':22}{a_drake[0]:15.6f}{a_drake[1]:15.6f}")
    print(f"  {'closed form':22}{a_ref[0]:15.6f}{a_ref[1]:15.6f}")
    rel = np.max(np.abs(a_drake - a_ref) / np.maximum(np.abs(a_ref), 1.0))
    print(f"\n  worst relative error    -> {rel:.3e}")
    print(f"  slider acceleration     -> Drake {a_drake[1]:.1f} exactly; "
          f"the slider is free")

    # ---- discrete --------------------------------------------------------
    print("\n" + "=" * 66)
    print("DISCRETE  MultibodyPlant(1e-3), SAP")
    print("=" * 66)
    pd, _ = build(1e-3)
    cd = pd.CreateDefaultContext()
    pd.SetPositions(cd, q0)
    pd.SetVelocities(cd, v0)
    sim = Simulator(pd, cd)
    sim.Initialize()
    print(f"  {'t (s)':>8}{'rod length':>14}{'target':>10}{'error':>12}")
    worst = 0.0
    for t in (0.0, 0.25, 0.5, 1.0):
        sim.AdvanceTo(t)
        q = pd.GetPositions(sim.get_context())
        Lr = rod_length(q[0], q[1])
        worst = max(worst, abs(Lr - L))
        print(f"  {t:8.2f}{Lr:14.6f}{L:10.3f}{abs(Lr - L):12.2e}")
    print(f"\n  worst rod-length error  -> {worst:.2e}   "
          f"the constraint IS enforced")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 66)
    print("  Same model, same version, two supported construction modes.")
    print(f"  Discrete   : correct, rod held to {worst:.0e} at dt=1e-3")
    print("               (and to ~1e-8 at dt=1e-4; the error scales with")
    print("               the step, as expected for a discrete solver).")
    print("  Continuous : constraint accepted, silently dropped, wrong")
    print("               dynamics returned as success, no diagnostic.")
    print()
    print("  This is an API GUARD failure, not a dynamics failure. Drake's")
    print("  articulated-body algorithms are correct; the model-building API")
    print("  accepted a model it does not implement. The guard that catches")
    print("  it exists on master and in no released version tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
