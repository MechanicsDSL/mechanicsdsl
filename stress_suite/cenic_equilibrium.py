"""
Does the shared equilibrium failure survive a structurally different integrator?

STATUS: OUTSIDE THE FROZEN CASE MATRIX
--------------------------------------
This script introduces an integrator that no scored case uses, so it cannot
change any frozen verdict. It is exploratory corroboration of the mechanism
claim, not an adjudicated case, and nothing here is reported in the paper
without a separate decision to do so.

WHY THIS EXISTS
---------------
The study pins one integrator -- scipy DOP853 at rtol 1e-10, atol 1e-12 -- for
every engine, so that no engine can be ranked on integration accuracy. That
pinning is load-bearing, but it leaves one question open about the central
positive finding: every engine and the reference report success while returning
a large trajectory for a system whose exact solution is no motion at all. Is
that a property of the DYNAMICS, or an artefact of the one integrator every
engine was driven through?

Drake 1.56 ships CENIC (Convex Error-controlled Numerical Integration for
Contact; Kurtz and Castro 2025, arXiv:2511.08771), an implicit error-controlled
integrator built on a convex Irrotational Contact Fields optimisation. It shares
no code, no author, no formulation and no error-control strategy with DOP853.

WHAT THE THREE MEASUREMENTS ESTABLISH
-------------------------------------
(1) ACCURACY INDEPENDENCE. Driven by an explicit method, the departure from the
    equilibrium is the same to three significant figures across six orders of
    magnitude of requested accuracy. A quantity that does not respond to
    integration tolerance is not integration error; it is the arithmetic
    mechanism of section 7.3.

(2) THE ULP CONTROL. Starting at exactly the representable pi, some stage
    combinations leave the state unchanged, because the per-step position
    increment falls below the ULP of pi (4.44e-16) and rounds away. That looks
    like immunity and is not: one ULP off pi, the runaway appears. Any claim
    that an integrator "does not show the failure" has to survive this control.

(3) THE RESOLUTION FLOOR. Sweeping the initial perturbation separates the two
    behaviours properly. An integrator that tracks the physics follows
    eps*cosh(lambda*T); one with an absolute resolution floor returns the
    perturbation unamplified below that floor.

NOTE ON COORDINATES
-------------------
`DrakeChain` takes ABSOLUTE angles measured from the downward vertical and
converts to Drake's relative joint coordinates internally. The inverted
equilibrium is every absolute angle = pi, i.e. relative q = [pi, 0, 0, ...].
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

from adapter_drake import DrakeChain
import reference

from pydrake.systems.analysis import Simulator, ResetIntegratorFromFlags

T_END = 10.0
N_SAMPLE = 50
PINNED_RTOL = 1e-10
PINNED_ATOL = 1e-12
PI = math.pi
ULP_PI = float(np.nextafter(PI, 4.0) - PI)      # 4.4409e-16
LAMBDA = math.sqrt(9.81)                        # unit length, unit mass


def _drake_run(theta0: float, scheme: str, accuracy: float,
               max_step: float = 1.0) -> tuple[float, int]:
    """Worst |theta - pi| over the horizon, and steps taken."""
    chain = DrakeChain(1)
    plant = chain.plant
    sim = Simulator(plant)
    ResetIntegratorFromFlags(sim, scheme, max_step)
    integ = sim.get_mutable_integrator()
    integ.set_target_accuracy(accuracy)
    ctx = sim.get_mutable_context()
    plant.SetPositions(ctx, [theta0])
    plant.SetVelocities(ctx, [0.0])
    sim.Initialize()
    worst = 0.0
    for k in range(N_SAMPLE + 1):
        sim.AdvanceTo(T_END * k / N_SAMPLE)
        worst = max(worst, abs(float(plant.GetPositions(ctx)[0]) - PI))
    return worst, int(integ.get_num_steps_taken())


def _pinned_run(theta0: float, which: str) -> float:
    """The study's pinned integrator, on the Drake plant or the reference."""
    from scipy.integrate import solve_ivp
    sysobj = DrakeChain(1) if which == "drake" else reference.NLinkChain(1)
    sol = solve_ivp(sysobj.rhs, (0.0, T_END), np.array([theta0, 0.0]),
                    t_eval=np.linspace(0.0, T_END, N_SAMPLE + 1),
                    method="DOP853", rtol=PINNED_RTOL, atol=PINNED_ATOL)
    return float(np.max(np.abs(sol.y[0, :] - PI)))


def measurement_1_accuracy_independence() -> list[dict]:
    print("(1) Is the departure integration error? Vary only the tolerance.\n")
    print(f"    {'scheme':<16}{'requested':>12}{'steps':>9}"
          f"{'worst departure':>20}")
    print("    " + "-" * 57)
    rows = []
    for scheme in ("runge_kutta3", "runge_kutta5", "cenic"):
        for acc in (1e-8, 1e-10, 1e-12, 1e-14):
            w, n = _drake_run(PI, scheme, acc)
            rows.append({"measurement": "accuracy_independence",
                         "scheme": scheme, "accuracy": acc,
                         "start": "pi", "steps": n, "worst": w})
            print(f"    {scheme:<16}{acc:>12.0e}{n:>9}{w:>20.4e}")
        print()
    return rows


def measurement_2_ulp_control() -> list[dict]:
    print(f"(2) The ULP control. ULP of pi = {ULP_PI:.4e}\n")
    starts = [("pi exactly", PI),
              ("pi + 1 ULP", float(np.nextafter(PI, 4.0))),
              ("pi - 1 ULP", float(np.nextafter(PI, 3.0))),
              ("pi + 10 ULP", PI + 10 * ULP_PI)]
    print(f"    {'start':<14}{'runge_kutta3':>16}{'runge_kutta5':>16}"
          f"{'cenic':>16}")
    print("    " + "-" * 62)
    rows = []
    for label, th in starts:
        cells = []
        for scheme in ("runge_kutta3", "runge_kutta5", "cenic"):
            w, n = _drake_run(th, scheme, 1e-10)
            rows.append({"measurement": "ulp_control", "scheme": scheme,
                         "start": label, "theta0": th, "worst": w})
            cells.append(f"{w:>16.4e}")
        print(f"    {label:<14}" + "".join(cells))
    print()
    return rows


def measurement_3_resolution_floor() -> list[dict]:
    cosh = math.cosh(LAMBDA * T_END)
    print(f"(3) Resolution floor. Exact linearised answer is "
          f"eps*cosh(lambda*T),\n    lambda = {LAMBDA:.4f}, "
          f"cosh(lambda*T) = {cosh:.3e}, saturating at a half turn.\n")
    print(f"    {'eps (rad)':>11}{'predicted':>14}{'runge_kutta5':>16}"
          f"{'cenic':>16}")
    print("    " + "-" * 57)
    rows = []
    for e in (1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3):
        pred = min(e * cosh, PI)
        r5, _ = _drake_run(PI + e, "runge_kutta5", 1e-10)
        cn, _ = _drake_run(PI + e, "cenic", 1e-10)
        rows.append({"measurement": "resolution_floor", "eps": e,
                     "predicted": pred, "runge_kutta5": r5, "cenic": cn})
        print(f"    {e:>11.0e}{pred:>14.4e}{r5:>16.4e}{cn:>16.4e}")
    print()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import importlib.metadata as md
    print("Inverted equilibrium under a structurally different integrator")
    print(f"  drake        : {md.version('drake')}")
    print(f"  system       : single pendulum, unit length and mass, g = 9.81")
    print(f"  horizon      : {T_END:g} s")
    print("  exact answer : started AT the equilibrium, nothing moves;")
    print("                 started eps away, it departs as eps*cosh(lambda t)\n")

    rows = []
    rows += [{"measurement": "pinned_baseline", "engine": e,
              "worst": _pinned_run(PI, e)} for e in ("reference", "drake")]
    print("(0) Pinned baseline at exactly pi (the study's integrator):")
    for r in rows:
        print(f"    {r['engine']:<12}DOP853 rtol 1e-10 "
              f"-> worst departure {r['worst']:.4e}")
    print()

    rows += measurement_1_accuracy_independence()
    rows += measurement_2_ulp_control()
    rows += measurement_3_resolution_floor()

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"drake": md.version("drake"), "horizon_s": T_END,
                       "ulp_pi": ULP_PI, "lambda": LAMBDA, "rows": rows},
                      f, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
