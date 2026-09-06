"""
Does the shared failure generalise beyond the pendulum chain?

WHY THIS EXISTS
---------------
The study's one positive finding rests on a single phenomenon in a single
family: at the chain's inverted equilibrium every engine and the reference
report success while returning a large trajectory whose exact solution is no
motion. One instance is a curiosity. If the same thing happens in a
structurally different mechanism, it is a class, and the paper's central claim
-- that differential testing is blind to shared failures -- rests on firmer
ground.

The cart-pole provides the test. It is a tree rather than a chain, carries mixed
joint types, and has a genuinely coupled configuration-dependent mass matrix.
Its pole-up configuration (theta = pi from the downward vertical, zero
velocities, cart at rest) is an unstable equilibrium of a different mechanism:
here the cart is free to recoil, so the instability couples two coordinates
rather than one.

THE TEST CAN FAIL
-----------------
If some engine stays at rest -- because its particular arithmetic happens to
produce an exact zero -- then the failure is NOT universal, and the paper's
claim would need narrowing to the chain. That outcome is as reportable as the
other.

Run in WSL, where all three engines import in one process:

    PYTHONPATH=../src ~/drake-venv/bin/python shared_failure_second_family.py
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

HORIZON = 10.0
RTOL, ATOL = 1e-10, 1e-12
MASS_RATIOS = (10.0, 1.0, 1e-2)


def integrate(accel_fn, y0, n):
    """Integrate with the study's pinned scheme, given an accelerations map."""
    def rhs(_t, y):
        out = np.empty_like(y)
        out[0::2] = y[1::2]
        out[1::2] = np.asarray(accel_fn(y), float).reshape(n)
        return out
    sol = solve_ivp(rhs, (0.0, HORIZON), y0, method="DOP853",
                    rtol=RTOL, atol=ATOL, dense_output=False)
    return sol


def main() -> int:
    import reference_cartpole as RCP
    import sweep_families as SF

    print("Unstable equilibrium in a second family: the cart-pole, pole up\n")
    print("  configuration : theta = pi (pole vertical), cart and pole at rest")
    print("  exact solution: nothing moves, for all time")
    print(f"  horizon       : {HORIZON}s, DOP853 rtol={RTOL:g} atol={ATOL:g}\n")
    print(f"{'M/m':>8}  {'engine':<14}{'reported':>10}{'max |dtheta|':>15}"
          f"{'max |dx|':>13}")
    print("-" * 62)

    rows = []
    for mr in MASS_RATIOS:
        cp = RCP.CartPole(mass_ratio=mr)
        th0 = math.pi
        y0 = np.array([0.0, 0.0, th0, 0.0])

        participants = [("reference", cp.accel)]
        for name, factory in SF.CARTPOLE_ENGINES:
            try:
                fn = factory(cp, th0)
                participants.append((name, fn[0] if isinstance(fn, tuple) else fn))
            except Exception as e:
                print(f"{mr:>8g}  {name:<14}  build failed: "
                      f"{type(e).__name__}: {str(e)[:32]}")

        for name, fn in participants:
            try:
                sol = integrate(fn, y0.copy(), 2)
                ok = bool(sol.success)
                dth = float(np.max(np.abs(sol.y[2] - th0)))
                dx = float(np.max(np.abs(sol.y[0])))
            except Exception as e:
                print(f"{mr:>8g}  {name:<14}  integration error: "
                      f"{type(e).__name__}")
                continue
            rows.append({"mass_ratio": mr, "engine": name, "success": ok,
                         "max_dtheta": dth, "max_dx": dx})
            print(f"{mr:>8g}  {name:<14}{('success' if ok else 'FAILED'):>10}"
                  f"{dth:>15.3e}{dx:>13.3e}", flush=True)
        print()

    moved = [r for r in rows if r["max_dtheta"] > 1e-3]
    stayed = [r for r in rows if r["max_dtheta"] <= 1e-3]
    all_success = all(r["success"] for r in rows)

    print(f"  participants that moved substantially : {len(moved)}/{len(rows)}")
    print(f"  participants that stayed at rest      : {len(stayed)}/{len(rows)}")
    print(f"  all reported success                  : {all_success}")
    if len(stayed) == 0 and all_success:
        print("\n  VERDICT: the failure is NOT specific to the chain. Every")
        print("           implementation, including the reference, reports success")
        print("           while returning motion the exact solution does not have.")
    else:
        print("\n  VERDICT: behaviour DIFFERS across implementations here --")
        print("           the claim must be narrowed. See the table above.")

    out = os.path.join(HERE, "out", "shared_failure_cartpole.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
