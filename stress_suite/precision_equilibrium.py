"""
Does the shared failure at an unstable equilibrium depend on binary64?

WHY THIS EXISTS
---------------
The paper's central positive finding is that every engine, and the independent
reference, reports success while returning a large trajectory for a system whose
exact solution is no motion at all. The explanation offered is arithmetic: pi is
not representable in binary64, sin(pi) evaluates to 1.2246e-16 rather than zero,
and an unstable equilibrium amplifies that residual exponentially.

That argument has an alternative the paper does not exclude. Every engine AND
the reference ran at the same precision on the same hardware, so a reader may
reasonably ask whether the failure is shared because the PHENOMENON is
structural or merely because the SUBSTRATE is. Those predict different things,
and the difference is measurable.

  substrate  -- the failure is an artefact of binary64; enough precision
                removes it.
  structural -- no finite precision removes it; precision only delays it, and
                the delay is logarithmic in the working precision.

THE PREDICTION
--------------
Near the inverted equilibrium of a simple pendulum, write phi = theta - pi. Then
sin(theta) = -sin(phi) ~= -phi, so

    phi'' = (g/l) phi,     lambda = sqrt(g/l) = 3.1321 s^-1

The initial residual is eps ~ 10^-p at p working digits, so the time to reach a
fixed threshold grows as

    t_divergence ~= ln(threshold / eps) / lambda ~= (p ln 10) / lambda

i.e. LINEAR in the number of digits, with slope ln(10)/lambda ~= 0.735 s per
digit, and never infinite. This script measures that line.

METHOD
------
The perturbation is integrated directly. Writing phi = theta - pi turns

    theta'' = -(g/l) sin(theta)      into      phi'' = (g/l) sin(phi)

which is the same dynamics with the equilibrium moved to the origin. This
matters for the measurement rather than the physics: integrating theta itself
near pi adds increments of order 10^-p to a value of order 3.14, and below about
p significant digits those increments vanish on addition. That makes the
equilibrium artificially sticky and, measured that way, 25 digits appears to
diverge EARLIER than 16 -- an impossibility that reveals the harness rather than
the phenomenon. Integrating phi removes the cancellation and leaves the question
being asked intact.

The initial perturbation is the residual any solver actually meets at this
precision, |sin(pi_p)|, with zero initial velocity. Fixed-step RK4 in mpmath, no
engine involved: this is a property of the problem.

Run:
    python precision_equilibrium.py
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mpmath import mp, mpf, sin as mp_sin  # noqa: E402

G, L = 9.81, 1.0
THRESHOLD = 0.1          # radians from the equilibrium that counts as diverged
STEP = mpf("0.005")
T_MAX = 200.0
PRECISIONS = (16, 25, 35, 50, 75, 100, 150)


def divergence_time(digits: int) -> dict:
    """Grow the residual perturbation at `digits` working precision."""
    mp.dps = digits
    g, l = mpf(G), mpf(L)
    lam = mp.sqrt(g / l)

    # The residual a solver actually meets: sin(pi) does not vanish at any
    # finite precision, and this is what seeds the divergence.
    residual = abs(mp_sin(mp.pi))

    phi, omega = residual, mpf(0)          # perturbation from the equilibrium

    def accel(p_):
        return (g / l) * mp_sin(p_)        # phi'' = (g/l) sin(phi)

    t, h, thresh = mpf(0), STEP, mpf(THRESHOLD)
    t_div = None
    while t < T_MAX:
        if abs(phi) > thresh:
            t_div = float(t)
            break
        k1p, k1w = omega, accel(phi)
        k2p, k2w = omega + h * k1w / 2, accel(phi + h * k1p / 2)
        k3p, k3w = omega + h * k2w / 2, accel(phi + h * k2p / 2)
        k4p, k4w = omega + h * k3w, accel(phi + h * k3p)
        phi += h * (k1p + 2 * k2p + 2 * k3p + k4p) / 6
        omega += h * (k1w + 2 * k2w + 2 * k3w + k4w) / 6
        t += h

    # phi(t) = phi_0 cosh(lambda t) while linear, so t = arccosh(thresh/phi_0)/lambda
    predicted = (float(mp.acosh(thresh / residual) / lam)
                 if residual > 0 else float("inf"))
    return {"digits": digits,
            "residual": float(residual) if residual > 0 else 0.0,
            "t_divergence": t_div,
            "t_predicted": predicted,
            "lambda": float(lam),
            "diverged": t_div is not None}


def main() -> int:
    print("Does the equilibrium failure depend on binary64?\n")
    print(f"  system     : simple pendulum, inverted, g={G} l={L}")
    print(f"  lambda     : {math.sqrt(G / L):.4f} 1/s  "
          f"(predicted slope {math.log(10) / math.sqrt(G / L):.3f} s/digit)")
    print(f"  threshold  : {THRESHOLD} rad, RK4 step {float(STEP)}, "
          f"perturbation coordinates\n")
    print(f"{'digits':>7}{'residual':>13}{'t_diverge':>12}"
          f"{'predicted':>11}  verdict")
    print("-" * 60)

    rows = []
    for p in PRECISIONS:
        r = divergence_time(p)
        rows.append(r)
        td = f"{r['t_divergence']:.2f}s" if r["diverged"] else f">{T_MAX:.0f}s"
        print(f"{p:>7}{r['residual']:>13.2e}{td:>12}"
              f"{r['t_predicted']:>10.2f}s  "
              f"{'DIVERGED' if r['diverged'] else 'no divergence'}", flush=True)

    got = [r for r in rows if r["diverged"]]
    print()
    monotone = all(got[i]["t_divergence"] <= got[i + 1]["t_divergence"]
                   for i in range(len(got) - 1))
    print(f"  monotone in precision: {monotone}")
    if len(got) >= 2:
        d0, d1 = got[0], got[-1]
        slope = ((d1["t_divergence"] - d0["t_divergence"])
                 / (d1["digits"] - d0["digits"]))
        print(f"  measured slope : {slope:.3f} s per digit")
        print(f"  predicted slope: {math.log(10) / math.sqrt(G / L):.3f} "
              "s per digit  (ln 10 / lambda)")
    print(f"  precisions at which the failure vanished: "
          f"{sum(1 for r in rows if not r['diverged'])} of {len(rows)}")

    out = os.path.join(HERE, "out", "precision_equilibrium.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
