"""
The amplitude axis -- initial angle dialled from quiescent to inverted.

WHY THIS AXIS EXISTS
--------------------
`SCOPE.md` names four knobs, one of which is "degenerate starting positions --
links perfectly aligned or fully extended." No axis in `systems.py` implements
it. The chain's initial conditions are hardcoded to 0.3 rad for the first link
and 0.15 for the rest.

Measurement showed why that matters. At those angles a perturbation of the
N-link chain grows by a factor of only 2 to 22 over ten seconds: polynomial
growth, quasi-periodic motion, the system's easy regime. At 3.0 rad the same
perturbation grows by 1.2e8 (N=2) to 1.3e11 (N=3) -- genuine chaos. The suite
has been testing the chain where it is well behaved.

Amplitude is the cheapest hard knob available. It is a number in the initial
conditions, so it transcribes into every engine without modelling choices,
unlike `mass_ratio` and `near_singular` which do not survive translation into a
rigid-body engine (TR-2026-03 section 5). It is therefore the only axis besides
`dof` that is portable to all three engines by direct transcription.

RELATIONSHIP TO THE FREEZE
--------------------------
The frozen case matrix -- six axes, 55 cases -- is NOT modified. `systems.py`
is untouched and its `all_cases()` returns exactly what it returned at
commit a8dc2b2, so the citable MechanicsDSL baseline of 44/0/1/10 stands.

This axis is additive and declared before measurement. Under the governance
rule in the freeze record, harness growth is permitted when it adds
adjudication without revising existing verdicts, and is gated on the frozen
matrix still reproducing bit-identically. Adding an axis prospectively is not
the same act as dropping an inconvenient case after seeing its result; the
first is extension, the second is selection on the outcome.

WHAT IS EXPECTED TO BREAK
-------------------------
Nothing, on the evidence so far -- which is exactly why it is worth running.
Three candidate effects:

  * Energy drift should worsen with amplitude for any engine whose integration
    tolerance does not adapt to the problem. This is the mechanism behind
    FIXES.md item 5.
  * Near theta = pi the chain starts inverted, an unstable equilibrium. Any
    engine that special-cases equilibria may report a frozen trajectory.
  * At exactly theta = pi with zero velocity the system is AT an equilibrium
    and the exact solution does not move.

CORRECTION, AFTER MEASUREMENT
-----------------------------
The third expectation above was stated wrongly when this axis was written, and
is left in place with this correction rather than quietly edited.

It said: "an engine that reports motion there is wrong; an engine that reports
no motion is right." That is false. The equilibrium at theta = pi is UNSTABLE,
and sin(pi) is not zero in floating point -- it is 1.22e-16. The resulting
acceleration residual of ~1.2e-15 is then amplified by the instability into
full-scale motion (about 20 radians) within the ten-second horizon.

All three engines do this. So does the library-independent reference in
`reference.py`, which shares no code with any of them. It is therefore a
property of the problem in finite precision, not a defect in any engine: no
finite-precision computation can remain at an unstable equilibrium.

The case is still worth keeping, because what it exposes is real and shared:
every engine returns success and hands back a large trajectory where the exact
answer is no motion at all, and NONE of them warns that the initial condition
is an unstable equilibrium whose evolution is dominated by round-off. That is
the study's thesis appearing as a universal blind spot rather than as a
difference between engines -- which makes it a weaker result than a
disagreement, and an honest one.
"""

from __future__ import annotations

import os
import sys
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

import systems

# Dialled from the suite's own quiescent value to the inverted equilibrium.
AMPLITUDES: List[float] = [0.15, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.pi]

# Three links: enough to be chaotic at large amplitude, small enough that every
# engine returns within the wall clock on the Lagrangian pathway.
N_LINKS = 3


def amplitude_dsl(amp: float, N: int = N_LINKS) -> str:
    """The N-link chain with every link started at `amp` radians.

    Built by taking `systems.n_pendulum_dsl` and replacing only the initial
    conditions, so the Lagrangian is character-for-character the one already
    frozen and measured. Nothing about the system changes except where it
    starts.
    """
    base = systems.n_pendulum_dsl(N)
    ic = ", ".join(f"theta{i}={amp!r}, theta{i}_dot=0.0" for i in range(N))
    out = []
    for line in base.splitlines():
        out.append(r"\initial{%s}" % ic if line.startswith(r"\initial")
                   else line)
    return "\n".join(out)


def all_cases() -> List[dict]:
    cases = []
    for level, amp in enumerate(AMPLITUDES):
        cases.append(dict(
            axis="amplitude", level=level, knob=float(amp),
            name=f"amp_{amp:.3f}".rstrip("0").rstrip("."),
            dsl=amplitude_dsl(amp), tools=["lagrangian", "hamiltonian"],
            formulation="unconstrained", conservative=True,
            # At exactly pi the chain starts at the inverted equilibrium with
            # zero velocity: the correct answer is that it does not move.
            expected_moving=(abs(amp - np.pi) > 1e-12),
            t_span=[0.0, 10.0], num_points=1500,
            N=N_LINKS))
    return cases


def initial_state(amp: float, N: int = N_LINKS) -> np.ndarray:
    """Interleaved [theta, thetadot] with every link at `amp`, at rest."""
    y0 = np.zeros(2 * N)
    y0[0::2] = amp
    return y0


if __name__ == "__main__":
    print(f"amplitude axis: {len(AMPLITUDES)} levels on a "
          f"{N_LINKS}-link chain\n")
    for c in all_cases():
        print(f"  {c['name']:<12} amp={c['knob']:.4f} rad  "
              f"expected_moving={c['expected_moving']}")
    print("\nLagrangian is identical to the frozen dof_N3 case; only the "
          "initial conditions differ.")
