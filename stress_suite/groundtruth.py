"""
Independent ground-truth equations of motion via SymPy's physics.mechanics.

This is a SEPARATE derivation path from MechanicsDSL: given the same Lagrangian
(and optional holonomic constraints) expressed in MechanicsDSL's symbol basis
{q, q_dot}, it uses sympy.physics.mechanics.LagrangesMethod -- a well-tested,
independently-maintained code path -- to produce the mass matrix and forcing.

We deliberately do NOT invert the mass matrix symbolically (that is exactly the
step that blows up in MechanicsDSL). Instead we keep M(q) and f(q, q_dot)
symbolic, lambdify them, and at each numeric probe state solve M a = f with
numpy. This lets the oracle scale far past the point where MechanicsDSL's own
symbolic solve times out, so we can check correctness on systems MechanicsDSL
can still handle.

The public entry point, build_truth(), returns an object with .accel(state)
giving the true coordinate accelerations at a state vector laid out as
[q0, q0_dot, q1, q1_dot, ...] -- the same layout MechanicsDSL's solver uses.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import sympy as sp
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols


class Truth:
    def __init__(self, accel: Callable[[np.ndarray], np.ndarray], n: int,
                 singular_ok: bool):
        self._accel = accel
        self.n = n
        self.singular_ok = singular_ok

    def accel(self, state: np.ndarray) -> np.ndarray:
        return self._accel(np.asarray(state, dtype=float))


def build_truth(
    lagrangian: sp.Expr,
    coordinates: List[str],
    parameters: Dict[str, float],
    constraints: Optional[List[sp.Expr]] = None,
) -> Truth:
    """Build the ground-truth acceleration function.

    Args:
        lagrangian: Lagrangian in MechanicsDSL symbols (q, q_dot as Symbol names).
        coordinates: ordered coordinate names.
        parameters: numeric parameter values.
        constraints: optional holonomic constraints g(q) = 0 in q symbols.
    """
    t = dynamicsymbols._t
    n = len(coordinates)

    # Map MechanicsDSL's flat symbols onto time-dependent dynamicsymbols.
    q_dyn = [dynamicsymbols(f"gtq{i}") for i in range(n)]
    subs = {}
    for i, name in enumerate(coordinates):
        subs[sp.Symbol(name, real=True)] = q_dyn[i]
        subs[sp.Symbol(f"{name}_dot", real=True)] = q_dyn[i].diff(t)

    L = lagrangian.subs(subs)
    # Substitute numeric parameters (anything left that is a plain Symbol).
    param_subs = {sp.Symbol(k, real=True): v for k, v in parameters.items()}
    L = L.subs(param_subs)

    hol = None
    if constraints:
        hol = [c.subs(subs).subs(param_subs) for c in constraints]

    if hol:
        lm = LagrangesMethod(L, q_dyn, hol_coneqs=hol)
    else:
        lm = LagrangesMethod(L, q_dyn)
    lm.form_lagranges_equations()

    # Full mass matrix / forcing include Lagrange-multiplier rows for the
    # constrained case. Coordinate accelerations are the first n unknowns.
    M = lm.mass_matrix_full
    F = lm.forcing_full

    # Unknown vector ordering from mechanics: [q', q'', lambda].
    # forcing_full / mass_matrix_full solve for d/dt of the state = [q, q']
    # plus multipliers. We want q'' = the derivative of the velocity block.
    # Build symbol lists for lambdify: positions and velocities.
    q_syms = q_dyn
    u_syms = [qi.diff(t) for qi in q_dyn]

    Mf = sp.lambdify([q_syms, u_syms], M, "numpy")
    Ff = sp.lambdify([q_syms, u_syms], F, "numpy")

    dim = M.shape[0]

    def accel(state: np.ndarray) -> np.ndarray:
        qv = state[0::2][:n]
        uv = state[1::2][:n]
        Mn = np.array(Mf(list(qv), list(uv)), dtype=float).reshape(dim, dim)
        Fn = np.array(Ff(list(qv), list(uv)), dtype=float).reshape(dim)
        try:
            sol = np.linalg.solve(Mn, Fn)
        except np.linalg.LinAlgError:
            # Redundant/rank-deficient constraints leave the Lagrange
            # multipliers undetermined but the coordinate accelerations
            # unique; least-squares recovers the min-norm solution whose
            # coordinate block is still correct.
            sol, *_ = np.linalg.lstsq(Mn, Fn, rcond=None)
        # State vector for mass_matrix_full is [q, u]; its derivative is
        # [u, u'] (+ multipliers). The velocity-derivative block (u') sits in
        # rows n..2n and equals the coordinate accelerations.
        return sol[n:2 * n]

    singular_ok = False
    return Truth(accel, n, singular_ok)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple pendulum: theta'' = -(g/l) sin(theta).
    th = sp.Symbol("theta0", real=True)
    thd = sp.Symbol("theta0_dot", real=True)
    m, l, g = 1.0, 1.0, 9.81
    L = sp.Rational(1, 2) * m * l**2 * thd**2 - m * g * l * (1 - sp.cos(th))
    tr = build_truth(L, ["theta0"], {"m": m, "l": l, "g": g})
    for theta in (0.3, 1.0, -0.7):
        got = tr.accel(np.array([theta, 0.2]))[0]
        exp = -(g / l) * np.sin(theta)
        print(f"pendulum theta={theta:+.2f}  truth={got:+.5f}  exact={exp:+.5f}  "
              f"{'OK' if abs(got - exp) < 1e-9 else 'MISMATCH'}")

    # Double pendulum sanity: just confirm it builds and returns 2 accels.
    t0, t1 = sp.symbols("theta0 theta1", real=True)
    d0, d1 = sp.symbols("theta0_dot theta1_dot", real=True)
    L2 = (0.5 * 2 * l**2 * d0**2 + 0.5 * l**2 * d1**2
          + l**2 * sp.cos(t0 - t1) * d0 * d1
          + 2 * g * l * sp.cos(t0) + g * l * sp.cos(t1))
    tr2 = build_truth(L2, ["theta0", "theta1"], {"m": m, "l": l, "g": g})
    a = tr2.accel(np.array([0.3, 0.0, 0.15, 0.0]))
    print("double-pendulum accel at rest-ish:", a, "(finite:", np.all(np.isfinite(a)), ")")
