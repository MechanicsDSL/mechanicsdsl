"""
Library-independent reference for the cart-pole -- the third problem family.

WHY THIS FAMILY
---------------
The first two families both required a closed loop or an explicit constraint,
which is exactly where Drake's continuous-mode API stops being usable (see
finding_drake_constraint.py). That makes them poor ground for a three-way
comparison: one engine is disqualified for a reason unrelated to its dynamics.

The cart-pole is a TREE -- a prismatic cart carrying a revolute pole, no loop,
no constraint -- so every engine handles it in its ordinary mode with nothing
excluded. It is still structurally new to the study:

  * mixed joint types (prismatic + revolute) in one mechanism, which the
    pendulum chain does not have;
  * a genuinely coupled, configuration-dependent mass matrix, unlike the two
    spring systems whose mass matrices are constant;
  * its own geometric degeneracy, described below.

GEOMETRY AND DYNAMICS
---------------------
Cart of mass M slides on a frictionless horizontal track at position x. A pole
of length l carries a point mass m at its end, pinned to the cart, at angle th
from the DOWNWARD vertical. The mass sits at

    (x + l sin th,  -l cos th)

giving velocity (xdot + l thdot cos th, l thdot sin th) and

    T = 1/2 (M+m) xdot^2 + m l xdot thdot cos th + 1/2 m l^2 thdot^2
    V = -m g l cos th

so with q = (x, th),

    M(q) = [ M+m         m l cos th ]        F(q, qdot) = [  m l thdot^2 sin th ]
           [ m l cos th  m l^2      ]                     [ -m g l sin th       ]

and M(q) qddot = F. No constraint, no multipliers, no symbolic algebra.

THE DEGENERACY
--------------
    det M(q) = m l^2 (M + m sin^2 th)

which is bounded away from zero for a heavy cart but approaches zero as the
cart is made light AND the pole passes through vertical (th = 0 or pi). With
M/m = 1e-6 the determinant collapses by six orders of magnitude twice per
swing.

This is a different mechanism from the slider-crank's. There the effective
inertia collapsed because a transmission ratio vanished; here the mass matrix
itself becomes near-singular because a light cart cannot resist the pole's
horizontal reaction. Both are geometric rather than hand-built, and they
degenerate for unrelated reasons, which is the point of testing both.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

__all__ = ["CartPole", "MASS_RATIOS", "ANGLES"]

# Cart-to-pole mass ratio M/m, dialled from heavy cart to nearly massless.
MASS_RATIOS: List[float] = [10.0, 1.0, 1e-2, 1e-4, 1e-6]

# Initial pole angles, including near-vertical where the determinant collapses.
ANGLES: List[float] = [0.3, 1.0, 2.0, 3.0]


class CartPole:
    def __init__(self, mass_ratio: float = 1.0, m: float = 1.0,
                 l: float = 1.0, g: float = 9.81) -> None:
        self.m = float(m)
        self.M = float(mass_ratio) * self.m
        self.mass_ratio = float(mass_ratio)
        self.l = float(l)
        self.g = float(g)
        self.n = 2

    def mass_matrix(self, q) -> np.ndarray:
        th = q[1]
        c = math.cos(th)
        return np.array([[self.M + self.m, self.m * self.l * c],
                         [self.m * self.l * c, self.m * self.l ** 2]])

    def forcing(self, q, v) -> np.ndarray:
        th, thd = q[1], v[1]
        s = math.sin(th)
        return np.array([self.m * self.l * thd * thd * s,
                         -self.m * self.g * self.l * s])

    def det_mass(self, th: float) -> float:
        """m l^2 (M + m sin^2 th). Collapses for a light cart at th = 0, pi."""
        return self.m * self.l ** 2 * (self.M + self.m * math.sin(th) ** 2)

    def degeneracy(self) -> float:
        """Ratio of largest to smallest det M over a full swing."""
        ths = np.linspace(0.0, 2 * math.pi, 2001)
        vals = np.array([self.det_mass(t) for t in ths])
        return float(vals.max() / max(vals.min(), 1e-300))

    def accel(self, state) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        q, v = s[0::2], s[1::2]
        try:
            return np.linalg.solve(self.mass_matrix(q), self.forcing(q, v))
        except np.linalg.LinAlgError:
            raise RuntimeError(
                f"cart-pole mass matrix singular at th={q[1]}") from None

    def rhs(self, _t, state) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        out = np.empty_like(s)
        out[0::2] = s[1::2]
        out[1::2] = self.accel(s)
        return out

    def energy(self, state) -> float:
        s = np.asarray(state, dtype=float)
        q, v = s[0::2], s[1::2]
        return (0.5 * float(v @ self.mass_matrix(q) @ v)
                - self.m * self.g * self.l * math.cos(q[1]))

    def initial_state(self, th0: float = 1.0) -> np.ndarray:
        return np.array([0.0, 0.0, th0, 0.0])

    def __repr__(self) -> str:
        return f"CartPole(M/m={self.mass_ratio:g})"


# ===========================================================================
# Self-tests
# ===========================================================================

def _test_heavy_cart_limit() -> None:
    """As M/m -> infinity the cart cannot move and the pole is a pendulum."""
    c = CartPole(mass_ratio=1e8)
    for th in (0.3, 1.0, -0.7):
        a = c.accel(np.array([0.0, 0.0, th, 0.0]))
        want = -(c.g / c.l) * math.sin(th)
        assert abs(a[1] - want) < 1e-6, (th, a[1], want)
        assert abs(a[0]) < 1e-6, f"cart moved: {a[0]:.2e}"
    print("  [ok] heavy-cart limit reduces to the simple pendulum")


def _test_momentum_conservation() -> None:
    """No horizontal force acts, so total horizontal momentum is conserved."""
    from scipy.integrate import solve_ivp
    for mr in (10.0, 1.0, 1e-2):
        c = CartPole(mass_ratio=mr)
        y0 = c.initial_state(1.0)
        sol = solve_ivp(c.rhs, (0.0, 5.0), y0, method="DOP853",
                        rtol=1e-12, atol=1e-14)
        assert sol.success
        def px(y):
            x, xd, th, thd = y
            return (c.M + c.m) * xd + c.m * c.l * thd * math.cos(th)
        p0 = px(sol.y[:, 0])
        worst = max(abs(px(sol.y[:, i]) - p0)
                    for i in range(sol.y.shape[1]))
        assert worst < 1e-8, f"M/m={mr:g}: momentum drift {worst:.2e}"
    print("  [ok] horizontal momentum conserved to 1e-8 for M/m = 10, 1, 1e-2")


def _test_determinant_law() -> None:
    """det M = m l^2 (M + m sin^2 th), checked against numpy."""
    rng = np.random.default_rng(5)
    for mr in MASS_RATIOS:
        c = CartPole(mass_ratio=mr)
        for _ in range(50):
            th = rng.uniform(-math.pi, math.pi)
            got = float(np.linalg.det(c.mass_matrix([0.0, th])))
            want = c.det_mass(th)
            assert abs(got - want) <= 1e-9 * max(abs(want), 1e-12), (mr, th)
    c = CartPole(mass_ratio=1e-6)
    assert c.degeneracy() > 1e5, f"expected collapse, got {c.degeneracy():.1e}"
    print(f"  [ok] determinant law holds; at M/m=1e-6 det M varies "
          f"{c.degeneracy():.2e}x over a swing")


def _test_energy() -> None:
    from scipy.integrate import solve_ivp
    for mr in (1.0, 1e-4):
        for th0 in (1.0, 3.0):
            c = CartPole(mass_ratio=mr)
            y0 = c.initial_state(th0)
            E0 = c.energy(y0)
            sol = solve_ivp(c.rhs, (0.0, 10.0), y0, method="DOP853",
                            rtol=1e-11, atol=1e-13)
            assert sol.success, f"M/m={mr:g} th0={th0}: {sol.message}"
            drift = max(abs(c.energy(sol.y[:, i]) - E0)
                        for i in range(sol.y.shape[1])) / max(abs(E0), 1e-12)
            assert drift < 1e-7, f"M/m={mr:g} th0={th0}: drift {drift:.2e}"
    print("  [ok] energy conserved to 1e-7 including at M/m = 1e-4")


def main() -> int:
    print("reference_cartpole.py self-tests\n")
    _test_heavy_cart_limit()
    _test_determinant_law()
    _test_momentum_conservation()
    _test_energy()
    print("\nAll self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
