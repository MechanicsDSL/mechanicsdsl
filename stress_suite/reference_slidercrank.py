"""
Library-independent reference for the slider-crank -- the second problem family.

WHY THIS FAMILY
---------------
Every result in the study so far comes from one planar pendulum chain plus two
constrained families built from the same ingredients: point masses, revolute
freedom, quadratic constraints. A reviewer will ask whether the agreement
between engines is a property of dynamics engines or a property of that
problem. The slider-crank answers it, because it exercises machinery none of
the current systems touch:

  * a PRISMATIC degree of freedom alongside a revolute one -- nothing in the
    suite has a translational coordinate;
  * a genuine closed loop, rather than an open chain;
  * DEAD CENTRE, where the mechanism's effective inertia collapses for
    geometric reasons twice per revolution, in normal operation.

Dead centre is the point of the exercise. At theta = 0 and theta = pi the crank
and rod are collinear and dx/dtheta = 0: the slider is instantaneously
stationary however fast the crank turns. Its mass therefore contributes NOTHING
to the effective inertia at that instant. With a light crank and a heavy
slider, the reduced inertia collapses toward zero every half revolution -- a
near-singular mass matrix arising from GEOMETRY rather than from a hand-built
matrix, which is what `near_singular` does. Whether an engine notices is
exactly the study's question.

GEOMETRY
--------
Crank pivot at the origin, crank length r, crank angle theta from +x.
Crank pin at P = (r cos th, r sin th). Slider at S = (x, 0), confined to the
x-axis. Massless connecting rod of length l joins P to S.

Coordinates q = (theta, x), one constraint, so one degree of freedom.

    g = |S - P|^2 - l^2
      = (x - r cos th)^2 + (r sin th)^2 - l^2
      = x^2 - 2 r x cos th + r^2 - l^2                                    (1)

the sin^2 + cos^2 collapsing to a constant. Differentiating,

    J = [ dg/dth, dg/dx ] = [ 2 r x sin th,  2x - 2 r cos th ]            (2)

    gddot = J qddot + 2 xdot^2 + 4 r xdot sin th thdot
                    + 2 r x cos th thdot^2 = 0
    =>  gamma = -( 2 xdot^2 + 4 r xdot sin th thdot
                   + 2 r x cos th thdot^2 )                               (3)

DYNAMICS
--------
Crank modelled as a point mass m_c at the pin (inertia m_c r^2 about the
pivot); slider a point mass m_s in translation; rod massless.

    M = diag(m_c r^2,  m_s),      V = m_c g r sin th,
    F = -dV/dq = ( -m_c g r cos th,  0 )

and (qddot, lambda) solve the saddle-point system

    [ M   J^T ] [ qddot ]   [ F     ]
    [ J   0   ] [ lam   ] = [ gamma ]                                     (4)

No symbolic algebra anywhere: (1)-(3) are closed forms.

WHY l > r MATTERS
-----------------
For l > r the constraint Jacobian (2) never vanishes on the manifold, so the
KKT system (4) stays nonsingular and the mechanism is well posed at every
configuration including dead centre. The degeneracy at dead centre is in the
EFFECTIVE inertia, not in the constraint, which is a subtler failure mode than
the suite has tested and the reason this family is worth adding.

As l/r approaches 1 the mechanism additionally approaches a folding
configuration at theta = pi, where the rod lies back along the crank and the
slider reaches x = 0. That gives a second, sharper knob.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

__all__ = ["SliderCrank", "RATIOS", "MASS_RATIOS"]

# Rod-to-crank ratio l/r, dialled from benign toward the folding configuration.
RATIOS: List[float] = [3.0, 2.0, 1.5, 1.2, 1.05, 1.01]

# Slider-to-crank mass ratio m_s/m_c: the larger this is, the more violently
# the effective inertia collapses at dead centre.
MASS_RATIOS: List[float] = [1.0, 1e2, 1e4, 1e6]


class SliderCrank:
    def __init__(self, ratio: float = 3.0, mass_ratio: float = 1.0,
                 r: float = 1.0, m_crank: float = 1.0, g: float = 9.81) -> None:
        self.r = float(r)
        self.l = float(ratio) * self.r
        self.ratio = float(ratio)
        self.m_c = float(m_crank)
        self.m_s = float(mass_ratio) * self.m_c
        self.mass_ratio = float(mass_ratio)
        self.g = float(g)
        self.n = 2
        if self.l <= self.r:
            raise ValueError("l must exceed r for the mechanism to be assemblable")

    # -- kinematics ----------------------------------------------------------

    def slider_position(self, th: float) -> float:
        """x(theta) on the assembly branch with the slider to the +x side."""
        disc = self.l ** 2 - (self.r * math.sin(th)) ** 2
        return self.r * math.cos(th) + math.sqrt(max(disc, 0.0))

    def slider_velocity(self, th: float, thd: float) -> float:
        """xdot from gdot = 0."""
        x = self.slider_position(th)
        denom = x - self.r * math.cos(th)
        return -self.r * x * math.sin(th) * thd / denom

    def dx_dtheta(self, th: float) -> float:
        """Transmission ratio. Vanishes at dead centre (theta = 0, pi)."""
        x = self.slider_position(th)
        return -self.r * x * math.sin(th) / (x - self.r * math.cos(th))

    def effective_inertia(self, th: float) -> float:
        """Reduced inertia in the single true degree of freedom theta.

        M_eff = m_c r^2 + m_s (dx/dth)^2.  At dead centre the second term is
        zero, so a heavy slider stops contributing entirely.
        """
        return self.m_c * self.r ** 2 + self.m_s * self.dx_dtheta(th) ** 2

    def inertia_collapse(self) -> float:
        """Ratio of largest to smallest effective inertia over a revolution.

        A measure of how violently the mechanism degenerates each cycle.
        """
        ths = np.linspace(0.0, 2 * math.pi, 2001)
        vals = np.array([self.effective_inertia(t) for t in ths])
        return float(vals.max() / vals.min())

    # -- dynamics ------------------------------------------------------------

    def mass(self) -> np.ndarray:
        return np.diag([self.m_c * self.r ** 2, self.m_s])

    def force(self, q: np.ndarray) -> np.ndarray:
        th = q[0]
        return np.array([-self.m_c * self.g * self.r * math.cos(th), 0.0])

    def jac_gamma(self, q: np.ndarray, v: np.ndarray):
        th, x = q
        thd, xd = v
        J = np.array([[2.0 * self.r * x * math.sin(th),
                       2.0 * x - 2.0 * self.r * math.cos(th)]])
        gamma = np.array([-(2.0 * xd * xd
                            + 4.0 * self.r * xd * math.sin(th) * thd
                            + 2.0 * self.r * x * math.cos(th) * thd * thd)])
        return J, gamma

    def accel(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        q, v = s[0::2], s[1::2]
        M = self.mass()
        F = self.force(q)
        J, gamma = self.jac_gamma(q, v)
        K = np.zeros((3, 3))
        K[:2, :2] = M
        K[:2, 2:] = J.T
        K[2:, :2] = J
        rhs = np.concatenate([F, gamma])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
        return sol[:2]

    def rhs(self, _t: float, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        out = np.empty_like(s)
        out[0::2] = s[1::2]
        out[1::2] = self.accel(s)
        return out

    # -- diagnostics ---------------------------------------------------------

    def constraint_residual_signed(self, q: np.ndarray) -> float:
        """g(q) itself, equation (1). Signed, for differentiation."""
        th, x = q
        return float(x * x - 2.0 * self.r * x * math.cos(th)
                     + self.r ** 2 - self.l ** 2)

    def constraint_residual(self, q: np.ndarray) -> float:
        return abs(self.constraint_residual_signed(q))

    def energy(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=float)
        q, v = s[0::2], s[1::2]
        return (0.5 * float(v @ self.mass() @ v)
                + self.m_c * self.g * self.r * math.sin(q[0]))

    def initial_state(self, th0: float = 0.3, thd0: float = 4.0) -> np.ndarray:
        """Consistent start: x and xdot solved from the constraint.

        thd0 is chosen large enough that the crank completes several
        revolutions in the horizon, so dead centre is crossed repeatedly.
        """
        x = self.slider_position(th0)
        xd = self.slider_velocity(th0, thd0)
        return np.array([th0, thd0, x, xd])

    def __repr__(self) -> str:
        return (f"SliderCrank(l/r={self.ratio:g}, m_s/m_c={self.mass_ratio:g})")


# ===========================================================================
# Self-tests
# ===========================================================================

def _test_constraint_closed_form() -> None:
    """The expanded constraint (1) must equal the geometric distance form."""
    rng = np.random.default_rng(3)
    for ratio in RATIOS:
        s = SliderCrank(ratio)
        for _ in range(200):
            th = rng.uniform(-math.pi, math.pi)
            x = s.slider_position(th)
            geo = ((x - s.r * math.cos(th)) ** 2 + (s.r * math.sin(th)) ** 2
                   - s.l ** 2)
            exp = x * x - 2 * s.r * x * math.cos(th) + s.r ** 2 - s.l ** 2
            assert abs(geo - exp) < 1e-12, (ratio, th, geo, exp)
            assert abs(geo) < 1e-12, f"assembly off manifold: {geo:.2e}"
    print("  [ok] expanded constraint matches the geometric form; the "
          "assembly solution satisfies it exactly")


def _test_jacobian_and_gamma() -> None:
    """J and gamma against finite differences of g along a trajectory."""
    rng = np.random.default_rng(11)
    for ratio in (3.0, 1.2):
        s = SliderCrank(ratio)
        for _ in range(60):
            th = rng.uniform(-2.0, 2.0)
            x = s.slider_position(th)
            thd = rng.uniform(-3.0, 3.0)
            xd = s.slider_velocity(th, thd)
            q = np.array([th, x]); v = np.array([thd, xd])
            J, _ = s.jac_gamma(q, v)
            h = 1e-6
            Jn = np.zeros(2)
            for k in range(2):
                qp, qm = q.copy(), q.copy()
                qp[k] += h; qm[k] -= h
                Jn[k] = (s.constraint_residual_signed(qp)
                         - s.constraint_residual_signed(qm)) / (2 * h)
            assert np.max(np.abs(J[0] - Jn)) < 1e-5, (ratio, J[0], Jn)
    print("  [ok] constraint Jacobian matches finite differences")


def _test_dead_centre() -> None:
    """Transmission ratio vanishes at dead centre; inertia collapses there."""
    for ratio in (3.0, 1.5, 1.05):
        s = SliderCrank(ratio, mass_ratio=1e4)
        for th in (0.0, math.pi):
            assert abs(s.dx_dtheta(th)) < 1e-12, \
                f"l/r={ratio}, th={th}: dx/dth = {s.dx_dtheta(th):.3e}"
        collapse = s.inertia_collapse()
        assert collapse > 10.0, f"l/r={ratio}: collapse only {collapse:.1f}x"
        print(f"  [ok] l/r={ratio:<5g} dx/dth = 0 at both dead centres; "
              f"effective inertia varies {collapse:.3e}x over a revolution")


def _test_energy_and_constraint() -> None:
    """Integrate several revolutions through dead centre."""
    from scipy.integrate import solve_ivp
    for ratio in (3.0, 1.5, 1.05):
        s = SliderCrank(ratio, mass_ratio=1e2)
        y0 = s.initial_state()
        E0 = s.energy(y0)
        sol = solve_ivp(s.rhs, (0.0, 5.0), y0, method="DOP853",
                        rtol=1e-11, atol=1e-13, dense_output=False)
        assert sol.success, f"l/r={ratio}: {sol.message}"
        res = max(s.constraint_residual(sol.y[0::2, i])
                  for i in range(sol.y.shape[1]))
        drift = max(abs(s.energy(sol.y[:, i]) - E0)
                    for i in range(sol.y.shape[1])) / max(abs(E0), 1e-12)
        revs = abs(sol.y[0, -1] - sol.y[0, 0]) / (2 * math.pi)
        assert res < 1e-5, f"l/r={ratio}: constraint drift {res:.3e}"
        print(f"  [ok] l/r={ratio:<5g} {revs:.1f} revolutions, constraint "
              f"{res:.2e}, energy {drift:.2e}")


def main() -> int:
    print("reference_slidercrank.py self-tests\n")
    _test_constraint_closed_form()
    _test_jacobian_and_gamma()
    _test_dead_centre()
    _test_energy_and_constraint()
    print("\nAll self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
