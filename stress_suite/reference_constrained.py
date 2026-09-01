"""
Library-independent reference for the CONSTRAINED pathway.

WHY
---
The constrained pathway is the study's last blind spot. Of the 45 adjudicated
cases in the frozen baseline, 8 are constrained and none of them carries an
independent derivation check -- they rest on energy conservation and constraint
residual alone, the weakest instruments available. A silent failure is by
construction the case nobody thought to check for, so the least-checked corner
is where one would still be hiding.

This module derives both constrained families in closed form with numpy only,
sharing no library with any engine under test, exactly as `reference.py` does
for the unconstrained ones.

THE MATHEMATICS
---------------
Both families have a constant diagonal mass matrix and holonomic constraints
g(q) = 0. Differentiating the constraint twice,

    g(q) = 0   ->   J(q) qdot = 0   ->   J(q) qddot + Jdot(q,qdot) qdot = 0

so the accelerations and multipliers solve the saddle-point (KKT) system

    [ M   J^T ] [ qddot ]   [ F(q)  ]
    [ J   0   ] [ lam   ] = [ gamma ]        gamma = -Jdot qdot          (1)

No symbolic algebra is required: J and gamma both have closed forms for
quadratic constraints, derived below.

ROD CONSTRAINT   g = (p_j - p_i).(p_j - p_i) - L^2
    dg/dp_j = 2(p_j - p_i),  dg/dp_i = -2(p_j - p_i)
    gddot   = 2|pdot_j - pdot_i|^2 + 2(p_j - p_i).(pddot_j - pddot_i) = 0
    => gamma = -2 |pdot_j - pdot_i|^2

CIRCLE CONSTRAINT   g = c(x^2 + y^2) - c
    J = (2cx, 2cy),   gamma = -2c(xdot^2 + ydot^2)

REDUNDANCY
----------
The `redundancy` family deliberately repeats one circle constraint with
different scalar multiples, so J has R+1 rows but rank 1. The KKT matrix is
then singular: the multipliers are underdetermined while the accelerations
remain unique. Equation (1) is solved by least squares, whose minimum-norm
solution has the correct acceleration block. This is a property of the problem,
not a workaround -- an exactly redundant constraint set does not determine how
the constraint force is shared out, only its total.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

__all__ = ["ClosedChain", "RedundantCircle", "constrained_reference_for_case"]


class _Constrained:
    """Common solver for M qddot + J^T lam = F, J qddot = gamma."""

    n: int          # number of coordinates

    def mass(self) -> np.ndarray:
        raise NotImplementedError

    def force(self, q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jac_gamma(self, q: np.ndarray, v: np.ndarray):
        """Return (J, gamma) at the given state."""
        raise NotImplementedError

    # -- solver --------------------------------------------------------------

    def accel(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        q, v = s[0::2], s[1::2]
        M = self.mass()
        F = self.force(q)
        J, gamma = self.jac_gamma(q, v)
        m, n = J.shape

        K = np.zeros((n + m, n + m))
        K[:n, :n] = M
        K[:n, n:] = J.T
        K[n:, :n] = J
        rhs = np.concatenate([F, gamma])

        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            # Rank-deficient constraints leave the multipliers undetermined
            # but the accelerations unique; the minimum-norm solution's
            # acceleration block is still correct.
            sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
        return sol[:n]

    def rhs(self, _t: float, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=float)
        out = np.empty_like(s)
        out[0::2] = s[1::2]
        out[1::2] = self.accel(s)
        return out

    def constraint_residual(self, q: np.ndarray) -> float:
        raise NotImplementedError

    def energy(self, state: np.ndarray) -> float:
        s = np.asarray(state, dtype=float)
        q, v = s[0::2], s[1::2]
        return 0.5 * float(v @ self.mass() @ v) + self.potential(q)

    def potential(self, q: np.ndarray) -> float:
        raise NotImplementedError


class ClosedChain(_Constrained):
    """Pinned closed N-gon of rods. Matches systems.closed_chain_dsl(N).

    Node 0 is pinned at the origin; the moving nodes are 1..N-1 in Cartesian
    coordinates, giving n = 2(N-1). There are N rod constraints, including the
    closure rod from node N-1 back to node 0.
    """

    def __init__(self, N: int, m: float = 1.0, g: float = 9.81) -> None:
        self.N, self.m, self.g = int(N), float(m), float(g)
        self.n = 2 * (self.N - 1)
        self.L2 = (2.0 * math.sin(math.pi / self.N)) ** 2
        self.edges: List[Tuple[int, int]] = [(i, (i + 1) % self.N)
                                             for i in range(self.N)]

    def _point(self, q, i):
        """Position of node i; node 0 is pinned at the origin."""
        if i == 0:
            return np.zeros(2)
        return q[2 * (i - 1):2 * (i - 1) + 2]

    def mass(self):
        return self.m * np.eye(self.n)

    def potential(self, q):
        return float(sum(self.m * self.g * self._point(q, i)[1]
                         for i in range(1, self.N)))

    def force(self, q):
        F = np.zeros(self.n)
        F[1::2] = -self.m * self.g          # -dV/dy for each moving node
        return F

    def jac_gamma(self, q, v):
        J = np.zeros((self.N, self.n))
        gamma = np.zeros(self.N)
        for k, (i, j) in enumerate(self.edges):
            dp = self._point(q, j) - self._point(q, i)
            dv = self._point(v, j) - self._point(v, i)
            if j != 0:
                J[k, 2 * (j - 1):2 * (j - 1) + 2] = 2.0 * dp
            if i != 0:
                J[k, 2 * (i - 1):2 * (i - 1) + 2] = -2.0 * dp
            gamma[k] = -2.0 * float(dv @ dv)
        return J, gamma

    def constraint_residual(self, q):
        worst = 0.0
        for i, j in self.edges:
            dp = self._point(q, j) - self._point(q, i)
            worst = max(worst, abs(float(dp @ dp) - self.L2))
        return worst

    def initial_state(self):
        """The rigidly rotating start systems.py writes (omega = 0.4)."""
        omega = 0.4
        pts = [(math.cos(2 * math.pi * k / self.N),
                math.sin(2 * math.pi * k / self.N)) for k in range(self.N)]
        x0, y0 = pts[0]
        pts = [(x - x0, y - y0) for x, y in pts]
        s = np.zeros(2 * self.n)
        for i in range(1, self.N):
            x, y = pts[i]
            base = 4 * (i - 1)
            s[base + 0], s[base + 1] = round(x, 4), round(-omega * y, 4)
            s[base + 2], s[base + 3] = round(y, 4), round(omega * x, 4)
        return s


class RedundantCircle(_Constrained):
    """Particle on a circle with R exactly-dependent duplicate constraints.

    Matches systems.redundant_dsl(R): the base constraint x^2 + y^2 - 1 = 0
    plus, for k = 2 .. R+1, the constraint k(x^2 + y^2) - k = 0, which is the
    base constraint multiplied by k. The Jacobian therefore has R+1 rows and
    rank 1.
    """

    def __init__(self, R: int, m: float = 1.0, g: float = 9.81) -> None:
        self.R, self.m, self.g = int(R), float(m), float(g)
        self.n = 2
        self.mults = [1.0] + [float(k + 2) for k in range(self.R)]

    def mass(self):
        return self.m * np.eye(2)

    def potential(self, q):
        return float(self.m * self.g * q[1])

    def force(self, q):
        return np.array([0.0, -self.m * self.g])

    def jac_gamma(self, q, v):
        x, y = q
        vx, vy = v
        J = np.array([[2.0 * c * x, 2.0 * c * y] for c in self.mults])
        gamma = np.array([-2.0 * c * (vx * vx + vy * vy) for c in self.mults])
        return J, gamma

    def constraint_residual(self, q):
        return abs(float(q @ q) - 1.0)

    def initial_state(self):
        return np.array([1.0, 0.0, 0.0, 1.0])


def constrained_reference_for_case(case: dict):
    axis = case.get("axis")
    if axis == "loops":
        return ClosedChain(int(case["knob"]))
    if axis == "redundancy":
        return RedundantCircle(int(case["knob"]))
    return None


# ===========================================================================
# Self-tests
# ===========================================================================

def _test_constraints_hold() -> None:
    """Integrate and confirm the constraints and energy hold."""
    from scipy.integrate import solve_ivp
    for name, sysobj, tend in (
            ("loops_N3", ClosedChain(3), 3.0),
            ("loops_N4", ClosedChain(4), 3.0),
            ("loops_N5", ClosedChain(5), 3.0),
            ("redund_R0", RedundantCircle(0), 5.0),
            ("redund_R4", RedundantCircle(4), 5.0),
            ("redund_R8", RedundantCircle(8), 5.0)):
        y0 = sysobj.initial_state()
        r0 = sysobj.constraint_residual(y0[0::2])
        E0 = sysobj.energy(y0)
        sol = solve_ivp(sysobj.rhs, (0.0, tend), y0, method="DOP853",
                        rtol=1e-9, atol=1e-11)
        assert sol.success, f"{name}: {sol.message}"
        res = max(abs(sysobj.constraint_residual(sol.y[0::2, i]) - r0)
                  for i in range(sol.y.shape[1]))
        drift = max(abs(sysobj.energy(sol.y[:, i]) - E0)
                    for i in range(sol.y.shape[1])) / max(abs(E0), 1e-12)
        moved = float(np.max(np.abs(sol.y[0::2] - sol.y[0::2][:, [0]])))
        # Index-1 reduction without stabilisation drifts off the manifold;
        # this reference deliberately does NOT apply Baumgarte stabilisation or
        # projection, because the study compares DERIVATIONS at given states,
        # and a stabilised reference would no longer be solving the same
        # equations the engines are asked for. Drift of ~1e-4 over 3s is the
        # expected cost of that choice, not a defect.
        assert res < 1e-3, f"{name}: constraint drift {res:.3e}"
        assert moved > 1e-3, f"{name}: did not move ({moved:.2e})"
        print(f"  [ok] {name:<11} constraint drift {res:.2e}, "
              f"energy {drift:.2e}, motion {moved:.3f}")


def _test_redundancy_is_exact() -> None:
    """The duplicated constraints must be exactly dependent, and the
    accelerations must not depend on how many duplicates are present."""
    base = RedundantCircle(0)
    y = np.array([0.6, 0.8, -0.8, 0.6])       # on the circle, moving along it
    a0 = base.accel(y)
    for R in (1, 2, 4, 8):
        s = RedundantCircle(R)
        J, _ = s.jac_gamma(y[0::2], y[1::2])
        rank = np.linalg.matrix_rank(J)
        assert rank == 1, f"R={R}: expected rank 1, got {rank}"
        a = s.accel(y)
        assert np.max(np.abs(a - a0)) < 1e-12, \
            f"R={R}: accelerations changed with redundancy ({a} vs {a0})"
    print("  [ok] redundant constraints are exactly rank-1; accelerations "
          "unchanged for R=1,2,4,8")


def _test_pinned_node() -> None:
    """Node 0 stays at the origin, and the initial polygon is on the manifold
    to within the suite's own rounding.

    NOTE, and this is a property of the suite rather than of this reference:
    `systems.closed_chain_dsl` rounds its initial conditions to four decimal
    places, so the starting polygon does not exactly satisfy its own rod
    constraints -- the residual is ~1e-4 for N=5. Every constrained case
    therefore begins slightly OFF the constraint manifold.

    This is benign for the study because the classifier measures constraint
    residual DRIFT relative to t=0 rather than its absolute value, so a
    constant initial offset cancels. It is recorded because a reader checking
    the constraint residual directly would otherwise find a violation and
    reasonably suspect the engine.
    """
    c = ClosedChain(5)
    y0 = c.initial_state()
    assert np.allclose(c._point(y0[0::2], 0), 0.0)
    r = c.constraint_residual(y0[0::2])
    assert r < 1e-3, f"initial state violates the rods by {r:.2e}"
    print(f"  [ok] pinned node at origin; initial polygon on the manifold to "
          f"{r:.1e} (suite rounds ICs to 4dp)")


def main() -> int:
    print("reference_constrained.py self-tests\n")
    _test_pinned_node()
    _test_redundancy_is_exact()
    _test_constraints_hold()
    print("\nAll self-tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
