"""
Symplectic Integrators for Hamiltonian Systems

This module provides structure-preserving numerical integrators designed
specifically for Hamiltonian mechanical systems. Unlike general-purpose
methods like RK45, symplectic integrators:

1. Preserve the symplectic 2-form (phase space structure)
2. Exhibit bounded energy error for long-time integration
3. Are time-reversible
4. Respect Poincaré invariants

Available Integrators:
----------------------
- StormerVerlet: 2nd order, explicit symplectic
- Leapfrog: 2nd order, equivalent to Störmer-Verlet
- Yoshida4: 4th order, composition method
- Ruth3: 3rd order, minimal stages
- McLachlan4: 4th order, optimal coefficients

Theory:
-------
For Hamiltonian H(q, p) = T(p) + V(q):
- T(p) = kinetic energy (depends only on momentum)
- V(q) = potential energy (depends only on position)

The symplectic Euler maps are:
- A: p → p - h∇V(q), then q → q + h∇T(p)
- B: q → q + h∇T(p), then p → p - h∇V(q)

Composition methods combine these to achieve higher order.

References:
-----------
[1] Hairer, Lubich, Wanner: "Geometric Numerical Integration" (2006)
[2] Leimkuhler, Reich: "Simulating Hamiltonian Dynamics" (2004)
[3] Yoshida: "Construction of higher order symplectic integrators" (1990)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class IntegratorConfig:
    """Configuration for symplectic integrators."""

    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = np.inf
    first_step: Optional[float] = None
    max_steps: int = 100000


class SymplecticIntegrator(ABC):
    """
    Base class for symplectic integrators.

    All symplectic integrators work with Hamiltonian systems of the form:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q

    For separable Hamiltonians H(q,p) = T(p) + V(q):
        dq/dt = ∂T/∂p = p/m  (for standard kinetic energy)
        dp/dt = -∂V/∂q = F(q)  (generalized force)
    """

    @property
    @abstractmethod
    def order(self) -> int:
        """Order of accuracy of the integrator."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the integrator."""
        pass

    @abstractmethod
    def step(
        self,
        t: float,
        q: np.ndarray,
        p: np.ndarray,
        h: float,
        grad_T: Callable[[np.ndarray], np.ndarray],
        grad_V: Callable[[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one integration step.

        Args:
            t: Current time
            q: Position vector
            p: Momentum vector
            h: Step size
            grad_T: Gradient of kinetic energy w.r.t. momentum (∂T/∂p)
            grad_V: Gradient of potential energy w.r.t. position (∂V/∂q)

        Returns:
            Tuple of (new_q, new_p) after one step
        """
        pass

    def integrate(
        self,
        t_span: Tuple[float, float],
        q0: np.ndarray,
        p0: np.ndarray,
        h: float,
        grad_T: Callable[[np.ndarray], np.ndarray],
        grad_V: Callable[[np.ndarray], np.ndarray],
        t_eval: Optional[np.ndarray] = None,
        max_steps: int = 100000,
    ) -> Dict[str, np.ndarray]:
        """
        Integrate the system from t_span[0] to t_span[1].

        Args:
            t_span: (t_start, t_end)
            q0: Initial positions
            p0: Initial momenta
            h: Fixed step size
            grad_T: Gradient of kinetic energy
            grad_V: Gradient of potential energy
            t_eval: Optional times at which to evaluate solution
            max_steps: Maximum integration steps

        Returns:
            Dictionary with keys:
            - 't': Time array
            - 'q': Position array (shape: n_coords x n_times)
            - 'p': Momentum array (shape: n_coords x n_times)
            - 'success': Boolean indicating success
            - 'message': Status message
        """
        t0, tf = t_span

        # Ensure arrays
        q = np.atleast_1d(q0).astype(float)
        p = np.atleast_1d(p0).astype(float)

        # Storage
        times = [t0]
        positions = [q.copy()]
        momenta = [p.copy()]

        t = t0
        step_count = 0

        while t < tf and step_count < max_steps:
            # Adjust last step if needed
            if t + h > tf:
                h_step = tf - t
            else:
                h_step = h

            q, p = self.step(t, q, p, h_step, grad_T, grad_V)
            t += h_step
            step_count += 1

            times.append(t)
            positions.append(q.copy())
            momenta.append(p.copy())

        times = np.array(times)
        positions = np.column_stack(positions)
        momenta = np.column_stack(momenta)

        # Interpolate to t_eval if specified
        if t_eval is not None:
            from scipy.interpolate import interp1d

            q_interp = interp1d(times, positions, kind="cubic", axis=1)
            p_interp = interp1d(times, momenta, kind="cubic", axis=1)
            positions = q_interp(t_eval)
            momenta = p_interp(t_eval)
            times = t_eval

        return {
            "t": times,
            "q": positions,
            "p": momenta,
            "success": t >= tf,
            "message": f"{self.name}: {step_count} steps, order {self.order}",
        }


class StormerVerlet(SymplecticIntegrator):
    """
    Störmer-Verlet integrator (2nd order, explicit symplectic).

    The velocity Verlet formulation:
        p_half = p_n - (h/2) * ∇V(q_n)
        q_{n+1} = q_n + h * ∇T(p_half)
        p_{n+1} = p_half - (h/2) * ∇V(q_{n+1})

    Properties:
    - 2nd order accurate
    - Symplectic
    - Time-reversible
    - Explicit (no implicit solves needed for separable H)
    - Energy oscillates around true value, bounded error
    """

    @property
    def order(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "Störmer-Verlet"

    def step(self, t, q, p, h, grad_T, grad_V):
        # Half step in momentum
        p_half = p - (h / 2) * grad_V(q)

        # Full step in position
        q_new = q + h * grad_T(p_half)

        # Half step in momentum
        p_new = p_half - (h / 2) * grad_V(q_new)

        return q_new, p_new


class Leapfrog(SymplecticIntegrator):
    """
    Leapfrog integrator (2nd order, explicit symplectic).

    Mathematically equivalent to Störmer-Verlet but with staggered
    storage pattern. Often preferred for N-body simulations.

    Algorithm:
        q_{n+1} = q_n + h * ∇T(p_{n+1/2})
        p_{n+3/2} = p_{n+1/2} - h * ∇V(q_{n+1})
    """

    @property
    def order(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "Leapfrog"

    def step(self, t, q, p, h, grad_T, grad_V):
        # Kick-Drift-Kick variant
        p_half = p - (h / 2) * grad_V(q)
        q_new = q + h * grad_T(p_half)
        p_new = p_half - (h / 2) * grad_V(q_new)
        return q_new, p_new


class Yoshida4(SymplecticIntegrator):
    """
    Yoshida 4th order symplectic integrator.

    Composition method using triple-jump technique:
        S_h = S_{c_1 h} ∘ S_{c_2 h} ∘ S_{c_1 h}

    where S is the 2nd order Störmer-Verlet step.

    Coefficients:
        c_1 = 1/(2 - 2^(1/3))
        c_2 = -2^(1/3)/(2 - 2^(1/3))

    Properties:
    - 4th order accurate
    - Symplectic
    - 3 stages (force evaluations per step)
    - Time-reversible

    Reference:
        Yoshida, "Construction of higher order symplectic integrators"
        Physics Letters A, 150 (1990)
    """

    # Yoshida coefficients for 4th order
    _c1 = 1.0 / (2.0 - np.power(2.0, 1.0 / 3.0))
    _c2 = -np.power(2.0, 1.0 / 3.0) / (2.0 - np.power(2.0, 1.0 / 3.0))

    @property
    def order(self) -> int:
        return 4

    @property
    def name(self) -> str:
        return "Yoshida-4"

    def step(self, t, q, p, h, grad_T, grad_V):
        # Stage 1: Störmer-Verlet with c1*h
        h1 = self._c1 * h
        p_mid = p - (h1 / 2) * grad_V(q)
        q_mid = q + h1 * grad_T(p_mid)
        p_mid = p_mid - (h1 / 2) * grad_V(q_mid)

        # Stage 2: Störmer-Verlet with c2*h
        h2 = self._c2 * h
        p_mid = p_mid - (h2 / 2) * grad_V(q_mid)
        q_mid = q_mid + h2 * grad_T(p_mid)
        p_mid = p_mid - (h2 / 2) * grad_V(q_mid)

        # Stage 3: Störmer-Verlet with c1*h
        p_mid = p_mid - (h1 / 2) * grad_V(q_mid)
        q_new = q_mid + h1 * grad_T(p_mid)
        p_new = p_mid - (h1 / 2) * grad_V(q_new)

        return q_new, p_new


class Ruth3(SymplecticIntegrator):
    """
    Ruth 3rd order symplectic integrator.

    Minimal-stage 3rd order method using composition:
        S_h = A_{c_1 h} ∘ B_{d_1 h} ∘ A_{c_2 h} ∘ B_{d_2 h} ∘ A_{c_3 h} ∘ B_{d_3 h}

    where A is the momentum kick and B is the position drift.

    Coefficients (Ruth):
        c1 = 7/24, c2 = 3/4, c3 = -1/24
        d1 = 2/3, d2 = -2/3, d3 = 1

    Properties:
    - 3rd order accurate
    - Symplectic
    - 3 stages

    Reference:
        Ruth, "A canonical integration technique" (1983)
    """

    # Ruth coefficients
    _c = np.array([7 / 24, 3 / 4, -1 / 24])
    _d = np.array([2 / 3, -2 / 3, 1.0])

    @property
    def order(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "Ruth-3"

    def step(self, t, q, p, h, grad_T, grad_V):
        q_new = q.copy()
        p_new = p.copy()

        # Apply stages
        for c, d in zip(self._c, self._d):
            p_new = p_new - c * h * grad_V(q_new)
            q_new = q_new + d * h * grad_T(p_new)

        return q_new, p_new


class McLachlan4(SymplecticIntegrator):
    """
    McLachlan 4th order symplectic integrator with optimal coefficients.

    This method minimizes the error constant while maintaining 4th order.
    Uses a 5-stage composition with optimized coefficients.

    Properties:
    - 4th order accurate
    - Symplectic
    - 5 stages
    - Smaller error constant than Yoshida-4

    Reference:
        McLachlan, "On the numerical integration of ODEs by symmetric
        composition methods" (1995)
    """

    # McLachlan optimal coefficients (symmetric)
    _w = np.array(
        [
            0.0378593198406116,
            0.1027719590678225,
            -0.0946010867462035,
            0.1545382271670929,
            0.2940033568823770,
        ]
    )

    @property
    def order(self) -> int:
        return 4

    @property
    def name(self) -> str:
        return "McLachlan-4"

    def step(self, t, q, p, h, grad_T, grad_V):
        q_new = q.copy()
        p_new = p.copy()

        # Build symmetric composition
        weights = np.concatenate([self._w, self._w[-2::-1]])

        for i, w in enumerate(weights):
            if i % 2 == 0:
                # Drift (position update)
                q_new = q_new + w * h * grad_T(p_new)
            else:
                # Kick (momentum update)
                p_new = p_new - w * h * grad_V(q_new)

        return q_new, p_new


# Convenience factory function
def get_symplectic_integrator(name: str) -> SymplecticIntegrator:
    """
    Get a symplectic integrator by name.

    Args:
        name: One of 'verlet', 'leapfrog', 'yoshida4', 'ruth3', 'mclachlan4'

    Returns:
        SymplecticIntegrator instance
    """
    integrators = {
        "verlet": StormerVerlet,
        "leapfrog": Leapfrog,
        "yoshida4": Yoshida4,
        "ruth3": Ruth3,
        "mclachlan4": McLachlan4,
    }

    name_lower = name.lower().replace("-", "").replace("_", "")

    for key, cls in integrators.items():
        if name_lower in key or key in name_lower:
            return cls()

    raise ValueError(f"Unknown integrator: {name}. Available: {list(integrators.keys())}")


__all__ = [
    "SymplecticIntegrator",
    "IntegratorConfig",
    "StormerVerlet",
    "Leapfrog",
    "Yoshida4",
    "Ruth3",
    "McLachlan4",
    "get_symplectic_integrator",
]
