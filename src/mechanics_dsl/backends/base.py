"""
Base backend interface for MechanicsDSL.

Defines the abstract interface that all simulation backends must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BackendCapabilities:
    """Describes what a backend can do."""

    supports_gpu: bool = False
    supports_autodiff: bool = False
    supports_jit: bool = False
    supports_batched: bool = False
    supports_events: bool = False
    supports_stiff: bool = True
    max_batch_size: int = 1000
    name: str = "base"


class Backend(ABC):
    """
    Abstract base class for simulation backends.

    Backends provide different execution strategies for solving ODEs:
    - SciPy (default): CPU-based, general purpose
    - JAX: GPU-accelerated, autodiff, JIT compiled
    - Numba: CPU JIT compiled for speed

    Example:
        class MyBackend(Backend):
            @property
            def capabilities(self):
                return BackendCapabilities(name="my_backend")

            def compile_equations(self, accelerations, coordinates):
                # Convert symbolic to callable
                ...

            def simulate(self, t_span, y0, **kwargs):
                # Run simulation
                ...
    """

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities."""

    @abstractmethod
    def compile_equations(
        self,
        accelerations: Dict[str, Any],  # Symbolic accelerations
        coordinates: List[str],
        parameters: Dict[str, float],
    ) -> Callable:
        """
        Compile symbolic equations to executable form.

        Args:
            accelerations: Dict of coordinate -> acceleration expression
            coordinates: List of coordinate names
            parameters: Dict of parameter values

        Returns:
            Callable that computes derivatives: (t, y) -> dydt
        """

    @abstractmethod
    def simulate(
        self,
        equations: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run simulation.

        Args:
            equations: Compiled equations callable
            t_span: (t_start, t_end)
            y0: Initial conditions array
            t_eval: Optional evaluation times
            **kwargs: Backend-specific options

        Returns:
            Dictionary with 't', 'y', 'success' keys
        """

    def simulate_batch(
        self,
        equations: Callable,
        t_span: Tuple[float, float],
        y0_batch: np.ndarray,  # (batch_size, state_dim)
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run multiple simulations with different initial conditions.

        Default implementation runs sequentially. Override for parallel.

        Args:
            equations: Compiled equations callable
            t_span: (t_start, t_end)
            y0_batch: Array of initial conditions
            **kwargs: Backend-specific options

        Returns:
            List of simulation results
        """
        results = []
        for y0 in y0_batch:
            results.append(self.simulate(equations, t_span, y0, **kwargs))
        return results

    def gradient(self, loss_fn: Callable, params: Dict[str, float]) -> Dict[str, float]:
        """
        Compute gradient of loss function w.r.t. parameters.

        Only available for backends with autodiff support.

        Args:
            loss_fn: Function (params) -> scalar loss
            params: Current parameter values

        Returns:
            Dictionary of parameter gradients

        Raises:
            NotImplementedError: If backend doesn't support autodiff
        """
        if not self.capabilities.supports_autodiff:
            raise NotImplementedError(f"{self.capabilities.name} backend does not support autodiff")
        raise NotImplementedError("Subclass must implement gradient()")

    def cleanup(self) -> None:
        """Release resources. Override for cleanup logic."""


class ScipyBackend(Backend):
    """Default SciPy-based backend."""

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="scipy",
            supports_stiff=True,
            supports_events=True,
        )

    def compile_equations(
        self, accelerations: Dict[str, Any], coordinates: List[str], parameters: Dict[str, float]
    ) -> Callable:
        """Use sympy.lambdify for compilation."""
        import sympy as sp

        # Build state vector order
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"{q}_dot"])

        # Create lambdified functions for each acceleration
        t_sym = sp.Symbol("t")
        state_symbols = [sp.Symbol(s) for s in state_vars]

        accel_funcs = {}
        for coord, accel_expr in accelerations.items():
            # Substitute parameter values
            for param, value in parameters.items():
                accel_expr = accel_expr.subs(sp.Symbol(param), value)

            accel_funcs[coord] = sp.lambdify([t_sym] + state_symbols, accel_expr, "numpy")

        def equations(t, state):
            dydt = np.zeros_like(state)

            for i, coord in enumerate(coordinates):
                dydt[2 * i] = state[2 * i + 1]  # velocity
                dydt[2 * i + 1] = accel_funcs[coord](t, *state)  # acceleration

            return dydt

        return equations

    def simulate(
        self,
        equations: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run simulation using scipy.integrate.solve_ivp."""
        from scipy.integrate import solve_ivp

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        solution = solve_ivp(
            equations, t_span, y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol, **kwargs
        )

        return {
            "success": solution.success,
            "t": solution.t,
            "y": solution.y,
            "message": getattr(solution, "message", ""),
            "nfev": getattr(solution, "nfev", 0),
        }


__all__ = [
    "BackendCapabilities",
    "Backend",
    "ScipyBackend",
]
