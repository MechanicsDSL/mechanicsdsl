"""
Parameter estimation for MechanicsDSL.

Fit model parameters to observed data using optimization.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, differential_evolution, minimize

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


@dataclass
class FitResult:
    """Result of parameter fitting."""

    success: bool
    parameters: Dict[str, float]
    residual: float
    iterations: int
    message: str
    jacobian: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None


class ParameterEstimator:
    """
    Estimate physical parameters from observed data.

    Solves the inverse problem: given observations of system behavior,
    find the parameters that best reproduce those observations.

    Example:
        compiler = PhysicsCompiler()
        compiler.compile_dsl(dsl_code)

        estimator = ParameterEstimator(compiler)

        # Fit mass and spring constant to data
        result = estimator.fit(
            observations,       # (n_times, n_coords) array
            t_obs,              # observation times
            ['m', 'k'],         # parameters to fit
            bounds={'m': (0.1, 10), 'k': (1, 100)}
        )

        print(f"Fitted m = {result.parameters['m']:.3f}")
        print(f"Fitted k = {result.parameters['k']:.3f}")
    """

    def __init__(self, compiler: "PhysicsCompiler"):
        self.compiler = compiler
        self._base_params = compiler.simulator.parameters.copy()

    def fit(
        self,
        observations: np.ndarray,
        t_obs: np.ndarray,
        params_to_fit: List[str],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
        weights: Optional[np.ndarray] = None,
        initial_guess: Optional[Dict[str, float]] = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> FitResult:
        """
        Fit parameters to minimize prediction error.

        Args:
            observations: Observed data, shape (n_times, n_coords) or (n_times,)
            t_obs: Observation times, shape (n_times,)
            params_to_fit: List of parameter names to optimize
            bounds: Optional bounds for each parameter
            method: Optimization method ('L-BFGS-B', 'Nelder-Mead', 'differential_evolution')
            weights: Optional weights for each observation time
            initial_guess: Initial parameter values (defaults to current)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            FitResult with optimized parameters
        """
        # Validate inputs
        observations = np.atleast_2d(observations)
        if observations.shape[1] == len(t_obs):
            observations = observations.T  # Transpose to (time, coords)

        n_times = len(t_obs)
        n_coords = observations.shape[1] if observations.ndim > 1 else 1

        if weights is None:
            weights = np.ones(n_times)

        # Set up bounds
        if bounds is None:
            bounds = {}

        bound_list = []
        for param in params_to_fit:
            if param in bounds:
                bound_list.append(bounds[param])
            else:
                # Default bounds
                current = self._base_params.get(param, 1.0)
                bound_list.append((current * 0.01, current * 100))

        # Initial guess
        if initial_guess is None:
            initial_guess = {}

        x0 = np.array([initial_guess.get(p, self._base_params.get(p, 1.0)) for p in params_to_fit])

        # Define objective function
        def objective(param_values):
            # Update parameters
            params = self._base_params.copy()
            for name, val in zip(params_to_fit, param_values):
                params[name] = val

            # Run simulation
            try:
                self.compiler.simulator.set_parameters(params)
                result = self.compiler.simulate(t_span=(t_obs[0], t_obs[-1]), num_points=len(t_obs))

                if not result["success"]:
                    return 1e10  # Penalty for failed simulation

                # Interpolate to observation times
                from scipy.interpolate import interp1d

                t_sim = result["t"]
                y_sim = result["y"]

                # Get position coordinates (every other row)
                coords = self.compiler.simulator.coordinates
                n_sim_coords = len(coords)

                predictions = np.zeros((n_times, min(n_coords, n_sim_coords)))
                for i in range(min(n_coords, n_sim_coords)):
                    interp = interp1d(t_sim, y_sim[2 * i], kind="cubic", fill_value="extrapolate")
                    predictions[:, i] = interp(t_obs)

                # Compute weighted residual
                residual = np.sum(
                    weights[:, None]
                    * (predictions - observations[:, : min(n_coords, n_sim_coords)]) ** 2
                )
                return residual

            except Exception as e:
                warnings.warn(f"Simulation failed: {e}")
                return 1e10

        # Optimize
        if method == "differential_evolution":
            result = differential_evolution(
                objective, bound_list, maxiter=max_iter, tol=tol, seed=42
            )
        else:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bound_list,
                options={"maxiter": max_iter, "ftol": tol},
            )

        # Build result
        fitted_params = {name: float(val) for name, val in zip(params_to_fit, result.x)}

        return FitResult(
            success=result.success,
            parameters=fitted_params,
            residual=float(result.fun),
            iterations=result.nit if hasattr(result, "nit") else 0,
            message=result.message if hasattr(result, "message") else str(result),
            jacobian=result.jac if hasattr(result, "jac") else None,
        )

    def fit_trajectory(
        self, trajectory: np.ndarray, t_trajectory: np.ndarray, params_to_fit: List[str], **kwargs
    ) -> FitResult:
        """
        Fit parameters to match a full trajectory.

        Convenient wrapper for fitting position/velocity time series.

        Args:
            trajectory: Shape (n_times, n_state) with positions and velocities
            t_trajectory: Time array
            params_to_fit: Parameters to optimize
            **kwargs: Additional arguments for fit()

        Returns:
            FitResult
        """
        return self.fit(trajectory, t_trajectory, params_to_fit, **kwargs)

    def grid_search(
        self,
        observations: np.ndarray,
        t_obs: np.ndarray,
        param_grids: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Exhaustive grid search over parameter space.

        Useful for visualizing the loss landscape or when optimization
        gets stuck in local minima.

        Args:
            observations: Observed data
            t_obs: Observation times
            param_grids: Dict of param_name -> array of values to try

        Returns:
            Tuple of (best_params, loss_grid)
        """
        param_names = list(param_grids.keys())
        grids = [param_grids[name] for name in param_names]

        # Create meshgrid
        mesh = np.meshgrid(*grids, indexing="ij")
        shape = mesh[0].shape

        # Evaluate at each point
        losses = np.zeros(shape)

        for idx in np.ndindex(shape):
            params = {name: mesh[i][idx] for i, name in enumerate(param_names)}

            # Update and simulate
            self.compiler.simulator.set_parameters(params)
            try:
                result = self.compiler.simulate(t_span=(t_obs[0], t_obs[-1]), num_points=len(t_obs))

                if result["success"]:
                    from scipy.interpolate import interp1d

                    y_pred = result["y"][0]  # First coordinate
                    interp = interp1d(result["t"], y_pred, fill_value="extrapolate")
                    pred = interp(t_obs)
                    losses[idx] = np.sum((pred - observations.flatten()[: len(pred)]) ** 2)
                else:
                    losses[idx] = np.inf
            except:
                losses[idx] = np.inf

        # Find minimum
        min_idx = np.unravel_index(np.argmin(losses), shape)
        best_params = {name: mesh[i][min_idx] for i, name in enumerate(param_names)}

        return best_params, losses


__all__ = [
    "FitResult",
    "ParameterEstimator",
]
