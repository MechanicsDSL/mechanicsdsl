"""
Sensitivity analysis for MechanicsDSL.

Determine how sensitive simulation outputs are to parameter changes.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


@dataclass
class SobolIndices:
    """Sobol sensitivity indices."""

    first_order: Dict[str, float]  # Main effects
    total_order: Dict[str, float]  # Total effects (including interactions)
    second_order: Optional[Dict[Tuple[str, str], float]] = None
    confidence: Optional[Dict[str, Tuple[float, float]]] = None


class SensitivityAnalyzer:
    """
    Global sensitivity analysis for physics simulations.

    Implements Sobol variance-based sensitivity analysis to identify
    which parameters have the greatest impact on simulation outputs.

    Example:
        analyzer = SensitivityAnalyzer(compiler)

        indices = analyzer.sobol_analysis(
            param_ranges={'m': (0.5, 2.0), 'k': (5, 20)},
            output_fn=lambda result: np.max(result['y'][0]),  # Max displacement
            n_samples=1000
        )

        print(f"Mass sensitivity: {indices.first_order['m']:.3f}")
        print(f"Spring sensitivity: {indices.first_order['k']:.3f}")
    """

    def __init__(self, compiler: "PhysicsCompiler"):
        self.compiler = compiler
        self._base_params = compiler.simulator.parameters.copy()

    def sobol_analysis(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        output_fn: Optional[Callable] = None,
        t_span: Tuple[float, float] = (0, 10),
        n_samples: int = 1024,
        calc_second_order: bool = False,
    ) -> SobolIndices:
        """
        Compute Sobol sensitivity indices.

        Uses Saltelli's sampling scheme for efficient estimation.

        Args:
            param_ranges: Dict of param_name -> (min, max)
            output_fn: Function to compute scalar output from simulation result.
                       Default: final energy if available, else RMS of first coord.
            t_span: Simulation time span
            n_samples: Number of base samples (actual evaluations = n*(2d+2))
            calc_second_order: Whether to compute pairwise interactions

        Returns:
            SobolIndices with first-order and total-order indices
        """
        param_names = list(param_ranges.keys())
        d = len(param_names)

        # Default output function
        if output_fn is None:

            def output_fn(result):
                if result["success"]:
                    return np.sqrt(np.mean(result["y"][0] ** 2))  # RMS of first coord
                return np.nan

        # Generate Saltelli samples
        samples_a, samples_b, samples_ab = self._saltelli_sample(param_ranges, n_samples)

        # Evaluate model at all sample points
        y_a = self._evaluate_samples(samples_a, param_names, t_span, output_fn)
        y_b = self._evaluate_samples(samples_b, param_names, t_span, output_fn)

        y_ab = {}
        for i, name in enumerate(param_names):
            y_ab[name] = self._evaluate_samples(samples_ab[i], param_names, t_span, output_fn)

        # Compute indices
        first_order = {}
        total_order = {}

        var_y = np.var(np.concatenate([y_a, y_b]))

        for i, name in enumerate(param_names):
            # First-order: V[E[Y|Xi]] / V[Y]
            f0 = np.mean(y_a)
            first_order[name] = (np.mean(y_b * (y_ab[name] - y_a)) / var_y) if var_y > 0 else 0.0

            # Total-order: E[V[Y|X~i]] / V[Y]
            total_order[name] = (
                (0.5 * np.mean((y_a - y_ab[name]) ** 2) / var_y) if var_y > 0 else 0.0
            )

        # Second-order (optional)
        second_order = None
        if calc_second_order and d > 1:
            second_order = {}
            for i, j in combinations(range(d), 2):
                name_i, name_j = param_names[i], param_names[j]
                # Simplified second-order estimate
                second_order[(name_i, name_j)] = max(
                    0,
                    total_order[name_i]
                    + total_order[name_j]
                    - first_order[name_i]
                    - first_order[name_j],
                )

        return SobolIndices(
            first_order=first_order, total_order=total_order, second_order=second_order
        )

    def _saltelli_sample(
        self, param_ranges: Dict[str, Tuple[float, float]], n: int
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Generate Saltelli's sampling matrices."""
        param_names = list(param_ranges.keys())
        d = len(param_names)

        # Base samples from Sobol sequence or random
        try:
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=d * 2, scramble=True)
            samples = sampler.random(n)

            # Split into A and B matrices
            samples_a = samples[:, :d]
            samples_b = samples[:, d:]
        except ImportError:
            # Fallback to random
            samples_a = np.random.random((n, d))
            samples_b = np.random.random((n, d))

        # Scale to parameter ranges
        for i, name in enumerate(param_names):
            low, high = param_ranges[name]
            samples_a[:, i] = low + samples_a[:, i] * (high - low)
            samples_b[:, i] = low + samples_b[:, i] * (high - low)

        # Create AB matrices (A with i-th column from B)
        samples_ab = []
        for i in range(d):
            ab = samples_a.copy()
            ab[:, i] = samples_b[:, i]
            samples_ab.append(ab)

        return samples_a, samples_b, samples_ab

    def _evaluate_samples(
        self,
        samples: np.ndarray,
        param_names: List[str],
        t_span: Tuple[float, float],
        output_fn: Callable,
    ) -> np.ndarray:
        """Evaluate model at all sample points."""
        n = samples.shape[0]
        outputs = np.zeros(n)

        for i in range(n):
            params = self._base_params.copy()
            for j, name in enumerate(param_names):
                params[name] = samples[i, j]

            self.compiler.simulator.set_parameters(params)

            try:
                result = self.compiler.simulate(t_span=t_span, num_points=100)
                outputs[i] = output_fn(result)
            except:
                outputs[i] = np.nan

        return outputs

    def local_sensitivity(
        self,
        param_name: str,
        output_fn: Optional[Callable] = None,
        delta: float = 0.01,
        t_span: Tuple[float, float] = (0, 10),
    ) -> float:
        """
        Compute local sensitivity (derivative) at current parameter values.

        Args:
            param_name: Parameter to perturb
            output_fn: Scalar output function
            delta: Relative perturbation size
            t_span: Simulation time span

        Returns:
            Normalized sensitivity: (dY/Y) / (dP/P)
        """
        if output_fn is None:

            def output_fn(result):
                return np.sqrt(np.mean(result["y"][0] ** 2))

        base_val = self._base_params[param_name]

        # Evaluate at base
        result_base = self.compiler.simulate(t_span=t_span, num_points=100)
        y_base = output_fn(result_base)

        # Evaluate at perturbed
        params_pert = self._base_params.copy()
        params_pert[param_name] = base_val * (1 + delta)
        self.compiler.simulator.set_parameters(params_pert)

        result_pert = self.compiler.simulate(t_span=t_span, num_points=100)
        y_pert = output_fn(result_pert)

        # Reset
        self.compiler.simulator.set_parameters(self._base_params)

        # Normalized sensitivity
        if y_base != 0:
            return ((y_pert - y_base) / y_base) / delta
        return 0.0

    def morris_screening(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        output_fn: Optional[Callable] = None,
        t_span: Tuple[float, float] = (0, 10),
        n_trajectories: int = 10,
        n_levels: int = 4,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Morris one-at-a-time screening method.

        Efficient for identifying non-influential parameters
        in systems with many parameters.

        Args:
            param_ranges: Parameter bounds
            output_fn: Scalar output function
            t_span: Simulation time span
            n_trajectories: Number of trajectories
            n_levels: Number of discretization levels

        Returns:
            Dict of param_name -> (mu*, sigma) where mu* is mean |effect|
        """
        param_names = list(param_ranges.keys())
        d = len(param_names)

        if output_fn is None:

            def output_fn(result):
                return np.sqrt(np.mean(result["y"][0] ** 2))

        effects = {name: [] for name in param_names}

        for _ in range(n_trajectories):
            # Random starting point
            x = np.random.randint(0, n_levels, d) / (n_levels - 1)

            # Scale to ranges
            scaled = np.array(
                [
                    param_ranges[name][0] + x[i] * (param_ranges[name][1] - param_ranges[name][0])
                    for i, name in enumerate(param_names)
                ]
            )

            # Evaluate base
            params = {name: scaled[i] for i, name in enumerate(param_names)}
            self.compiler.simulator.set_parameters(params)
            result = self.compiler.simulate(t_span=t_span, num_points=100)
            y_base = output_fn(result)

            # Perturb each parameter
            delta = 1.0 / (n_levels - 1)
            for i, name in enumerate(param_names):
                x_new = x.copy()
                x_new[i] = min(1, x[i] + delta)

                scaled_new = [
                    param_ranges[n][0] + x_new[j] * (param_ranges[n][1] - param_ranges[n][0])
                    for j, n in enumerate(param_names)
                ]

                params_new = {n: scaled_new[j] for j, n in enumerate(param_names)}
                self.compiler.simulator.set_parameters(params_new)
                result_new = self.compiler.simulate(t_span=t_span, num_points=100)
                y_new = output_fn(result_new)

                # Elementary effect
                effect = (y_new - y_base) / delta
                effects[name].append(effect)

                y_base = y_new
                x = x_new

        # Compute mu* and sigma
        return {
            name: (np.mean(np.abs(effects[name])), np.std(effects[name])) for name in param_names
        }


__all__ = [
    "SobolIndices",
    "SensitivityAnalyzer",
]
