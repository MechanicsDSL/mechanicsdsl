"""
Uncertainty quantification for MechanicsDSL.

Bayesian inference and bootstrap methods for parameter uncertainty.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


@dataclass
class MCMCResult:
    """Result of MCMC sampling."""

    samples: np.ndarray  # (n_samples, n_params)
    param_names: List[str]
    acceptance_rate: float
    log_likelihood: np.ndarray
    mean: Dict[str, float]
    std: Dict[str, float]
    credible_intervals: Dict[str, Tuple[float, float]]  # 95% CI

    def get_chain(self, param: str) -> np.ndarray:
        """Get MCMC chain for a specific parameter."""
        idx = self.param_names.index(param)
        return self.samples[:, idx]


@dataclass
class BootstrapResult:
    """Result of bootstrap uncertainty estimation."""

    estimates: np.ndarray  # (n_bootstrap, n_params)
    param_names: List[str]
    mean: Dict[str, float]
    std: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    percentiles: Dict[str, Dict[str, float]]  # 2.5, 25, 50, 75, 97.5


class UncertaintyQuantifier:
    """
    Uncertainty quantification for fitted parameters.

    Provides Bayesian (MCMC) and frequentist (bootstrap) methods
    for estimating parameter uncertainty.

    Example:
        uq = UncertaintyQuantifier(compiler)

        # Bootstrap confidence intervals
        bootstrap = uq.bootstrap_uncertainty(
            observations, t_obs,
            fitted_params={'m': 1.0, 'k': 10.0},
            params_to_fit=['m', 'k'],
            n_bootstrap=1000
        )
        print(f"m = {bootstrap.mean['m']:.2f} ± {bootstrap.std['m']:.2f}")

        # Bayesian MCMC
        mcmc = uq.mcmc_inference(
            observations, t_obs,
            ['m', 'k'],
            priors={'m': ('uniform', 0.1, 5), 'k': ('normal', 10, 2)},
            n_samples=10000
        )
    """

    def __init__(self, compiler: "PhysicsCompiler"):
        self.compiler = compiler
        self._base_params = compiler.simulator.parameters.copy()

    def bootstrap_uncertainty(
        self,
        observations: np.ndarray,
        t_obs: np.ndarray,
        fitted_params: Dict[str, float],
        params_to_fit: List[str],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        method: str = "nonparametric",
    ) -> BootstrapResult:
        """
        Estimate parameter uncertainty using bootstrap resampling.

        Args:
            observations: Observed data
            t_obs: Observation times
            fitted_params: Point estimates from fitting
            params_to_fit: Parameters that were fitted
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            method: 'nonparametric' or 'parametric'

        Returns:
            BootstrapResult with distributions and intervals
        """
        from .estimator import ParameterEstimator

        observations = np.atleast_2d(observations)
        if observations.shape[1] == len(t_obs):
            observations = observations.T

        n_times = len(t_obs)
        n_params = len(params_to_fit)

        estimates = np.zeros((n_bootstrap, n_params))

        for b in range(n_bootstrap):
            if method == "nonparametric":
                # Resample observations with replacement
                indices = np.random.choice(n_times, n_times, replace=True)
                obs_boot = observations[indices]
                t_boot = t_obs[indices]

                # Sort by time
                sort_idx = np.argsort(t_boot)
                obs_boot = obs_boot[sort_idx]
                t_boot = t_boot[sort_idx]
            else:
                # Parametric: add noise to observations
                noise_std = np.std(observations, axis=0) * 0.1
                obs_boot = observations + np.random.randn(*observations.shape) * noise_std
                t_boot = t_obs

            # Refit
            try:
                estimator = ParameterEstimator(self.compiler)
                result = estimator.fit(
                    obs_boot, t_boot, params_to_fit, initial_guess=fitted_params, max_iter=100
                )

                for i, name in enumerate(params_to_fit):
                    estimates[b, i] = result.parameters.get(name, fitted_params[name])
            except:
                # Use original estimate if fit fails
                for i, name in enumerate(params_to_fit):
                    estimates[b, i] = fitted_params[name]

        # Compute statistics
        alpha = 1 - confidence_level

        mean = {name: np.mean(estimates[:, i]) for i, name in enumerate(params_to_fit)}
        std = {name: np.std(estimates[:, i]) for i, name in enumerate(params_to_fit)}

        ci = {}
        percentiles = {}
        for i, name in enumerate(params_to_fit):
            lower = np.percentile(estimates[:, i], alpha / 2 * 100)
            upper = np.percentile(estimates[:, i], (1 - alpha / 2) * 100)
            ci[name] = (lower, upper)

            percentiles[name] = {
                "2.5": np.percentile(estimates[:, i], 2.5),
                "25": np.percentile(estimates[:, i], 25),
                "50": np.percentile(estimates[:, i], 50),
                "75": np.percentile(estimates[:, i], 75),
                "97.5": np.percentile(estimates[:, i], 97.5),
            }

        return BootstrapResult(
            estimates=estimates,
            param_names=params_to_fit,
            mean=mean,
            std=std,
            confidence_intervals=ci,
            percentiles=percentiles,
        )

    def mcmc_inference(
        self,
        observations: np.ndarray,
        t_obs: np.ndarray,
        params_to_fit: List[str],
        priors: Optional[Dict[str, Tuple]] = None,
        n_samples: int = 10000,
        n_burn: int = 1000,
        proposal_scale: float = 0.1,
        noise_std: Optional[float] = None,
    ) -> MCMCResult:
        """
        Bayesian parameter inference using MCMC.

        Uses Metropolis-Hastings algorithm with Gaussian proposals.

        Args:
            observations: Observed data
            t_obs: Observation times
            params_to_fit: Parameters to estimate
            priors: Dict of param -> (dist_type, *args)
                    e.g., {'m': ('uniform', 0.1, 5), 'k': ('normal', 10, 2)}
            n_samples: Number of MCMC samples
            n_burn: Burn-in samples to discard
            proposal_scale: Scale of proposal distribution
            noise_std: Assumed observation noise (estimated if None)

        Returns:
            MCMCResult with posterior samples
        """
        observations = np.atleast_2d(observations)
        if observations.shape[1] == len(t_obs):
            observations = observations.T

        n_params = len(params_to_fit)

        # Default priors (wide uniform)
        if priors is None:
            priors = {}
        for name in params_to_fit:
            if name not in priors:
                current = self._base_params.get(name, 1.0)
                priors[name] = ("uniform", current * 0.01, current * 100)

        # Estimate noise if not provided
        if noise_std is None:
            noise_std = np.std(observations) * 0.1

        def log_prior(params: Dict[str, float]) -> float:
            """Compute log prior probability."""
            log_p = 0.0
            for name, value in params.items():
                prior = priors[name]
                if prior[0] == "uniform":
                    low, high = prior[1], prior[2]
                    if value < low or value > high:
                        return -np.inf
                    log_p += -np.log(high - low)
                elif prior[0] == "normal":
                    mean, std = prior[1], prior[2]
                    log_p += -0.5 * ((value - mean) / std) ** 2
                elif prior[0] == "lognormal":
                    mean, std = prior[1], prior[2]
                    if value <= 0:
                        return -np.inf
                    log_p += -0.5 * ((np.log(value) - mean) / std) ** 2 - np.log(value)
            return log_p

        def log_likelihood(params: Dict[str, float]) -> float:
            """Compute log likelihood of observations given params."""
            self.compiler.simulator.set_parameters(params)

            try:
                result = self.compiler.simulate(t_span=(t_obs[0], t_obs[-1]), num_points=len(t_obs))

                if not result["success"]:
                    return -np.inf

                # Interpolate predictions
                from scipy.interpolate import interp1d

                y_pred = result["y"][0]
                interp = interp1d(result["t"], y_pred, fill_value="extrapolate")
                predictions = interp(t_obs)

                # Gaussian likelihood
                residuals = observations[:, 0] - predictions
                log_lik = -0.5 * np.sum((residuals / noise_std) ** 2)
                log_lik -= len(residuals) * np.log(noise_std * np.sqrt(2 * np.pi))

                return log_lik
            except:
                return -np.inf

        # Initialize chain
        samples = np.zeros((n_samples + n_burn, n_params))
        log_likes = np.zeros(n_samples + n_burn)

        # Start from prior mean or current values
        current = {}
        for name in params_to_fit:
            prior = priors[name]
            if prior[0] == "uniform":
                current[name] = (prior[1] + prior[2]) / 2
            else:
                current[name] = prior[1]

        current_log_posterior = log_prior(current) + log_likelihood(current)

        n_accepted = 0

        # MCMC loop
        for i in range(n_samples + n_burn):
            # Propose new values
            proposed = current.copy()
            for j, name in enumerate(params_to_fit):
                scale = np.abs(current[name]) * proposal_scale
                proposed[name] = current[name] + np.random.randn() * scale

            # Compute acceptance ratio
            proposed_log_posterior = log_prior(proposed) + log_likelihood(proposed)
            log_alpha = proposed_log_posterior - current_log_posterior

            # Accept/reject
            if np.log(np.random.random()) < log_alpha:
                current = proposed
                current_log_posterior = proposed_log_posterior
                n_accepted += 1

            # Store sample
            for j, name in enumerate(params_to_fit):
                samples[i, j] = current[name]
            log_likes[i] = current_log_posterior

        # Discard burn-in
        samples = samples[n_burn:]
        log_likes = log_likes[n_burn:]

        # Compute statistics
        mean = {name: np.mean(samples[:, i]) for i, name in enumerate(params_to_fit)}
        std = {name: np.std(samples[:, i]) for i, name in enumerate(params_to_fit)}
        ci = {
            name: (np.percentile(samples[:, i], 2.5), np.percentile(samples[:, i], 97.5))
            for i, name in enumerate(params_to_fit)
        }

        return MCMCResult(
            samples=samples,
            param_names=params_to_fit,
            acceptance_rate=n_accepted / (n_samples + n_burn),
            log_likelihood=log_likes,
            mean=mean,
            std=std,
            credible_intervals=ci,
        )


__all__ = [
    "MCMCResult",
    "BootstrapResult",
    "UncertaintyQuantifier",
]
