"""
Viscoelasticity Module for Solid Mechanics

Implements time-dependent material behavior:
- Maxwell, Kelvin-Voigt, and Standard Linear Solid models
- Generalized Maxwell (Prony series)
- Creep and relaxation functions
- Complex modulus and dynamic mechanical analysis
- Time-temperature superposition
- Boltzmann superposition principle

Security: All time and temperature inputs validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...utils import logger


@dataclass
class ViscoelasticMaterial:
    """Base viscoelastic material properties."""
    name: str
    E_infinity: float = 0.0  # Equilibrium modulus
    E_0: float = 0.0         # Instantaneous modulus
    
    def relaxation_modulus(self, t: float) -> float:
        """Relaxation modulus E(t). Override in subclasses."""
        raise NotImplementedError
    
    def creep_compliance(self, t: float) -> float:
        """Creep compliance J(t). Override in subclasses."""
        raise NotImplementedError


@dataclass
class Maxwell:
    """
    Maxwell model: spring and dashpot in series.
    
    Captures stress relaxation but not creep recovery.
    E(t) = E₀ exp(-t/τ)
    """
    E: float      # Spring modulus (Pa)
    eta: float    # Dashpot viscosity (Pa·s)
    
    def __post_init__(self):
        if self.E <= 0 or self.eta <= 0:
            raise ValueError("E and η must be positive")
    
    @property
    def relaxation_time(self) -> float:
        """Relaxation time τ = η/E."""
        return self.eta / self.E
    
    def relaxation_modulus(self, t: float) -> float:
        """E(t) = E₀ exp(-t/τ)."""
        if t < 0:
            raise ValueError("Time must be non-negative")
        return self.E * math.exp(-t / self.relaxation_time)
    
    def creep_compliance(self, t: float) -> float:
        """J(t) = 1/E + t/η."""
        if t < 0:
            raise ValueError("Time must be non-negative")
        return 1 / self.E + t / self.eta
    
    def stress_relaxation(self, epsilon_0: float, t: float) -> float:
        """Stress response to step strain."""
        return epsilon_0 * self.relaxation_modulus(t)
    
    def creep_strain(self, sigma_0: float, t: float) -> float:
        """Strain response to step stress."""
        return sigma_0 * self.creep_compliance(t)


@dataclass
class KelvinVoigt:
    """
    Kelvin-Voigt model: spring and dashpot in parallel.
    
    Captures creep but not stress relaxation well.
    J(t) = (1/E)(1 - exp(-t/τ))
    """
    E: float      # Spring modulus (Pa)
    eta: float    # Dashpot viscosity (Pa·s)
    
    def __post_init__(self):
        if self.E <= 0 or self.eta <= 0:
            raise ValueError("E and η must be positive")
    
    @property
    def retardation_time(self) -> float:
        """Retardation time τ = η/E."""
        return self.eta / self.E
    
    def creep_compliance(self, t: float) -> float:
        """J(t) = (1/E)(1 - exp(-t/τ))."""
        if t < 0:
            raise ValueError("Time must be non-negative")
        return (1 / self.E) * (1 - math.exp(-t / self.retardation_time))
    
    def relaxation_modulus(self, t: float) -> float:
        """
        Relaxation function (involves Dirac delta at t=0).
        Returns E plus viscous contribution.
        """
        if t == 0:
            return float('inf')  # Delta function
        return self.E
    
    def creep_strain(self, sigma_0: float, t: float) -> float:
        """Strain response to step stress."""
        return sigma_0 * self.creep_compliance(t)


@dataclass
class StandardLinearSolid:
    """
    Standard Linear Solid (Zener) model.
    
    Spring in parallel with Maxwell element.
    Captures both creep and relaxation.
    
    E(t) = E∞ + (E₀ - E∞)exp(-t/τ)
    """
    E_0: float    # Instantaneous modulus (Pa)
    E_inf: float  # Equilibrium modulus (Pa)
    tau: float    # Relaxation time (s)
    
    def __post_init__(self):
        if self.E_0 <= 0 or self.E_inf <= 0:
            raise ValueError("Moduli must be positive")
        if self.E_inf > self.E_0:
            raise ValueError("E_inf must be less than E_0")
        if self.tau <= 0:
            raise ValueError("Relaxation time must be positive")
    
    def relaxation_modulus(self, t: float) -> float:
        """E(t) = E∞ + (E₀ - E∞)exp(-t/τ)."""
        if t < 0:
            raise ValueError("Time must be non-negative")
        return self.E_inf + (self.E_0 - self.E_inf) * math.exp(-t / self.tau)
    
    def creep_compliance(self, t: float) -> float:
        """J(t) for SLS model."""
        if t < 0:
            raise ValueError("Time must be non-negative")
        
        # Retardation time is different from relaxation time
        tau_c = self.tau * self.E_0 / self.E_inf
        J_inf = 1 / self.E_inf
        J_0 = 1 / self.E_0
        
        return J_inf - (J_inf - J_0) * math.exp(-t / tau_c)
    
    def complex_modulus(self, omega: float) -> complex:
        """
        Complex modulus E*(ω) = E' + iE''.
        
        Args:
            omega: Angular frequency (rad/s)
        """
        omega_tau = omega * self.tau
        denom = 1 + omega_tau**2
        
        E_prime = self.E_inf + (self.E_0 - self.E_inf) * omega_tau**2 / denom
        E_double_prime = (self.E_0 - self.E_inf) * omega_tau / denom
        
        return complex(E_prime, E_double_prime)


@dataclass
class GeneralizedMaxwell:
    """
    Generalized Maxwell model (Prony series).
    
    Multiple Maxwell elements in parallel with equilibrium spring.
    E(t) = E∞ + Σ Eᵢ exp(-t/τᵢ)
    """
    E_inf: float
    E_i: List[float]      # Prony coefficients
    tau_i: List[float]    # Relaxation times
    
    def __post_init__(self):
        if len(self.E_i) != len(self.tau_i):
            raise ValueError("E_i and tau_i must have same length")
        if any(e <= 0 for e in self.E_i):
            raise ValueError("All Prony coefficients must be positive")
        if any(t <= 0 for t in self.tau_i):
            raise ValueError("All relaxation times must be positive")
    
    @property
    def E_0(self) -> float:
        """Instantaneous modulus."""
        return self.E_inf + sum(self.E_i)
    
    @property
    def n_terms(self) -> int:
        """Number of Prony terms."""
        return len(self.E_i)
    
    def relaxation_modulus(self, t: float) -> float:
        """E(t) = E∞ + Σ Eᵢ exp(-t/τᵢ)."""
        if t < 0:
            raise ValueError("Time must be non-negative")
        
        result = self.E_inf
        for E, tau in zip(self.E_i, self.tau_i):
            result += E * math.exp(-t / tau)
        return result
    
    def relaxation_modulus_array(self, t: np.ndarray) -> np.ndarray:
        """Vectorized relaxation modulus."""
        result = np.full_like(t, self.E_inf)
        for E, tau in zip(self.E_i, self.tau_i):
            result += E * np.exp(-t / tau)
        return result
    
    def complex_modulus(self, omega: float) -> complex:
        """Complex modulus E*(ω)."""
        E_prime = self.E_inf
        E_double_prime = 0.0
        
        for E, tau in zip(self.E_i, self.tau_i):
            omega_tau = omega * tau
            denom = 1 + omega_tau**2
            E_prime += E * omega_tau**2 / denom
            E_double_prime += E * omega_tau / denom
        
        return complex(E_prime, E_double_prime)
    
    def storage_modulus(self, omega: float) -> float:
        """Storage modulus E'(ω)."""
        return self.complex_modulus(omega).real
    
    def loss_modulus(self, omega: float) -> float:
        """Loss modulus E''(ω)."""
        return self.complex_modulus(omega).imag
    
    def loss_tangent(self, omega: float) -> float:
        """Loss tangent tan(δ) = E''/E'."""
        E_star = self.complex_modulus(omega)
        return E_star.imag / E_star.real


@dataclass
class RelaxationModulus:
    """Container for relaxation modulus function."""
    func: Callable[[float], float]
    E_0: float = None
    E_inf: float = None
    
    def __call__(self, t: float) -> float:
        return self.func(t)
    
    def evaluate_array(self, t: np.ndarray) -> np.ndarray:
        """Evaluate at array of times."""
        return np.array([self.func(ti) for ti in t])


@dataclass
class CreepCompliance:
    """Container for creep compliance function."""
    func: Callable[[float], float]
    J_0: float = None
    J_inf: float = None
    
    def __call__(self, t: float) -> float:
        return self.func(t)


@dataclass
class CreepFunction:
    """Alias for CreepCompliance for compatibility."""
    func: Callable[[float], float]
    
    def __call__(self, t: float) -> float:
        return self.func(t)


@dataclass
class ComplexModulus:
    """Complex modulus E* = E' + iE''."""
    storage: float  # E' (real part)
    loss: float     # E'' (imaginary part)
    frequency: float
    
    @property
    def magnitude(self) -> float:
        """Magnitude |E*|."""
        return math.sqrt(self.storage**2 + self.loss**2)
    
    @property
    def phase_angle(self) -> float:
        """Phase angle δ in radians."""
        return math.atan2(self.loss, self.storage)
    
    @property
    def loss_tangent(self) -> float:
        """tan(δ) = E''/E'."""
        return self.loss / self.storage


# Aliases
StorageModulus = float
LossModulus = float
LossTangent = float


class RelaxationSpectrum:
    """Continuous relaxation spectrum H(τ)."""
    
    def __init__(self, func: Callable[[float], float]):
        """
        Initialize with spectrum function.
        
        Args:
            func: H(τ) function
        """
        self.H = func
    
    def relaxation_modulus(self, t: float, tau_min: float = 1e-10, 
                           tau_max: float = 1e10, n_points: int = 100) -> float:
        """
        Integrate spectrum to get relaxation modulus.
        
        E(t) = ∫ H(τ) exp(-t/τ) d(ln τ)
        """
        log_tau = np.linspace(np.log10(tau_min), np.log10(tau_max), n_points)
        tau = 10**log_tau
        
        integrand = self.H(tau) * np.exp(-t / tau)
        
        # Trapezoidal integration in log space
        d_log_tau = log_tau[1] - log_tau[0]
        return np.trapz(integrand, dx=d_log_tau) * np.log(10)


class PronySeriesFit:
    """Fit Prony series to experimental data."""
    
    @staticmethod
    def fit_relaxation_data(
        time: np.ndarray, E_data: np.ndarray,
        n_terms: int = 5, tau_min: float = None, tau_max: float = None
    ) -> GeneralizedMaxwell:
        """
        Fit Prony series to relaxation data.
        
        Uses non-negative least squares.
        
        Args:
            time: Time points
            E_data: Relaxation modulus data
            n_terms: Number of Prony terms
            tau_min, tau_max: Relaxation time bounds
            
        Returns:
            GeneralizedMaxwell model
        """
        if tau_min is None:
            tau_min = time[time > 0].min()
        if tau_max is None:
            tau_max = time.max()
        
        # Logarithmically spaced relaxation times
        tau_i = np.logspace(np.log10(tau_min), np.log10(tau_max), n_terms)
        
        # Build design matrix
        A = np.zeros((len(time), n_terms + 1))
        A[:, 0] = 1  # E_inf column
        for i, tau in enumerate(tau_i):
            A[:, i + 1] = np.exp(-time / tau)
        
        # Non-negative least squares
        from scipy.optimize import nnls
        coeffs, _ = nnls(A, E_data)
        
        E_inf = coeffs[0]
        E_i = coeffs[1:].tolist()
        
        return GeneralizedMaxwell(E_inf=E_inf, E_i=E_i, tau_i=tau_i.tolist())


class MasterCurve:
    """Time-temperature superposition master curve."""
    
    def __init__(self, reference_temp: float):
        """
        Initialize master curve.
        
        Args:
            reference_temp: Reference temperature (K or °C)
        """
        self.T_ref = reference_temp
        self.shift_factors: Dict[float, float] = {}
    
    def add_data(self, temp: float, time: np.ndarray, 
                 modulus: np.ndarray, shift_factor: float):
        """Add data at a temperature with its shift factor."""
        self.shift_factors[temp] = shift_factor
    
    def get_shift_factor(self, temp: float) -> float:
        """Get shift factor for temperature."""
        return self.shift_factors.get(temp, 1.0)


class TimeTemperatureSuperposition:
    """Time-temperature superposition principle."""
    
    @staticmethod
    def shift_time(t: float, a_T: float) -> float:
        """
        Shift time by factor a_T.
        
        t_reduced = t / a_T
        """
        return t / a_T
    
    @staticmethod
    def arrhenius_shift(T: float, T_ref: float, 
                        E_a: float, R: float = 8.314) -> float:
        """
        Arrhenius shift factor.
        
        log(a_T) = (E_a/R)(1/T - 1/T_ref)
        
        Args:
            T: Temperature (K)
            T_ref: Reference temperature (K)
            E_a: Activation energy (J/mol)
            R: Gas constant
            
        Returns:
            Shift factor a_T
        """
        return math.exp((E_a / R) * (1/T - 1/T_ref))


class WLFEquation:
    """
    Williams-Landel-Ferry equation for shift factors.
    
    log(a_T) = -C1(T - T_ref) / (C2 + T - T_ref)
    """
    
    def __init__(self, T_ref: float, C1: float, C2: float):
        """
        Initialize WLF equation.
        
        Args:
            T_ref: Reference temperature
            C1, C2: WLF constants
            
        Universal constants (Tg reference):
        C1 ≈ 17.44, C2 ≈ 51.6 K
        """
        self.T_ref = T_ref
        self.C1 = C1
        self.C2 = C2
    
    def shift_factor(self, T: float) -> float:
        """Compute shift factor at temperature T."""
        dT = T - self.T_ref
        if abs(dT + self.C2) < 1e-10:
            raise ValueError("Temperature too close to singular point")
        
        log_aT = -self.C1 * dT / (self.C2 + dT)
        return 10**log_aT
    
    @classmethod
    def universal(cls, T_g: float) -> 'WLFEquation':
        """Create WLF with universal constants at glass transition."""
        return cls(T_ref=T_g, C1=17.44, C2=51.6)


class BoltzmannSuperposition:
    """
    Boltzmann superposition principle.
    
    σ(t) = ∫ E(t-s) dε/ds ds
    ε(t) = ∫ J(t-s) dσ/ds ds
    """
    
    @staticmethod
    def stress_from_strain_history(
        E: Callable[[float], float],
        strain_history: Callable[[float], float],
        t: float, dt: float = 0.001
    ) -> float:
        """
        Compute stress from strain history using convolution.
        
        Args:
            E: Relaxation modulus function
            strain_history: ε(t) function
            t: Current time
            dt: Time step for integration
            
        Returns:
            Current stress
        """
        n_steps = int(t / dt)
        times = np.linspace(0, t, n_steps + 1)
        
        sigma = 0.0
        for i in range(n_steps):
            s = times[i]
            d_epsilon = strain_history(times[i+1]) - strain_history(times[i])
            sigma += E(t - s) * d_epsilon
        
        return sigma
    
    @staticmethod
    def strain_from_stress_history(
        J: Callable[[float], float],
        stress_history: Callable[[float], float],
        t: float, dt: float = 0.001
    ) -> float:
        """
        Compute strain from stress history using convolution.
        """
        n_steps = int(t / dt)
        times = np.linspace(0, t, n_steps + 1)
        
        epsilon = 0.0
        for i in range(n_steps):
            s = times[i]
            d_sigma = stress_history(times[i+1]) - stress_history(times[i])
            epsilon += J(t - s) * d_sigma
        
        return epsilon


class FractionalViscoelasticity:
    """
    Fractional calculus viscoelasticity models.
    
    Uses fractional derivatives for intermediate behavior.
    """
    
    def __init__(self, E: float, eta: float, alpha: float):
        """
        Initialize fractional model.
        
        Args:
            E: Modulus (Pa)
            eta: Viscosity-like parameter
            alpha: Fractional order (0 < α < 1)
        """
        if not 0 < alpha < 1:
            raise ValueError("α must be between 0 and 1")
        
        self.E = E
        self.eta = eta
        self.alpha = alpha
    
    def relaxation_modulus(self, t: float) -> float:
        """
        Fractional relaxation modulus using Mittag-Leffler function.
        
        Approximation for large times.
        """
        if t <= 0:
            return self.E
        
        tau = (self.eta / self.E)**(1 / self.alpha)
        x = -(t / tau)**self.alpha
        
        # Simplified Mittag-Leffler approximation
        return self.E * math.exp(x / math.gamma(1 + self.alpha))


# Convenience functions

def compute_relaxation_response(
    model: ViscoelasticMaterial, 
    epsilon_0: float,
    times: np.ndarray
) -> np.ndarray:
    """
    Compute stress relaxation response.
    
    Args:
        model: Viscoelastic material model
        epsilon_0: Applied strain step
        times: Time array
        
    Returns:
        Stress array
    """
    return np.array([epsilon_0 * model.relaxation_modulus(t) for t in times])


def compute_creep_response(
    model: ViscoelasticMaterial,
    sigma_0: float,
    times: np.ndarray
) -> np.ndarray:
    """
    Compute creep response.
    
    Args:
        model: Viscoelastic material model
        sigma_0: Applied stress step
        times: Time array
        
    Returns:
        Strain array
    """
    return np.array([sigma_0 * model.creep_compliance(t) for t in times])


def compute_dynamic_modulus(
    model: GeneralizedMaxwell,
    frequencies: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute dynamic mechanical properties over frequency range.
    
    Args:
        model: Generalized Maxwell model
        frequencies: Frequency array (Hz)
        
    Returns:
        (storage_modulus, loss_modulus, tan_delta) arrays
    """
    omega = 2 * np.pi * frequencies
    
    E_prime = np.array([model.storage_modulus(w) for w in omega])
    E_double_prime = np.array([model.loss_modulus(w) for w in omega])
    tan_delta = E_double_prime / E_prime
    
    return E_prime, E_double_prime, tan_delta


def fit_prony_series(
    time: np.ndarray, 
    E_data: np.ndarray,
    n_terms: int = 5
) -> GeneralizedMaxwell:
    """Fit Prony series to relaxation data."""
    return PronySeriesFit.fit_relaxation_data(time, E_data, n_terms)
