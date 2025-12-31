"""
Quantum Mechanics Domain for MechanicsDSL

Provides tools for semiclassical quantum mechanics, including:
- WKB approximation
- Bohr-Sommerfeld quantization
- Ehrenfest theorem (quantum-classical correspondence)
- Quantum harmonic oscillator
- Path integral formulation basics
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Union
import sympy as sp
import numpy as np
from scipy import integrate
from ..base import PhysicsDomain


# Physical constants (SI units)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
PLANCK_H = 6.62607015e-34  # Planck constant (J·s)


class QuantumState(Enum):
    """Classification of quantum states."""
    BOUND = "bound"
    SCATTERING = "scattering"
    RESONANCE = "resonance"


@dataclass
class EnergyLevel:
    """
    Represents a quantized energy level.
    
    Attributes:
        n: Principal quantum number
        energy: Energy eigenvalue
        degeneracy: Degeneracy of the level
    """
    n: int
    energy: float
    degeneracy: int = 1


class WKBApproximation:
    """
    Implements the WKB (Wentzel-Kramers-Brillouin) approximation.
    
    Valid in the semiclassical limit where the de Broglie wavelength
    varies slowly compared to the potential.
    
    The WKB wavefunction is:
        ψ(x) ≈ C/√p(x) * exp(±i/ℏ ∫p(x')dx')
    
    where p(x) = √(2m(E-V(x))) is the classical momentum.
    
    Example:
        >>> wkb = WKBApproximation(potential=lambda x: 0.5*x**2, mass=1.0)
        >>> levels = wkb.bohr_sommerfeld_levels(n_max=10)
    """
    
    def __init__(self, potential: Callable[[float], float], mass: float = 1.0,
                 hbar: float = 1.0):
        """
        Initialize WKB approximation.
        
        Args:
            potential: Potential energy function V(x)
            mass: Particle mass
            hbar: Reduced Planck constant (default 1 for natural units)
        """
        self.V = potential
        self.mass = mass
        self.hbar = hbar
    
    def classical_momentum(self, x: float, E: float) -> float:
        """
        Calculate classical momentum p(x) = √(2m(E-V(x))).
        
        Returns 0 if E < V(x) (classically forbidden region).
        """
        diff = E - self.V(x)
        if diff < 0:
            return 0.0
        return np.sqrt(2 * self.mass * diff)
    
    def find_turning_points(self, E: float, x_range: Tuple[float, float],
                           n_points: int = 1000) -> List[float]:
        """
        Find classical turning points where E = V(x).
        
        Args:
            E: Total energy
            x_range: (x_min, x_max) search range
            n_points: Number of grid points
            
        Returns:
            List of turning point positions
        """
        x_array = np.linspace(x_range[0], x_range[1], n_points)
        diff = np.array([E - self.V(x) for x in x_array])
        
        # Find sign changes
        turning_points = []
        for i in range(len(diff) - 1):
            if diff[i] * diff[i+1] < 0:
                # Linear interpolation
                x_tp = x_array[i] - diff[i] * (x_array[i+1] - x_array[i]) / (diff[i+1] - diff[i])
                turning_points.append(x_tp)
        
        return turning_points
    
    def action_integral(self, E: float, x1: float, x2: float) -> float:
        """
        Compute action integral ∫p(x)dx between turning points.
        
        Args:
            E: Total energy
            x1: Left turning point
            x2: Right turning point
            
        Returns:
            Action integral value
        """
        def integrand(x):
            return self.classical_momentum(x, E)
        
        result, _ = integrate.quad(integrand, x1, x2)
        return result
    
    def bohr_sommerfeld_condition(self, E: float, x_range: Tuple[float, float]) -> float:
        """
        Evaluate Bohr-Sommerfeld quantization condition.
        
        For bound states: ∮p dx = (n + 1/2)h
        
        Returns the value that should equal (n + 1/2) for valid energies.
        
        Args:
            E: Trial energy
            x_range: Search range for turning points
            
        Returns:
            Quantization condition value
        """
        turning_points = self.find_turning_points(E, x_range)
        
        if len(turning_points) < 2:
            return float('nan')
        
        x1, x2 = turning_points[0], turning_points[-1]
        action = self.action_integral(E, x1, x2)
        
        # Full cycle = 2 * one-way action
        return 2 * action / (2 * np.pi * self.hbar)
    
    def find_energy_level(self, n: int, E_range: Tuple[float, float],
                         x_range: Tuple[float, float]) -> float:
        """
        Find the n-th energy level using Bohr-Sommerfeld quantization.
        
        Args:
            n: Quantum number (0, 1, 2, ...)
            E_range: (E_min, E_max) search range
            x_range: Spatial range for turning points
            
        Returns:
            Energy eigenvalue
        """
        from scipy.optimize import brentq
        
        target = n + 0.5  # Bohr-Sommerfeld: n + 1/2
        
        def objective(E):
            return self.bohr_sommerfeld_condition(E, x_range) - target
        
        try:
            E_n = brentq(objective, E_range[0], E_range[1])
            return E_n
        except ValueError:
            return float('nan')
    
    def bohr_sommerfeld_levels(self, n_max: int, E_range: Tuple[float, float],
                              x_range: Tuple[float, float]) -> List[EnergyLevel]:
        """
        Compute multiple energy levels.
        
        Args:
            n_max: Maximum quantum number
            E_range: Energy search range
            x_range: Spatial range
            
        Returns:
            List of EnergyLevel objects
        """
        levels = []
        
        # Subdivide energy range for each level
        E_min, E_max = E_range
        dE = (E_max - E_min) / (n_max + 2)
        
        for n in range(n_max + 1):
            E_n = self.find_energy_level(n, (E_min + n*dE*0.5, E_max), x_range)
            if not np.isnan(E_n):
                levels.append(EnergyLevel(n=n, energy=E_n))
        
        return levels


class QuantumHarmonicOscillator:
    """
    Exact quantum harmonic oscillator solution.
    
    H = p²/(2m) + (1/2)mω²x²
    
    Energy levels: E_n = ℏω(n + 1/2)
    """
    
    def __init__(self, mass: float = 1.0, omega: float = 1.0, hbar: float = 1.0):
        """
        Initialize quantum harmonic oscillator.
        
        Args:
            mass: Particle mass
            omega: Angular frequency
            hbar: Reduced Planck constant
        """
        self.mass = mass
        self.omega = omega
        self.hbar = hbar
    
    def energy_level(self, n: int) -> float:
        """
        Exact energy eigenvalue.
        
        E_n = ℏω(n + 1/2)
        """
        return self.hbar * self.omega * (n + 0.5)
    
    def zero_point_energy(self) -> float:
        """Ground state energy E_0 = ℏω/2."""
        return self.hbar * self.omega / 2
    
    def characteristic_length(self) -> float:
        """
        Characteristic length scale a = √(ℏ/(mω)).
        
        This is the ground state width.
        """
        return np.sqrt(self.hbar / (self.mass * self.omega))
    
    def classical_amplitude(self, n: int) -> float:
        """
        Classical turning point for energy level n.
        
        x_max = √(2E_n / (mω²))
        """
        E_n = self.energy_level(n)
        return np.sqrt(2 * E_n / (self.mass * self.omega**2))
    
    def wavefunction(self, x: np.ndarray, n: int) -> np.ndarray:
        """
        Normalized wavefunction ψ_n(x).
        
        Uses Hermite polynomials.
        """
        from scipy.special import hermite
        from math import factorial
        
        a = self.characteristic_length()
        xi = x / a
        
        # Hermite polynomial
        H_n = hermite(n)
        
        # Normalization
        norm = 1.0 / np.sqrt(2**n * factorial(n)) * (1 / (np.pi * a**2))**0.25
        
        return norm * np.exp(-xi**2 / 2) * H_n(xi)
    
    def probability_density(self, x: np.ndarray, n: int) -> np.ndarray:
        """Probability density |ψ_n(x)|²."""
        psi = self.wavefunction(x, n)
        return np.abs(psi)**2
    
    def position_expectation(self, n: int) -> float:
        """Expectation value <x> = 0 for all n."""
        return 0.0
    
    def position_variance(self, n: int) -> float:
        """
        Variance <x²> = a²(n + 1/2).
        """
        a = self.characteristic_length()
        return a**2 * (n + 0.5)
    
    def momentum_variance(self, n: int) -> float:
        """
        Variance <p²> = (ℏ/a)²(n + 1/2).
        """
        a = self.characteristic_length()
        return (self.hbar / a)**2 * (n + 0.5)
    
    def uncertainty_product(self, n: int) -> float:
        """
        Uncertainty product Δx·Δp = ℏ(n + 1/2).
        
        Minimum (ℏ/2) for ground state n=0.
        """
        return self.hbar * (n + 0.5)


class EhrenfestDynamics:
    """
    Ehrenfest theorem: quantum-classical correspondence.
    
    d<x>/dt = <p>/m
    d<p>/dt = -<dV/dx>
    
    Expectation values follow classical equations for quadratic potentials.
    """
    
    def __init__(self, potential: Callable[[float], float], 
                 potential_derivative: Callable[[float], float],
                 mass: float = 1.0):
        """
        Initialize Ehrenfest dynamics.
        
        Args:
            potential: V(x)
            potential_derivative: dV/dx
            mass: Particle mass
        """
        self.V = potential
        self.dV = potential_derivative
        self.mass = mass
    
    def classical_force(self, x: float) -> float:
        """Classical force F = -dV/dx."""
        return -self.dV(x)
    
    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Classical equations for expectation values.
        
        Args:
            t: Time
            y: State vector [<x>, <p>]
            
        Returns:
            Derivatives [d<x>/dt, d<p>/dt]
        """
        x_exp = y[0]
        p_exp = y[1]
        
        dx_dt = p_exp / self.mass
        dp_dt = self.classical_force(x_exp)
        
        return np.array([dx_dt, dp_dt])
    
    def propagate(self, x0: float, p0: float, t_span: Tuple[float, float],
                 n_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Propagate expectation values in time.
        
        Args:
            x0: Initial <x>
            p0: Initial <p>
            t_span: (t_start, t_end)
            n_points: Number of output points
            
        Returns:
            Dictionary with t, x_exp, p_exp arrays
        """
        from scipy.integrate import solve_ivp
        
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        sol = solve_ivp(self.equations_of_motion, t_span, [x0, p0], 
                       t_eval=t_eval, method='RK45')
        
        return {
            't': sol.t,
            'x_exp': sol.y[0],
            'p_exp': sol.y[1]
        }


class InfiniteSquareWell:
    """
    Particle in an infinite square well (1D box).
    
    V(x) = 0 for 0 < x < L
    V(x) = ∞ otherwise
    
    Energy levels: E_n = n²π²ℏ²/(2mL²)
    """
    
    def __init__(self, length: float = 1.0, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize infinite square well.
        
        Args:
            length: Well width L
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        self.L = length
        self.mass = mass
        self.hbar = hbar
    
    def energy_level(self, n: int) -> float:
        """
        Energy eigenvalue for quantum number n (n = 1, 2, 3, ...).
        
        E_n = n²π²ℏ²/(2mL²)
        """
        if n < 1:
            raise ValueError("n must be >= 1 for infinite square well")
        return (n**2 * np.pi**2 * self.hbar**2) / (2 * self.mass * self.L**2)
    
    def wavefunction(self, x: np.ndarray, n: int) -> np.ndarray:
        """
        Normalized wavefunction ψ_n(x) = √(2/L) sin(nπx/L).
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        
        psi = np.sqrt(2 / self.L) * np.sin(n * np.pi * x / self.L)
        # Zero outside the well
        psi = np.where((x >= 0) & (x <= self.L), psi, 0.0)
        return psi
    
    def probability_density(self, x: np.ndarray, n: int) -> np.ndarray:
        """Probability density |ψ_n(x)|²."""
        return np.abs(self.wavefunction(x, n))**2
    
    def position_expectation(self, n: int) -> float:
        """<x> = L/2 for all n."""
        return self.L / 2
    
    def position_variance(self, n: int) -> float:
        """<x²> - <x>² for level n."""
        x_sq = self.L**2 * (1/3 - 1/(2*n**2*np.pi**2))
        return x_sq - (self.L/2)**2


# Convenience functions

def de_broglie_wavelength(momentum: float, hbar: float = HBAR) -> float:
    """
    Calculate de Broglie wavelength λ = h/p = 2πℏ/p.
    
    Args:
        momentum: Particle momentum
        hbar: Reduced Planck constant
        
    Returns:
        de Broglie wavelength
    """
    return 2 * np.pi * hbar / momentum


def compton_wavelength(mass: float, hbar: float = HBAR, c: float = 299792458.0) -> float:
    """
    Calculate Compton wavelength λ_C = h/(mc) = 2πℏ/(mc).
    
    Args:
        mass: Particle mass
        hbar: Reduced Planck constant
        c: Speed of light
        
    Returns:
        Compton wavelength
    """
    return 2 * np.pi * hbar / (mass * c)


def heisenberg_minimum(hbar: float = 1.0) -> float:
    """
    Minimum uncertainty product Δx·Δp ≥ ℏ/2.
    
    Returns:
        Minimum uncertainty product
    """
    return hbar / 2


__all__ = [
    'HBAR',
    'PLANCK_H',
    'QuantumState',
    'EnergyLevel',
    'WKBApproximation',
    'QuantumHarmonicOscillator',
    'EhrenfestDynamics',
    'InfiniteSquareWell',
    'de_broglie_wavelength',
    'compton_wavelength',
    'heisenberg_minimum',
]
