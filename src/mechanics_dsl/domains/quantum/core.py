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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate

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

    def __init__(self, potential: Callable[[float], float], mass: float = 1.0, hbar: float = 1.0):
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

    def find_turning_points(
        self, E: float, x_range: Tuple[float, float], n_points: int = 1000
    ) -> List[float]:
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
            if diff[i] * diff[i + 1] < 0:
                # Linear interpolation
                x_tp = x_array[i] - diff[i] * (x_array[i + 1] - x_array[i]) / (
                    diff[i + 1] - diff[i]
                )
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
            return float("nan")

        x1, x2 = turning_points[0], turning_points[-1]
        action = self.action_integral(E, x1, x2)

        # Full cycle = 2 * one-way action
        return 2 * action / (2 * np.pi * self.hbar)

    def find_energy_level(
        self, n: int, E_range: Tuple[float, float], x_range: Tuple[float, float]
    ) -> float:
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
            return float("nan")

    def bohr_sommerfeld_levels(
        self, n_max: int, E_range: Tuple[float, float], x_range: Tuple[float, float]
    ) -> List[EnergyLevel]:
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
            E_n = self.find_energy_level(n, (E_min + n * dE * 0.5, E_max), x_range)
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
        from math import factorial

        from scipy.special import hermite

        a = self.characteristic_length()
        xi = x / a

        # Hermite polynomial
        H_n = hermite(n)

        # Normalization
        norm = 1.0 / np.sqrt(2**n * factorial(n)) * (1 / (np.pi * a**2)) ** 0.25

        return norm * np.exp(-(xi**2) / 2) * H_n(xi)

    def probability_density(self, x: np.ndarray, n: int) -> np.ndarray:
        """Probability density |ψ_n(x)|²."""
        psi = self.wavefunction(x, n)
        return np.abs(psi) ** 2

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
        return (self.hbar / a) ** 2 * (n + 0.5)

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

    def __init__(
        self,
        potential: Callable[[float], float],
        potential_derivative: Callable[[float], float],
        mass: float = 1.0,
    ):
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

    def propagate(
        self, x0: float, p0: float, t_span: Tuple[float, float], n_points: int = 100
    ) -> Dict[str, np.ndarray]:
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

        sol = solve_ivp(self.equations_of_motion, t_span, [x0, p0], t_eval=t_eval, method="RK45")

        return {"t": sol.t, "x_exp": sol.y[0], "p_exp": sol.y[1]}


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
        return np.abs(self.wavefunction(x, n)) ** 2

    def position_expectation(self, n: int) -> float:
        """<x> = L/2 for all n."""
        return self.L / 2

    def position_variance(self, n: int) -> float:
        """<x²> - <x>² for level n."""
        x_sq = self.L**2 * (1 / 3 - 1 / (2 * n**2 * np.pi**2))
        return x_sq - (self.L / 2) ** 2


class FiniteSquareWell:
    """
    Particle in a finite square well (bound states).

    V(x) = -V₀ for |x| < a/2
    V(x) = 0   otherwise

    Bound state energies satisfy transcendental equations:
    - Even parity: k tan(ka/2) = κ
    - Odd parity:  k cot(ka/2) = -κ

    where k = √(2m(E+V₀))/ℏ and κ = √(-2mE)/ℏ

    Always has at least one bound state.

    Example:
        >>> well = FiniteSquareWell(depth=10.0, width=2.0)
        >>> energies = well.find_bound_states()
    """

    def __init__(self, depth: float, width: float, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize finite square well.

        Args:
            depth: Well depth V₀ (positive)
            width: Well width a
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        if depth <= 0:
            raise ValueError("Well depth must be positive")
        self.V0 = depth
        self.a = width
        self.mass = mass
        self.hbar = hbar

    def dimensionless_parameter(self) -> float:
        """
        Compute dimensionless well parameter z₀ = a√(2mV₀)/(2ℏ).

        The number of bound states is ⌊z₀/π⌋ + 1.
        """
        return (self.a / 2) * np.sqrt(2 * self.mass * self.V0) / self.hbar

    def max_bound_states(self) -> int:
        """Maximum number of bound states."""
        z0 = self.dimensionless_parameter()
        return int(np.floor(z0 / (np.pi / 2))) + 1

    def _even_parity_equation(self, E: float) -> float:
        """Transcendental equation for even parity states: k tan(ka/2) - κ = 0."""
        if E >= 0 or E < -self.V0:
            return float("inf")

        k = np.sqrt(2 * self.mass * (E + self.V0)) / self.hbar
        kappa = np.sqrt(-2 * self.mass * E) / self.hbar

        return k * np.tan(k * self.a / 2) - kappa

    def _odd_parity_equation(self, E: float) -> float:
        """Transcendental equation for odd parity states: -k cot(ka/2) - κ = 0."""
        if E >= 0 or E < -self.V0:
            return float("inf")

        k = np.sqrt(2 * self.mass * (E + self.V0)) / self.hbar
        kappa = np.sqrt(-2 * self.mass * E) / self.hbar

        tan_val = np.tan(k * self.a / 2)
        if abs(tan_val) < 1e-10:
            return float("inf")

        return -k / tan_val - kappa

    def find_bound_states(self, n_search: int = 100) -> List[EnergyLevel]:
        """
        Find all bound state energies by solving transcendental equations.

        Args:
            n_search: Number of search points in energy grid

        Returns:
            List of EnergyLevel objects for bound states
        """
        from scipy.optimize import brentq

        levels = []
        E_grid = np.linspace(-self.V0 * 0.999, -1e-10, n_search)

        # Find even parity states
        for i in range(len(E_grid) - 1):
            try:
                f1 = self._even_parity_equation(E_grid[i])
                f2 = self._even_parity_equation(E_grid[i + 1])
                if f1 * f2 < 0 and np.isfinite(f1) and np.isfinite(f2):
                    E = brentq(self._even_parity_equation, E_grid[i], E_grid[i + 1])
                    levels.append(EnergyLevel(n=len(levels), energy=E))
            except (ValueError, RuntimeError):
                pass

        # Find odd parity states
        for i in range(len(E_grid) - 1):
            try:
                f1 = self._odd_parity_equation(E_grid[i])
                f2 = self._odd_parity_equation(E_grid[i + 1])
                if f1 * f2 < 0 and np.isfinite(f1) and np.isfinite(f2):
                    E = brentq(self._odd_parity_equation, E_grid[i], E_grid[i + 1])
                    levels.append(EnergyLevel(n=len(levels), energy=E))
            except (ValueError, RuntimeError):
                pass

        # Sort by energy
        levels.sort(key=lambda x: x.energy)
        for i, level in enumerate(levels):
            level.n = i

        return levels

    def transmission_coefficient(self, E: float) -> float:
        """
        Transmission coefficient for scattering states (E > 0).

        T = 1 / (1 + V₀²sin²(k₁a)/(4E(E+V₀)))

        where k₁ = √(2m(E+V₀))/ℏ
        """
        if E <= 0:
            return 0.0

        k1 = np.sqrt(2 * self.mass * (E + self.V0)) / self.hbar
        sin_term = np.sin(k1 * self.a) ** 2

        denominator = 1 + (self.V0**2 * sin_term) / (4 * E * (E + self.V0))
        return 1.0 / denominator


class StepPotential:
    """
    Quantum step potential (transmission and reflection).

    V(x) = 0   for x < 0
    V(x) = V₀  for x ≥ 0

    Exact transmission and reflection coefficients:

    For E > V₀:
        R = ((k₁ - k₂)/(k₁ + k₂))²
        T = 4k₁k₂/(k₁ + k₂)²

    For E < V₀:
        R = 1 (total reflection)
        T = 0

    where k₁ = √(2mE)/ℏ, k₂ = √(2m(E-V₀))/ℏ

    Example:
        >>> step = StepPotential(height=5.0)
        >>> R, T = step.reflection_transmission(E=10.0)
    """

    def __init__(self, height: float, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize step potential.

        Args:
            height: Step height V₀
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        self.V0 = height
        self.mass = mass
        self.hbar = hbar

    def reflection_transmission(self, E: float) -> Tuple[float, float]:
        """
        Calculate reflection and transmission coefficients.

        Args:
            E: Particle energy (must be > 0)

        Returns:
            Tuple of (R, T) where R + T = 1
        """
        if E <= 0:
            raise ValueError("Energy must be positive")

        k1 = np.sqrt(2 * self.mass * E) / self.hbar

        if E > self.V0:
            # Above barrier: partial transmission
            k2 = np.sqrt(2 * self.mass * (E - self.V0)) / self.hbar
            R = ((k1 - k2) / (k1 + k2)) ** 2
            T = 4 * k1 * k2 / (k1 + k2) ** 2
        else:
            # Below barrier: total reflection (no tunneling for step)
            R = 1.0
            T = 0.0

        return R, T

    def penetration_depth(self, E: float) -> float:
        """
        Calculate penetration depth into forbidden region.

        δ = ℏ / √(2m(V₀-E))

        Args:
            E: Particle energy (E < V₀)

        Returns:
            Penetration depth
        """
        if E >= self.V0:
            return float("inf")

        kappa = np.sqrt(2 * self.mass * (self.V0 - E)) / self.hbar
        return 1.0 / kappa


class DeltaFunctionBarrier:
    """
    Delta function potential barrier: V(x) = λδ(x)

    A thin, infinitely high barrier with finite area.

    Transmission coefficient:
        T = 1 / (1 + mλ²/(2ℏ²E))

    Reflection coefficient:
        R = 1 - T = (mλ²/(2ℏ²E)) / (1 + mλ²/(2ℏ²E))

    Example:
        >>> barrier = DeltaFunctionBarrier(strength=1.0)
        >>> T = barrier.transmission(E=0.5)
    """

    def __init__(self, strength: float, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize delta function barrier.

        Args:
            strength: Barrier strength λ (positive for barrier, negative for well)
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        self.lambda_ = strength
        self.mass = mass
        self.hbar = hbar

    def transmission(self, E: float) -> float:
        """
        Transmission coefficient T(E).

        T = 1 / (1 + mλ²/(2ℏ²E))
        """
        if E <= 0:
            return 0.0

        factor = self.mass * self.lambda_**2 / (2 * self.hbar**2 * E)
        return 1.0 / (1 + factor)

    def reflection(self, E: float) -> float:
        """Reflection coefficient R(E) = 1 - T(E)."""
        return 1.0 - self.transmission(E)

    def bound_state_energy(self) -> Optional[float]:
        """
        Bound state energy for attractive delta well (λ < 0).

        E = -mλ²/(2ℏ²)

        Returns:
            Bound state energy, or None if λ ≥ 0
        """
        if self.lambda_ >= 0:
            return None

        return -self.mass * self.lambda_**2 / (2 * self.hbar**2)


class HydrogenAtom:
    """
    Hydrogen atom energy levels and wavefunctions.

    Exact Bohr model energies:
        E_n = -13.6 eV / n² = -m_e e⁴ / (2(4πε₀)²ℏ²n²)

    Bohr radius:
        a₀ = 4πε₀ℏ² / (m_e e²) ≈ 0.529 Å

    Example:
        >>> hydrogen = HydrogenAtom()
        >>> E1 = hydrogen.energy_level(n=1)  # Ground state: -13.6 eV
    """

    # Physical constants (SI)
    ELECTRON_MASS = 9.109e-31  # kg
    ELEMENTARY_CHARGE = 1.602e-19  # C
    BOHR_RADIUS = 5.292e-11  # m
    RYDBERG_ENERGY = 13.6  # eV

    def __init__(self, Z: int = 1, reduced_mass: Optional[float] = None):
        """
        Initialize hydrogen-like atom.

        Args:
            Z: Nuclear charge (Z=1 for hydrogen)
            reduced_mass: Reduced mass (default: electron mass)
        """
        self.Z = Z
        self.mu = reduced_mass if reduced_mass else self.ELECTRON_MASS

    def energy_level(self, n: int) -> float:
        """
        Energy eigenvalue for principal quantum number n.

        E_n = -Z² × 13.6 eV / n²

        Args:
            n: Principal quantum number (n ≥ 1)

        Returns:
            Energy in eV (negative for bound states)
        """
        if n < 1:
            raise ValueError("n must be ≥ 1")

        return -self.Z**2 * self.RYDBERG_ENERGY / n**2

    def energy_level_joules(self, n: int) -> float:
        """Energy in Joules."""
        return self.energy_level(n) * self.ELEMENTARY_CHARGE

    def bohr_radius_n(self, n: int) -> float:
        """
        Most probable radius for state n.

        r_n = n² × a₀ / Z
        """
        return n**2 * self.BOHR_RADIUS / self.Z

    def ionization_energy(self) -> float:
        """Ionization energy (energy to remove electron from ground state)."""
        return -self.energy_level(1)

    def degeneracy(self, n: int) -> int:
        """
        Degeneracy of energy level n.

        g_n = 2n² (including spin)
        """
        return 2 * n**2

    def orbital_angular_momentum(self, l: int) -> float:
        """
        Orbital angular momentum magnitude.

        L = √(l(l+1)) × ℏ
        """
        return np.sqrt(l * (l + 1)) * HBAR

    def radial_probability_max(self, n: int, l: int) -> float:
        """
        Most probable radius for quantum numbers (n, l).

        For l = n-1 (circular orbits): r_max = n² × a₀ / Z
        """
        if l >= n or l < 0:
            raise ValueError("l must satisfy 0 ≤ l < n")

        # For general (n, l), approximate using Bohr model
        return n**2 * self.BOHR_RADIUS / self.Z

    def transition_wavelength(self, n_initial: int, n_final: int) -> float:
        """
        Wavelength of photon emitted/absorbed in transition.

        1/λ = R_H × Z² × |1/n_f² - 1/n_i²|

        where R_H = 1.097e7 m⁻¹ (Rydberg constant)

        Returns:
            Wavelength in meters
        """
        R_H = 1.097e7  # Rydberg constant

        delta_inv = abs(1 / n_final**2 - 1 / n_initial**2)
        inv_lambda = R_H * self.Z**2 * delta_inv

        return 1.0 / inv_lambda

    def transition_energy(self, n_initial: int, n_final: int) -> float:
        """Energy of photon in transition (eV)."""
        return abs(self.energy_level(n_initial) - self.energy_level(n_final))

    def spectral_series(self, n_final: int, n_max: int = 7) -> Dict[str, float]:
        """
        Calculate spectral series wavelengths.

        Args:
            n_final: Final state (1=Lyman, 2=Balmer, 3=Paschen, etc.)
            n_max: Maximum initial state

        Returns:
            Dictionary of wavelengths keyed by transition name
        """
        series_names = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett", 5: "Pfund"}
        series = {}

        for n_i in range(n_final + 1, n_max + 1):
            name = f"{series_names.get(n_final, f'n={n_final}')}_{n_i}->{n_final}"
            series[name] = self.transition_wavelength(n_i, n_final)

        return series


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


class QuantumTunneling:
    """
    Quantum tunneling calculations for potential barriers.

    Implements:
    - WKB tunneling approximation: T ≈ exp(-2∫κ dx)
    - Exact rectangular barrier solution
    - Gamow tunneling factor for alpha decay
    - Double-well tunneling splitting

    The WKB transmission coefficient through a barrier:
        T ≈ exp(-2/ℏ ∫√(2m(V(x)-E)) dx)

    where the integral is over the classically forbidden region.

    Example:
        >>> tunneling = QuantumTunneling(mass=1.0, hbar=1.0)
        >>> T = tunneling.rectangular_barrier(E=1.0, V0=2.0, width=1.0)
    """

    def __init__(self, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize tunneling calculator.

        Args:
            mass: Particle mass
            hbar: Reduced Planck constant
        """
        self.mass = mass
        self.hbar = hbar

    def decay_constant(self, E: float, V: float) -> float:
        """
        Calculate decay constant κ = √(2m(V-E))/ℏ in forbidden region.

        Args:
            E: Particle energy
            V: Barrier potential

        Returns:
            Decay constant κ (imaginary wavevector)
        """
        if V <= E:
            return 0.0
        return np.sqrt(2 * self.mass * (V - E)) / self.hbar

    def rectangular_barrier(self, E: float, V0: float, width: float) -> float:
        """
        Exact transmission coefficient for rectangular barrier.

        For E < V0:
            T = 1 / (1 + (V0²sinh²(κa))/(4E(V0-E)))

        where κ = √(2m(V0-E))/ℏ and a = width.

        For E > V0 (above barrier):
            T = 1 / (1 + (V0²sin²(ka))/(4E(E-V0)))

        Args:
            E: Particle energy (E > 0)
            V0: Barrier height
            width: Barrier width

        Returns:
            Transmission probability T ∈ [0, 1]
        """
        if E <= 0:
            return 0.0

        if E < V0:
            # Tunneling regime
            kappa = np.sqrt(2 * self.mass * (V0 - E)) / self.hbar
            kappa_a = kappa * width

            # Prevent overflow for large barriers
            if kappa_a > 50:
                return 0.0

            sinh_term = np.sinh(kappa_a) ** 2
            denominator = 1 + (V0**2 * sinh_term) / (4 * E * (V0 - E))
            return 1.0 / denominator

        else:
            # Above-barrier scattering
            k = np.sqrt(2 * self.mass * (E - V0)) / self.hbar
            ka = k * width
            sin_term = np.sin(ka) ** 2

            if E == V0:
                return 1.0

            denominator = 1 + (V0**2 * sin_term) / (4 * E * (E - V0))
            return 1.0 / denominator

    def wkb_transmission(
        self,
        E: float,
        potential: Callable[[float], float],
        x1: float,
        x2: float,
        n_points: int = 1000,
    ) -> float:
        """
        WKB tunneling transmission coefficient.

        T ≈ exp(-2γ) where γ = (1/ℏ) ∫_{x1}^{x2} √(2m(V(x)-E)) dx

        Args:
            E: Particle energy
            potential: Potential function V(x)
            x1: Left turning point (entry into barrier)
            x2: Right turning point (exit from barrier)
            n_points: Integration points

        Returns:
            WKB transmission coefficient
        """
        from scipy.integrate import quad

        def integrand(x):
            V = potential(x)
            if V > E:
                return np.sqrt(2 * self.mass * (V - E))
            return 0.0

        gamma, _ = quad(integrand, x1, x2)
        gamma /= self.hbar

        # Transmission coefficient
        if gamma > 50:  # Prevent underflow
            return 0.0
        return np.exp(-2 * gamma)

    def gamow_factor(self, E: float, Z1: int, Z2: int, R_nuclear: float = 1e-14) -> float:
        """
        Gamow tunneling factor for alpha decay / nuclear reactions.

        For Coulomb barrier with V(r) = Z1*Z2*e²/(4πε₀r):

        G = exp(-2π * η) where η = Z1*Z2*e²/(4πε₀*ℏ*v)

        Simplified formula:
        G ≈ exp(-2π * Z1*Z2 * sqrt(m/(2E)) * e²/(4πε₀*ℏ))

        Args:
            E: Kinetic energy (Joules)
            Z1, Z2: Atomic numbers
            R_nuclear: Nuclear radius (≈ 1 fm)

        Returns:
            Gamow penetration factor
        """
        # Physical constants (SI)
        e = 1.602e-19  # Elementary charge
        epsilon_0 = 8.854e-12  # Permittivity
        k_coulomb = 1 / (4 * np.pi * epsilon_0)

        # Sommerfeld parameter
        v = np.sqrt(2 * E / self.mass)  # Velocity
        eta = Z1 * Z2 * k_coulomb * e**2 / (self.hbar * v)

        return np.exp(-2 * np.pi * eta)

    def tunneling_time_wkb(
        self, E: float, potential: Callable[[float], float], x1: float, x2: float
    ) -> float:
        """
        Estimate tunneling traversal time (Büttiker-Landauer time).

        τ = m * ∫_{x1}^{x2} dx / √(2m(V(x)-E))

        Note: Tunneling time is a subtle concept with multiple definitions.
        This gives the "dwell time" in the barrier region.

        Args:
            E: Particle energy
            potential: Potential function V(x)
            x1: Left turning point
            x2: Right turning point

        Returns:
            Characteristic tunneling time
        """
        from scipy.integrate import quad

        def integrand(x):
            V = potential(x)
            if V > E:
                kappa = np.sqrt(2 * self.mass * (V - E))
                return self.mass / kappa
            return 0.0

        tau, _ = quad(integrand, x1, x2)
        return tau

    def double_well_splitting(
        self, omega: float, barrier_height: float, well_separation: float
    ) -> float:
        """
        Tunnel splitting for symmetric double-well potential.

        For V(x) = V0 * ((x/a)² - 1)² with minima at x = ±a:

        ΔE ≈ ℏω * exp(-S_inst/ℏ)

        where S_inst is the instanton action.

        Approximate formula:
        ΔE ≈ ℏω * (8V0/(ℏω))^(1/2) * exp(-πV0/(ℏω))

        Args:
            omega: Ground state angular frequency
            barrier_height: Height of central barrier V0
            well_separation: Distance between well minima 2a

        Returns:
            Energy splitting between symmetric/antisymmetric states
        """
        ratio = barrier_height / (self.hbar * omega)

        if ratio > 50:  # Prevent underflow
            return 0.0

        prefactor = self.hbar * omega * np.sqrt(8 * ratio / np.pi)
        exponent = -np.pi * ratio / 2

        return prefactor * np.exp(exponent)

    def resonant_tunneling_peaks(
        self,
        V0_left: float,
        V0_right: float,
        well_width: float,
        barrier_widths: Tuple[float, float],
        n_max: int = 5,
    ) -> List[float]:
        """
        Find resonant tunneling energies for double-barrier structure.

        Resonances occur when the phase condition is satisfied:
        2*k*w + φ_L + φ_R = 2πn

        At resonance, T → 1 (unity transmission).

        Args:
            V0_left, V0_right: Barrier heights
            well_width: Width of central well
            barrier_widths: (left_barrier_width, right_barrier_width)
            n_max: Maximum quantum number

        Returns:
            List of resonance energies
        """
        resonances = []

        # Approximate: quantized levels in finite well
        # E_n ≈ n²π²ℏ²/(2m*w²) for infinitely deep well
        for n in range(1, n_max + 1):
            E_approx = (n**2 * np.pi**2 * self.hbar**2) / (2 * self.mass * well_width**2)

            # Only include if below barrier
            if E_approx < min(V0_left, V0_right):
                resonances.append(E_approx)

        return resonances


def tunneling_probability_rectangular(
    E: float, V0: float, width: float, mass: float = 1.0, hbar: float = 1.0
) -> float:
    """
    Convenience function for rectangular barrier tunneling.

    Args:
        E: Particle energy
        V0: Barrier height
        width: Barrier width
        mass: Particle mass
        hbar: Reduced Planck constant

    Returns:
        Transmission probability
    """
    tunneling = QuantumTunneling(mass=mass, hbar=hbar)
    return tunneling.rectangular_barrier(E, V0, width)


def alpha_decay_rate(
    E_alpha: float,
    Z_daughter: int,
    R_nuclear: float = 1.4e-15 * 4 ** (1 / 3),
    mass_alpha: float = 6.644e-27,
) -> float:
    """
    Estimate alpha decay rate using Gamow formula.

    λ ≈ (v/2R) * G²

    where G is the Gamow factor and v is alpha velocity.

    Args:
        E_alpha: Alpha particle kinetic energy (J)
        Z_daughter: Atomic number of daughter nucleus
        R_nuclear: Nuclear radius (default for A≈4)
        mass_alpha: Alpha particle mass

    Returns:
        Decay rate (s⁻¹)
    """
    tunneling = QuantumTunneling(mass=mass_alpha, hbar=HBAR)

    # Alpha velocity
    v = np.sqrt(2 * E_alpha / mass_alpha)

    # Gamow factor (Z_alpha = 2)
    G = tunneling.gamow_factor(E_alpha, Z1=2, Z2=Z_daughter)

    # Attempt frequency ~ v / (2R)
    attempt_freq = v / (2 * R_nuclear)

    return attempt_freq * G**2


__all__ = [
    "HBAR",
    "PLANCK_H",
    "QuantumState",
    "EnergyLevel",
    "WKBApproximation",
    "QuantumHarmonicOscillator",
    "EhrenfestDynamics",
    "InfiniteSquareWell",
    "FiniteSquareWell",
    "StepPotential",
    "DeltaFunctionBarrier",
    "HydrogenAtom",
    "QuantumTunneling",
    "de_broglie_wavelength",
    "compton_wavelength",
    "heisenberg_minimum",
    "tunneling_probability_rectangular",
    "alpha_decay_rate",
]
