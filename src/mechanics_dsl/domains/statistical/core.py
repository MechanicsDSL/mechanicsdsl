"""
Statistical Mechanics Domain for MechanicsDSL

Provides tools for statistical mechanics calculations, including:
- Microcanonical, canonical, and grand canonical ensembles
- Partition functions
- Boltzmann distribution
- Entropy and free energy
- Ising model
- Ideal gas
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import factorial

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
AVOGADRO_NUMBER = 6.02214076e23  # 1/mol
GAS_CONSTANT = 8.314462  # J/(mol·K)
PLANCK_CONSTANT = 6.62607015e-34  # J·s


class EnsembleType(Enum):
    """Statistical ensemble types."""

    MICROCANONICAL = "microcanonical"  # Fixed E, N, V
    CANONICAL = "canonical"  # Fixed T, N, V
    GRAND_CANONICAL = "grand_canonical"  # Fixed T, μ, V
    ISOTHERMAL_ISOBARIC = "npt"  # Fixed T, P, N


@dataclass
class ThermodynamicState:
    """
    Represents a thermodynamic state.

    Attributes:
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        volume: Volume (m³)
        particle_count: Number of particles
        energy: Internal energy (J)
        entropy: Entropy (J/K)
    """

    temperature: float = 300.0
    pressure: float = 101325.0
    volume: float = 1.0
    particle_count: int = 6.02e23
    energy: Optional[float] = None
    entropy: Optional[float] = None


class BoltzmannDistribution:
    """
    Boltzmann distribution for thermal equilibrium.

    P(E) ∝ g(E) × exp(-E/(k_B T))

    where g(E) is the degeneracy of state with energy E.

    Example:
        >>> boltz = BoltzmannDistribution(temperature=300)
        >>> prob_ratio = boltz.probability_ratio(E1=0.1, E2=0.2)
    """

    def __init__(self, temperature: float, k_B: float = BOLTZMANN_CONSTANT):
        """
        Initialize Boltzmann distribution.

        Args:
            temperature: Temperature (K)
            k_B: Boltzmann constant
        """
        self.T = temperature
        self.k_B = k_B
        self.beta = 1 / (k_B * temperature)  # Inverse temperature

    def boltzmann_factor(self, energy: float) -> float:
        """
        Calculate Boltzmann factor exp(-βE).

        Args:
            energy: Energy of state (J)
        """
        return np.exp(-self.beta * energy)

    def probability_ratio(self, E1: float, E2: float) -> float:
        """
        Ratio of occupation probabilities P(E1)/P(E2).

        P(E1)/P(E2) = exp(-(E1-E2)/(k_B T))
        """
        return np.exp(-self.beta * (E1 - E2))

    def average_energy(
        self, energies: np.ndarray, degeneracies: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate ensemble average energy.

        <E> = Σ g_i E_i exp(-βE_i) / Z
        """
        if degeneracies is None:
            degeneracies = np.ones_like(energies)

        weights = degeneracies * np.exp(-self.beta * energies)
        Z = np.sum(weights)

        return np.sum(energies * weights) / Z

    def partition_function(
        self, energies: np.ndarray, degeneracies: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate partition function Z = Σ g_i exp(-βE_i).
        """
        if degeneracies is None:
            degeneracies = np.ones_like(energies)

        return np.sum(degeneracies * np.exp(-self.beta * energies))

    def thermal_wavelength(self, mass: float) -> float:
        """
        Thermal de Broglie wavelength.

        λ = h / √(2πmk_B T)
        """
        return PLANCK_CONSTANT / np.sqrt(2 * np.pi * mass * self.k_B * self.T)

    def maxwell_speed_distribution(self, v: np.ndarray, mass: float) -> np.ndarray:
        """
        Maxwell-Boltzmann speed distribution.

        f(v) = 4π (m/(2πk_B T))^(3/2) v² exp(-mv²/(2k_B T))
        """
        A = 4 * np.pi * (mass / (2 * np.pi * self.k_B * self.T)) ** 1.5
        return A * v**2 * np.exp(-mass * v**2 / (2 * self.k_B * self.T))

    def most_probable_speed(self, mass: float) -> float:
        """Most probable speed v_p = √(2k_B T/m)."""
        return np.sqrt(2 * self.k_B * self.T / mass)

    def mean_speed(self, mass: float) -> float:
        """Mean speed <v> = √(8k_B T/(πm))."""
        return np.sqrt(8 * self.k_B * self.T / (np.pi * mass))

    def rms_speed(self, mass: float) -> float:
        """RMS speed v_rms = √(3k_B T/m)."""
        return np.sqrt(3 * self.k_B * self.T / mass)


class IdealGas:
    """
    Ideal gas model.

    PV = NkT = nRT

    Example:
        >>> gas = IdealGas(n_moles=1.0, temperature=300, volume=0.0224)
        >>> P = gas.pressure()  # ~1 atm at STP
    """

    def __init__(
        self,
        n_particles: Optional[int] = None,
        n_moles: Optional[float] = None,
        temperature: float = 300.0,
        volume: float = 1.0,
        pressure: Optional[float] = None,
    ):
        """
        Initialize ideal gas.

        Specify two of (temperature, volume, pressure) to determine the third.
        """
        self.T = temperature

        if n_moles is not None:
            self.N = n_moles * AVOGADRO_NUMBER
            self.n = n_moles
        elif n_particles is not None:
            self.N = n_particles
            self.n = n_particles / AVOGADRO_NUMBER
        else:
            self.N = AVOGADRO_NUMBER
            self.n = 1.0

        if pressure is None:
            self.V = volume
            self.P = self.N * BOLTZMANN_CONSTANT * temperature / volume
        elif volume is None:
            self.P = pressure
            self.V = self.N * BOLTZMANN_CONSTANT * temperature / pressure
        else:
            self.V = volume
            self.P = pressure

    def pressure(self) -> float:
        """Calculate pressure from ideal gas law."""
        return self.N * BOLTZMANN_CONSTANT * self.T / self.V

    def internal_energy(self, degrees_of_freedom: int = 3) -> float:
        """
        Internal energy U = (f/2) NkT.

        Args:
            degrees_of_freedom: f=3 (monoatomic), f=5 (diatomic), f=6 (polyatomic)
        """
        return 0.5 * degrees_of_freedom * self.N * BOLTZMANN_CONSTANT * self.T

    def heat_capacity_V(self, degrees_of_freedom: int = 3) -> float:
        """Heat capacity at constant volume C_V = (f/2)Nk."""
        return 0.5 * degrees_of_freedom * self.N * BOLTZMANN_CONSTANT

    def heat_capacity_P(self, degrees_of_freedom: int = 3) -> float:
        """Heat capacity at constant pressure C_P = C_V + Nk."""
        return self.heat_capacity_V(degrees_of_freedom) + self.N * BOLTZMANN_CONSTANT

    def entropy(self, degrees_of_freedom: int = 3) -> float:
        """
        Sackur-Tetrode entropy for ideal monoatomic gas.

        S = Nk [ln(V/N λ³) + 5/2]

        where λ is thermal wavelength.
        """
        # Use simplified form
        return self.N * BOLTZMANN_CONSTANT * (np.log(self.V / self.N) + 1.5 * np.log(self.T) + 2.5)

    def helmholtz_free_energy(self, degrees_of_freedom: int = 3) -> float:
        """Helmholtz free energy F = U - TS."""
        return self.internal_energy(degrees_of_freedom) - self.T * self.entropy(degrees_of_freedom)

    def gibbs_free_energy(self, degrees_of_freedom: int = 3) -> float:
        """Gibbs free energy G = F + PV = U - TS + PV."""
        return self.helmholtz_free_energy(degrees_of_freedom) + self.P * self.V

    def chemical_potential(self) -> float:
        """Chemical potential μ = -kT ln(V/(N λ³))."""
        boltz = BoltzmannDistribution(self.T)
        m_approx = 4.65e-26  # ~Nitrogen mass
        lambda_th = boltz.thermal_wavelength(m_approx)
        return -BOLTZMANN_CONSTANT * self.T * np.log(self.V / (self.N * lambda_th**3))


class QuantumHarmonicOscillatorEnsemble:
    """
    Quantum harmonic oscillator in canonical ensemble.

    Energy levels: E_n = ℏω(n + 1/2)

    Partition function: Z = exp(-βℏω/2) / (1 - exp(-βℏω))
    """

    def __init__(self, omega: float, temperature: float, hbar: float = 1.055e-34):
        self.omega = omega
        self.T = temperature
        self.hbar = hbar
        self.beta = 1 / (BOLTZMANN_CONSTANT * temperature)

    def partition_function(self) -> float:
        """Partition function Z."""
        x = self.beta * self.hbar * self.omega
        return np.exp(-x / 2) / (1 - np.exp(-x))

    def average_energy(self) -> float:
        """
        Average energy <E> = ℏω(n̄ + 1/2).

        n̄ = 1/(exp(βℏω) - 1) is Bose-Einstein occupation
        """
        x = self.beta * self.hbar * self.omega
        n_bar = 1 / (np.exp(x) - 1)
        return self.hbar * self.omega * (n_bar + 0.5)

    def average_occupation(self) -> float:
        """Average quantum number n̄ = 1/(exp(βℏω) - 1)."""
        x = self.beta * self.hbar * self.omega
        return 1 / (np.exp(x) - 1)

    def heat_capacity(self) -> float:
        """
        Heat capacity C = k_B (βℏω)² exp(βℏω)/(exp(βℏω)-1)²
        """
        x = self.beta * self.hbar * self.omega
        return BOLTZMANN_CONSTANT * x**2 * np.exp(x) / (np.exp(x) - 1) ** 2

    def entropy(self) -> float:
        """Entropy S = k_B [βℏω n̄ - ln(1 - exp(-βℏω))]."""
        x = self.beta * self.hbar * self.omega
        n_bar = self.average_occupation()
        return BOLTZMANN_CONSTANT * (x * n_bar - np.log(1 - np.exp(-x)))


class IsingModel:
    """
    Ising model for magnetic systems.

    H = -J Σ<i,j> s_i s_j - h Σ_i s_i

    where s_i = ±1 are spin variables.

    Example:
        >>> ising = IsingModel(L=10, J=1.0, h=0.0)
        >>> ising.initialize_random()
        >>> M = ising.magnetization()
    """

    def __init__(
        self, L: int, J: float = 1.0, h: float = 0.0, temperature: float = 2.0, dimension: int = 2
    ):
        """
        Initialize Ising model.

        Args:
            L: Linear size of lattice
            J: Coupling constant (J > 0 ferromagnetic)
            h: External magnetic field
            temperature: Temperature in units of J/k_B
            dimension: 1, 2, or 3
        """
        self.L = L
        self.J = J
        self.h = h
        self.T = temperature
        self.beta = 1 / temperature
        self.dim = dimension

        if dimension == 1:
            self.spins = np.ones(L, dtype=np.int8)
        elif dimension == 2:
            self.spins = np.ones((L, L), dtype=np.int8)
        else:
            self.spins = np.ones((L, L, L), dtype=np.int8)

    def initialize_random(self) -> None:
        """Initialize spins randomly."""
        self.spins = np.random.choice([-1, 1], size=self.spins.shape).astype(np.int8)

    def initialize_ordered(self, up: bool = True) -> None:
        """Initialize all spins aligned."""
        self.spins = np.ones_like(self.spins) * (1 if up else -1)

    def energy(self) -> float:
        """Calculate total energy."""
        E = 0.0

        if self.dim == 1:
            # Nearest neighbor sum
            E = -self.J * np.sum(self.spins * np.roll(self.spins, 1))
        elif self.dim == 2:
            # Sum over horizontal and vertical bonds
            E = -self.J * np.sum(
                self.spins * np.roll(self.spins, 1, axis=0)
                + self.spins * np.roll(self.spins, 1, axis=1)
            )
        else:
            E = -self.J * np.sum(
                self.spins * np.roll(self.spins, 1, axis=0)
                + self.spins * np.roll(self.spins, 1, axis=1)
                + self.spins * np.roll(self.spins, 1, axis=2)
            )

        # External field contribution
        E -= self.h * np.sum(self.spins)

        return E

    def magnetization(self) -> float:
        """Calculate total magnetization M = Σ s_i."""
        return np.sum(self.spins)

    def magnetization_density(self) -> float:
        """Magnetization per spin m = M/N."""
        return np.mean(self.spins)

    def metropolis_step(self) -> bool:
        """
        Perform one Metropolis Monte Carlo step.

        Returns:
            True if flip was accepted
        """
        # Choose random site
        if self.dim == 1:
            i = np.random.randint(self.L)
            neighbors_sum = self.spins[(i - 1) % self.L] + self.spins[(i + 1) % self.L]
            delta_E = 2 * self.J * self.spins[i] * neighbors_sum + 2 * self.h * self.spins[i]

            if delta_E <= 0 or np.random.random() < np.exp(-self.beta * delta_E):
                self.spins[i] *= -1
                return True

        elif self.dim == 2:
            i, j = np.random.randint(self.L), np.random.randint(self.L)
            neighbors_sum = (
                self.spins[(i - 1) % self.L, j]
                + self.spins[(i + 1) % self.L, j]
                + self.spins[i, (j - 1) % self.L]
                + self.spins[i, (j + 1) % self.L]
            )
            delta_E = 2 * self.J * self.spins[i, j] * neighbors_sum + 2 * self.h * self.spins[i, j]

            if delta_E <= 0 or np.random.random() < np.exp(-self.beta * delta_E):
                self.spins[i, j] *= -1
                return True

        return False

    def monte_carlo_sweep(self) -> Tuple[float, float]:
        """
        Perform N Monte Carlo steps (one sweep).

        Returns:
            (acceptance_rate, energy)
        """
        N = self.spins.size
        accepted = 0

        for _ in range(N):
            if self.metropolis_step():
                accepted += 1

        return accepted / N, self.energy()

    def critical_temperature_2d(self) -> float:
        """
        Exact critical temperature for 2D Ising model.

        T_c = 2J / (k_B ln(1 + √2)) ≈ 2.269 J/k_B
        """
        return 2 * self.J / np.log(1 + np.sqrt(2))


class FermiDirac:
    """
    Fermi-Dirac distribution for fermions.

    f(E) = 1 / (exp((E-μ)/(k_B T)) + 1)
    """

    def __init__(
        self, temperature: float, chemical_potential: float, k_B: float = BOLTZMANN_CONSTANT
    ):
        self.T = temperature
        self.mu = chemical_potential
        self.k_B = k_B
        self.beta = 1 / (k_B * temperature)

    def occupation(self, energy: float) -> float:
        """Fermi-Dirac occupation number."""
        x = self.beta * (energy - self.mu)
        if x > 700:  # Prevent overflow
            return 0.0
        elif x < -700:
            return 1.0
        return 1 / (np.exp(x) + 1)

    def fermi_energy(self, n: float, m: float, V: float = 1.0) -> float:
        """
        Fermi energy at T=0.

        E_F = (ℏ²/2m)(3π²n)^(2/3)

        Args:
            n: Number density
            m: Particle mass
            V: Volume
        """
        hbar = 1.055e-34
        return (hbar**2 / (2 * m)) * (3 * np.pi**2 * n) ** (2 / 3)


class BoseEinstein:
    """
    Bose-Einstein distribution for bosons.

    n(E) = 1 / (exp((E-μ)/(k_B T)) - 1)
    """

    def __init__(
        self, temperature: float, chemical_potential: float = 0.0, k_B: float = BOLTZMANN_CONSTANT
    ):
        self.T = temperature
        self.mu = chemical_potential
        self.k_B = k_B
        self.beta = 1 / (k_B * temperature)

    def occupation(self, energy: float) -> float:
        """Bose-Einstein occupation number."""
        x = self.beta * (energy - self.mu)
        if x > 700:
            return 0.0
        if x <= 0:
            return np.inf  # Condensation
        return 1 / (np.exp(x) - 1)

    def critical_temperature(self, n: float, m: float) -> float:
        """
        BEC critical temperature.

        T_c = (2πℏ²/m k_B)(n/ζ(3/2))^(2/3)

        where ζ(3/2) ≈ 2.612
        """
        hbar = 1.055e-34
        zeta_32 = 2.612
        return (2 * np.pi * hbar**2 / (m * self.k_B)) * (n / zeta_32) ** (2 / 3)


__all__ = [
    "BOLTZMANN_CONSTANT",
    "AVOGADRO_NUMBER",
    "GAS_CONSTANT",
    "PLANCK_CONSTANT",
    "EnsembleType",
    "ThermodynamicState",
    "BoltzmannDistribution",
    "IdealGas",
    "QuantumHarmonicOscillatorEnsemble",
    "IsingModel",
    "FermiDirac",
    "BoseEinstein",
]
