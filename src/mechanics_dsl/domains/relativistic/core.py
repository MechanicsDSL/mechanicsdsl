"""
Special Relativistic Mechanics Domain for MechanicsDSL

Provides tools for relativistic particle dynamics, including:
- Lorentz transformations
- Relativistic momentum and energy
- Four-vectors and invariants
- Relativistic collisions
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from ..base import PhysicsDomain

# Physical constants (SI units)
SPEED_OF_LIGHT = 299792458.0  # m/s


class RelativisticParticle(PhysicsDomain):
    """
    Special relativistic point particle dynamics.

    Uses the relativistic Lagrangian:
        L = -mc²√(1 - v²/c²) - V(r)

    or equivalently minimizes proper time along worldline.

    Example:
        >>> particle = RelativisticParticle(mass=1.0)
        >>> particle.set_parameter('c', 1.0)  # Natural units
        >>> gamma = particle.lorentz_factor(0.8)  # v = 0.8c
    """

    def __init__(self, mass: float = 1.0, name: str = "relativistic_particle"):
        super().__init__(name)
        self.rest_mass = mass
        self.parameters["m"] = mass
        self.parameters["c"] = SPEED_OF_LIGHT

        self.coordinates = ["x", "y", "z"]

        # Symbols
        self.x, self.y, self.z = sp.symbols("x y z", real=True)
        self.vx, self.vy, self.vz = sp.symbols("x_dot y_dot z_dot", real=True)
        self.t = sp.Symbol("t", real=True)
        self.m = sp.Symbol("m", positive=True)
        self.c = sp.Symbol("c", positive=True)

        # Potential (default: free particle)
        self.potential: sp.Expr = sp.Integer(0)

    def set_potential(self, V: sp.Expr) -> None:
        """Set the potential energy function."""
        self.potential = V

    def lorentz_factor(self, v: float) -> float:
        """
        Calculate Lorentz factor γ = 1/√(1 - v²/c²).

        Args:
            v: Speed (magnitude of velocity)

        Returns:
            Lorentz factor γ
        """
        c = self.parameters.get("c", SPEED_OF_LIGHT)
        beta = v / c
        if beta >= 1.0:
            raise ValueError(f"Speed {v} must be less than c = {c}")
        return 1.0 / np.sqrt(1 - beta**2)

    def relativistic_momentum(self, v: float) -> float:
        """
        Calculate relativistic momentum p = γmv.

        Args:
            v: Speed

        Returns:
            Relativistic momentum
        """
        gamma = self.lorentz_factor(v)
        return gamma * self.rest_mass * v

    def relativistic_energy(self, v: float) -> float:
        """
        Calculate total relativistic energy E = γmc².

        Args:
            v: Speed

        Returns:
            Total energy
        """
        c = self.parameters.get("c", SPEED_OF_LIGHT)
        gamma = self.lorentz_factor(v)
        return gamma * self.rest_mass * c**2

    def kinetic_energy(self, v: float) -> float:
        """
        Calculate relativistic kinetic energy T = (γ-1)mc².

        Args:
            v: Speed

        Returns:
            Kinetic energy
        """
        c = self.parameters.get("c", SPEED_OF_LIGHT)
        gamma = self.lorentz_factor(v)
        return (gamma - 1) * self.rest_mass * c**2

    def rest_energy(self) -> float:
        """Calculate rest energy E₀ = mc²."""
        c = self.parameters.get("c", SPEED_OF_LIGHT)
        return self.rest_mass * c**2

    def define_lagrangian(self) -> sp.Expr:
        """
        Relativistic Lagrangian:
        L = -mc²√(1 - v²/c²) - V(r)

        In the low-velocity limit, this reduces to (1/2)mv² - V plus a constant.
        """
        v_squared = self.vx**2 + self.vy**2 + self.vz**2
        gamma_inv = sp.sqrt(1 - v_squared / self.c**2)

        L = -self.m * self.c**2 * gamma_inv - self.potential
        return L

    def define_hamiltonian(self) -> sp.Expr:
        """
        Relativistic Hamiltonian:
        H = √(p²c² + m²c⁴) + V(r)

        where p is the relativistic momentum.
        """
        px, py, pz = sp.symbols("p_x p_y p_z", real=True)
        p_squared = px**2 + py**2 + pz**2

        H = sp.sqrt(p_squared * self.c**2 + self.m**2 * self.c**4) + self.potential
        return H

    def derive_equations_of_motion(self) -> Dict[str, sp.Expr]:
        """
        Derive relativistic equations of motion.

        dp/dt = F = -∇V

        Returns:
            Dictionary with momentum rate equations
        """
        # Force from potential
        Fx = -sp.diff(self.potential, self.x)
        Fy = -sp.diff(self.potential, self.y)
        Fz = -sp.diff(self.potential, self.z)

        # Relativistic momentum components
        v_squared = self.vx**2 + self.vy**2 + self.vz**2
        gamma = 1 / sp.sqrt(1 - v_squared / self.c**2)

        # d(γmv)/dt = F leads to complicated acceleration expressions
        # For numerical work, it's better to evolve momentum directly
        return {
            "px_dot": Fx,
            "py_dot": Fy,
            "pz_dot": Fz,
        }

    def get_state_variables(self) -> List[str]:
        """State variables: positions and momenta."""
        return ["x", "y", "z", "px", "py", "pz"]

    def get_required_parameters(self) -> List[str]:
        return ["m", "c"]

    def get_conserved_quantities(self) -> Dict[str, sp.Expr]:
        """Total energy and momentum (if no external forces)."""
        p_squared = (self.m * self.vx) ** 2 + (self.m * self.vy) ** 2 + (self.m * self.vz) ** 2
        v_squared = self.vx**2 + self.vy**2 + self.vz**2
        gamma = 1 / sp.sqrt(1 - v_squared / self.c**2)

        return {
            "total_energy": gamma * self.m * self.c**2 + self.potential,
            "momentum_x": gamma * self.m * self.vx,
            "momentum_y": gamma * self.m * self.vy,
            "momentum_z": gamma * self.m * self.vz,
        }


@dataclass
class FourVector:
    """
    Represents a four-vector in special relativity.

    Components: (ct, x, y, z) with metric signature (+,-,-,-).
    """

    ct: Union[float, sp.Expr]
    x: Union[float, sp.Expr]
    y: Union[float, sp.Expr]
    z: Union[float, sp.Expr]

    def __post_init__(self):
        """Convert to float if numeric."""
        pass

    def invariant(self) -> Union[float, sp.Expr]:
        """
        Calculate Lorentz invariant (squared interval).

        s² = (ct)² - x² - y² - z²
        """
        return self.ct**2 - self.x**2 - self.y**2 - self.z**2

    def magnitude(self) -> Union[float, sp.Expr]:
        """Magnitude √|s²|."""
        inv = self.invariant()
        if isinstance(inv, (int, float)):
            return np.sqrt(abs(inv))
        return sp.sqrt(sp.Abs(inv))

    def is_timelike(self) -> bool:
        """Check if interval is timelike (s² > 0)."""
        inv = self.invariant()
        if isinstance(inv, (int, float)):
            return inv > 0
        return None  # Cannot determine symbolically

    def is_spacelike(self) -> bool:
        """Check if interval is spacelike (s² < 0)."""
        inv = self.invariant()
        if isinstance(inv, (int, float)):
            return inv < 0
        return None

    def is_lightlike(self) -> bool:
        """Check if interval is lightlike/null (s² = 0)."""
        inv = self.invariant()
        if isinstance(inv, (int, float)):
            return abs(inv) < 1e-10
        return None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([float(self.ct), float(self.x), float(self.y), float(self.z)])


class LorentzTransform:
    """
    Lorentz transformation tools.

    Transforms four-vectors between inertial reference frames.
    """

    @staticmethod
    def boost_x(four_vector: FourVector, v: float, c: float = SPEED_OF_LIGHT) -> FourVector:
        """
        Apply Lorentz boost along x-axis.

        Args:
            four_vector: Input four-vector
            v: Relative velocity of new frame
            c: Speed of light

        Returns:
            Transformed four-vector
        """
        beta = v / c
        gamma = 1.0 / np.sqrt(1 - beta**2)

        ct_new = gamma * (four_vector.ct - beta * four_vector.x)
        x_new = gamma * (four_vector.x - beta * four_vector.ct)

        return FourVector(ct_new, x_new, four_vector.y, four_vector.z)

    @staticmethod
    def boost_matrix(v: float, direction: np.ndarray, c: float = SPEED_OF_LIGHT) -> np.ndarray:
        """
        Construct 4x4 Lorentz boost matrix for arbitrary direction.

        Args:
            v: Speed of new frame
            direction: Unit vector for boost direction
            c: Speed of light

        Returns:
            4x4 boost matrix
        """
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        beta = v / c
        gamma = 1.0 / np.sqrt(1 - beta**2)

        # Boost matrix components
        n = direction
        Lambda = np.eye(4)

        Lambda[0, 0] = gamma
        Lambda[0, 1:4] = -gamma * beta * n
        Lambda[1:4, 0] = -gamma * beta * n

        outer = np.outer(n, n)
        Lambda[1:4, 1:4] = np.eye(3) + (gamma - 1) * outer

        return Lambda

    @staticmethod
    def velocity_addition(v1: float, v2: float, c: float = SPEED_OF_LIGHT) -> float:
        """
        Relativistic velocity addition formula.

        u = (v1 + v2) / (1 + v1*v2/c²)

        Args:
            v1: First velocity
            v2: Second velocity
            c: Speed of light

        Returns:
            Combined velocity
        """
        return (v1 + v2) / (1 + v1 * v2 / c**2)

    @staticmethod
    def time_dilation(proper_time: float, v: float, c: float = SPEED_OF_LIGHT) -> float:
        """
        Calculate dilated time interval.

        Δt = γΔτ

        Args:
            proper_time: Proper time interval Δτ
            v: Relative velocity
            c: Speed of light

        Returns:
            Dilated time interval
        """
        gamma = 1.0 / np.sqrt(1 - (v / c) ** 2)
        return gamma * proper_time

    @staticmethod
    def length_contraction(proper_length: float, v: float, c: float = SPEED_OF_LIGHT) -> float:
        """
        Calculate contracted length.

        L = L₀/γ

        Args:
            proper_length: Proper length L₀
            v: Relative velocity
            c: Speed of light

        Returns:
            Contracted length
        """
        gamma = 1.0 / np.sqrt(1 - (v / c) ** 2)
        return proper_length / gamma


class RelativisticCollision:
    """
    Analyzes relativistic particle collisions.

    Conserves four-momentum: Σpᵢ = Σpf
    """

    @staticmethod
    def invariant_mass(
        particles: List[Tuple[float, np.ndarray]], c: float = SPEED_OF_LIGHT
    ) -> float:
        """
        Calculate invariant mass of a system of particles.

        s = (Σp)² = (ΣE/c)² - (Σp⃗)²
        M = √s / c²

        Args:
            particles: List of (rest_mass, velocity_3vector) tuples
            c: Speed of light

        Returns:
            Invariant mass of the system
        """
        total_E = 0.0
        total_p = np.zeros(3)

        for m, v in particles:
            v_mag = np.linalg.norm(v)
            gamma = 1.0 / np.sqrt(1 - (v_mag / c) ** 2)
            E = gamma * m * c**2
            p = gamma * m * v

            total_E += E
            total_p += p

        s = (total_E / c) ** 2 - np.dot(total_p, total_p)
        return np.sqrt(s) / c if s > 0 else 0.0

    @staticmethod
    def threshold_energy(
        m_target: float, m_products: List[float], c: float = SPEED_OF_LIGHT
    ) -> float:
        """
        Calculate threshold kinetic energy for particle production.

        For a + b → c + d + ...
        with b at rest, minimum KE of a is:

        KE_threshold = [(Σm_products)² - m_a² - m_b²]c² / (2m_b)

        Args:
            m_target: Rest mass of target (at rest)
            m_products: List of product masses
            c: Speed of light

        Returns:
            Threshold kinetic energy
        """
        M_sum = sum(m_products)
        # This is simplified; full formula depends on projectile mass
        return (M_sum**2 - m_target**2) * c**2 / (2 * m_target)


# Convenience functions


def gamma(v: float, c: float = SPEED_OF_LIGHT) -> float:
    """Calculate Lorentz factor."""
    return 1.0 / np.sqrt(1 - (v / c) ** 2)


def beta(v: float, c: float = SPEED_OF_LIGHT) -> float:
    """Calculate β = v/c."""
    return v / c


def rapidity(v: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Calculate rapidity η = arctanh(v/c).

    Rapidities add linearly: η₁₂ = η₁ + η₂
    """
    return np.arctanh(v / c)


class DopplerEffect:
    """
    Relativistic Doppler effect for light and sound.

    Handles both longitudinal and transverse Doppler shifts.
    """

    @staticmethod
    def longitudinal_frequency(
        f_source: float, v: float, c: float = SPEED_OF_LIGHT, approaching: bool = True
    ) -> float:
        """
        Calculate observed frequency for source moving toward/away.

        f_obs = f_source * √((1 + β)/(1 - β)) for approaching
        f_obs = f_source * √((1 - β)/(1 + β)) for receding

        Args:
            f_source: Source frequency
            v: Relative velocity
            c: Speed of light
            approaching: True if source approaches observer

        Returns:
            Observed frequency
        """
        beta_val = v / c
        if approaching:
            return f_source * np.sqrt((1 + beta_val) / (1 - beta_val))
        else:
            return f_source * np.sqrt((1 - beta_val) / (1 + beta_val))

    @staticmethod
    def transverse_frequency(f_source: float, v: float, c: float = SPEED_OF_LIGHT) -> float:
        """
        Calculate transverse Doppler shift (time dilation effect).

        f_obs = f_source / γ

        Occurs when source moves perpendicular to line of sight.
        """
        gamma_val = 1.0 / np.sqrt(1 - (v / c) ** 2)
        return f_source / gamma_val

    @staticmethod
    def redshift(v: float, c: float = SPEED_OF_LIGHT) -> float:
        """
        Calculate cosmological redshift z for receding source.

        z = √((1 + β)/(1 - β)) - 1
        """
        beta_val = v / c
        return np.sqrt((1 + beta_val) / (1 - beta_val)) - 1

    @staticmethod
    def velocity_from_redshift(z: float, c: float = SPEED_OF_LIGHT) -> float:
        """
        Calculate velocity from observed redshift.

        β = ((1+z)² - 1) / ((1+z)² + 1)
        """
        factor = (1 + z) ** 2
        beta_val = (factor - 1) / (factor + 1)
        return beta_val * c


def relativistic_aberration(theta_source: float, v: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Calculate relativistic stellar aberration.

    cos(θ_obs) = (cos(θ_source) - β) / (1 - β*cos(θ_source))

    Args:
        theta_source: Angle in source frame (radians)
        v: Relative velocity
        c: Speed of light

    Returns:
        Observed angle (radians)
    """
    beta_val = v / c
    cos_source = np.cos(theta_source)
    cos_obs = (cos_source - beta_val) / (1 - beta_val * cos_source)
    return np.arccos(np.clip(cos_obs, -1, 1))


def compton_wavelength_shift(
    theta: float, m_e: float = 9.109e-31, c: float = SPEED_OF_LIGHT, h: float = 6.626e-34
) -> float:
    """
    Calculate Compton wavelength shift for photon-electron scattering.

    Δλ = (h/m_e c)(1 - cos(θ))

    Args:
        theta: Scattering angle (radians)
        m_e: Electron mass
        c: Speed of light
        h: Planck constant

    Returns:
        Wavelength shift (m)
    """
    lambda_c = h / (m_e * c)  # Compton wavelength
    return lambda_c * (1 - np.cos(theta))


class SynchrotronRadiation:
    """
    Synchrotron radiation from relativistic charged particles.

    When charged particles are accelerated (e.g., in magnetic fields),
    they emit electromagnetic radiation. For relativistic particles,
    this radiation is beamed in the forward direction.

    Key formulas:
    - Radiated power: P = (q²c/6πε₀)(γ⁴/R²)β⁴  (circular motion)
    - Critical frequency: ω_c = (3/2) γ³ c/R
    - Beaming angle: θ ≈ 1/γ

    Example:
        >>> synch = SynchrotronRadiation(charge=1.6e-19, mass=9.1e-31)
        >>> power = synch.radiated_power(gamma=1000, radius=1.0)
    """

    EPSILON_0 = 8.854e-12

    def __init__(self, charge: float, mass: float, c: float = SPEED_OF_LIGHT):
        """
        Initialize synchrotron radiation calculator.

        Args:
            charge: Particle charge (C)
            mass: Particle rest mass (kg)
            c: Speed of light (m/s)
        """
        self.q = abs(charge)
        self.m = mass
        self.c = c

    def radiated_power(self, gamma_factor: float, radius: float) -> float:
        """
        Power radiated by relativistic particle in circular motion.

        P = (q²c/6πε₀)(γ⁴/R²)β⁴= (q²c/6πε₀)(γ⁴/R²)(1 - 1/γ²)²

        Simplified for γ >> 1:
        P ≈ (q²c γ⁴)/(6πε₀ R²)

        Args:
            gamma_factor: Lorentz factor γ
            radius: Radius of circular motion (m)

        Returns:
            Radiated power (W)
        """
        beta_sq = 1 - 1 / gamma_factor**2
        prefactor = self.q**2 * self.c / (6 * np.pi * self.EPSILON_0)
        return prefactor * (gamma_factor**4 / radius**2) * beta_sq**2

    def critical_frequency(self, gamma_factor: float, radius: float) -> float:
        """
        Critical frequency of synchrotron spectrum.

        ω_c = (3/2) γ³ c/R

        Half the radiated power is above this frequency.
        """
        return (3 / 2) * gamma_factor**3 * self.c / radius

    def characteristic_angle(self, gamma_factor: float) -> float:
        """
        Characteristic opening angle of radiation cone.

        θ ≈ 1/γ (radians)
        """
        return 1.0 / gamma_factor

    def energy_loss_per_turn(self, gamma_factor: float, radius: float) -> float:
        """
        Energy lost per revolution in circular accelerator.

        ΔE = (4π/3)(r_e/R)(γ⁴/m_e c²) for electrons
        """
        circumference = 2 * np.pi * radius
        power = self.radiated_power(gamma_factor, radius)
        period = circumference / (self.c * np.sqrt(1 - 1 / gamma_factor**2))
        return power * period


class ThomasPrecession:
    """
    Thomas precession of spinning particles.

    When a spinning particle accelerates, its spin axis precesses
    even in the absence of torques (relativistic kinematic effect).

    Precession frequency: Ω_T = (γ-1)/v² × (a × v)

    For circular motion: Ω_T = (γ-1) ω

    Example:
        >>> thomas = ThomasPrecession()
        >>> omega_T = thomas.precession_frequency(v=0.1*c, a=1e10, theta_av=np.pi/2)
    """

    def __init__(self, c: float = SPEED_OF_LIGHT):
        """
        Initialize Thomas precession calculator.

        Args:
            c: Speed of light
        """
        self.c = c

    def precession_frequency(self, v: float, a: float, theta_av: float) -> float:
        """
        Thomas precession angular frequency.

        |Ω_T| = (γ-1)/v² × |a × v| = (γ-1) a sin(θ)/v

        Args:
            v: Particle speed
            a: Magnitude of acceleration
            theta_av: Angle between a and v (radians)

        Returns:
            Precession angular frequency (rad/s)
        """
        gamma_val = 1.0 / np.sqrt(1 - (v / self.c) ** 2)
        return (gamma_val - 1) * a * np.sin(theta_av) / v

    def circular_motion_precession(self, omega_orbit: float, v: float) -> float:
        """
        Thomas precession for circular motion.

        Ω_T = (γ - 1) × ω_orbit

        Args:
            omega_orbit: Orbital angular frequency
            v: Orbital speed
        """
        gamma_val = 1.0 / np.sqrt(1 - (v / self.c) ** 2)
        return (gamma_val - 1) * omega_orbit


class TwinParadox:
    """
    Twin paradox calculations.

    Calculates proper time for various worldlines to show that
    the traveling twin ages less than the stationary twin.

    For constant velocity travel at speed v for distance L:
    - Earth time: t_E = 2L/v
    - Traveler proper time: τ = 2L/(γv) = t_E/γ

    Example:
        >>> paradox = TwinParadox()
        >>> ages = paradox.constant_velocity(distance=4.0, speed=0.8, c=1.0)
    """

    def __init__(self, c: float = SPEED_OF_LIGHT):
        """
        Initialize twin paradox calculator.

        Args:
            c: Speed of light
        """
        self.c = c

    def constant_velocity(self, distance: float, speed: float) -> Dict[str, float]:
        """
        Simple twin paradox with constant velocity travel.

        Args:
            distance: One-way distance (m)
            speed: Travel speed (m/s)

        Returns:
            Dictionary with 'earth_time', 'proper_time', 'age_difference'
        """
        gamma_val = 1.0 / np.sqrt(1 - (speed / self.c) ** 2)
        earth_time = 2 * distance / speed
        proper_time = earth_time / gamma_val

        return {
            "earth_time": earth_time,
            "proper_time": proper_time,
            "age_difference": earth_time - proper_time,
            "gamma": gamma_val,
        }

    def constant_acceleration(self, distance: float, acceleration: float) -> Dict[str, float]:
        """
        Twin paradox with constant proper acceleration.

        Uses hyperbolic motion for more realistic scenario.

        For constant acceleration a, traveling to distance d and back:

        τ = (4c/a) × arcsinh(√(ad/c²))

        Args:
            distance: One-way distance (m)
            acceleration: Proper acceleration (m/s²)

        Returns:
            Dictionary with timing information
        """
        # Time for one leg (accelerate to midpoint, decelerate to end)
        # Using hyperbolic motion
        c = self.c
        a = acceleration
        d = distance / 2  # Midpoint

        # Coordinate time for one quarter trip
        t_quarter = (c / a) * np.sqrt((2 * a * d / c**2) + (a * d / c**2) ** 2)
        t_total = 4 * t_quarter  # Four quarters (there, back)

        # Proper time
        tau_quarter = (c / a) * np.arcsinh(a * t_quarter / c)
        tau_total = 4 * tau_quarter

        return {
            "earth_time": t_total,
            "proper_time": tau_total,
            "age_difference": t_total - tau_total,
        }


def proper_acceleration(
    coordinate_acceleration: float, v: float, c: float = SPEED_OF_LIGHT
) -> float:
    """
    Convert coordinate acceleration to proper acceleration.

    a_proper = γ³ × a_coordinate (for parallel motion)

    Args:
        coordinate_acceleration: Acceleration measured in lab frame
        v: Current velocity
        c: Speed of light

    Returns:
        Proper acceleration felt by accelerating observer
    """
    gamma_val = 1.0 / np.sqrt(1 - (v / c) ** 2)
    return gamma_val**3 * coordinate_acceleration


def relativistic_mass(rest_mass: float, v: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Relativistic mass (deprecated concept but sometimes useful).

    m_rel = γ × m_0

    Note: Modern physics prefers to use rest mass and momentum.
    """
    gamma_val = 1.0 / np.sqrt(1 - (v / c) ** 2)
    return gamma_val * rest_mass


def gravitational_redshift(delta_phi: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Gravitational redshift (weak field approximation).

    z = Δφ/c² (for Δφ << c²)

    where Δφ is the gravitational potential difference.

    Args:
        delta_phi: Potential difference (J/kg) = GM(1/r₁ - 1/r₂)
        c: Speed of light

    Returns:
        Redshift z = Δλ/λ
    """
    return delta_phi / c**2


def relativistic_kinetic_energy(mass: float, v: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Relativistic kinetic energy T = (γ-1)mc².

    Args:
        mass: Rest mass
        v: Velocity
        c: Speed of light
    """
    gamma_val = 1.0 / np.sqrt(1 - (v / c) ** 2)
    return (gamma_val - 1) * mass * c**2


def velocity_from_kinetic_energy(T: float, mass: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Calculate velocity from relativistic kinetic energy.

    From T = (γ-1)mc², solve for v.

    Args:
        T: Kinetic energy
        mass: Rest mass
        c: Speed of light

    Returns:
        Velocity
    """
    gamma_val = T / (mass * c**2) + 1
    beta_sq = 1 - 1 / gamma_val**2
    return c * np.sqrt(beta_sq)


def momentum_from_energy(E: float, mass: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Calculate momentum from total energy using E² = (pc)² + (mc²)².

    p = √(E² - (mc²)²) / c
    """
    E0 = mass * c**2  # Rest energy
    if E < E0:
        raise ValueError("Energy must be >= rest energy")
    return np.sqrt(E**2 - E0**2) / c


def mandelstam_s(p1: FourVector, p2: FourVector) -> float:
    """
    Mandelstam s variable (center of mass energy squared).

    s = (p₁ + p₂)²

    For collision of particles with 4-momenta p1 and p2.
    """
    # Sum the four-vectors
    sum_vec = FourVector(ct=p1.ct + p2.ct, x=p1.x + p2.x, y=p1.y + p2.y, z=p1.z + p2.z)
    return sum_vec.invariant()


__all__ = [
    "SPEED_OF_LIGHT",
    "RelativisticParticle",
    "FourVector",
    "LorentzTransform",
    "RelativisticCollision",
    "DopplerEffect",
    "SynchrotronRadiation",
    "ThomasPrecession",
    "TwinParadox",
    "gamma",
    "beta",
    "rapidity",
    "relativistic_aberration",
    "compton_wavelength_shift",
    "proper_acceleration",
    "relativistic_mass",
    "gravitational_redshift",
    "relativistic_kinetic_energy",
    "velocity_from_kinetic_energy",
    "momentum_from_energy",
    "mandelstam_s",
]
