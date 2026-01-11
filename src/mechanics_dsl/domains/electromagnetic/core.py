"""
Electromagnetic Domain for MechanicsDSL

Provides tools for charged particle dynamics in electromagnetic fields,
including Lorentz force, electromagnetic potentials, and radiation.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import sympy as sp
import numpy as np
from ..base import PhysicsDomain


class FieldType(Enum):
    """Types of electromagnetic fields."""
    UNIFORM = "uniform"
    DIPOLE = "dipole"
    POINT_CHARGE = "point_charge"
    CUSTOM = "custom"


@dataclass
class ElectromagneticField:
    """
    Represents an electromagnetic field configuration.
    
    Attributes:
        E: Electric field vector (Ex, Ey, Ez) as SymPy expressions or constants
        B: Magnetic field vector (Bx, By, Bz) as SymPy expressions or constants
        field_type: Classification of the field
    """
    E: Tuple[sp.Expr, sp.Expr, sp.Expr]
    B: Tuple[sp.Expr, sp.Expr, sp.Expr]
    field_type: FieldType = FieldType.CUSTOM


class ChargedParticle(PhysicsDomain):
    """
    Dynamics of a charged particle in electromagnetic fields.
    
    Implements the Lorentz force law:
        F = q(E + v × B)
    
    And the corresponding Lagrangian:
        L = (1/2)m*v² - q*φ + q*v·A
    
    where φ is the scalar potential and A is the vector potential.
    
    Example:
        >>> particle = ChargedParticle(mass=1.0, charge=1.0)
        >>> particle.set_uniform_magnetic_field(Bz=1.0)
        >>> eom = particle.derive_equations_of_motion()
    """
    
    def __init__(self, mass: float = 1.0, charge: float = 1.0, name: str = "charged_particle"):
        super().__init__(name)
        self.mass = mass
        self.charge = charge
        self.parameters['m'] = mass
        self.parameters['q'] = charge
        
        # 3D coordinates
        self.coordinates = ['x', 'y', 'z']
        
        # Field potentials (scalar φ and vector A)
        self.scalar_potential: sp.Expr = sp.Integer(0)
        self.vector_potential: Tuple[sp.Expr, sp.Expr, sp.Expr] = (
            sp.Integer(0), sp.Integer(0), sp.Integer(0)
        )
        
        # Define coordinate symbols
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.vx, self.vy, self.vz = sp.symbols('x_dot y_dot z_dot', real=True)
        self.t = sp.Symbol('t', real=True)
        self.m = sp.Symbol('m', positive=True)
        self.q = sp.Symbol('q', real=True)
    
    def set_uniform_electric_field(self, Ex: float = 0, Ey: float = 0, Ez: float = 0) -> None:
        """Set a uniform electric field E = (Ex, Ey, Ez)."""
        # For uniform E, we can use φ = -E·r
        self.scalar_potential = -(Ex * self.x + Ey * self.y + Ez * self.z)
    
    def set_uniform_magnetic_field(self, Bx: float = 0, By: float = 0, Bz: float = 0) -> None:
        """
        Set a uniform magnetic field B = (Bx, By, Bz).
        
        Uses the symmetric gauge: A = (1/2) B × r
        """
        # Symmetric gauge for uniform B
        Ax = sp.Rational(1, 2) * (By * self.z - Bz * self.y)
        Ay = sp.Rational(1, 2) * (Bz * self.x - Bx * self.z)
        Az = sp.Rational(1, 2) * (Bx * self.y - By * self.x)
        self.vector_potential = (Ax, Ay, Az)
    
    def set_potentials(self, phi: sp.Expr, A: Tuple[sp.Expr, sp.Expr, sp.Expr]) -> None:
        """Set custom scalar and vector potentials."""
        self.scalar_potential = phi
        self.vector_potential = A
    
    def define_lagrangian(self) -> sp.Expr:
        """
        Lagrangian for charged particle in EM field:
        L = (1/2)m*v² - q*φ + q*v·A
        """
        # Kinetic energy
        T = sp.Rational(1, 2) * self.m * (self.vx**2 + self.vy**2 + self.vz**2)
        
        # Potential energy from scalar potential
        V = self.q * self.scalar_potential
        
        # Coupling to vector potential
        Ax, Ay, Az = self.vector_potential
        coupling = self.q * (self.vx * Ax + self.vy * Ay + self.vz * Az)
        
        return T - V + coupling
    
    def define_hamiltonian(self) -> sp.Expr:
        """
        Hamiltonian for charged particle in EM field:
        H = (1/2m)(p - qA)² + qφ
        """
        px, py, pz = sp.symbols('p_x p_y p_z', real=True)
        Ax, Ay, Az = self.vector_potential
        
        # Mechanical momentum π = p - qA
        pi_x = px - self.q * Ax
        pi_y = py - self.q * Ay
        pi_z = pz - self.q * Az
        
        # Kinetic energy
        T = (pi_x**2 + pi_y**2 + pi_z**2) / (2 * self.m)
        
        # Potential energy
        V = self.q * self.scalar_potential
        
        return T + V
    
    def derive_equations_of_motion(self) -> Dict[str, sp.Expr]:
        """
        Derive equations of motion from Lorentz force.
        
        Returns:
            Dictionary with x_ddot, y_ddot, z_ddot expressions
        """
        # Electric field from potential: E = -∇φ - ∂A/∂t
        Ex = -sp.diff(self.scalar_potential, self.x)
        Ey = -sp.diff(self.scalar_potential, self.y)
        Ez = -sp.diff(self.scalar_potential, self.z)
        
        # Magnetic field from vector potential: B = ∇ × A
        Ax, Ay, Az = self.vector_potential
        Bx = sp.diff(Az, self.y) - sp.diff(Ay, self.z)
        By = sp.diff(Ax, self.z) - sp.diff(Az, self.x)
        Bz = sp.diff(Ay, self.x) - sp.diff(Ax, self.y)
        
        # Lorentz force: F = q(E + v × B)
        Fx = self.q * (Ex + self.vy * Bz - self.vz * By)
        Fy = self.q * (Ey + self.vz * Bx - self.vx * Bz)
        Fz = self.q * (Ez + self.vx * By - self.vy * Bx)
        
        # Newton's second law
        return {
            'x_ddot': Fx / self.m,
            'y_ddot': Fy / self.m,
            'z_ddot': Fz / self.m
        }
    
    def get_state_variables(self) -> List[str]:
        """Get state variables: positions and velocities."""
        return ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']
    
    def get_required_parameters(self) -> List[str]:
        return ['m', 'q']
    
    def cyclotron_frequency(self, B_magnitude: float) -> float:
        """
        Calculate cyclotron frequency ωc = qB/m.
        
        Args:
            B_magnitude: Magnitude of magnetic field
            
        Returns:
            Cyclotron angular frequency
        """
        return abs(self.charge) * B_magnitude / self.mass
    
    def larmor_radius(self, v_perp: float, B_magnitude: float) -> float:
        """
        Calculate Larmor (gyro) radius rL = mv_⊥/(qB).
        
        Args:
            v_perp: Velocity perpendicular to B
            B_magnitude: Magnitude of magnetic field
            
        Returns:
            Larmor radius
        """
        return self.mass * v_perp / (abs(self.charge) * B_magnitude)


class CyclotronMotion:
    """
    Analyzes cyclotron motion of charged particles in magnetic fields.
    
    Provides exact solutions for uniform magnetic field and
    perturbative corrections for non-uniform fields.
    """
    
    def __init__(self, particle: ChargedParticle):
        self.particle = particle
    
    def exact_trajectory(self, 
                         v0: Tuple[float, float, float],
                         r0: Tuple[float, float, float],
                         B: float,
                         t_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute exact cyclotron trajectory for uniform B along z.
        
        Args:
            v0: Initial velocity (vx0, vy0, vz0)
            r0: Initial position (x0, y0, z0)
            B: Magnetic field magnitude (along z)
            t_array: Time points
            
        Returns:
            Dictionary with x, y, z, vx, vy, vz arrays
        """
        m = self.particle.mass
        q = self.particle.charge
        
        # Cyclotron frequency
        omega_c = q * B / m
        
        vx0, vy0, vz0 = v0
        x0, y0, z0 = r0
        
        # Perpendicular velocity magnitude
        v_perp = np.sqrt(vx0**2 + vy0**2)
        
        # Phase angle
        if v_perp > 0:
            phi0 = np.arctan2(vy0, vx0)
        else:
            phi0 = 0
        
        # Larmor radius
        r_L = v_perp / abs(omega_c) if omega_c != 0 else 0
        
        # Guiding center
        if omega_c != 0:
            xc = x0 + vy0 / omega_c
            yc = y0 - vx0 / omega_c
        else:
            xc, yc = x0, y0
        
        # Trajectory
        if omega_c != 0:
            x = xc - r_L * np.sin(omega_c * t_array + phi0)
            y = yc + r_L * np.cos(omega_c * t_array + phi0)
        else:
            x = x0 + vx0 * t_array
            y = y0 + vy0 * t_array
        
        z = z0 + vz0 * t_array
        
        # Velocities
        if omega_c != 0:
            vx = -r_L * omega_c * np.cos(omega_c * t_array + phi0)
            vy = -r_L * omega_c * np.sin(omega_c * t_array + phi0)
        else:
            vx = np.full_like(t_array, vx0)
            vy = np.full_like(t_array, vy0)
        
        vz = np.full_like(t_array, vz0)
        
        return {
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            't': t_array
        }


class DipoleTrap:
    """
    Models magnetic dipole traps (mirror machines, magnetic bottles).
    
    Analyzes adiabatic invariants and mirror points.
    """
    
    def __init__(self, B0: float, L: float):
        """
        Initialize dipole trap.
        
        Args:
            B0: Field strength at center
            L: Characteristic length scale
        """
        self.B0 = B0
        self.L = L
    
    def magnetic_field(self, z: float) -> float:
        """
        Magnetic field magnitude along axis.
        
        B(z) = B0 * (1 + (z/L)²)
        """
        return self.B0 * (1 + (z / self.L)**2)
    
    def mirror_ratio(self, z_mirror: float) -> float:
        """Mirror ratio R = B_mirror / B_min."""
        return self.magnetic_field(z_mirror) / self.B0
    
    def loss_cone_angle(self, z_mirror: float) -> float:
        """
        Loss cone angle for particles that escape.
        
        sin²(θ_loss) = B_min / B_mirror = 1/R
        """
        R = self.mirror_ratio(z_mirror)
        return np.arcsin(1.0 / np.sqrt(R))
    
    def is_trapped(self, pitch_angle: float, z_mirror: float) -> bool:
        """
        Determine if a particle with given pitch angle is trapped.
        
        Args:
            pitch_angle: Angle between v and B (radians)
            z_mirror: Position of mirror point
            
        Returns:
            True if particle is trapped
        """
        return pitch_angle > self.loss_cone_angle(z_mirror)


class PenningTrap:
    """
    Models a Penning trap for charged particle confinement.
    
    Uses a uniform magnetic field for radial confinement and
    a quadrupole electric field for axial confinement.
    
    The motion consists of three independent oscillations:
    - Axial oscillation (ωz)
    - Modified cyclotron motion (ω+)
    - Magnetron motion (ω-)
    """
    
    def __init__(self, B: float, V0: float, d: float, m: float, q: float):
        """
        Initialize Penning trap.
        
        Args:
            B: Magnetic field magnitude (T)
            V0: Trap voltage (V)
            d: Characteristic trap dimension (m)
            m: Particle mass (kg)
            q: Particle charge (C)
        """
        self.B = B
        self.V0 = V0
        self.d = d
        self.m = m
        self.q = q
    
    def cyclotron_frequency(self) -> float:
        """Free cyclotron frequency ωc = qB/m."""
        return abs(self.q) * self.B / self.m
    
    def axial_frequency(self) -> float:
        """Axial oscillation frequency ωz = √(qV0/(md²))."""
        return np.sqrt(abs(self.q) * self.V0 / (self.m * self.d**2))
    
    def modified_cyclotron_frequency(self) -> float:
        """
        Modified cyclotron frequency ω+ = (ωc/2) + √((ωc/2)² - ωz²/2).
        """
        omega_c = self.cyclotron_frequency()
        omega_z = self.axial_frequency()
        
        discriminant = (omega_c / 2)**2 - omega_z**2 / 2
        if discriminant < 0:
            raise ValueError("Trap is unstable: ωz² > ωc²/2")
        
        return omega_c / 2 + np.sqrt(discriminant)
    
    def magnetron_frequency(self) -> float:
        """
        Magnetron frequency ω- = (ωc/2) - √((ωc/2)² - ωz²/2).
        """
        omega_c = self.cyclotron_frequency()
        omega_z = self.axial_frequency()
        
        discriminant = (omega_c / 2)**2 - omega_z**2 / 2
        if discriminant < 0:
            raise ValueError("Trap is unstable: ωz² > ωc²/2")
        
        return omega_c / 2 - np.sqrt(discriminant)
    
    def is_stable(self) -> bool:
        """Check if trap configuration is stable."""
        omega_c = self.cyclotron_frequency()
        omega_z = self.axial_frequency()
        return omega_z**2 <= omega_c**2 / 2
    
    def invariant_theorem(self) -> float:
        """
        Verify Brown-Gabrielse invariant: ω+ + ω- = ωc.
        
        Returns the ratio (should be 1.0 for valid trap).
        """
        omega_c = self.cyclotron_frequency()
        omega_plus = self.modified_cyclotron_frequency()
        omega_minus = self.magnetron_frequency()
        return (omega_plus + omega_minus) / omega_c


class GradientDrift:
    """
    Analyzes particle drifts in non-uniform magnetic fields.
    
    Implements:
    - Gradient-B drift
    - Curvature drift
    - Combined grad-B and curvature drift
    """
    
    @staticmethod
    def grad_b_drift_velocity(m: float, v_perp: float, q: float,
                               B: float, grad_B: float) -> float:
        """
        Calculate gradient-B drift velocity.
        
        v_∇B = (mv⊥²/2qB²) × (B × ∇B)/|B|
        
        Returns magnitude of drift velocity.
        """
        return m * v_perp**2 * abs(grad_B) / (2 * abs(q) * B**2)
    
    @staticmethod
    def curvature_drift_velocity(m: float, v_parallel: float, q: float,
                                  B: float, R_c: float) -> float:
        """
        Calculate curvature drift velocity.
        
        v_R = (mv∥²/(qBR_c)) in direction perpendicular to B and R_c.
        
        Args:
            R_c: Radius of curvature of field line
            
        Returns magnitude of drift velocity.
        """
        return m * v_parallel**2 / (abs(q) * B * R_c)
    
    @staticmethod
    def polarization_drift_velocity(m: float, q: float, B: float, 
                                     dE_dt: float) -> float:
        """
        Calculate polarization drift from time-varying E field.
        
        v_p = (m/qB²) dE/dt
        """
        return m * abs(dE_dt) / (abs(q) * B**2)


# Convenience functions

def uniform_crossed_fields(E: float, B: float) -> ChargedParticle:
    """
    Create a particle in crossed E and B fields (E×B drift).
    
    E along y, B along z.
    
    Args:
        E: Electric field magnitude
        B: Magnetic field magnitude
        
    Returns:
        Configured ChargedParticle
    """
    particle = ChargedParticle(name="crossed_fields")
    particle.set_uniform_electric_field(Ey=E)
    particle.set_uniform_magnetic_field(Bz=B)
    return particle


def calculate_drift_velocity(E: float, B: float) -> float:
    """
    Calculate E×B drift velocity.
    
    v_drift = E/B (perpendicular to both E and B)
    """
    return E / B if B != 0 else float('inf')


def magnetic_moment(m: float, v_perp: float, B: float) -> float:
    """
    Calculate magnetic moment (first adiabatic invariant).
    
    μ = mv⊥²/(2B)
    """
    return m * v_perp**2 / (2 * B)


def plasma_frequency(n_e: float, m_e: float = 9.109e-31, 
                     epsilon_0: float = 8.854e-12) -> float:
    """
    Calculate electron plasma frequency.
    
    ωp = √(n_e * e² / (ε₀ * m_e))
    
    Args:
        n_e: Electron density (m⁻³)
        m_e: Electron mass
        epsilon_0: Permittivity of free space
    """
    e = 1.602e-19
    return np.sqrt(n_e * e**2 / (epsilon_0 * m_e))


class ElectromagneticWave:
    """
    Plane electromagnetic wave propagation.
    
    E = E₀ cos(kx - ωt)
    B = B₀ cos(kx - ωt)
    
    where |E₀| = c|B₀| and E ⊥ B ⊥ k.
    
    Example:
        >>> wave = ElectromagneticWave(frequency=1e9)  # 1 GHz
        >>> wavelength = wave.wavelength()  # ~30 cm
    """
    
    # Physical constants
    SPEED_OF_LIGHT = 299792458.0  # m/s
    MU_0 = 4 * np.pi * 1e-7  # H/m
    EPSILON_0 = 8.854e-12  # F/m
    
    def __init__(self, frequency: float = 1e9, amplitude_E: float = 1.0,
                 medium_epsilon_r: float = 1.0, medium_mu_r: float = 1.0):
        """
        Initialize plane EM wave.
        
        Args:
            frequency: Wave frequency (Hz)
            amplitude_E: Electric field amplitude (V/m)
            medium_epsilon_r: Relative permittivity
            medium_mu_r: Relative permeability
        """
        self.f = frequency
        self.omega = 2 * np.pi * frequency
        self.E0 = amplitude_E
        self.epsilon_r = medium_epsilon_r
        self.mu_r = medium_mu_r
    
    def phase_velocity(self) -> float:
        """Phase velocity v = c / √(εᵣμᵣ)."""
        return self.SPEED_OF_LIGHT / np.sqrt(self.epsilon_r * self.mu_r)
    
    def wavelength(self) -> float:
        """Wavelength λ = v/f."""
        return self.phase_velocity() / self.f
    
    def wavenumber(self) -> float:
        """Wavenumber k = 2π/λ = ω/v."""
        return self.omega / self.phase_velocity()
    
    def impedance(self) -> float:
        """
        Wave impedance η = √(μ/ε).
        
        In vacuum: η₀ ≈ 377 Ω
        """
        mu = self.mu_r * self.MU_0
        eps = self.epsilon_r * self.EPSILON_0
        return np.sqrt(mu / eps)
    
    def magnetic_amplitude(self) -> float:
        """Magnetic field amplitude B₀ = E₀/c."""
        return self.E0 / self.phase_velocity()
    
    def intensity(self) -> float:
        """
        Time-averaged intensity (Poynting vector magnitude).
        
        I = (1/2) ε₀ c E₀² = E₀²/(2η)
        """
        return self.E0**2 / (2 * self.impedance())
    
    def energy_density(self) -> float:
        """
        Time-averaged energy density.
        
        u = (1/2) ε₀ E₀²
        """
        eps = self.epsilon_r * self.EPSILON_0
        return 0.5 * eps * self.E0**2
    
    def momentum_density(self) -> float:
        """Momentum density g = S/c²."""
        return self.intensity() / self.SPEED_OF_LIGHT**2
    
    def radiation_pressure(self, absorption: float = 1.0) -> float:
        """
        Radiation pressure on a surface.
        
        P = I/c (absorption) or P = 2I/c (reflection)
        
        Args:
            absorption: 0 for perfect reflection, 1 for perfect absorption
        """
        return self.intensity() * (1 + (1 - absorption)) / self.SPEED_OF_LIGHT
    
    def electric_field(self, x: np.ndarray, t: float) -> np.ndarray:
        """E(x, t) = E₀ cos(kx - ωt)."""
        k = self.wavenumber()
        return self.E0 * np.cos(k * x - self.omega * t)
    
    def refractive_index(self) -> float:
        """Refractive index n = √(εᵣμᵣ) = c/v."""
        return np.sqrt(self.epsilon_r * self.mu_r)


class Antenna:
    """
    Antenna radiation patterns and parameters.
    
    Implements:
    - Hertzian (short) dipole
    - Half-wave dipole
    - Radiation resistance
    - Directivity patterns
    
    Example:
        >>> antenna = Antenna.hertzian_dipole(length=0.01, frequency=1e9)
        >>> P_rad = antenna.radiated_power(current=1.0)
    """
    
    SPEED_OF_LIGHT = 299792458.0
    MU_0 = 4 * np.pi * 1e-7
    IMPEDANCE_FREE_SPACE = 376.73  # Ω
    
    def __init__(self, radiation_resistance: float, directivity: float = 1.5):
        """
        Initialize antenna.
        
        Args:
            radiation_resistance: Radiation resistance R_rad (Ω)
            directivity: Directivity D
        """
        self.R_rad = radiation_resistance
        self.D = directivity
    
    @classmethod
    def hertzian_dipole(cls, length: float, frequency: float) -> 'Antenna':
        """
        Create Hertzian (short) dipole antenna.
        
        R_rad = 80π² (l/λ)² ≈ 790 (l/λ)² Ω
        
        Args:
            length: Dipole length (must be << λ)
            frequency: Operating frequency (Hz)
        """
        wavelength = cls.SPEED_OF_LIGHT / frequency
        l_over_lambda = length / wavelength
        
        R_rad = 80 * np.pi**2 * l_over_lambda**2
        return cls(radiation_resistance=R_rad, directivity=1.5)
    
    @classmethod
    def half_wave_dipole(cls, frequency: float) -> 'Antenna':
        """
        Create half-wave dipole antenna.
        
        R_rad ≈ 73 Ω
        D ≈ 1.64
        """
        return cls(radiation_resistance=73.1, directivity=1.64)
    
    def radiated_power(self, current: float) -> float:
        """
        Radiated power P_rad = (1/2) I² R_rad.
        
        Args:
            current: RMS current (A)
        """
        return 0.5 * current**2 * self.R_rad
    
    def gain(self, efficiency: float = 1.0) -> float:
        """Gain G = η × D."""
        return efficiency * self.D
    
    def effective_aperture(self, wavelength: float) -> float:
        """
        Effective aperture A_e = G λ² / (4π).
        """
        return self.D * wavelength**2 / (4 * np.pi)


class Waveguide:
    """
    Rectangular waveguide analysis.
    
    Supports TE and TM mode analysis with cutoff frequencies.
    
    For TEₘₙ modes in waveguide with dimensions a × b:
        f_c = (c/2) √((m/a)² + (n/b)²)
    
    Example:
        >>> wg = Waveguide(a=0.023, b=0.010)  # WR-90
        >>> fc = wg.cutoff_frequency(m=1, n=0)  # TE₁₀ mode
    """
    
    SPEED_OF_LIGHT = 299792458.0
    
    def __init__(self, a: float, b: float, epsilon_r: float = 1.0):
        """
        Initialize rectangular waveguide.
        
        Args:
            a: Width (larger dimension, m)
            b: Height (smaller dimension, m)
            epsilon_r: Relative permittivity of fill material
        """
        if b > a:
            a, b = b, a  # Ensure a > b
        self.a = a
        self.b = b
        self.epsilon_r = epsilon_r
    
    def cutoff_frequency(self, m: int, n: int) -> float:
        """
        Cutoff frequency for TEₘₙ or TMₘₙ mode.
        
        f_c = (c/(2√εᵣ)) √((m/a)² + (n/b)²)
        """
        c = self.SPEED_OF_LIGHT / np.sqrt(self.epsilon_r)
        return (c / 2) * np.sqrt((m / self.a)**2 + (n / self.b)**2)
    
    def cutoff_wavelength(self, m: int, n: int) -> float:
        """Cutoff wavelength λ_c = c / f_c."""
        return self.SPEED_OF_LIGHT / self.cutoff_frequency(m, n)
    
    def guide_wavelength(self, frequency: float, m: int = 1, n: int = 0) -> float:
        """
        Guide wavelength λ_g = λ / √(1 - (f_c/f)²).
        
        Only valid for f > f_c.
        """
        fc = self.cutoff_frequency(m, n)
        if frequency <= fc:
            raise ValueError(f"Frequency {frequency} Hz below cutoff {fc} Hz")
        
        lambda_0 = self.SPEED_OF_LIGHT / frequency
        return lambda_0 / np.sqrt(1 - (fc / frequency)**2)
    
    def phase_velocity(self, frequency: float, m: int = 1, n: int = 0) -> float:
        """
        Phase velocity v_p = c / √(1 - (f_c/f)²).
        
        Note: v_p > c (but no information travels faster than c).
        """
        fc = self.cutoff_frequency(m, n)
        if frequency <= fc:
            return 0.0
        return self.SPEED_OF_LIGHT / np.sqrt(1 - (fc / frequency)**2)
    
    def group_velocity(self, frequency: float, m: int = 1, n: int = 0) -> float:
        """
        Group velocity v_g = c √(1 - (f_c/f)²).
        
        Energy travels at v_g < c.
        """
        fc = self.cutoff_frequency(m, n)
        if frequency <= fc:
            return 0.0
        return self.SPEED_OF_LIGHT * np.sqrt(1 - (fc / frequency)**2)
    
    def wave_impedance(self, frequency: float, m: int = 1, n: int = 0,
                       mode: str = 'TE') -> float:
        """
        Wave impedance in waveguide.
        
        TE: Z = η₀ / √(1 - (f_c/f)²)
        TM: Z = η₀ × √(1 - (f_c/f)²)
        """
        eta_0 = 376.73  # Free space impedance
        fc = self.cutoff_frequency(m, n)
        
        if frequency <= fc:
            return float('inf') if mode == 'TE' else 0.0
        
        factor = np.sqrt(1 - (fc / frequency)**2)
        
        if mode.upper() == 'TE':
            return eta_0 / factor
        else:  # TM
            return eta_0 * factor
    
    def dominant_mode_range(self) -> Tuple[float, float]:
        """
        Frequency range for single-mode (TE₁₀) operation.
        
        Returns (f_c10, f_c20) or (f_c10, f_c01) whichever is lower.
        """
        fc10 = self.cutoff_frequency(1, 0)
        fc20 = self.cutoff_frequency(2, 0)
        fc01 = self.cutoff_frequency(0, 1)
        
        upper = min(fc20, fc01)
        return (fc10, upper)


class SkinEffect:
    """
    Skin effect calculations for conductors.
    
    At high frequencies, current flows in a thin layer near the surface.
    
    Skin depth: δ = √(2ρ / (ωμ))
    
    Example:
        >>> skin = SkinEffect(conductivity=5.8e7)  # Copper
        >>> delta = skin.skin_depth(frequency=1e6)  # ~66 μm at 1 MHz
    """
    
    MU_0 = 4 * np.pi * 1e-7
    
    def __init__(self, conductivity: float, mu_r: float = 1.0):
        """
        Initialize skin effect calculator.
        
        Args:
            conductivity: Electrical conductivity σ (S/m)
            mu_r: Relative permeability
        """
        self.sigma = conductivity
        self.mu_r = mu_r
        self.rho = 1.0 / conductivity  # Resistivity
    
    def skin_depth(self, frequency: float) -> float:
        """
        Calculate skin depth δ = √(2ρ / (ωμ)).
        
        Args:
            frequency: Frequency (Hz)
            
        Returns:
            Skin depth (m)
        """
        omega = 2 * np.pi * frequency
        mu = self.mu_r * self.MU_0
        return np.sqrt(2 * self.rho / (omega * mu))
    
    def surface_resistance(self, frequency: float) -> float:
        """
        Surface resistance R_s = ρ / δ = √(ωμρ/2).
        
        For a conductor with skin effect, the AC resistance
        per unit length is R_s / (perimeter).
        """
        delta = self.skin_depth(frequency)
        return self.rho / delta
    
    def ac_resistance_ratio(self, wire_radius: float, frequency: float) -> float:
        """
        Ratio of AC to DC resistance for a round wire.
        
        R_AC / R_DC ≈ r / (2δ) for r >> δ
        
        Args:
            wire_radius: Wire radius (m)
            frequency: Frequency (Hz)
        """
        delta = self.skin_depth(frequency)
        
        if delta >= wire_radius:
            # Low frequency: uniform current
            return 1.0
        else:
            # High frequency: skin effect
            return wire_radius / (2 * delta)


# Physical constants for EM
SPEED_OF_LIGHT = 299792458.0  # m/s
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
IMPEDANCE_FREE_SPACE = np.sqrt(MU_0 / EPSILON_0)  # ≈ 377 Ω


def debye_length(n_e: float, T_e: float, epsilon_0: float = EPSILON_0) -> float:
    """
    Calculate Debye length (plasma shielding distance).
    
    λ_D = √(ε₀ k_B T_e / (n_e e²))
    
    Args:
        n_e: Electron density (m⁻³)
        T_e: Electron temperature (K)
        epsilon_0: Permittivity of free space
    """
    k_B = 1.381e-23  # Boltzmann constant
    e = 1.602e-19
    return np.sqrt(epsilon_0 * k_B * T_e / (n_e * e**2))


def cyclotron_resonance_frequency(B: float, m: float, q: float) -> float:
    """
    Cyclotron resonance frequency.
    
    f_c = |q|B / (2πm)
    """
    return abs(q) * B / (2 * np.pi * m)


def biot_savart_field(current: float, dl: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Magnetic field contribution from current element (Biot-Savart law).
    
    dB = (μ₀/4π) × (I dl × r̂) / r²
    
    Args:
        current: Current (A)
        dl: Current element vector (m)
        r: Position vector from element to field point (m)
        
    Returns:
        Magnetic field contribution (T)
    """
    r_mag = np.linalg.norm(r)
    if r_mag < 1e-15:
        return np.zeros(3)
    
    r_hat = r / r_mag
    dB = (MU_0 / (4 * np.pi)) * current * np.cross(dl, r_hat) / r_mag**2
    return dB


__all__ = [
    'FieldType',
    'ElectromagneticField',
    'ChargedParticle',
    'CyclotronMotion',
    'DipoleTrap',
    'PenningTrap',
    'GradientDrift',
    'ElectromagneticWave',
    'Antenna',
    'Waveguide',
    'SkinEffect',
    'uniform_crossed_fields',
    'calculate_drift_velocity',
    'magnetic_moment',
    'plasma_frequency',
    'debye_length',
    'cyclotron_resonance_frequency',
    'biot_savart_field',
    'SPEED_OF_LIGHT',
    'MU_0',
    'EPSILON_0',
    'IMPEDANCE_FREE_SPACE',
]


