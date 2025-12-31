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


__all__ = [
    'FieldType',
    'ElectromagneticField',
    'ChargedParticle',
    'CyclotronMotion',
    'DipoleTrap',
    'uniform_crossed_fields',
    'calculate_drift_velocity',
]
