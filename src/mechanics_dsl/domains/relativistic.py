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
import sympy as sp
import numpy as np
from .base import PhysicsDomain


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
        self.parameters['m'] = mass
        self.parameters['c'] = SPEED_OF_LIGHT
        
        self.coordinates = ['x', 'y', 'z']
        
        # Symbols
        self.x, self.y, self.z = sp.symbols('x y z', real=True)
        self.vx, self.vy, self.vz = sp.symbols('x_dot y_dot z_dot', real=True)
        self.t = sp.Symbol('t', real=True)
        self.m = sp.Symbol('m', positive=True)
        self.c = sp.Symbol('c', positive=True)
        
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
        c = self.parameters.get('c', SPEED_OF_LIGHT)
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
        c = self.parameters.get('c', SPEED_OF_LIGHT)
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
        c = self.parameters.get('c', SPEED_OF_LIGHT)
        gamma = self.lorentz_factor(v)
        return (gamma - 1) * self.rest_mass * c**2
    
    def rest_energy(self) -> float:
        """Calculate rest energy E₀ = mc²."""
        c = self.parameters.get('c', SPEED_OF_LIGHT)
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
        px, py, pz = sp.symbols('p_x p_y p_z', real=True)
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
            'px_dot': Fx,
            'py_dot': Fy,
            'pz_dot': Fz,
        }
    
    def get_state_variables(self) -> List[str]:
        """State variables: positions and momenta."""
        return ['x', 'y', 'z', 'px', 'py', 'pz']
    
    def get_required_parameters(self) -> List[str]:
        return ['m', 'c']
    
    def get_conserved_quantities(self) -> Dict[str, sp.Expr]:
        """Total energy and momentum (if no external forces)."""
        p_squared = (self.m * self.vx)**2 + (self.m * self.vy)**2 + (self.m * self.vz)**2
        v_squared = self.vx**2 + self.vy**2 + self.vz**2
        gamma = 1 / sp.sqrt(1 - v_squared / self.c**2)
        
        return {
            'total_energy': gamma * self.m * self.c**2 + self.potential,
            'momentum_x': gamma * self.m * self.vx,
            'momentum_y': gamma * self.m * self.vy,
            'momentum_z': gamma * self.m * self.vz,
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
        gamma = 1.0 / np.sqrt(1 - (v/c)**2)
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
        gamma = 1.0 / np.sqrt(1 - (v/c)**2)
        return proper_length / gamma


class RelativisticCollision:
    """
    Analyzes relativistic particle collisions.
    
    Conserves four-momentum: Σpᵢ = Σpf
    """
    
    @staticmethod
    def invariant_mass(particles: List[Tuple[float, np.ndarray]], c: float = SPEED_OF_LIGHT) -> float:
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
            gamma = 1.0 / np.sqrt(1 - (v_mag/c)**2)
            E = gamma * m * c**2
            p = gamma * m * v
            
            total_E += E
            total_p += p
        
        s = (total_E / c)**2 - np.dot(total_p, total_p)
        return np.sqrt(s) / c if s > 0 else 0.0
    
    @staticmethod
    def threshold_energy(m_target: float, m_products: List[float], 
                        c: float = SPEED_OF_LIGHT) -> float:
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
    return 1.0 / np.sqrt(1 - (v/c)**2)


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
    def longitudinal_frequency(f_source: float, v: float, 
                               c: float = SPEED_OF_LIGHT,
                               approaching: bool = True) -> float:
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
    def transverse_frequency(f_source: float, v: float,
                             c: float = SPEED_OF_LIGHT) -> float:
        """
        Calculate transverse Doppler shift (time dilation effect).
        
        f_obs = f_source / γ
        
        Occurs when source moves perpendicular to line of sight.
        """
        gamma_val = 1.0 / np.sqrt(1 - (v/c)**2)
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
        factor = (1 + z)**2
        beta_val = (factor - 1) / (factor + 1)
        return beta_val * c


def relativistic_aberration(theta_source: float, v: float, 
                            c: float = SPEED_OF_LIGHT) -> float:
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


def compton_wavelength_shift(theta: float, m_e: float = 9.109e-31,
                             c: float = SPEED_OF_LIGHT,
                             h: float = 6.626e-34) -> float:
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


__all__ = [
    'SPEED_OF_LIGHT',
    'RelativisticParticle',
    'FourVector',
    'LorentzTransform',
    'RelativisticCollision',
    'DopplerEffect',
    'gamma',
    'beta',
    'rapidity',
    'relativistic_aberration',
    'compton_wavelength_shift',
]

