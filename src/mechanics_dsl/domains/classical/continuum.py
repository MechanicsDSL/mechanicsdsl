"""
Continuous Systems and Field Mechanics

This module implements:
- Lagrangian density for continuous media
- Wave equation derivation
- Vibrating string dynamics
- Membrane vibrations
- Stress-energy tensor
- Field Euler-Lagrange equations

For a continuous system with field φ(x, t):
Action S = ∫∫ L(φ, ∂φ/∂t, ∂φ/∂x) dx dt

The Euler-Lagrange equation becomes:
∂L/∂φ - ∂/∂t(∂L/∂φ_t) - ∂/∂x(∂L/∂φ_x) = 0
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp

from ...utils import logger


class FieldType(Enum):
    """Types of continuous fields."""
    SCALAR = "scalar"       # φ(x,t)
    VECTOR = "vector"       # A(x,t)
    TENSOR = "tensor"       # T_ij(x,t)


@dataclass
class FieldConfiguration:
    """
    Configuration of a continuous field.
    
    Attributes:
        dimension: Spatial dimension (1, 2, or 3)
        field_type: Type of field
        boundary_conditions: Dictionary of BC specifications
    """
    dimension: int = 1
    field_type: FieldType = FieldType.SCALAR
    boundary_conditions: Dict[str, str] = field(default_factory=dict)


@dataclass
class WaveMode:
    """
    A normal mode of a wave system.
    
    Attributes:
        mode_number: Mode index (n)
        frequency: Angular frequency ω
        wavenumber: Wave number k
        amplitude_function: Spatial mode shape
    """
    mode_number: int
    frequency: float
    wavenumber: float
    amplitude_function: Optional[sp.Expr] = None


class LagrangianDensity:
    """
    Lagrangian density for continuous systems.
    
    L = T - V where T and V are kinetic and potential
    energy densities.
    
    Example:
        >>> # Vibrating string
        >>> L = (1/2)*rho*phi_t**2 - (1/2)*T*phi_x**2
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._function_cache: Dict[str, sp.Function] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def get_field(self, name: str, *args) -> sp.Function:
        """Get or create a field function."""
        if name not in self._function_cache:
            self._function_cache[name] = sp.Function(name, real=True)
        return self._function_cache[name](*args)
    
    def string_lagrangian(self) -> sp.Expr:
        """
        Get Lagrangian density for vibrating string.
        
        L = (1/2)*μ*(∂u/∂t)² - (1/2)*T*(∂u/∂x)²
        
        where μ is linear density and T is tension.
        
        Returns:
            Lagrangian density expression
        """
        x = self.get_symbol('x')
        t = self.get_symbol('t')
        mu = self.get_symbol('mu', positive=True)  # Linear density
        T = self.get_symbol('T', positive=True)    # Tension
        
        u = self.get_field('u', x, t)
        u_t = sp.diff(u, t)
        u_x = sp.diff(u, x)
        
        L = sp.Rational(1, 2) * mu * u_t**2 - sp.Rational(1, 2) * T * u_x**2
        
        return L
    
    def membrane_lagrangian(self) -> sp.Expr:
        """
        Get Lagrangian density for vibrating membrane.
        
        L = (1/2)*σ*(∂u/∂t)² - (1/2)*T*[(∂u/∂x)² + (∂u/∂y)²]
        
        Returns:
            Lagrangian density
        """
        x = self.get_symbol('x')
        y = self.get_symbol('y')
        t = self.get_symbol('t')
        sigma = self.get_symbol('sigma', positive=True)  # Surface density
        T = self.get_symbol('T', positive=True)          # Tension
        
        u = self.get_field('u', x, y, t)
        u_t = sp.diff(u, t)
        u_x = sp.diff(u, x)
        u_y = sp.diff(u, y)
        
        L = sp.Rational(1, 2) * sigma * u_t**2 - \
            sp.Rational(1, 2) * T * (u_x**2 + u_y**2)
        
        return L
    
    def klein_gordon_lagrangian(self) -> sp.Expr:
        """
        Get Klein-Gordon Lagrangian density.
        
        L = (1/2)[(∂φ/∂t)² - c²(∂φ/∂x)² - m²c⁴φ²]
        
        Returns:
            Lagrangian density
        """
        x = self.get_symbol('x')
        t = self.get_symbol('t')
        c = self.get_symbol('c', positive=True)
        m = self.get_symbol('m', positive=True)
        
        phi = self.get_field('phi', x, t)
        phi_t = sp.diff(phi, t)
        phi_x = sp.diff(phi, x)
        
        L = sp.Rational(1, 2) * (phi_t**2 - c**2 * phi_x**2 - m**2 * c**4 * phi**2)
        
        return L


class FieldEulerLagrange:
    """
    Derive field equations from Lagrangian density.
    
    For L(φ, φ_t, φ_x, φ_xx, ...):
    
    ∂L/∂φ - ∂/∂t(∂L/∂φ_t) - ∂/∂x(∂L/∂φ_x) + ... = 0
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def derive_field_equation(self, lagrangian: sp.Expr,
                               field: sp.Function,
                               coordinates: List[sp.Symbol]) -> sp.Expr:
        """
        Derive Euler-Lagrange field equation.
        
        δS/δφ = ∂L/∂φ - Σᵢ ∂/∂xᵢ(∂L/∂(∂φ/∂xᵢ)) = 0
        
        Args:
            lagrangian: Lagrangian density L
            field: Field function φ(x,t)
            coordinates: List of independent variables [x, t, ...]
            
        Returns:
            Field equation
        """
        # ∂L/∂φ
        dL_dphi = sp.diff(lagrangian, field)
        
        # Σᵢ ∂/∂xᵢ(∂L/∂φᵢ)
        derivative_terms = sp.S.Zero
        for coord in coordinates:
            phi_i = sp.diff(field, coord)
            dL_dphi_i = sp.diff(lagrangian, phi_i)
            derivative_terms += sp.diff(dL_dphi_i, coord)
        
        equation = dL_dphi - derivative_terms
        
        return sp.simplify(equation)
    
    def wave_equation_1d(self) -> sp.Expr:
        """
        Derive 1D wave equation from string Lagrangian.
        
        ∂²u/∂t² = c² ∂²u/∂x²
        
        where c² = T/μ
        
        Returns:
            Wave equation
        """
        x = self.get_symbol('x')
        t = self.get_symbol('t')
        c = self.get_symbol('c', positive=True)
        
        u = sp.Function('u', real=True)(x, t)
        
        # Wave equation: u_tt = c² u_xx
        wave_eq = sp.diff(u, t, 2) - c**2 * sp.diff(u, x, 2)
        
        return sp.Eq(wave_eq, 0)


class VibratingString:
    """
    Solver for vibrating string problems.
    
    Boundary conditions:
    - Fixed ends: u(0,t) = u(L,t) = 0
    - Free ends: ∂u/∂x|₀ = ∂u/∂x|_L = 0
    
    Normal modes: u_n(x,t) = A_n sin(nπx/L) cos(ω_n t)
    where ω_n = nπc/L
    """
    
    def __init__(self, length: float, wave_speed: float):
        """
        Initialize vibrating string.
        
        Args:
            length: String length L
            wave_speed: Wave speed c = √(T/μ)
        """
        self.L = length
        self.c = wave_speed
    
    def fundamental_frequency(self) -> float:
        """
        Get fundamental (n=1) frequency.
        
        f₁ = c / (2L)
        
        Returns:
            Fundamental frequency in Hz
        """
        return self.c / (2 * self.L)
    
    def mode_frequency(self, n: int) -> float:
        """
        Get frequency of mode n.
        
        f_n = n * c / (2L)
        
        Args:
            n: Mode number (1, 2, 3, ...)
            
        Returns:
            Frequency in Hz
        """
        return n * self.c / (2 * self.L)
    
    def mode_angular_frequency(self, n: int) -> float:
        """
        Get angular frequency of mode n.
        
        ω_n = nπc/L
        
        Args:
            n: Mode number
            
        Returns:
            Angular frequency
        """
        return n * np.pi * self.c / self.L
    
    def wavenumber(self, n: int) -> float:
        """
        Get wavenumber of mode n.
        
        k_n = nπ/L
        
        Args:
            n: Mode number
            
        Returns:
            Wavenumber
        """
        return n * np.pi / self.L
    
    def mode_shape(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get mode shape (spatial pattern).
        
        φ_n(x) = sin(nπx/L)
        
        Args:
            n: Mode number
            x: Position(s)
            
        Returns:
            Mode amplitude(s)
        """
        return np.sin(n * np.pi * x / self.L)
    
    def compute_modes(self, n_modes: int = 5) -> List[WaveMode]:
        """
        Compute first n normal modes.
        
        Args:
            n_modes: Number of modes to compute
            
        Returns:
            List of WaveMode objects
        """
        modes = []
        for n in range(1, n_modes + 1):
            mode = WaveMode(
                mode_number=n,
                frequency=self.mode_angular_frequency(n),
                wavenumber=self.wavenumber(n)
            )
            modes.append(mode)
        return modes
    
    def fourier_coefficients(self, initial_shape: np.ndarray,
                              x_points: np.ndarray,
                              n_modes: int = 10) -> np.ndarray:
        """
        Compute Fourier coefficients for initial displacement.
        
        A_n = (2/L) ∫₀ᴸ f(x) sin(nπx/L) dx
        
        Args:
            initial_shape: Initial displacement u(x, 0)
            x_points: x coordinates
            n_modes: Number of modes
            
        Returns:
            Array of Fourier coefficients
        """
        dx = x_points[1] - x_points[0]
        coeffs = np.zeros(n_modes)
        
        for n in range(1, n_modes + 1):
            mode_shape = np.sin(n * np.pi * x_points / self.L)
            coeffs[n-1] = (2 / self.L) * np.sum(initial_shape * mode_shape) * dx
        
        return coeffs
    
    def solution(self, x: np.ndarray, t: np.ndarray,
                 coefficients: np.ndarray) -> np.ndarray:
        """
        Compute string displacement using modal superposition.
        
        u(x,t) = Σ A_n sin(nπx/L) cos(ω_n t)
        
        Args:
            x: Spatial points
            t: Time points
            coefficients: Fourier coefficients A_n
            
        Returns:
            2D array u[i,j] = u(x[i], t[j])
        """
        X, T = np.meshgrid(x, t, indexing='ij')
        u = np.zeros_like(X)
        
        for n in range(1, len(coefficients) + 1):
            omega_n = self.mode_angular_frequency(n)
            spatial = np.sin(n * np.pi * X / self.L)
            temporal = np.cos(omega_n * T)
            u += coefficients[n-1] * spatial * temporal
        
        return u


class VibratingMembrane:
    """
    Solver for rectangular vibrating membrane.
    
    Boundary conditions: u = 0 on all edges (fixed boundary)
    
    Normal modes: u_{mn}(x,y,t) = sin(mπx/a) sin(nπy/b) cos(ω_{mn} t)
    where ω_{mn} = πc√((m/a)² + (n/b)²)
    """
    
    def __init__(self, length_x: float, length_y: float, wave_speed: float):
        """
        Initialize rectangular membrane.
        
        Args:
            length_x: Length in x direction (a)
            length_y: Length in y direction (b)
            wave_speed: Wave speed c
        """
        self.a = length_x
        self.b = length_y
        self.c = wave_speed
    
    def mode_frequency(self, m: int, n: int) -> float:
        """
        Get frequency of mode (m,n).
        
        ω_{mn} = πc√((m/a)² + (n/b)²)
        
        Args:
            m, n: Mode numbers
            
        Returns:
            Angular frequency
        """
        return np.pi * self.c * np.sqrt((m/self.a)**2 + (n/self.b)**2)
    
    def mode_shape(self, m: int, n: int,
                   x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get 2D mode shape.
        
        φ_{mn}(x,y) = sin(mπx/a) sin(nπy/b)
        
        Args:
            m, n: Mode numbers
            x, y: Coordinate arrays
            
        Returns:
            2D array of mode amplitudes
        """
        X, Y = np.meshgrid(x, y, indexing='ij')
        return np.sin(m * np.pi * X / self.a) * np.sin(n * np.pi * Y / self.b)
    
    def compute_modes(self, max_m: int = 3, 
                      max_n: int = 3) -> List[Tuple[int, int, float]]:
        """
        Compute modes up to specified indices.
        
        Args:
            max_m, max_n: Maximum mode indices
            
        Returns:
            List of (m, n, frequency) tuples sorted by frequency
        """
        modes = []
        for m in range(1, max_m + 1):
            for n in range(1, max_n + 1):
                freq = self.mode_frequency(m, n)
                modes.append((m, n, freq))
        
        # Sort by frequency
        modes.sort(key=lambda x: x[2])
        return modes


class StressEnergyTensor:
    """
    Stress-energy tensor for continuous systems.
    
    T^{μν} = (∂L/∂(∂_μφ)) ∂^νφ - η^{μν} L
    
    For a 1+1 dimensional system:
    - T^{00} = energy density
    - T^{01} = T^{10} = energy flux
    - T^{11} = stress (negative pressure)
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def energy_density(self, lagrangian: sp.Expr,
                       field: sp.Function,
                       time: sp.Symbol) -> sp.Expr:
        """
        Compute energy density T^{00}.
        
        T^{00} = (∂L/∂φ_t) φ_t - L
        
        This is the Hamiltonian density.
        
        Args:
            lagrangian: Lagrangian density
            field: Field function
            time: Time variable
            
        Returns:
            Energy density
        """
        phi_t = sp.diff(field, time)
        
        dL_dphi_t = sp.diff(lagrangian, phi_t)
        
        T00 = dL_dphi_t * phi_t - lagrangian
        
        return sp.simplify(T00)
    
    def momentum_density(self, lagrangian: sp.Expr,
                         field: sp.Function,
                         time: sp.Symbol,
                         space: sp.Symbol) -> sp.Expr:
        """
        Compute momentum density T^{01}.
        
        T^{01} = (∂L/∂φ_t) φ_x
        
        Args:
            lagrangian: Lagrangian density
            field: Field function
            time: Time variable
            space: Space variable
            
        Returns:
            Momentum density
        """
        phi_t = sp.diff(field, time)
        phi_x = sp.diff(field, space)
        
        dL_dphi_t = sp.diff(lagrangian, phi_t)
        
        T01 = dL_dphi_t * phi_x
        
        return sp.simplify(T01)


# Convenience functions

def string_mode_frequencies(length: float, tension: float,
                            density: float, n_modes: int = 5) -> List[float]:
    """
    Compute normal mode frequencies for a string.
    
    f_n = (n/2L)√(T/μ)
    
    Args:
        length: String length
        tension: String tension
        density: Linear mass density
        n_modes: Number of modes
        
    Returns:
        List of frequencies in Hz
    """
    c = np.sqrt(tension / density)
    return [(n * c) / (2 * length) for n in range(1, n_modes + 1)]


def wave_speed(tension: float, density: float) -> float:
    """
    Compute wave speed in a string.
    
    c = √(T/μ)
    
    Args:
        tension: String tension
        density: Linear mass density
        
    Returns:
        Wave speed
    """
    return np.sqrt(tension / density)
