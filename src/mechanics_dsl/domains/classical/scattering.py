"""
Scattering Theory for Classical Mechanics

This module implements:
- Rutherford scattering
- Impact parameter to scattering angle
- Differential cross-section
- Total cross-section
- Scattering in central force fields

Classical scattering: a particle approaches from infinity,
interacts with a potential, and escapes to infinity at
a deflection angle θ.
"""
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import sympy as sp
import numpy as np
from scipy import integrate

from ...utils import logger


@dataclass
class ScatteringResult:
    """
    Result of scattering calculation.
    
    Attributes:
        impact_parameter: Impact parameter b
        scattering_angle: Deflection angle θ
        closest_approach: Distance of closest approach r_min
        differential_cross_section: dσ/dΩ at this angle
    """
    impact_parameter: float
    scattering_angle: float
    closest_approach: float
    differential_cross_section: Optional[float] = None


class ScatteringAnalyzer:
    """
    Analyzer for classical scattering problems.
    
    For a central force potential V(r), computes:
    - Scattering angle as function of impact parameter
    - Differential cross-section dσ/dΩ
    - Total cross-section
    
    The scattering angle is:
    θ = π - 2∫[r_min to ∞] (b/r²)/√(1 - b²/r² - V(r)/E) dr
    
    Example:
        >>> analyzer = ScatteringAnalyzer()
        >>> # Coulomb scattering
        >>> result = analyzer.coulomb_scattering(E=1.0, b=0.5, k=1.0)
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
    
    def compute_scattering_angle(self, potential: Callable[[float], float],
                                  energy: float,
                                  impact_parameter: float,
                                  mass: float = 1.0) -> ScatteringResult:
        """
        Compute scattering angle for arbitrary central potential.
        
        θ = π - 2 ∫[r_min to ∞] (b/r²)/√(1 - b²/r² - 2V(r)/(mv²)) dr
        
        Args:
            potential: V(r) function
            energy: Kinetic energy at infinity
            impact_parameter: Impact parameter b
            mass: Particle mass
            
        Returns:
            ScatteringResult with angle and related quantities
        """
        b = impact_parameter
        E = energy
        
        # Find closest approach r_min (turning point)
        # E = (1/2)mv² = L²/(2mr²) + V(r) at r_min
        # where L = mvb
        
        v_inf = np.sqrt(2 * E / mass)
        L = mass * v_inf * b
        
        def effective_potential(r):
            return L**2 / (2 * mass * r**2) + potential(r)
        
        def radial_energy(r):
            return E - effective_potential(r)
        
        # Find r_min by bisection
        r_min = self._find_turning_point(radial_energy, b)
        
        if r_min is None:
            logger.warning("Could not find turning point")
            return ScatteringResult(
                impact_parameter=b,
                scattering_angle=0.0,
                closest_approach=float('inf')
            )
        
        # Integrate for scattering angle
        def integrand(r):
            denominator = r**2 * np.sqrt(radial_energy(r))
            if denominator < 1e-10:
                return 0.0
            return b / denominator
        
        try:
            integral, _ = integrate.quad(integrand, r_min * 1.001, 100 * b, limit=200)
            theta = np.pi - 2 * integral
        except Exception as e:
            logger.warning(f"Integration failed: {e}")
            theta = 0.0
        
        # Keep angle in [0, 2π]
        theta = theta % (2 * np.pi)
        if theta > np.pi:
            theta = 2 * np.pi - theta
        
        return ScatteringResult(
            impact_parameter=b,
            scattering_angle=theta,
            closest_approach=r_min
        )
    
    def _find_turning_point(self, radial_energy: Callable[[float], float],
                            b: float,
                            r_max: float = 1000.0) -> Optional[float]:
        """Find turning point where radial energy = 0."""
        # Use bisection to find r where radial_energy(r) = 0
        
        # Start from a small r and find where energy goes negative
        r_test = np.logspace(-3, np.log10(r_max), 1000)
        
        for i in range(len(r_test) - 1):
            try:
                e1 = radial_energy(r_test[i])
                e2 = radial_energy(r_test[i + 1])
                if e1 * e2 < 0:
                    # Found bracket, use bisection
                    from scipy.optimize import brentq
                    return brentq(radial_energy, r_test[i], r_test[i + 1])
            except:
                continue
        
        return b  # Default to impact parameter
    
    def coulomb_scattering(self, energy: float, impact_parameter: float,
                           k: float, mass: float = 1.0) -> ScatteringResult:
        """
        Rutherford scattering for Coulomb potential V = k/r.
        
        Analytical result:
        θ = 2 * arctan(k / (2*E*b))
        
        Args:
            energy: Kinetic energy
            impact_parameter: Impact parameter b
            k: Coulomb constant (positive for repulsion)
            mass: Particle mass
            
        Returns:
            ScatteringResult
        """
        b = impact_parameter
        E = energy
        
        # Rutherford formula
        if b > 0:
            theta = 2 * np.arctan(k / (2 * E * b))
        else:
            theta = np.pi  # Head-on collision
        
        # Closest approach (for repulsive potential)
        # E = k/r_min + L²/(2m*r_min²)
        # For head-on (b=0): r_min = k/E
        if k > 0:
            r_min = (k / (2 * E)) * (1 + np.sqrt(1 + (2 * E * b / k)**2))
        else:
            # Attractive: need to solve quadratic
            r_min = abs(k) / E  # Approximate
        
        # Differential cross-section
        dcs = self.rutherford_differential_cross_section(k, E, theta, mass)
        
        return ScatteringResult(
            impact_parameter=b,
            scattering_angle=theta,
            closest_approach=r_min,
            differential_cross_section=dcs
        )
    
    def rutherford_differential_cross_section(self, k: float, energy: float,
                                               theta: float, 
                                               mass: float = 1.0) -> float:
        """
        Rutherford differential cross-section.
        
        dσ/dΩ = (k / (4E))² / sin⁴(θ/2)
        
        Args:
            k: Coulomb constant
            energy: Kinetic energy
            theta: Scattering angle
            mass: Particle mass
            
        Returns:
            Differential cross-section
        """
        if abs(theta) < 1e-10:
            return float('inf')
        
        sin_half = np.sin(theta / 2)
        if abs(sin_half) < 1e-10:
            return float('inf')
        
        return (k / (4 * energy))**2 / sin_half**4
    
    def hard_sphere_scattering(self, radius: float, 
                                impact_parameter: float) -> ScatteringResult:
        """
        Scattering from hard sphere of radius R.
        
        θ = π - 2*arcsin(b/R) for b ≤ R
        θ = 0 for b > R
        
        Args:
            radius: Sphere radius
            impact_parameter: Impact parameter
            
        Returns:
            ScatteringResult
        """
        b = impact_parameter
        R = radius
        
        if b > R:
            theta = 0.0
            r_min = b
        else:
            theta = np.pi - 2 * np.arcsin(b / R)
            r_min = R
        
        # Differential cross-section for hard sphere
        # dσ/dΩ = R²/4 (isotropic in CM frame)
        dcs = R**2 / 4
        
        return ScatteringResult(
            impact_parameter=b,
            scattering_angle=theta,
            closest_approach=r_min,
            differential_cross_section=dcs
        )
    
    def total_cross_section(self, differential_cs: Callable[[float], float],
                            theta_min: float = 0.001,
                            theta_max: float = np.pi) -> float:
        """
        Integrate differential cross-section to get total.
        
        σ_total = 2π ∫ (dσ/dΩ) sin(θ) dθ
        
        Args:
            differential_cs: dσ/dΩ as function of θ
            theta_min: Minimum angle (avoid singularity)
            theta_max: Maximum angle
            
        Returns:
            Total cross-section
        """
        def integrand(theta):
            return differential_cs(theta) * np.sin(theta)
        
        try:
            result, _ = integrate.quad(integrand, theta_min, theta_max)
            return 2 * np.pi * result
        except Exception as e:
            logger.warning(f"Cross-section integration failed: {e}")
            return float('inf')


class SymbolicScattering:
    """
    Symbolic scattering calculations using SymPy.
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
    
    def rutherford_formula(self) -> Dict[str, sp.Expr]:
        """
        Derive Rutherford scattering formulas symbolically.
        
        Returns:
            Dictionary of symbolic expressions
        """
        k = self.get_symbol('k')
        E = self.get_symbol('E', positive=True)
        b = self.get_symbol('b', positive=True)
        theta = self.get_symbol('theta', positive=True)
        
        # Impact parameter to angle relation
        # tan(θ/2) = k / (2Eb)
        theta_expr = 2 * sp.atan(k / (2 * E * b))
        
        # Differential cross-section
        # dσ/dΩ = (k / (4E))² / sin⁴(θ/2)
        dcs = (k / (4 * E))**2 / sp.sin(theta / 2)**4
        
        # Closest approach (repulsive)
        a = k / (2 * E)
        r_min = a * (1 + sp.sqrt(1 + (b / a)**2))
        
        return {
            'scattering_angle': theta_expr,
            'differential_cross_section': dcs,
            'closest_approach': r_min,
            'half_distance_of_closest_approach': a
        }
    
    def orbit_equation(self) -> sp.Expr:
        """
        Derive the orbit equation for scattering.
        
        1/r = (1/p) * (1 + e*cos(φ))
        
        where p = L²/(mk) and e = √(1 + 2EL²/(mk²))
        
        Returns:
            Orbit equation
        """
        r = self.get_symbol('r', positive=True)
        phi = self.get_symbol('phi')
        L = self.get_symbol('L', positive=True)
        m = self.get_symbol('m', positive=True)
        k = self.get_symbol('k')
        E = self.get_symbol('E')
        
        p = L**2 / (m * sp.Abs(k))
        e = sp.sqrt(1 + 2 * E * L**2 / (m * k**2))
        
        # 1/r = (1/p)(1 + e*cos(φ))
        orbit = sp.Eq(1/r, (1/p) * (1 + e * sp.cos(phi)))
        
        return orbit


# Convenience functions

def rutherford_angle(energy: float, impact_parameter: float, 
                     k: float) -> float:
    """
    Compute Rutherford scattering angle.
    
    θ = 2 * arctan(k / (2Eb))
    
    Args:
        energy: Kinetic energy
        impact_parameter: Impact parameter b
        k: Coulomb constant
        
    Returns:
        Scattering angle in radians
    """
    if impact_parameter <= 0:
        return np.pi
    return 2 * np.arctan(k / (2 * energy * impact_parameter))


def rutherford_cross_section(k: float, energy: float, 
                              theta: float) -> float:
    """
    Compute Rutherford differential cross-section.
    
    dσ/dΩ = (k / (4E))² / sin⁴(θ/2)
    
    Args:
        k: Coulomb constant
        energy: Kinetic energy
        theta: Scattering angle
        
    Returns:
        Differential cross-section
    """
    sin_half = np.sin(theta / 2)
    if abs(sin_half) < 1e-10:
        return float('inf')
    return (k / (4 * energy))**2 / sin_half**4
