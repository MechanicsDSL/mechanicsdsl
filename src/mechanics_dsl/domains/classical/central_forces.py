"""
Central Force Problems and Orbital Mechanics

This module implements:
- Effective potential for central force problems
- Orbit classification (bounded, unbounded, circular, elliptical)
- Turning point computation
- Kepler problem specialization
- Precession rate for non-Keplerian potentials

For a central force problem with potential V(r):
    L = (1/2)*m*(ṙ² + r²*φ̇²) - V(r)

The effective potential is:
    V_eff(r) = V(r) + L²/(2mr²)

where L = mr²φ̇ is the conserved angular momentum.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ...utils import logger


class OrbitType(Enum):
    """Classification of orbit types."""
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"
    BOUNDED = "bounded"          # Generic bounded orbit
    UNBOUNDED = "unbounded"      # Generic escape orbit
    COLLISION = "collision"      # Falls to center
    UNKNOWN = "unknown"


@dataclass
class OrbitalElements:
    """
    Keplerian orbital elements.
    
    Attributes:
        semi_major_axis: Semi-major axis a
        eccentricity: Orbital eccentricity e
        period: Orbital period T
        angular_momentum: Angular momentum L
        energy: Total mechanical energy E
        perihelion: Closest approach r_min
        aphelion: Farthest distance r_max (None for unbounded)
        orbit_type: Classification of orbit
    """
    semi_major_axis: float
    eccentricity: float
    period: Optional[float] = None
    angular_momentum: float = 0.0
    energy: float = 0.0
    perihelion: float = 0.0
    aphelion: Optional[float] = None
    orbit_type: OrbitType = OrbitType.UNKNOWN
    
    def __repr__(self) -> str:
        return (f"OrbitalElements(a={self.semi_major_axis:.4f}, e={self.eccentricity:.4f}, "
                f"T={self.period}, type={self.orbit_type.value})")


@dataclass
class TurningPoints:
    """
    Turning points of effective potential motion.
    
    Attributes:
        r_min: Inner turning point (perihelion)
        r_max: Outer turning point (aphelion), None if unbounded
        is_bounded: Whether orbit is bounded
    """
    r_min: float
    r_max: Optional[float]
    is_bounded: bool
    
    def __repr__(self) -> str:
        if self.is_bounded:
            return f"TurningPoints(r_min={self.r_min:.4f}, r_max={self.r_max:.4f}, bounded)"
        else:
            return f"TurningPoints(r_min={self.r_min:.4f}, unbounded)"


class EffectivePotential:
    """
    Effective potential for central force problems.
    
    For a particle with angular momentum L in a central potential V(r):
        V_eff(r) = V(r) + L²/(2mr²)
    
    The radial motion is equivalent to 1D motion in V_eff(r).
    
    Example:
        >>> # Kepler problem
        >>> V = -G*M*m/r
        >>> eff = EffectivePotential(V, L=1.0, m=1.0)
        >>> turning_points = eff.find_turning_points(E=-0.5)
    """
    
    def __init__(self, potential: sp.Expr, angular_momentum: float = 1.0,
                 mass: float = 1.0, r_symbol: Optional[sp.Symbol] = None):
        """
        Initialize effective potential.
        
        Args:
            potential: Symbolic potential V(r)
            angular_momentum: Angular momentum L
            mass: Particle mass m
            r_symbol: Symbol for radial coordinate (auto-detected if None)
        """
        self.potential = potential
        self.L = angular_momentum
        self.m = mass
        
        # Detect or use provided r symbol
        if r_symbol is not None:
            self.r = r_symbol
        else:
            symbols = list(potential.free_symbols)
            if len(symbols) == 1:
                self.r = symbols[0]
            else:
                self.r = sp.Symbol('r', real=True, positive=True)
        
        # Build effective potential
        self._effective = self._build_effective_potential()
        self._numerical_func = None
    
    def _build_effective_potential(self) -> sp.Expr:
        """Build the effective potential expression."""
        centrifugal = self.L**2 / (2 * self.m * self.r**2)
        return self.potential + centrifugal
    
    def get_expression(self) -> sp.Expr:
        """Get symbolic effective potential expression."""
        return self._effective
    
    def evaluate(self, r: float) -> float:
        """
        Evaluate effective potential at radius r.
        
        Args:
            r: Radial distance
            
        Returns:
            V_eff(r)
        """
        if self._numerical_func is None:
            self._numerical_func = sp.lambdify(self.r, self._effective, 'numpy')
        
        return float(self._numerical_func(r))
    
    def derivative(self, r: float) -> float:
        """
        Evaluate dV_eff/dr at radius r.
        
        Args:
            r: Radial distance
            
        Returns:
            dV_eff/dr
        """
        dV_eff = sp.diff(self._effective, self.r)
        func = sp.lambdify(self.r, dV_eff, 'numpy')
        return float(func(r))
    
    def find_circular_orbit_radius(self) -> Optional[float]:
        """
        Find radius of circular orbit where dV_eff/dr = 0 and d²V_eff/dr² > 0.
        
        Returns:
            Circular orbit radius, or None if not found
        """
        dV_eff = sp.diff(self._effective, self.r)
        
        # Solve dV_eff/dr = 0
        try:
            solutions = sp.solve(dV_eff, self.r)
        except Exception as e:
            logger.warning(f"Could not solve for circular orbit: {e}")
            return None
        
        for sol in solutions:
            try:
                r_val = float(sol.evalf())
                if r_val > 0:
                    # Check stability (d²V/dr² > 0)
                    d2V = sp.diff(self._effective, self.r, 2)
                    d2V_val = float(d2V.subs(self.r, r_val).evalf())
                    if d2V_val > 0:
                        return r_val
            except (TypeError, ValueError):
                continue
        
        return None
    
    def find_turning_points(self, energy: float, 
                            r_min_search: float = 0.01,
                            r_max_search: float = 100.0) -> TurningPoints:
        """
        Find turning points where E = V_eff(r).
        
        Args:
            energy: Total mechanical energy
            r_min_search: Minimum r to search
            r_max_search: Maximum r to search
            
        Returns:
            TurningPoints object with inner and outer turning points
        """
        def f(r):
            return self.evaluate(r) - energy
        
        # Find roots of V_eff(r) - E = 0
        turning_points = []
        
        # Sample to find sign changes
        r_vals = np.logspace(np.log10(r_min_search), np.log10(r_max_search), 1000)
        f_vals = np.array([f(r) for r in r_vals])
        
        # Find sign changes
        for i in range(len(r_vals) - 1):
            if f_vals[i] * f_vals[i+1] < 0:
                try:
                    r_root = brentq(f, r_vals[i], r_vals[i+1])
                    turning_points.append(r_root)
                except ValueError:
                    pass
        
        if len(turning_points) == 0:
            # No turning points - check if always below or always above
            if f(r_min_search) > 0:
                # E < V_eff everywhere - forbidden
                logger.warning("Energy below potential everywhere")
                return TurningPoints(r_min=0, r_max=None, is_bounded=False)
            else:
                # E > V_eff everywhere - unbounded
                return TurningPoints(r_min=r_min_search, r_max=None, is_bounded=False)
        
        elif len(turning_points) == 1:
            # One turning point - either inner or outer
            r_tp = turning_points[0]
            if f(r_tp * 0.9) < 0:
                # Inner turning point (perihelion)
                return TurningPoints(r_min=r_tp, r_max=None, is_bounded=False)
            else:
                # Outer turning point (aphelion) - shouldn't happen for central forces
                return TurningPoints(r_min=r_min_search, r_max=r_tp, is_bounded=True)
        
        else:
            # Two or more turning points - bounded orbit
            return TurningPoints(
                r_min=min(turning_points),
                r_max=max(turning_points),
                is_bounded=True
            )


class CentralForceAnalyzer:
    """
    Analyzer for central force problems.
    
    Supports:
    - Effective potential analysis
    - Orbit classification
    - Turning point computation
    - Precession rate calculation
    
    Example:
        >>> analyzer = CentralForceAnalyzer()
        >>> # Gravitational potential
        >>> V = -G*M*m/r
        >>> orbit = analyzer.classify_orbit(V, E, L, m)
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True, 'positive': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def classify_orbit(self, potential: sp.Expr, energy: float,
                       angular_momentum: float, mass: float = 1.0) -> OrbitType:
        """
        Classify the type of orbit based on energy and angular momentum.
        
        Args:
            potential: Central potential V(r)
            energy: Total mechanical energy
            angular_momentum: Angular momentum
            mass: Particle mass
            
        Returns:
            OrbitType classification
        """
        eff_pot = EffectivePotential(potential, angular_momentum, mass)
        
        # Find turning points
        turning = eff_pot.find_turning_points(energy)
        
        if not turning.is_bounded:
            if turning.r_min > 0:
                return OrbitType.UNBOUNDED
            else:
                return OrbitType.COLLISION
        
        # Check for circular orbit
        if turning.r_max is not None:
            if abs(turning.r_max - turning.r_min) / turning.r_min < 0.01:
                return OrbitType.CIRCULAR
        
        return OrbitType.BOUNDED
    
    def compute_orbital_elements_kepler(self, G: float, M: float, m: float,
                                         energy: float, angular_momentum: float) -> OrbitalElements:
        """
        Compute Keplerian orbital elements.
        
        For the gravitational potential V = -GMm/r:
        - e = sqrt(1 + 2EL²/(m(GMm)²))
        - a = -GMm/(2E) for E < 0
        - T = 2π*sqrt(a³/(GM))
        
        Args:
            G: Gravitational constant
            M: Central mass
            m: Orbiting mass
            energy: Total energy
            angular_momentum: Angular momentum
            
        Returns:
            OrbitalElements object
        """
        mu = G * M * m
        
        # Eccentricity
        e_squared = 1 + 2 * energy * angular_momentum**2 / (m * mu**2)
        e = np.sqrt(max(0, e_squared))
        
        # Semi-major axis (for elliptical orbits)
        if energy < 0:
            a = -mu / (2 * energy)
            period = 2 * np.pi * np.sqrt(a**3 / (G * M))
            r_min = a * (1 - e)
            r_max = a * (1 + e)
            
            if e < 1e-6:
                orbit_type = OrbitType.CIRCULAR
            else:
                orbit_type = OrbitType.ELLIPTICAL
        elif abs(energy) < 1e-10:
            # Parabolic
            a = float('inf')
            period = None
            r_min = angular_momentum**2 / (2 * m * mu)
            r_max = None
            orbit_type = OrbitType.PARABOLIC
        else:
            # Hyperbolic
            a = mu / (2 * energy)  # Negative for hyperbolic
            period = None
            r_min = a * (e - 1)  # Closest approach
            r_max = None
            orbit_type = OrbitType.HYPERBOLIC
        
        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=e,
            period=period,
            angular_momentum=angular_momentum,
            energy=energy,
            perihelion=r_min,
            aphelion=r_max,
            orbit_type=orbit_type
        )
    
    def compute_precession_rate(self, potential: sp.Expr, 
                                 angular_momentum: float,
                                 mass: float,
                                 energy: float) -> float:
        """
        Compute the precession rate for a non-Keplerian potential.
        
        For a potential V(r) = -k/r + δV(r), the precession per orbit is:
        Δφ = 2π * (additional angular displacement)
        
        Uses numerical integration of the orbit equation.
        
        Args:
            potential: Central potential V(r)
            angular_momentum: Angular momentum L
            mass: Particle mass
            energy: Total energy
            
        Returns:
            Precession rate in radians per orbit
        """
        eff = EffectivePotential(potential, angular_momentum, mass)
        turning = eff.find_turning_points(energy)
        
        if not turning.is_bounded or turning.r_max is None:
            logger.warning("Cannot compute precession for unbounded orbit")
            return 0.0
        
        r_min = turning.r_min
        r_max = turning.r_max
        
        # Integrate dφ = L/(mr²) * dr/ṙ from r_min to r_max
        # ṙ² = (2/m)*(E - V_eff)
        def integrand(r):
            V_eff = eff.evaluate(r)
            kinetic = energy - V_eff
            if kinetic <= 0:
                return 0.0
            r_dot = np.sqrt(2 * kinetic / mass)
            return angular_momentum / (mass * r**2 * r_dot)
        
        from scipy.integrate import quad
        
        # Integrate with care near turning points
        try:
            delta_phi, _ = quad(integrand, r_min * 1.001, r_max * 0.999, limit=100)
            # Full orbit is 2 * delta_phi
            total_phi = 2 * delta_phi
            # Precession is deviation from 2π
            precession = total_phi - 2 * np.pi
            return precession
        except Exception as e:
            logger.warning(f"Precession integration failed: {e}")
            return 0.0
    
    def compute_radial_period(self, potential: sp.Expr,
                               angular_momentum: float,
                               mass: float,
                               energy: float) -> float:
        """
        Compute the radial period (time for r to go from r_min to r_max and back).
        
        T_r = 2 * ∫[r_min to r_max] dr/ṙ
        
        Args:
            potential: Central potential
            angular_momentum: Angular momentum
            mass: Particle mass
            energy: Total energy
            
        Returns:
            Radial period
        """
        eff = EffectivePotential(potential, angular_momentum, mass)
        turning = eff.find_turning_points(energy)
        
        if not turning.is_bounded or turning.r_max is None:
            return float('inf')
        
        def integrand(r):
            V_eff = eff.evaluate(r)
            kinetic = energy - V_eff
            if kinetic <= 0:
                return 0.0
            return 1.0 / np.sqrt(2 * kinetic / mass)
        
        from scipy.integrate import quad
        
        try:
            half_period, _ = quad(integrand, turning.r_min * 1.001, 
                                   turning.r_max * 0.999, limit=100)
            return 2 * half_period
        except Exception as e:
            logger.warning(f"Period integration failed: {e}")
            return float('inf')


class KeplerProblem:
    """
    Specialized solver for the Kepler two-body problem.
    
    Potential: V(r) = -GMm/r
    
    Provides analytical solutions for:
    - Orbital elements
    - Position as function of time
    - Kepler's equation solver
    
    Example:
        >>> kepler = KeplerProblem(G=6.674e-11, M=1.989e30, m=5.972e24)
        >>> elements = kepler.compute_elements(E, L)
        >>> r, phi = kepler.position_at_time(t, elements)
    """
    
    def __init__(self, G: float, M: float, m: float):
        """
        Initialize Kepler problem.
        
        Args:
            G: Gravitational constant
            M: Central mass
            m: Orbiting mass
        """
        self.G = G
        self.M = M
        self.m = m
        self.mu = G * M  # Gravitational parameter
        
        self._analyzer = CentralForceAnalyzer()
    
    def compute_elements(self, energy: float, 
                         angular_momentum: float) -> OrbitalElements:
        """Compute orbital elements from energy and angular momentum."""
        return self._analyzer.compute_orbital_elements_kepler(
            self.G, self.M, self.m, energy, angular_momentum
        )
    
    def solve_kepler_equation(self, mean_anomaly: float, 
                               eccentricity: float,
                               tol: float = 1e-10) -> float:
        """
        Solve Kepler's equation: M = E - e*sin(E)
        
        Args:
            mean_anomaly: Mean anomaly M
            eccentricity: Orbital eccentricity e
            tol: Convergence tolerance
            
        Returns:
            Eccentric anomaly E
        """
        # Newton-Raphson iteration
        E = mean_anomaly  # Initial guess
        
        for _ in range(100):
            f = E - eccentricity * np.sin(E) - mean_anomaly
            f_prime = 1 - eccentricity * np.cos(E)
            delta = f / f_prime
            E = E - delta
            if abs(delta) < tol:
                break
        
        return E
    
    def position_at_time(self, t: float, elements: OrbitalElements,
                          t0: float = 0.0) -> Tuple[float, float]:
        """
        Compute position (r, φ) at time t.
        
        Args:
            t: Time
            elements: Orbital elements
            t0: Initial time (perihelion passage)
            
        Returns:
            Tuple of (radius, true_anomaly)
        """
        if elements.orbit_type not in [OrbitType.CIRCULAR, OrbitType.ELLIPTICAL]:
            raise ValueError("Position calculation only for bound orbits")
        
        a = elements.semi_major_axis
        e = elements.eccentricity
        T = elements.period
        
        # Mean anomaly
        M = 2 * np.pi * (t - t0) / T
        M = M % (2 * np.pi)
        
        # Eccentric anomaly from Kepler's equation
        E = self.solve_kepler_equation(M, e)
        
        # True anomaly
        phi = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2),
                             np.sqrt(1 - e) * np.cos(E/2))
        
        # Radius
        r = a * (1 - e * np.cos(E))
        
        return r, phi
    
    def velocity_at_position(self, r: float, phi: float,
                              elements: OrbitalElements) -> Tuple[float, float]:
        """
        Compute velocity components at position.
        
        Args:
            r: Radial distance
            phi: True anomaly
            elements: Orbital elements
            
        Returns:
            Tuple of (radial_velocity, tangential_velocity)
        """
        a = elements.semi_major_axis
        e = elements.eccentricity
        L = elements.angular_momentum
        
        # Radial velocity from vis-viva
        v_squared = self.mu * (2/r - 1/a)
        v_phi = L / (self.m * r)  # Tangential
        v_r = np.sqrt(max(0, v_squared - v_phi**2))  # Radial
        
        # Sign of radial velocity depends on whether moving inward or outward
        # This is simplified; full determination needs orbit phase
        
        return v_r, v_phi
