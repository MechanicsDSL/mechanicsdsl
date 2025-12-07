"""
Collision Dynamics for Classical Mechanics

This module implements:
- Elastic and inelastic collisions
- Center of mass frame transformations
- Coefficient of restitution
- Impulse calculations
- Multi-body collision handling

For two-body collisions:
- Elastic: kinetic energy conserved
- Inelastic: kinetic energy lost, momentum conserved
- Perfectly inelastic: bodies stick together
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np

from ...utils import logger


class CollisionType(Enum):
    """Types of collisions."""
    ELASTIC = "elastic"                   # e = 1
    INELASTIC = "inelastic"              # 0 < e < 1
    PERFECTLY_INELASTIC = "perfectly_inelastic"  # e = 0
    SUPERELASTIC = "superelastic"        # e > 1 (explosive)


@dataclass
class CollisionResult:
    """
    Result of a collision calculation.
    
    Attributes:
        v1_final: Final velocity of body 1
        v2_final: Final velocity of body 2
        impulse: Impulse delivered during collision
        energy_loss: Kinetic energy lost
        collision_type: Type of collision
    """
    v1_final: np.ndarray
    v2_final: np.ndarray
    impulse: np.ndarray
    energy_loss: float
    collision_type: CollisionType
    
    @property
    def is_elastic(self) -> bool:
        """Check if collision was elastic."""
        return abs(self.energy_loss) < 1e-10


@dataclass
class Particle:
    """
    Represents a particle for collision calculations.
    
    Attributes:
        mass: Particle mass
        position: Position vector
        velocity: Velocity vector
    """
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    
    @property
    def momentum(self) -> np.ndarray:
        """Get momentum p = mv."""
        return self.mass * self.velocity
    
    @property
    def kinetic_energy(self) -> float:
        """Get kinetic energy T = mv²/2."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)


class CollisionSolver:
    """
    Solver for two-body collision problems.
    
    Supports:
    - 1D, 2D, and 3D collisions
    - Variable coefficient of restitution
    - Center of mass frame analysis
    - Impulse calculations
    
    Example:
        >>> solver = CollisionSolver()
        >>> p1 = Particle(mass=1.0, position=np.zeros(3), velocity=np.array([1, 0, 0]))
        >>> p2 = Particle(mass=2.0, position=np.array([1, 0, 0]), velocity=np.zeros(3))
        >>> result = solver.solve(p1, p2, e=1.0)  # Elastic collision
    """
    
    def solve(self, particle1: Particle, particle2: Particle,
              e: float = 1.0,
              normal: Optional[np.ndarray] = None) -> CollisionResult:
        """
        Solve collision between two particles.
        
        Uses conservation of momentum and coefficient of restitution:
        - p₁ + p₂ = p₁' + p₂' (momentum conservation)
        - e = -(v₁' - v₂')·n / (v₁ - v₂)·n (restitution)
        
        Args:
            particle1: First particle
            particle2: Second particle
            e: Coefficient of restitution (0 to 1 for normal collisions)
            normal: Collision normal direction (auto-computed if None)
            
        Returns:
            CollisionResult with final velocities
        """
        m1, m2 = particle1.mass, particle2.mass
        v1, v2 = particle1.velocity.copy(), particle2.velocity.copy()
        
        # Compute collision normal if not provided
        if normal is None:
            delta_r = particle2.position - particle1.position
            norm = np.linalg.norm(delta_r)
            if norm > 1e-10:
                normal = delta_r / norm
            else:
                # Default to x-direction for coincident particles
                normal = np.array([1.0, 0.0, 0.0])[:len(v1)]
        
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Relative velocity along normal
        v_rel = v1 - v2
        v_rel_n = np.dot(v_rel, normal)
        
        # If particles are separating, no collision
        if v_rel_n >= 0:
            return CollisionResult(
                v1_final=v1,
                v2_final=v2,
                impulse=np.zeros_like(v1),
                energy_loss=0.0,
                collision_type=CollisionType.ELASTIC
            )
        
        # Impulse magnitude (from momentum conservation + restitution)
        # J = -(1 + e) * v_rel_n / (1/m1 + 1/m2)
        J_mag = -(1 + e) * v_rel_n / (1/m1 + 1/m2)
        J = J_mag * normal
        
        # Apply impulse
        v1_final = v1 + J / m1
        v2_final = v2 - J / m2
        
        # Initial and final kinetic energies
        KE_initial = particle1.kinetic_energy + particle2.kinetic_energy
        KE_final = 0.5 * m1 * np.dot(v1_final, v1_final) + \
                   0.5 * m2 * np.dot(v2_final, v2_final)
        energy_loss = KE_initial - KE_final
        
        # Determine collision type
        if abs(e - 1.0) < 1e-10:
            collision_type = CollisionType.ELASTIC
        elif abs(e) < 1e-10:
            collision_type = CollisionType.PERFECTLY_INELASTIC
        elif e > 1.0:
            collision_type = CollisionType.SUPERELASTIC
        else:
            collision_type = CollisionType.INELASTIC
        
        return CollisionResult(
            v1_final=v1_final,
            v2_final=v2_final,
            impulse=J,
            energy_loss=energy_loss,
            collision_type=collision_type
        )
    
    def solve_1d(self, m1: float, v1: float, 
                 m2: float, v2: float,
                 e: float = 1.0) -> Tuple[float, float]:
        """
        Solve 1D collision (simplified interface).
        
        Args:
            m1, v1: Mass and velocity of particle 1
            m2, v2: Mass and velocity of particle 2
            e: Coefficient of restitution
            
        Returns:
            Tuple of (v1_final, v2_final)
        """
        # Closed-form solution for 1D
        v1_final = ((m1 - e*m2)*v1 + (1 + e)*m2*v2) / (m1 + m2)
        v2_final = ((m2 - e*m1)*v2 + (1 + e)*m1*v1) / (m1 + m2)
        
        return v1_final, v2_final
    
    def center_of_mass_frame(self, particle1: Particle, 
                              particle2: Particle) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform to center of mass reference frame.
        
        v_cm = (m1*v1 + m2*v2) / (m1 + m2)
        
        Args:
            particle1, particle2: The two particles
            
        Returns:
            Tuple of velocities in CM frame (v1_cm, v2_cm)
        """
        m1, m2 = particle1.mass, particle2.mass
        v1, v2 = particle1.velocity, particle2.velocity
        
        v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)
        
        v1_cm = v1 - v_cm
        v2_cm = v2 - v_cm
        
        return v1_cm, v2_cm
    
    def reduced_mass(self, m1: float, m2: float) -> float:
        """
        Compute reduced mass μ = m1*m2/(m1 + m2).
        
        Args:
            m1, m2: Particle masses
            
        Returns:
            Reduced mass
        """
        return m1 * m2 / (m1 + m2)


class SymbolicCollisionSolver:
    """
    Symbolic collision solver using SymPy.
    
    For deriving collision formulas with arbitrary parameters.
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
    
    def solve_1d_elastic(self) -> Dict[str, sp.Expr]:
        """
        Derive 1D elastic collision formulas symbolically.
        
        Returns:
            Dictionary with v1_final and v2_final expressions
        """
        m1 = self.get_symbol('m_1', positive=True)
        m2 = self.get_symbol('m_2', positive=True)
        v1 = self.get_symbol('v_1')
        v2 = self.get_symbol('v_2')
        
        # Conservation of momentum: m1*v1 + m2*v2 = m1*v1' + m2*v2'
        # Conservation of energy: m1*v1² + m2*v2² = m1*v1'² + m2*v2'²
        
        v1_final = ((m1 - m2)*v1 + 2*m2*v2) / (m1 + m2)
        v2_final = ((m2 - m1)*v2 + 2*m1*v1) / (m1 + m2)
        
        return {
            'v1_final': sp.simplify(v1_final),
            'v2_final': sp.simplify(v2_final)
        }
    
    def solve_1d_inelastic(self, e: Optional[sp.Symbol] = None) -> Dict[str, sp.Expr]:
        """
        Derive 1D inelastic collision formulas with restitution e.
        
        Args:
            e: Coefficient of restitution symbol (created if None)
            
        Returns:
            Dictionary with final velocity expressions
        """
        m1 = self.get_symbol('m_1', positive=True)
        m2 = self.get_symbol('m_2', positive=True)
        v1 = self.get_symbol('v_1')
        v2 = self.get_symbol('v_2')
        
        if e is None:
            e = self.get_symbol('e', positive=True)
        
        v1_final = ((m1 - e*m2)*v1 + (1 + e)*m2*v2) / (m1 + m2)
        v2_final = ((m2 - e*m1)*v2 + (1 + e)*m1*v1) / (m1 + m2)
        
        return {
            'v1_final': sp.simplify(v1_final),
            'v2_final': sp.simplify(v2_final),
            'restitution': e
        }
    
    def energy_loss(self) -> sp.Expr:
        """
        Derive expression for energy loss in inelastic collision.
        
        ΔKE = (1/2) * μ * (1 - e²) * v_rel²
        
        Returns:
            Symbolic energy loss expression
        """
        m1 = self.get_symbol('m_1', positive=True)
        m2 = self.get_symbol('m_2', positive=True)
        v1 = self.get_symbol('v_1')
        v2 = self.get_symbol('v_2')
        e = self.get_symbol('e', positive=True)
        
        mu = m1 * m2 / (m1 + m2)  # Reduced mass
        v_rel = v1 - v2
        
        delta_KE = sp.Rational(1, 2) * mu * (1 - e**2) * v_rel**2
        
        return sp.simplify(delta_KE)


class ImpulseCalculator:
    """
    Calculate impulses for collision and impact problems.
    """
    
    @staticmethod
    def impulse_momentum(mass: float, delta_v: np.ndarray) -> np.ndarray:
        """
        Calculate impulse from change in velocity.
        
        J = m * Δv
        
        Args:
            mass: Particle mass
            delta_v: Change in velocity
            
        Returns:
            Impulse vector
        """
        return mass * delta_v
    
    @staticmethod
    def impulse_from_force(force: np.ndarray, duration: float) -> np.ndarray:
        """
        Calculate impulse from average force.
        
        J = F_avg * Δt
        
        Args:
            force: Average force during collision
            duration: Collision duration
            
        Returns:
            Impulse vector
        """
        return force * duration
    
    @staticmethod
    def angular_impulse(torque: np.ndarray, duration: float) -> np.ndarray:
        """
        Calculate angular impulse.
        
        H = τ_avg * Δt
        
        Args:
            torque: Average torque during collision
            duration: Collision duration
            
        Returns:
            Angular impulse vector
        """
        return torque * duration


# Convenience functions

def elastic_collision_1d(m1: float, v1: float, 
                         m2: float, v2: float) -> Tuple[float, float]:
    """
    Compute final velocities for 1D elastic collision.
    
    Args:
        m1, v1: Mass and velocity of particle 1
        m2, v2: Mass and velocity of particle 2
        
    Returns:
        Tuple of (v1_final, v2_final)
    """
    solver = CollisionSolver()
    return solver.solve_1d(m1, v1, m2, v2, e=1.0)


def inelastic_collision_1d(m1: float, v1: float,
                           m2: float, v2: float,
                           e: float) -> Tuple[float, float]:
    """
    Compute final velocities for 1D inelastic collision.
    
    Args:
        m1, v1: Mass and velocity of particle 1
        m2, v2: Mass and velocity of particle 2
        e: Coefficient of restitution
        
    Returns:
        Tuple of (v1_final, v2_final)
    """
    solver = CollisionSolver()
    return solver.solve_1d(m1, v1, m2, v2, e=e)


def perfectly_inelastic_1d(m1: float, v1: float,
                            m2: float, v2: float) -> float:
    """
    Compute final velocity for perfectly inelastic collision.
    
    Bodies stick together: v_final = (m1*v1 + m2*v2)/(m1 + m2)
    
    Args:
        m1, v1: Mass and velocity of particle 1
        m2, v2: Mass and velocity of particle 2
        
    Returns:
        Final combined velocity
    """
    return (m1 * v1 + m2 * v2) / (m1 + m2)
