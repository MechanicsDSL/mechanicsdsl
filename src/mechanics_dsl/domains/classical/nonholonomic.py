"""
Non-Holonomic Constraints for Classical Mechanics

This module implements:
- Velocity-dependent constraints (Pfaffian constraints)
- d'Alembert-Lagrange equations
- Appell's equations
- Rolling without slipping
- Chaplygin sleigh dynamics

Non-holonomic constraints have the form:
    Σᵢ aᵢⱼ(q)q̇ⱼ + bᵢ(q) = 0

These cannot be integrated to position-only constraints.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np

from ...utils import logger


class ConstraintType(Enum):
    """Types of mechanical constraints."""
    HOLONOMIC = "holonomic"           # f(q, t) = 0
    NONHOLONOMIC = "nonholonomic"     # Σ aᵢ(q)q̇ᵢ + b(q,t) = 0
    SEMI_HOLONOMIC = "semi_holonomic" # Integrable non-holonomic
    RHEONOMIC = "rheonomic"           # Explicitly time-dependent
    SCLERONOMIC = "scleronomic"       # Time-independent


@dataclass
class NonholonomicConstraint:
    """
    Represents a non-holonomic (velocity) constraint.
    
    Form: Σᵢ aᵢ(q)q̇ᵢ + b(q,t) = 0
    
    Attributes:
        coefficients: Dictionary mapping coordinate to coefficient aᵢ(q)
        inhomogeneous: The b(q,t) term
        name: Optional constraint name
    """
    coefficients: Dict[str, sp.Expr]
    inhomogeneous: sp.Expr = sp.S.Zero
    name: str = "constraint"
    
    def evaluate(self, state: Dict[str, float]) -> float:
        """Evaluate constraint violation at given state."""
        result = float(self.inhomogeneous.subs(
            {sp.Symbol(k): v for k, v in state.items()}
        ).evalf())
        
        for coord, coeff in self.coefficients.items():
            vel_name = f"{coord}_dot"
            if vel_name in state:
                coeff_val = float(coeff.subs(
                    {sp.Symbol(k): v for k, v in state.items()}
                ).evalf())
                result += coeff_val * state[vel_name]
        
        return result
    
    def as_matrix_form(self, coordinates: List[str]) -> Tuple[sp.Matrix, sp.Expr]:
        """
        Convert to matrix form A·q̇ + b = 0.
        
        Returns:
            Tuple of (A row vector, b scalar)
        """
        A = []
        for coord in coordinates:
            if coord in self.coefficients:
                A.append(self.coefficients[coord])
            else:
                A.append(sp.S.Zero)
        
        return sp.Matrix([A]), self.inhomogeneous


class NonholonomicSystem:
    """
    System with non-holonomic constraints.
    
    Uses the d'Alembert-Lagrange principle:
    d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = Σⱼ λⱼ aⱼᵢ
    
    where λⱼ are Lagrange multipliers and aⱼᵢ are constraint coefficients.
    
    Example:
        >>> system = NonholonomicSystem()
        >>> system.set_lagrangian(L)
        >>> system.add_rolling_constraint('x', 'theta', radius=R)
        >>> eom = system.derive_equations_of_motion(['x', 'theta'])
    """
    
    def __init__(self):
        self._lagrangian: Optional[sp.Expr] = None
        self._constraints: List[NonholonomicConstraint] = []
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._multipliers: List[sp.Symbol] = []
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def set_lagrangian(self, lagrangian: sp.Expr) -> None:
        """Set the system Lagrangian."""
        self._lagrangian = lagrangian
    
    def add_constraint(self, constraint: NonholonomicConstraint) -> sp.Symbol:
        """
        Add a non-holonomic constraint.
        
        Args:
            constraint: NonholonomicConstraint object
            
        Returns:
            Lagrange multiplier symbol
        """
        self._constraints.append(constraint)
        
        # Create Lagrange multiplier
        idx = len(self._multipliers)
        lam = self.get_symbol(f'lambda_nh_{idx}')
        self._multipliers.append(lam)
        
        logger.info(f"Added non-holonomic constraint: {constraint.name}")
        return lam
    
    def add_rolling_constraint(self, linear_coord: str, angular_coord: str,
                                radius: sp.Expr,
                                direction: str = 'forward') -> sp.Symbol:
        """
        Add rolling without slipping constraint.
        
        v = R·ω (or v = -R·ω for reverse)
        
        In differential form: dx - R·dθ = 0
        In velocity form: ẋ - R·θ̇ = 0
        
        Args:
            linear_coord: Linear position coordinate
            angular_coord: Angular coordinate
            radius: Rolling radius
            direction: 'forward' or 'reverse'
            
        Returns:
            Lagrange multiplier
        """
        sign = 1 if direction == 'forward' else -1
        
        constraint = NonholonomicConstraint(
            coefficients={
                linear_coord: sp.S.One,
                angular_coord: -sign * radius
            },
            name=f"rolling_{linear_coord}_{angular_coord}"
        )
        
        return self.add_constraint(constraint)
    
    def add_knife_edge_constraint(self, x_coord: str, y_coord: str,
                                   theta_coord: str) -> sp.Symbol:
        """
        Add knife-edge (Chaplygin sleigh) constraint.
        
        The velocity must be along the body direction:
        ẏ·cos(θ) - ẋ·sin(θ) = 0
        
        Args:
            x_coord: x position coordinate
            y_coord: y position coordinate
            theta_coord: orientation angle
            
        Returns:
            Lagrange multiplier
        """
        theta = self.get_symbol(theta_coord)
        
        constraint = NonholonomicConstraint(
            coefficients={
                x_coord: -sp.sin(theta),
                y_coord: sp.cos(theta)
            },
            name="knife_edge"
        )
        
        return self.add_constraint(constraint)
    
    def derive_equations_of_motion(self, 
                                    coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Derive equations of motion with non-holonomic constraints.
        
        Uses d'Alembert-Lagrange equations:
        d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = Σⱼ λⱼ aⱼᵢ
        
        Plus constraint equations.
        
        Args:
            coordinates: List of generalized coordinates
            
        Returns:
            Dictionary of equations
        """
        if self._lagrangian is None:
            raise ValueError("Lagrangian not set")
        
        equations = {}
        L = self._lagrangian
        
        # Build constraint force matrix
        # Each constraint j contributes λⱼ·aⱼᵢ to equation i
        
        for i, q in enumerate(coordinates):
            q_sym = self.get_symbol(q)
            q_dot = self.get_symbol(f"{q}_dot")
            
            # Standard Euler-Lagrange terms
            dL_dq = sp.diff(L, q_sym)
            dL_dq_dot = sp.diff(L, q_dot)
            
            # Constraint forces
            constraint_force = sp.S.Zero
            for j, (constraint, lam) in enumerate(zip(self._constraints, 
                                                       self._multipliers)):
                if q in constraint.coefficients:
                    constraint_force += lam * constraint.coefficients[q]
            
            equations[f"{q}_eq"] = sp.Eq(
                sp.Symbol(f"EL_{q}"),  # Placeholder for d/dt(∂L/∂q̇) - ∂L/∂q
                constraint_force
            )
        
        # Add constraint equations
        for j, constraint in enumerate(self._constraints):
            expr = constraint.inhomogeneous
            for coord, coeff in constraint.coefficients.items():
                q_dot = self.get_symbol(f"{coord}_dot")
                expr += coeff * q_dot
            equations[f"constraint_{j}"] = sp.Eq(expr, 0)
        
        return equations
    
    def get_constraint_matrix(self, 
                               coordinates: List[str]) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Get constraint coefficient matrix A and vector b.
        
        Constraints: A·q̇ + b = 0
        
        Args:
            coordinates: List of coordinates
            
        Returns:
            Tuple of (A matrix, b vector)
        """
        n_constraints = len(self._constraints)
        n_coords = len(coordinates)
        
        A = sp.zeros(n_constraints, n_coords)
        b = sp.zeros(n_constraints, 1)
        
        for j, constraint in enumerate(self._constraints):
            for i, coord in enumerate(coordinates):
                if coord in constraint.coefficients:
                    A[j, i] = constraint.coefficients[coord]
            b[j] = constraint.inhomogeneous
        
        return A, b


class AppellEquations:
    """
    Appell's equations for non-holonomic systems.
    
    Uses the Gibbs-Appell function (acceleration energy):
    S = (1/2) Σᵢ mᵢ aᵢ²
    
    Appell's equations: ∂S/∂q̈ⱼ = Qⱼ
    
    Especially useful for non-holonomic constraints.
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
    
    def compute_acceleration_energy(self, 
                                      kinetic_energy: sp.Expr,
                                      coordinates: List[str]) -> sp.Expr:
        """
        Compute Gibbs-Appell function from kinetic energy.
        
        S = (1/2) Σᵢⱼ Mᵢⱼ q̈ᵢ q̈ⱼ + terms linear in q̈
        
        Args:
            kinetic_energy: Kinetic energy T
            coordinates: List of coordinates
            
        Returns:
            Gibbs-Appell function S
        """
        # T = (1/2) Σᵢⱼ Mᵢⱼ(q) q̇ᵢ q̇ⱼ
        # S is obtained by replacing q̇ with q̈ in T
        
        S = sp.S.Zero
        
        for qi in coordinates:
            qi_dot = self.get_symbol(f"{qi}_dot")
            qi_ddot = self.get_symbol(f"{qi}_ddot")
            
            for qj in coordinates:
                qj_dot = self.get_symbol(f"{qj}_dot")
                qj_ddot = self.get_symbol(f"{qj}_ddot")
                
                # Extract Mᵢⱼ coefficient
                Mij = sp.diff(sp.diff(kinetic_energy, qi_dot), qj_dot)
                
                S += sp.Rational(1, 2) * Mij * qi_ddot * qj_ddot
        
        return sp.simplify(S)
    
    def derive_appell_equations(self, acceleration_energy: sp.Expr,
                                 generalized_forces: Dict[str, sp.Expr],
                                 pseudo_coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Derive Appell's equations of motion.
        
        ∂S/∂q̈ⱼ = Qⱼ
        
        Args:
            acceleration_energy: Gibbs-Appell function S
            generalized_forces: Dictionary of generalized forces
            pseudo_coordinates: Quasi-velocities
            
        Returns:
            Dictionary of equations
        """
        equations = {}
        
        for coord in pseudo_coordinates:
            q_ddot = self.get_symbol(f"{coord}_ddot")
            
            dS_dq_ddot = sp.diff(acceleration_energy, q_ddot)
            Q = generalized_forces.get(coord, sp.S.Zero)
            
            equations[coord] = sp.Eq(dS_dq_ddot, Q)
        
        return equations


class MaggiEquations:
    """
    Maggi's equations for non-holonomic systems.
    
    An alternative to Lagrange multipliers that projects
    equations onto the constraint manifold.
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
    
    def project_equations(self, euler_lagrange: List[sp.Expr],
                          constraint_matrix: sp.Matrix,
                          coordinates: List[str]) -> List[sp.Expr]:
        """
        Project Euler-Lagrange equations onto constraint manifold.
        
        Uses null space of constraint matrix to eliminate multipliers.
        
        Args:
            euler_lagrange: EL equations (one per coordinate)
            constraint_matrix: A matrix from A·q̇ = 0
            coordinates: List of coordinates
            
        Returns:
            Projected equations (fewer than original)
        """
        # Find null space of A
        null_space = constraint_matrix.nullspace()
        
        projected = []
        for null_vec in null_space:
            # Project: nᵀ · (EL equations) = 0
            proj_eq = sp.S.Zero
            for i, eq in enumerate(euler_lagrange):
                proj_eq += null_vec[i] * eq
            projected.append(sp.simplify(proj_eq))
        
        return projected


# Convenience functions

def rolling_constraint(linear: str, angular: str, 
                       radius: sp.Expr) -> NonholonomicConstraint:
    """Create a rolling without slipping constraint."""
    return NonholonomicConstraint(
        coefficients={linear: sp.S.One, angular: -radius},
        name=f"rolling_{linear}"
    )


def knife_edge_constraint(x: str, y: str, theta: str) -> NonholonomicConstraint:
    """Create a knife-edge constraint."""
    theta_sym = sp.Symbol(theta, real=True)
    return NonholonomicConstraint(
        coefficients={x: -sp.sin(theta_sym), y: sp.cos(theta_sym)},
        name="knife_edge"
    )
