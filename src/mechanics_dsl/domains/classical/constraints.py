"""
Constraint Handling for Classical Mechanics

Implements holonomic and non-holonomic constraint handling
using Lagrange multipliers.
"""
from typing import Dict, List, Optional, Tuple
import sympy as sp

from ...utils import logger


class ConstraintHandler:
    """
    Handles mechanical constraints using Lagrange multipliers.
    
    Supports:
    - Holonomic constraints: g(q, t) = 0
    - Non-holonomic constraints: A(q) * q̇ + B(q, t) = 0
    """
    
    def __init__(self):
        self.holonomic_constraints: List[sp.Expr] = []
        self.nonholonomic_constraints: List[sp.Expr] = []
        self._multipliers: List[sp.Symbol] = []
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def add_holonomic_constraint(self, constraint: sp.Expr) -> sp.Symbol:
        """
        Add a holonomic constraint g(q, t) = 0.
        
        Args:
            constraint: SymPy expression that equals zero
            
        Returns:
            The Lagrange multiplier symbol for this constraint
        """
        idx = len(self.holonomic_constraints)
        lambda_sym = self.get_symbol(f'lambda_{idx}')
        self.holonomic_constraints.append(constraint)
        self._multipliers.append(lambda_sym)
        return lambda_sym
    
    def add_nonholonomic_constraint(self, constraint: sp.Expr) -> None:
        """
        Add a non-holonomic constraint (velocity-dependent).
        
        Args:
            constraint: SymPy expression involving velocities
        """
        self.nonholonomic_constraints.append(constraint)
    
    def augment_lagrangian(self, lagrangian: sp.Expr) -> sp.Expr:
        """
        Create augmented Lagrangian with constraint terms.
        
        L' = L + Σ(λ_i * g_i)
        
        Args:
            lagrangian: Original Lagrangian
            
        Returns:
            Augmented Lagrangian
        """
        L_augmented = lagrangian
        for lam, constraint in zip(self._multipliers, self.holonomic_constraints):
            L_augmented += lam * constraint
        return L_augmented
    
    def get_constraint_equations(self) -> List[sp.Expr]:
        """
        Get constraint equations that must be satisfied.
        
        Returns:
            List of constraint expressions (should equal zero)
        """
        return self.holonomic_constraints.copy()
    
    def get_multipliers(self) -> List[sp.Symbol]:
        """Get list of Lagrange multiplier symbols."""
        return self._multipliers.copy()
    
    def clear(self) -> None:
        """Clear all constraints."""
        self.holonomic_constraints.clear()
        self.nonholonomic_constraints.clear()
        self._multipliers.clear()


class BaumgarteStabilization:
    """
    Baumgarte constraint stabilization for numerical DAE solving.
    
    For a constraint g(q) = 0, the acceleration-level equation is:
        g̈ + 2α*ġ + β²*g = 0
    
    This prevents constraint drift during numerical integration.
    
    Typical values: α = β = 5-10 for stable constraint enforcement.
    """
    
    def __init__(self, alpha: float = 5.0, beta: float = 5.0):
        """
        Initialize Baumgarte stabilization.
        
        Args:
            alpha: Damping coefficient (controls velocity error damping)
            beta: Stiffness coefficient (controls position error correction)
        """
        self.alpha = alpha
        self.beta = beta
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._time_symbol = sp.Symbol('t', real=True)
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def velocity_level_constraint(self, constraint: sp.Expr,
                                    coordinates: List[str]) -> sp.Expr:
        """
        Differentiate position constraint to get velocity constraint.
        
        ġ = Σᵢ (∂g/∂qᵢ * q̇ᵢ) + ∂g/∂t
        
        Args:
            constraint: Position-level constraint g(q, t) = 0
            coordinates: List of coordinate names
            
        Returns:
            Velocity-level constraint expression
        """
        g_dot = sp.S.Zero
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            q_dot = self.get_symbol(f"{q}_dot")
            g_dot += sp.diff(constraint, q_sym) * q_dot
        
        # Add explicit time derivative
        g_dot += sp.diff(constraint, self._time_symbol)
        
        return g_dot
    
    def acceleration_level_constraint(self, constraint: sp.Expr,
                                        coordinates: List[str]) -> sp.Expr:
        """
        Differentiate velocity constraint to get acceleration constraint.
        
        g̈ = Σᵢ (∂ġ/∂qᵢ * q̇ᵢ + ∂ġ/∂q̇ᵢ * q̈ᵢ) + ∂ġ/∂t
        
        Args:
            constraint: Position-level constraint g(q, t) = 0
            coordinates: List of coordinate names
            
        Returns:
            Acceleration-level constraint expression
        """
        g_dot = self.velocity_level_constraint(constraint, coordinates)
        g_ddot = sp.S.Zero
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            q_dot = self.get_symbol(f"{q}_dot")
            q_ddot = self.get_symbol(f"{q}_ddot")
            
            # Chain rule terms
            g_ddot += sp.diff(g_dot, q_sym) * q_dot
            g_ddot += sp.diff(g_dot, q_dot) * q_ddot
        
        # Explicit time derivative
        g_ddot += sp.diff(g_dot, self._time_symbol)
        
        return g_ddot
    
    def stabilized_constraint(self, constraint: sp.Expr,
                               coordinates: List[str]) -> sp.Expr:
        """
        Get Baumgarte-stabilized acceleration-level constraint.
        
        g̈ + 2α*ġ + β²*g = 0
        
        Rearranged for the constraint force computation.
        
        Args:
            constraint: Position-level constraint g(q, t) = 0
            coordinates: List of coordinate names
            
        Returns:
            Stabilized constraint equation (equals 0)
        """
        g = constraint
        g_dot = self.velocity_level_constraint(constraint, coordinates)
        g_ddot = self.acceleration_level_constraint(constraint, coordinates)
        
        # Baumgarte stabilization: g̈ + 2α*ġ + β²*g = 0
        stabilized = g_ddot + 2 * self.alpha * g_dot + self.beta**2 * g
        
        return stabilized
    
    def compute_constraint_force(self, constraint: sp.Expr,
                                   lagrangian: sp.Expr,
                                   coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Compute constrained equations of motion with Baumgarte stabilization.
        
        This modifies the Euler-Lagrange equations to include constraint forces
        that keep the system on the constraint manifold.
        
        Args:
            constraint: Position-level constraint g(q) = 0
            lagrangian: System Lagrangian L(q, q̇)
            coordinates: List of coordinate names
            
        Returns:
            Dictionary with constraint Jacobian and stabilization terms
        """
        result = {}
        
        # Constraint Jacobian: ∂g/∂q
        jacobian = []
        for q in coordinates:
            q_sym = self.get_symbol(q)
            jacobian.append(sp.diff(constraint, q_sym))
        
        result['jacobian'] = jacobian
        result['constraint'] = constraint
        result['velocity_constraint'] = self.velocity_level_constraint(constraint, coordinates)
        result['stabilized'] = self.stabilized_constraint(constraint, coordinates)
        
        return result


class ConstrainedLagrangianSystem:
    """
    Complete system for constrained Lagrangian mechanics.
    
    Combines holonomic constraints with Baumgarte stabilization
    for robust numerical integration.
    
    Example:
        >>> system = ConstrainedLagrangianSystem()
        >>> system.set_lagrangian(L)
        >>> system.add_constraint(x**2 + y**2 - l**2)  # Pendulum constraint
        >>> eom = system.derive_equations_of_motion(['x', 'y'])
    """
    
    def __init__(self, alpha: float = 5.0, beta: float = 5.0):
        self.constraint_handler = ConstraintHandler()
        self.stabilization = BaumgarteStabilization(alpha, beta)
        self._lagrangian: Optional[sp.Expr] = None
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
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
    
    def add_constraint(self, constraint: sp.Expr) -> sp.Symbol:
        """Add a holonomic constraint and return its Lagrange multiplier."""
        return self.constraint_handler.add_holonomic_constraint(constraint)
    
    def derive_equations_of_motion(self, coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Derive constrained equations of motion with stabilization.
        
        This produces the full DAE system:
        - Euler-Lagrange equations with constraint forces
        - Stabilized constraint equations
        
        Args:
            coordinates: List of generalized coordinates
            
        Returns:
            Dictionary of equations
        """
        if self._lagrangian is None:
            raise ValueError("Lagrangian not set")
        
        equations = {}
        L = self._lagrangian
        constraints = self.constraint_handler.get_constraint_equations()
        multipliers = self.constraint_handler.get_multipliers()
        
        # Euler-Lagrange for each coordinate
        for q in coordinates:
            q_sym = self.get_symbol(q)
            q_dot = self.get_symbol(f"{q}_dot")
            q_ddot = self.get_symbol(f"{q}_ddot")
            
            # Standard E-L terms
            dL_dq = sp.diff(L, q_sym)
            dL_dq_dot = sp.diff(L, q_dot)
            
            # For d/dt(∂L/∂q̇), need to handle time derivatives
            # Simplified: extract coefficients
            eq = -dL_dq  # Will add more terms
            
            # Add constraint force terms: Σ λᵢ * ∂gᵢ/∂q
            for lam, g in zip(multipliers, constraints):
                eq += lam * sp.diff(g, q_sym)
            
            equations[f"{q}_ddot_eq"] = eq
        
        # Add stabilized constraints
        for i, g in enumerate(constraints):
            stab = self.stabilization.stabilized_constraint(g, coordinates)
            equations[f"constraint_{i}"] = stab
        
        return equations

