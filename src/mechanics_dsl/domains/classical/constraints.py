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
