"""
Lagrangian Mechanics Implementation

Provides tools for deriving equations of motion from Lagrangians using
the Euler-Lagrange equations.
"""
from typing import Dict, List, Optional
import sympy as sp

from ..base import PhysicsDomain
from ...utils import logger


class LagrangianMechanics(PhysicsDomain):
    """
    Lagrangian mechanics domain for classical mechanical systems.
    
    Uses L = T - V (kinetic - potential energy) to derive equations of motion
    via the Euler-Lagrange equations.
    
    Example:
        >>> domain = LagrangianMechanics("pendulum")
        >>> domain.add_coordinate("theta")
        >>> domain.set_parameters({"m": 1.0, "l": 1.0, "g": 9.81})
    """
    
    def __init__(self, name: str = "lagrangian_system"):
        super().__init__(name)
        self._lagrangian: Optional[sp.Expr] = None
        self._kinetic_energy: Optional[sp.Expr] = None
        self._potential_energy: Optional[sp.Expr] = None
        self._time_symbol = sp.Symbol('t', real=True)
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def set_kinetic_energy(self, expr: sp.Expr) -> None:
        """Set the kinetic energy expression T."""
        self._kinetic_energy = expr
        self._update_lagrangian()
    
    def set_potential_energy(self, expr: sp.Expr) -> None:
        """Set the potential energy expression V."""
        self._potential_energy = expr
        self._update_lagrangian()
    
    def set_lagrangian(self, expr: sp.Expr) -> None:
        """Set the Lagrangian directly."""
        self._lagrangian = expr
    
    def _update_lagrangian(self) -> None:
        """Update L = T - V when T or V changes."""
        if self._kinetic_energy is not None and self._potential_energy is not None:
            self._lagrangian = self._kinetic_energy - self._potential_energy
    
    def define_lagrangian(self) -> sp.Expr:
        """Return the Lagrangian expression."""
        if self._lagrangian is None:
            raise ValueError("Lagrangian not defined. Set T and V or use set_lagrangian()")
        return self._lagrangian
    
    def define_hamiltonian(self) -> sp.Expr:
        """
        Convert Lagrangian to Hamiltonian via Legendre transform.
        
        H = Σ(p_i * q̇_i) - L
        """
        if self._lagrangian is None:
            raise ValueError("Cannot derive Hamiltonian: Lagrangian not defined")
        
        L = self._lagrangian
        H = sp.S.Zero
        
        for q in self.coordinates:
            q_dot = self.get_symbol(f"{q}_dot")
            p = self.get_symbol(f"p_{q}")
            
            # Conjugate momentum: p = ∂L/∂q̇
            momentum = sp.diff(L, q_dot)
            
            # Try to solve for q̇ in terms of p
            try:
                q_dot_solution = sp.solve(momentum - p, q_dot)
                if q_dot_solution:
                    H += p * q_dot_solution[0]
            except (ValueError, NotImplementedError):
                logger.warning(f"Could not solve for {q}_dot, using symbolic form")
                H += p * q_dot
        
        # H = Σ(p * q̇) - L, then substitute q̇ expressions
        H = H - L
        
        return sp.simplify(H)
    
    def derive_equations_of_motion(self) -> Dict[str, sp.Expr]:
        """
        Derive equations of motion using Euler-Lagrange equations.
        
        d/dt(∂L/∂q̇) - ∂L/∂q = 0
        
        Returns:
            Dictionary mapping acceleration symbols to expressions
        """
        if self._lagrangian is None:
            raise ValueError("Lagrangian not defined")
        
        L = self._lagrangian
        equations = []
        
        for q in self.coordinates:
            q_sym = self.get_symbol(q)
            q_dot = self.get_symbol(f"{q}_dot")
            q_ddot = self.get_symbol(f"{q}_ddot")
            
            # Create time-dependent function for derivation
            q_func = sp.Function(q)(self._time_symbol)
            
            # Substitute into Lagrangian
            L_sub = L.subs(q_sym, q_func)
            L_sub = L_sub.subs(q_dot, sp.diff(q_func, self._time_symbol))
            
            # Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q = 0
            dL_dq_dot = sp.diff(L_sub, sp.diff(q_func, self._time_symbol))
            d_dt = sp.diff(dL_dq_dot, self._time_symbol)
            dL_dq = sp.diff(L_sub, q_func)
            
            eq = d_dt - dL_dq
            
            # Replace derivatives back with symbols
            eq = eq.subs(sp.diff(q_func, self._time_symbol, 2), q_ddot)
            eq = eq.subs(sp.diff(q_func, self._time_symbol), q_dot)
            eq = eq.subs(q_func, q_sym)
            
            equations.append((q, eq))
        
        # Solve for accelerations
        accelerations = {}
        for q, eq in equations:
            q_ddot = self.get_symbol(f"{q}_ddot")
            eq_expanded = sp.expand(eq)
            
            # Linear extraction: A * q̈ + B = 0 → q̈ = -B/A
            A = sp.diff(eq_expanded, q_ddot)
            B = eq_expanded.subs(q_ddot, 0)
            
            if A != 0:
                accelerations[f"{q}_ddot"] = sp.simplify(-B / A)
            else:
                logger.warning(f"Could not solve for {q}_ddot")
                accelerations[f"{q}_ddot"] = sp.S.Zero
        
        self.equations = accelerations
        self._is_compiled = True
        return accelerations
    
    def get_state_variables(self) -> List[str]:
        """Get state variables for ODE system."""
        state = []
        for q in self.coordinates:
            state.extend([q, f"{q}_dot"])
        return state
    
    def get_required_parameters(self) -> List[str]:
        """Override to specify required parameters."""
        return []
    
    def get_conserved_quantities(self) -> Dict[str, sp.Expr]:
        """Return energy as a conserved quantity for autonomous systems."""
        if self._kinetic_energy is not None and self._potential_energy is not None:
            return {
                'total_energy': self._kinetic_energy + self._potential_energy,
                'kinetic_energy': self._kinetic_energy,
                'potential_energy': self._potential_energy,
            }
        return {}
