"""
Dissipation and Non-Conservative Forces for Classical Mechanics

This module implements:
- Rayleigh dissipation function for velocity-dependent damping
- Friction models (Coulomb, viscous, Stribeck)
- Generalized non-conservative forces

The modified Euler-Lagrange equations with dissipation are:
    d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ + ∂F/∂q̇ᵢ = Qᵢ

where F is the Rayleigh dissipation function and Qᵢ are generalized forces.
"""
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np

from ...utils import logger


class FrictionType(Enum):
    """Types of friction models available."""
    VISCOUS = "viscous"      # F = -b*v
    COULOMB = "coulomb"      # F = -μ*N*sign(v)
    STRIBECK = "stribeck"    # Combines static, Coulomb, and viscous


@dataclass
class FrictionModel:
    """
    Friction model for mechanical systems.
    
    Supports viscous, Coulomb, and Stribeck friction models.
    
    Attributes:
        friction_type: Type of friction model
        coefficients: Dictionary of friction coefficients
            - viscous: {'b': damping_coefficient}
            - coulomb: {'mu': friction_coefficient, 'N': normal_force}
            - stribeck: {'mu_s': static, 'mu_k': kinetic, 'b': viscous, 'v_s': Stribeck velocity}
    """
    friction_type: FrictionType
    coefficients: Dict[str, float] = field(default_factory=dict)
    
    def get_force_expression(self, velocity: sp.Symbol) -> sp.Expr:
        """
        Get symbolic expression for friction force.
        
        Args:
            velocity: Symbolic velocity variable
            
        Returns:
            SymPy expression for friction force
        """
        if self.friction_type == FrictionType.VISCOUS:
            b = self.coefficients.get('b', 1.0)
            return -b * velocity
            
        elif self.friction_type == FrictionType.COULOMB:
            mu = self.coefficients.get('mu', 0.3)
            N = self.coefficients.get('N', 1.0)
            # Use sign function for Coulomb friction
            return -mu * N * sp.sign(velocity)
            
        elif self.friction_type == FrictionType.STRIBECK:
            mu_s = self.coefficients.get('mu_s', 0.4)  # Static friction
            mu_k = self.coefficients.get('mu_k', 0.3)  # Kinetic friction
            b = self.coefficients.get('b', 0.1)        # Viscous coefficient
            v_s = self.coefficients.get('v_s', 0.01)   # Stribeck velocity
            N = self.coefficients.get('N', 1.0)        # Normal force
            
            # Stribeck curve: F = (mu_k + (mu_s - mu_k)*exp(-|v|/v_s)) * N * sign(v) + b*v
            stribeck_term = (mu_k + (mu_s - mu_k) * sp.exp(-sp.Abs(velocity) / v_s))
            return -stribeck_term * N * sp.sign(velocity) - b * velocity
            
        else:
            logger.warning(f"Unknown friction type: {self.friction_type}")
            return sp.S.Zero
    
    def get_numerical_force(self, velocity: float) -> float:
        """
        Compute numerical friction force value.
        
        Args:
            velocity: Numerical velocity value
            
        Returns:
            Friction force value
        """
        if self.friction_type == FrictionType.VISCOUS:
            b = self.coefficients.get('b', 1.0)
            return -b * velocity
            
        elif self.friction_type == FrictionType.COULOMB:
            mu = self.coefficients.get('mu', 0.3)
            N = self.coefficients.get('N', 1.0)
            return -mu * N * np.sign(velocity) if abs(velocity) > 1e-10 else 0.0
            
        elif self.friction_type == FrictionType.STRIBECK:
            mu_s = self.coefficients.get('mu_s', 0.4)
            mu_k = self.coefficients.get('mu_k', 0.3)
            b = self.coefficients.get('b', 0.1)
            v_s = self.coefficients.get('v_s', 0.01)
            N = self.coefficients.get('N', 1.0)
            
            if abs(velocity) < 1e-10:
                return 0.0
            stribeck = mu_k + (mu_s - mu_k) * np.exp(-abs(velocity) / v_s)
            return -stribeck * N * np.sign(velocity) - b * velocity
            
        return 0.0


@dataclass
class GeneralizedForce:
    """
    Represents a generalized non-conservative force.
    
    The generalized force Qᵢ appears on the RHS of the Euler-Lagrange equations:
        d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = Qᵢ
    
    Attributes:
        coordinate: The generalized coordinate this force acts on
        expression: Symbolic expression for the force Q(q, q̇, t)
        numerical_func: Optional numerical function for evaluation
    """
    coordinate: str
    expression: sp.Expr
    numerical_func: Optional[Callable] = None
    
    def evaluate(self, state: Dict[str, float], t: float = 0.0) -> float:
        """
        Evaluate the generalized force numerically.
        
        Args:
            state: Dictionary of state variable values
            t: Current time
            
        Returns:
            Force value
        """
        if self.numerical_func is not None:
            return self.numerical_func(state, t)
        
        # Fall back to symbolic evaluation
        subs_dict = {sp.Symbol(k): v for k, v in state.items()}
        subs_dict[sp.Symbol('t')] = t
        result = self.expression.subs(subs_dict)
        return float(result.evalf())


class RayleighDissipation:
    """
    Rayleigh dissipation function for velocity-dependent damping.
    
    The Rayleigh dissipation function F is defined as:
        F = (1/2) * Σᵢⱼ bᵢⱼ * q̇ᵢ * q̇ⱼ
    
    For simple damping with coefficient b:
        F = (1/2) * b * q̇²
    
    The dissipative force is: Q_dissipative = -∂F/∂q̇
    
    The power dissipated is: P = 2F (always positive)
    
    Example:
        >>> dissipation = RayleighDissipation()
        >>> dissipation.add_damping('theta', 0.1)
        >>> F = dissipation.get_dissipation_function()
    """
    
    def __init__(self):
        self._damping_coefficients: Dict[str, Dict[str, float]] = {}
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._dissipation_function: Optional[sp.Expr] = None
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def add_damping(self, coordinate: str, coefficient: float, 
                    cross_coordinate: Optional[str] = None) -> None:
        """
        Add damping to a coordinate or cross-damping between coordinates.
        
        Args:
            coordinate: Primary coordinate q
            coefficient: Damping coefficient b
            cross_coordinate: Optional second coordinate for cross-damping bᵢⱼ
        """
        if coordinate not in self._damping_coefficients:
            self._damping_coefficients[coordinate] = {}
        
        key = cross_coordinate if cross_coordinate else coordinate
        self._damping_coefficients[coordinate][key] = coefficient
        
        # Make symmetric for cross-damping
        if cross_coordinate and cross_coordinate != coordinate:
            if cross_coordinate not in self._damping_coefficients:
                self._damping_coefficients[cross_coordinate] = {}
            self._damping_coefficients[cross_coordinate][coordinate] = coefficient
        
        # Invalidate cached function
        self._dissipation_function = None
        logger.debug(f"Added damping: b[{coordinate},{key}] = {coefficient}")
    
    def add_friction(self, coordinate: str, friction: FrictionModel) -> None:
        """
        Add friction model to a coordinate.
        
        Note: Friction is handled separately from Rayleigh dissipation
        as it may be nonlinear. Use GeneralizedForce for the friction term.
        
        Args:
            coordinate: Coordinate to add friction to
            friction: FrictionModel instance
        """
        velocity = self.get_symbol(f"{coordinate}_dot")
        force_expr = friction.get_force_expression(velocity)
        logger.info(f"Friction force for {coordinate}: {force_expr}")
        # Return as GeneralizedForce for use in equations
        return GeneralizedForce(coordinate, force_expr)
    
    def get_dissipation_function(self) -> sp.Expr:
        """
        Build and return the Rayleigh dissipation function.
        
        F = (1/2) * Σᵢⱼ bᵢⱼ * q̇ᵢ * q̇ⱼ
        
        Returns:
            SymPy expression for the dissipation function
        """
        if self._dissipation_function is not None:
            return self._dissipation_function
        
        F = sp.S.Zero
        
        for coord_i, damping_dict in self._damping_coefficients.items():
            q_dot_i = self.get_symbol(f"{coord_i}_dot")
            for coord_j, b_ij in damping_dict.items():
                q_dot_j = self.get_symbol(f"{coord_j}_dot")
                # Factor of 1/2 for diagonal, but we double-count off-diagonal
                if coord_i == coord_j:
                    F += sp.Rational(1, 2) * b_ij * q_dot_i * q_dot_j
                else:
                    # Off-diagonal: only add once (already symmetric)
                    if coord_i < coord_j:  # Lexicographic ordering
                        F += sp.Rational(1, 2) * b_ij * q_dot_i * q_dot_j
        
        self._dissipation_function = F
        return F
    
    def get_dissipative_force(self, coordinate: str) -> sp.Expr:
        """
        Get the dissipative force for a coordinate.
        
        Q_dissipative = -∂F/∂q̇
        
        Args:
            coordinate: Generalized coordinate name
            
        Returns:
            SymPy expression for the dissipative force
        """
        F = self.get_dissipation_function()
        q_dot = self.get_symbol(f"{coordinate}_dot")
        return -sp.diff(F, q_dot)
    
    def get_power_dissipated(self) -> sp.Expr:
        """
        Get the power dissipated by all damping.
        
        P = 2F = Σᵢⱼ bᵢⱼ * q̇ᵢ * q̇ⱼ
        
        Returns:
            SymPy expression for dissipated power (always >= 0)
        """
        return 2 * self.get_dissipation_function()
    
    def get_coordinates(self) -> List[str]:
        """Get list of coordinates with damping."""
        return list(self._damping_coefficients.keys())
    
    def clear(self) -> None:
        """Clear all damping coefficients."""
        self._damping_coefficients.clear()
        self._dissipation_function = None


class DissipativeLagrangianMechanics:
    """
    Extension of Lagrangian mechanics with dissipation and non-conservative forces.
    
    Implements the modified Euler-Lagrange equations:
        d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ + ∂F/∂q̇ᵢ = Qᵢ
    
    where:
        L = Lagrangian (T - V)
        F = Rayleigh dissipation function
        Qᵢ = Generalized non-conservative forces
    
    Example:
        >>> system = DissipativeLagrangianMechanics("damped_pendulum")
        >>> system.set_lagrangian(L)
        >>> system.add_damping('theta', b=0.1)
        >>> eom = system.derive_equations_of_motion()
    """
    
    def __init__(self, name: str = "dissipative_system"):
        self.name = name
        self.coordinates: List[str] = []
        self.parameters: Dict[str, float] = {}
        self._lagrangian: Optional[sp.Expr] = None
        self._dissipation = RayleighDissipation()
        self._generalized_forces: Dict[str, sp.Expr] = {}
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._time_symbol = sp.Symbol('t', real=True)
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def add_coordinate(self, name: str) -> None:
        """Add a generalized coordinate."""
        if name not in self.coordinates:
            self.coordinates.append(name)
    
    def set_lagrangian(self, expr: sp.Expr) -> None:
        """Set the Lagrangian L = T - V."""
        self._lagrangian = expr
    
    def add_damping(self, coordinate: str, coefficient: float,
                    cross_coordinate: Optional[str] = None) -> None:
        """
        Add Rayleigh damping to a coordinate.
        
        Args:
            coordinate: Generalized coordinate
            coefficient: Damping coefficient b
            cross_coordinate: Optional coordinate for cross-damping
        """
        self._dissipation.add_damping(coordinate, coefficient, cross_coordinate)
    
    def add_generalized_force(self, coordinate: str, force: sp.Expr) -> None:
        """
        Add a generalized non-conservative force.
        
        Args:
            coordinate: Generalized coordinate the force acts on
            force: Symbolic expression Q(q, q̇, t)
        """
        if coordinate in self._generalized_forces:
            self._generalized_forces[coordinate] += force
        else:
            self._generalized_forces[coordinate] = force
    
    def add_friction(self, coordinate: str, friction: FrictionModel) -> None:
        """
        Add friction to a coordinate.
        
        Args:
            coordinate: Generalized coordinate
            friction: FrictionModel instance
        """
        velocity = self.get_symbol(f"{coordinate}_dot")
        force = friction.get_force_expression(velocity)
        self.add_generalized_force(coordinate, force)
    
    def add_driving_force(self, coordinate: str, amplitude: float, 
                          frequency: float, phase: float = 0.0) -> None:
        """
        Add a sinusoidal driving force.
        
        Q = A * cos(ω*t + φ)
        
        Args:
            coordinate: Generalized coordinate
            amplitude: Force amplitude A
            frequency: Angular frequency ω
            phase: Phase offset φ
        """
        t = self._time_symbol
        force = amplitude * sp.cos(frequency * t + phase)
        self.add_generalized_force(coordinate, force)
    
    def get_dissipation_function(self) -> sp.Expr:
        """Get the Rayleigh dissipation function."""
        return self._dissipation.get_dissipation_function()
    
    def derive_equations_of_motion(self) -> Dict[str, sp.Expr]:
        """
        Derive the modified Euler-Lagrange equations with dissipation.
        
        d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ + ∂F/∂q̇ᵢ = Qᵢ
        
        Rearranged to solve for accelerations:
        q̈ᵢ = f(q, q̇, t)
        
        Returns:
            Dictionary mapping acceleration variables to expressions
        """
        if self._lagrangian is None:
            raise ValueError("Lagrangian not defined")
        
        L = self._lagrangian
        F = self.get_dissipation_function()
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
            
            # Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q
            dL_dq_dot = sp.diff(L_sub, sp.diff(q_func, self._time_symbol))
            d_dt = sp.diff(dL_dq_dot, self._time_symbol)
            dL_dq = sp.diff(L_sub, q_func)
            
            eq = d_dt - dL_dq
            
            # Add dissipation term: +∂F/∂q̇
            dissipation_force = sp.diff(F, q_dot)
            
            # Substitute for dissipation (doesn't need time derivative)
            # Replace derivatives back with symbols first
            eq = eq.subs(sp.diff(q_func, self._time_symbol, 2), q_ddot)
            eq = eq.subs(sp.diff(q_func, self._time_symbol), q_dot)
            eq = eq.subs(q_func, q_sym)
            
            # Add dissipation
            eq = eq + dissipation_force
            
            # Subtract generalized forces (move to LHS)
            Q_i = self._generalized_forces.get(q, sp.S.Zero)
            eq = eq - Q_i
            
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
        
        return accelerations
    
    def compute_energy_rate(self) -> sp.Expr:
        """
        Compute the rate of change of mechanical energy.
        
        For a dissipative system:
        dE/dt = -2F + Σᵢ Qᵢ * q̇ᵢ
        
        where F is the Rayleigh function and Qᵢ are generalized forces.
        
        Returns:
            Expression for dE/dt (negative for energy loss)
        """
        # Power dissipated by Rayleigh function
        power_dissipated = self._dissipation.get_power_dissipated()
        
        # Power input from generalized forces
        power_input = sp.S.Zero
        for q, Q in self._generalized_forces.items():
            q_dot = self.get_symbol(f"{q}_dot")
            power_input += Q * q_dot
        
        return -power_dissipated + power_input
    
    def is_dissipative(self) -> bool:
        """Check if the system has any dissipation."""
        return len(self._dissipation.get_coordinates()) > 0
    
    def get_state_variables(self) -> List[str]:
        """Get state variables for ODE system."""
        state = []
        for q in self.coordinates:
            state.extend([q, f"{q}_dot"])
        return state
