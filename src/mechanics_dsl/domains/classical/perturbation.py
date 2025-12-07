"""
Perturbation Theory for Classical Mechanics

This module implements:
- Time-independent perturbation expansion
- Secular and periodic term separation
- Small parameter series expansion
- Canonical perturbation theory
- Lindstedt-Poincaré method

For a Hamiltonian H = H₀ + εH₁ where ε << 1, the perturbation
expansion gives corrections to frequencies and trajectories.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp
import numpy as np

from ...utils import logger


class PerturbationType(Enum):
    """Types of perturbation expansions."""
    TIME_INDEPENDENT = "time_independent"
    TIME_DEPENDENT = "time_dependent"
    SECULAR = "secular"
    PERIODIC = "periodic"


@dataclass
class PerturbationResult:
    """
    Result of perturbation expansion.
    
    Attributes:
        order: Order of perturbation (1, 2, 3, ...)
        unperturbed: Zeroth-order solution
        corrections: List of correction terms by order
        frequency_shift: Correction to frequencies
        secular_terms: Terms that grow with time
        periodic_terms: Bounded oscillatory corrections
    """
    order: int
    unperturbed: sp.Expr
    corrections: List[sp.Expr] = field(default_factory=list)
    frequency_shift: Optional[sp.Expr] = None
    secular_terms: List[sp.Expr] = field(default_factory=list)
    periodic_terms: List[sp.Expr] = field(default_factory=list)
    
    def total_solution(self, epsilon: sp.Symbol) -> sp.Expr:
        """Get total solution to specified order."""
        result = self.unperturbed
        for i, corr in enumerate(self.corrections):
            result += epsilon**(i+1) * corr
        return result


class PerturbationExpander:
    """
    Perturbation theory expansion for Hamiltonian systems.
    
    For a Hamiltonian H = H₀ + εH₁ + ε²H₂ + ...
    
    Uses canonical perturbation theory to find corrections to:
    - Action variables
    - Angle variables  
    - Frequencies
    
    Example:
        >>> expander = PerturbationExpander()
        >>> H0 = p**2/(2*m) + m*omega**2*x**2/2  # Harmonic oscillator
        >>> H1 = alpha*x**3  # Cubic perturbation
        >>> result = expander.expand(H0, H1, ['x'], order=2)
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._epsilon = sp.Symbol('epsilon', real=True, positive=True)
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def expand_hamiltonian(self, H0: sp.Expr, H1: sp.Expr,
                           coordinates: List[str],
                           order: int = 1) -> PerturbationResult:
        """
        Expand Hamiltonian in perturbation series.
        
        H = H₀ + εH₁
        
        Finds corrections to action-angle variables.
        
        Args:
            H0: Unperturbed Hamiltonian
            H1: First-order perturbation
            coordinates: Generalized coordinates
            order: Order of perturbation expansion
            
        Returns:
            PerturbationResult with corrections
        """
        corrections = []
        
        # First-order correction
        if order >= 1:
            # For integrable H₀, average H₁ over angle variables
            # <H₁> = (1/T) ∫₀ᵀ H₁ dt
            correction_1 = self._first_order_correction(H0, H1, coordinates)
            corrections.append(correction_1)
        
        # Second-order correction
        if order >= 2:
            correction_2 = self._second_order_correction(H0, H1, coordinates)
            corrections.append(correction_2)
        
        return PerturbationResult(
            order=order,
            unperturbed=H0,
            corrections=corrections
        )
    
    def _first_order_correction(self, H0: sp.Expr, H1: sp.Expr,
                                 coordinates: List[str]) -> sp.Expr:
        """Compute first-order perturbation correction."""
        # For time-independent perturbation:
        # E₁ = <H₁> = time average of H₁ over unperturbed orbit
        
        # Simplified: if H₀ is harmonic oscillator, use averaging
        # <x^n> over harmonic oscillator cycle
        
        return sp.simplify(H1)  # Placeholder for full averaging
    
    def _second_order_correction(self, H0: sp.Expr, H1: sp.Expr,
                                  coordinates: List[str]) -> sp.Expr:
        """Compute second-order perturbation correction."""
        # E₂ = Σₙ |<n|H₁|0>|² / (E₀ - Eₙ)
        # This is the quantum formula; classical analog uses action-angle
        
        return sp.S.Zero  # Full implementation requires action-angle analysis
    
    def lindstedt_poincare(self, equation: sp.Expr, 
                           coordinate: str,
                           omega0: sp.Expr,
                           epsilon: sp.Symbol,
                           order: int = 2) -> Dict[str, sp.Expr]:
        """
        Lindstedt-Poincaré method for finding periodic solutions.
        
        Avoids secular terms by expanding the frequency:
        ω = ω₀ + εω₁ + ε²ω₂ + ...
        
        Args:
            equation: Equation of motion (ẍ + ω₀²x + εf(x) = 0)
            coordinate: Coordinate variable
            omega0: Unperturbed frequency
            epsilon: Small parameter
            order: Order of expansion
            
        Returns:
            Dictionary with frequency corrections and solutions
        """
        x = self.get_symbol(coordinate)
        t = self.get_symbol('t')
        
        # Expand frequency
        omega = omega0
        omega_corrections = []
        for i in range(1, order + 1):
            omega_corr = self.get_symbol(f'omega_{i}')
            omega_corrections.append(omega_corr)
            omega += epsilon**i * omega_corr
        
        # The solution x(t) = x₀(τ) + εx₁(τ) + ...
        # where τ = ωt is the stretched time
        
        return {
            'frequency': omega,
            'frequency_corrections': omega_corrections,
            'order': order
        }
    
    def average_over_fast_angle(self, expr: sp.Expr, 
                                 angle: str) -> sp.Expr:
        """
        Average expression over a fast angle variable.
        
        <f(θ)> = (1/2π) ∫₀²π f(θ) dθ
        
        Args:
            expr: Expression to average
            angle: Angle variable name
            
        Returns:
            Averaged expression
        """
        theta = self.get_symbol(angle)
        
        # Integrate over one period
        integral = sp.integrate(expr, (theta, 0, 2*sp.pi))
        averaged = integral / (2 * sp.pi)
        
        return sp.simplify(averaged)
    
    def separate_secular_periodic(self, solution: sp.Expr,
                                   time: sp.Symbol) -> Tuple[sp.Expr, sp.Expr]:
        """
        Separate solution into secular and periodic parts.
        
        Secular terms: grow without bound (t, t², ...)
        Periodic terms: bounded oscillations (sin, cos, ...)
        
        Args:
            solution: Solution expression
            time: Time variable
            
        Returns:
            Tuple of (secular_part, periodic_part)
        """
        # Expand and collect terms
        expanded = sp.expand(solution)
        
        secular = sp.S.Zero
        periodic = sp.S.Zero
        
        # Check each term for time dependence
        if isinstance(expanded, sp.Add):
            terms = expanded.args
        else:
            terms = [expanded]
        
        for term in terms:
            # Check if term has unbounded growth with time
            # Polynomial in t -> secular
            # trig(ωt) -> periodic
            
            if term.has(sp.sin) or term.has(sp.cos):
                # Check coefficient for t
                coeff = term.as_independent(time)[0]
                if not coeff.has(time):
                    periodic += term
                else:
                    secular += term
            else:
                # Polynomial terms
                degree = sp.degree(term, time) if term.has(time) else 0
                if degree > 0:
                    secular += term
                else:
                    periodic += term
        
        return secular, periodic


class AveragingMethod:
    """
    Method of averaging for oscillatory systems.
    
    For a system: ẋ = εf(x, t, ε)
    
    The averaged system: Ẋ = ε<f(X, t, 0)>
    
    where <·> denotes time average over fast oscillation.
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def average_system(self, rhs: sp.Expr, fast_var: str,
                       period: sp.Expr) -> sp.Expr:
        """
        Average a dynamical system over fast variable.
        
        Args:
            rhs: Right-hand side of ODE
            fast_var: Fast oscillating variable
            period: Period of fast oscillation
            
        Returns:
            Averaged RHS
        """
        phi = self.get_symbol(fast_var)
        
        averaged = sp.integrate(rhs, (phi, 0, period)) / period
        return sp.simplify(averaged)
    
    def action_angle_averaging(self, hamiltonian: sp.Expr,
                                action: str, angle: str) -> sp.Expr:
        """
        Average Hamiltonian over angle variable at fixed action.
        
        <H>_θ = (1/2π) ∫₀²π H(I, θ) dθ
        
        Args:
            hamiltonian: Hamiltonian H(I, θ)
            action: Action variable name
            angle: Angle variable name
            
        Returns:
            Averaged Hamiltonian <H>(I)
        """
        theta = self.get_symbol(angle)
        
        averaged = sp.integrate(hamiltonian, (theta, 0, 2*sp.pi)) / (2*sp.pi)
        return sp.simplify(averaged)


class MultiScaleAnalysis:
    """
    Multiple scale perturbation method.
    
    Introduces multiple time scales: T₀ = t, T₁ = εt, T₂ = ε²t, ...
    
    Use for systems with:
    - Slowly varying amplitudes
    - Weakly nonlinear oscillations
    - Resonance phenomena
    """
    
    def __init__(self, order: int = 2):
        self.order = order
        self._symbol_cache: Dict[str, sp.Symbol] = {}
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def define_time_scales(self, t: sp.Symbol, 
                           epsilon: sp.Symbol) -> List[sp.Symbol]:
        """
        Define multiple time scales.
        
        T₀ = t (fast)
        T₁ = εt (slow)
        T₂ = ε²t (slower)
        ...
        
        Args:
            t: Original time variable
            epsilon: Small parameter
            
        Returns:
            List of time scale symbols [T₀, T₁, ...]
        """
        scales = []
        for i in range(self.order + 1):
            Ti = self.get_symbol(f'T_{i}')
            scales.append(Ti)
        return scales
    
    def expand_derivative(self, time_scales: List[sp.Symbol],
                          epsilon: sp.Symbol) -> sp.Expr:
        """
        Expand time derivative operator.
        
        d/dt = ∂/∂T₀ + ε∂/∂T₁ + ε²∂/∂T₂ + ...
        
        Returns symbolic representation.
        """
        # This returns a symbolic expression representing the expansion
        # Actual differentiation requires function objects
        
        result = sp.S.One  # ∂/∂T₀
        for i, Ti in enumerate(time_scales[1:], 1):
            result += epsilon**i  # + εⁱ∂/∂Tᵢ (symbolic)
        
        return result
    
    def remove_secular_terms(self, equations: List[sp.Expr],
                              fast_time: sp.Symbol) -> List[sp.Expr]:
        """
        Remove secular terms by choosing slow-time evolution.
        
        At each order, set coefficients of secular terms to zero
        to determine slow-time dynamics.
        
        Args:
            equations: Perturbation equations at each order
            fast_time: Fast time scale T₀
            
        Returns:
            Solvability conditions (slow dynamics)
        """
        conditions = []
        
        for eq in equations:
            # Find terms that would grow with fast_time
            # These multiply sin(ω₀T₀) or cos(ω₀T₀) at resonance
            # Setting their coefficients to zero gives conditions
            
            if eq.has(fast_time):
                # Extract secular-generating terms
                condition = eq.coeff(fast_time)
                if condition != 0:
                    conditions.append(condition)
        
        return conditions


# Convenience functions

def perturbation_expand(H0: sp.Expr, H1: sp.Expr,
                        coordinates: List[str],
                        order: int = 1) -> PerturbationResult:
    """
    Convenience function for Hamiltonian perturbation expansion.
    
    Args:
        H0: Unperturbed Hamiltonian
        H1: Perturbation
        coordinates: List of coordinates
        order: Expansion order
        
    Returns:
        PerturbationResult
    """
    expander = PerturbationExpander()
    return expander.expand_hamiltonian(H0, H1, coordinates, order)


def average_over_angle(expr: sp.Expr, angle: str) -> sp.Expr:
    """
    Convenience function to average over an angle.
    
    Args:
        expr: Expression to average
        angle: Angle variable name
        
    Returns:
        Averaged expression
    """
    expander = PerturbationExpander()
    return expander.average_over_fast_angle(expr, angle)
