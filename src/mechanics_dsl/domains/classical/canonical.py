"""
Canonical Transformations for Hamiltonian Mechanics

This module implements:
- Generating functions (all 4 types)
- Canonical transformation application
- Poisson bracket verification
- Action-angle variables for integrable systems
- Hamilton-Jacobi equation

A canonical transformation preserves the form of Hamilton's equations:
    dQ/dt = ∂K/∂P
    dP/dt = -∂K/∂Q

where K(Q, P, t) is the transformed Hamiltonian.

Generating Functions:
- F1(q, Q, t): p = ∂F1/∂q, P = -∂F1/∂Q
- F2(q, P, t): p = ∂F2/∂q, Q = ∂F2/∂P
- F3(p, Q, t): q = -∂F3/∂p, P = -∂F3/∂Q
- F4(p, P, t): q = -∂F4/∂p, Q = ∂F4/∂P
"""
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import sympy as sp

from ...utils import logger


class GeneratingFunctionType(Enum):
    """Types of generating functions for canonical transformations."""
    F1 = "F1"  # F1(q, Q, t)
    F2 = "F2"  # F2(q, P, t)
    F3 = "F3"  # F3(p, Q, t)
    F4 = "F4"  # F4(p, P, t)


@dataclass
class GeneratingFunction:
    """
    Generating function for a canonical transformation.
    
    Attributes:
        expression: Symbolic expression for the generating function
        function_type: Type of generating function (F1-F4)
        old_coords: Old coordinate names (q)
        new_coords: New coordinate names (Q)
    """
    expression: sp.Expr
    function_type: GeneratingFunctionType
    old_coords: List[str]
    new_coords: List[str]
    
    def get_transformation_relations(self) -> Dict[str, sp.Expr]:
        """
        Get the transformation relations from the generating function.
        
        Returns:
            Dictionary mapping old and new momenta to their expressions
        """
        relations = {}
        
        for q, Q in zip(self.old_coords, self.new_coords):
            q_sym = sp.Symbol(q, real=True)
            Q_sym = sp.Symbol(Q, real=True)
            p_sym = sp.Symbol(f"p_{q}", real=True)
            P_sym = sp.Symbol(f"P_{Q}", real=True)
            
            if self.function_type == GeneratingFunctionType.F1:
                # p = ∂F1/∂q, P = -∂F1/∂Q
                relations[f"p_{q}"] = sp.diff(self.expression, q_sym)
                relations[f"P_{Q}"] = -sp.diff(self.expression, Q_sym)
                
            elif self.function_type == GeneratingFunctionType.F2:
                # p = ∂F2/∂q, Q = ∂F2/∂P
                relations[f"p_{q}"] = sp.diff(self.expression, q_sym)
                relations[Q] = sp.diff(self.expression, P_sym)
                
            elif self.function_type == GeneratingFunctionType.F3:
                # q = -∂F3/∂p, P = -∂F3/∂Q
                relations[q] = -sp.diff(self.expression, p_sym)
                relations[f"P_{Q}"] = -sp.diff(self.expression, Q_sym)
                
            elif self.function_type == GeneratingFunctionType.F4:
                # q = -∂F4/∂p, Q = ∂F4/∂P
                relations[q] = -sp.diff(self.expression, p_sym)
                relations[Q] = sp.diff(self.expression, P_sym)
        
        return relations


@dataclass
class CanonicalTransformationResult:
    """
    Result of applying a canonical transformation.
    
    Attributes:
        new_hamiltonian: Hamiltonian in new coordinates K(Q, P, t)
        coordinate_map: Mapping from old to new coordinates
        momentum_map: Mapping from old to new momenta
        is_time_dependent: Whether transformation involves explicit time
    """
    new_hamiltonian: sp.Expr
    coordinate_map: Dict[str, sp.Expr]
    momentum_map: Dict[str, sp.Expr]
    is_time_dependent: bool = False


class CanonicalTransformation:
    """
    Canonical transformation tools for Hamiltonian mechanics.
    
    Provides methods for:
    - Applying generating function transformations
    - Verifying canonicity via Poisson brackets
    - Common canonical transformations (point, exchange, scaling)
    
    Example:
        >>> ct = CanonicalTransformation()
        >>> # Point transformation Q = q²
        >>> result = ct.apply_point_transformation(H, {'q': 'q**2'}, ['q'])
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._time_symbol = sp.Symbol('t', real=True)
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol with caching."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def poisson_bracket(self, f: sp.Expr, g: sp.Expr, 
                        coordinates: List[str]) -> sp.Expr:
        """
        Compute the Poisson bracket {f, g}.
        
        {f, g} = Σᵢ (∂f/∂qᵢ * ∂g/∂pᵢ - ∂f/∂pᵢ * ∂g/∂qᵢ)
        
        Args:
            f: First function
            g: Second function
            coordinates: List of coordinate names
            
        Returns:
            Poisson bracket expression
        """
        result = sp.S.Zero
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            p_sym = self.get_symbol(f"p_{q}")
            
            df_dq = sp.diff(f, q_sym)
            df_dp = sp.diff(f, p_sym)
            dg_dq = sp.diff(g, q_sym)
            dg_dp = sp.diff(g, p_sym)
            
            result += df_dq * dg_dp - df_dp * dg_dq
        
        return sp.simplify(result)
    
    def verify_canonical(self, Q: List[sp.Expr], P: List[sp.Expr],
                         old_coords: List[str]) -> bool:
        """
        Verify that a transformation is canonical using Poisson brackets.
        
        For a canonical transformation:
        - {Qᵢ, Qⱼ} = 0
        - {Pᵢ, Pⱼ} = 0
        - {Qᵢ, Pⱼ} = δᵢⱼ
        
        Args:
            Q: New coordinates as functions of old (q, p)
            P: New momenta as functions of old (q, p)
            old_coords: Names of old coordinates
            
        Returns:
            True if transformation is canonical
        """
        n = len(Q)
        
        for i in range(n):
            for j in range(n):
                # Check {Qᵢ, Qⱼ} = 0
                bracket_QQ = self.poisson_bracket(Q[i], Q[j], old_coords)
                if not sp.simplify(bracket_QQ) == 0:
                    logger.warning(f"{{Q{i}, Q{j}}} = {bracket_QQ} ≠ 0")
                    return False
                
                # Check {Pᵢ, Pⱼ} = 0
                bracket_PP = self.poisson_bracket(P[i], P[j], old_coords)
                if not sp.simplify(bracket_PP) == 0:
                    logger.warning(f"{{P{i}, P{j}}} = {bracket_PP} ≠ 0")
                    return False
                
                # Check {Qᵢ, Pⱼ} = δᵢⱼ
                expected = 1 if i == j else 0
                bracket_QP = self.poisson_bracket(Q[i], P[j], old_coords)
                if not sp.simplify(bracket_QP - expected) == 0:
                    logger.warning(f"{{Q{i}, P{j}}} = {bracket_QP} ≠ {expected}")
                    return False
        
        return True
    
    def apply_generating_function(self, hamiltonian: sp.Expr,
                                   gen_func: GeneratingFunction) -> CanonicalTransformationResult:
        """
        Apply a canonical transformation using a generating function.
        
        Args:
            hamiltonian: Original Hamiltonian H(q, p, t)
            gen_func: Generating function specification
            
        Returns:
            CanonicalTransformationResult with new Hamiltonian
        """
        relations = gen_func.get_transformation_relations()
        
        # Build substitution mappings
        coord_map = {}
        mom_map = {}
        
        for q, Q in zip(gen_func.old_coords, gen_func.new_coords):
            q_sym = self.get_symbol(q)
            Q_sym = self.get_symbol(Q)
            p_sym = self.get_symbol(f"p_{q}")
            P_sym = self.get_symbol(f"P_{Q}")
            
            # Depending on generating function type, solve for old variables
            # This is simplified; full implementation requires symbolic inversion
            coord_map[q] = Q_sym  # Placeholder
            mom_map[f"p_{q}"] = P_sym  # Placeholder
        
        # Transform Hamiltonian
        # K = H + ∂F/∂t for time-dependent transformations
        new_H = hamiltonian
        for old_var, new_expr in {**coord_map, **mom_map}.items():
            old_sym = self.get_symbol(old_var)
            new_H = new_H.subs(old_sym, new_expr)
        
        # Add time derivative of generating function if time-dependent
        dF_dt = sp.diff(gen_func.expression, self._time_symbol)
        is_time_dep = not (dF_dt.equals(sp.S.Zero) or sp.simplify(dF_dt) == 0)
        
        if is_time_dep:
            new_H = new_H + dF_dt
        
        return CanonicalTransformationResult(
            new_hamiltonian=sp.simplify(new_H),
            coordinate_map=coord_map,
            momentum_map=mom_map,
            is_time_dependent=is_time_dep
        )
    
    def point_transformation(self, hamiltonian: sp.Expr,
                              coord_transform: Dict[str, sp.Expr],
                              coordinates: List[str]) -> CanonicalTransformationResult:
        """
        Apply a point transformation Q = f(q).
        
        For point transformations, momenta transform as:
        P = (∂q/∂Q) * p
        
        Args:
            hamiltonian: Original Hamiltonian
            coord_transform: Dictionary mapping old coords to new coord expressions
            coordinates: List of original coordinate names
            
        Returns:
            CanonicalTransformationResult
        """
        coord_map = {}
        mom_map = {}
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            p_sym = self.get_symbol(f"p_{q}")
            
            if q in coord_transform:
                Q_expr = coord_transform[q]
                # Invert to get q(Q) - this is simplified
                Q_sym = list(Q_expr.free_symbols - {q_sym})[0] if Q_expr.free_symbols - {q_sym} else sp.Symbol(f"Q_{q}")
                
                # For simple transformations, compute Jacobian
                dQ_dq = sp.diff(Q_expr, q_sym)
                
                coord_map[q] = Q_sym
                mom_map[f"p_{q}"] = self.get_symbol(f"P_{q}") / dQ_dq
            else:
                coord_map[q] = q_sym
                mom_map[f"p_{q}"] = p_sym
        
        # Transform Hamiltonian
        new_H = hamiltonian
        for old_var, new_expr in {**coord_map, **mom_map}.items():
            old_sym = self.get_symbol(old_var)
            new_H = new_H.subs(old_sym, new_expr)
        
        return CanonicalTransformationResult(
            new_hamiltonian=sp.simplify(new_H),
            coordinate_map=coord_map,
            momentum_map=mom_map
        )
    
    def exchange_transformation(self, hamiltonian: sp.Expr,
                                  coordinates: List[str]) -> CanonicalTransformationResult:
        """
        Apply the exchange transformation Q = p, P = -q.
        
        This swaps coordinates and momenta.
        
        Args:
            hamiltonian: Original Hamiltonian
            coordinates: List of coordinate names
            
        Returns:
            CanonicalTransformationResult
        """
        coord_map = {}
        mom_map = {}
        
        new_H = hamiltonian
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            p_sym = self.get_symbol(f"p_{q}")
            Q_sym = self.get_symbol(f"Q_{q}")
            P_sym = self.get_symbol(f"P_{q}")
            
            # Q = p, P = -q
            coord_map[q] = P_sym
            mom_map[f"p_{q}"] = Q_sym
            
            # Substitute: q → -P, p → Q
            new_H = new_H.subs(q_sym, -P_sym)
            new_H = new_H.subs(p_sym, Q_sym)
        
        return CanonicalTransformationResult(
            new_hamiltonian=sp.simplify(new_H),
            coordinate_map=coord_map,
            momentum_map=mom_map
        )


class ActionAngleVariables:
    """
    Action-angle variable computation for integrable systems.
    
    For a 1D integrable system with Hamiltonian H(q, p):
    - Action: J = (1/2π) ∮ p dq
    - Angle: θ = ∂S/∂J where S is Hamilton's principal function
    
    In action-angle variables:
    - H = H(J) depends only on action
    - θ̇ = ω(J) = ∂H/∂J (constant frequency)
    
    Example:
        >>> aa = ActionAngleVariables()
        >>> J, omega = aa.compute_action_frequency(H, 'q', E)
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
    
    def compute_action(self, hamiltonian: sp.Expr, coordinate: str,
                       energy: float, parameters: Dict[str, float],
                       q_min: float = -10, q_max: float = 10) -> float:
        """
        Compute the action variable J = (1/2π) ∮ p dq numerically.
        
        Args:
            hamiltonian: Hamiltonian H(q, p)
            coordinate: Coordinate name
            energy: Energy level
            parameters: System parameters
            q_min, q_max: Search range for turning points
            
        Returns:
            Action variable value
        """
        import numpy as np
        from scipy.integrate import quad
        
        q_sym = self.get_symbol(coordinate)
        p_sym = self.get_symbol(f"p_{coordinate}")
        
        # Solve H(q, p) = E for p
        # For simple systems: p = sqrt(2m(E - V(q))) or similar
        p_squared_expr = sp.solve(hamiltonian - energy, p_sym**2)
        
        if not p_squared_expr:
            # Try solving for p directly
            p_expr = sp.solve(hamiltonian - energy, p_sym)
            if not p_expr:
                logger.warning("Could not solve for momentum")
                return 0.0
            p_func = sp.lambdify(q_sym, p_expr[0].subs(parameters), 'numpy')
        else:
            # p = sqrt(p²)
            p_squared_func = sp.lambdify(q_sym, p_squared_expr[0].subs(parameters), 'numpy')
            def p_func(q):
                p2 = p_squared_func(q)
                return np.sqrt(max(0, p2)) if np.isfinite(p2) else 0.0
        
        # Find turning points (where p = 0)
        q_vals = np.linspace(q_min, q_max, 1000)
        try:
            p_vals = np.array([abs(p_func(q)) for q in q_vals])
        except Exception:
            return 0.0
        
        # Find sign changes or zeros
        turning_points = []
        for i in range(len(q_vals) - 1):
            if p_vals[i] * p_vals[i+1] < 0 or p_vals[i] < 1e-10:
                turning_points.append(q_vals[i])
        
        if len(turning_points) < 2:
            logger.warning("Could not find two turning points")
            return 0.0
        
        q_left = min(turning_points)
        q_right = max(turning_points)
        
        # Integrate p dq from q_left to q_right and back
        try:
            integral, _ = quad(p_func, q_left, q_right, limit=100)
            J = integral / np.pi  # Factor of 2π for full cycle, but we only do half
            return J
        except Exception as e:
            logger.warning(f"Action integral failed: {e}")
            return 0.0
    
    def compute_frequency(self, hamiltonian: sp.Expr, coordinate: str,
                          action: float, delta_J: float = 0.01) -> float:
        """
        Compute the angular frequency ω = ∂H/∂J numerically.
        
        Uses finite difference: ω ≈ (H(J+δ) - H(J-δ)) / (2δ)
        
        This requires knowing H(J), which involves inverting action integral.
        For simple cases, use the period: ω = 2π/T
        
        Args:
            hamiltonian: Hamiltonian expression
            coordinate: Coordinate name
            action: Action value J
            delta_J: Finite difference step
            
        Returns:
            Angular frequency
        """
        # This is a placeholder - full implementation requires 
        # computing the period from the action integral derivative
        logger.warning("compute_frequency not fully implemented; use period-based method")
        return 0.0


class HamiltonJacobi:
    """
    Hamilton-Jacobi equation solver.
    
    The Hamilton-Jacobi equation is:
        H(q, ∂S/∂q) + ∂S/∂t = 0
    
    For time-independent H, we look for S = W(q) - Et:
        H(q, ∂W/∂q) = E
    
    This is useful for finding action-angle variables and
    solving integrable systems.
    """
    
    def __init__(self):
        self._symbol_cache: Dict[str, sp.Symbol] = {}
        self._time_symbol = sp.Symbol('t', real=True)
    
    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a symbol."""
        if name not in self._symbol_cache:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self._symbol_cache[name] = sp.Symbol(name, **default_assumptions)
        return self._symbol_cache[name]
    
    def characteristic_function_1d(self, hamiltonian: sp.Expr,
                                    coordinate: str,
                                    energy: sp.Symbol) -> Optional[sp.Expr]:
        """
        Compute Hamilton's characteristic function W(q, E) for 1D system.
        
        W = ∫ p(q, E) dq where H(q, p) = E
        
        Args:
            hamiltonian: Hamiltonian H(q, p)
            coordinate: Coordinate name
            energy: Energy symbol (constant of motion)
            
        Returns:
            Characteristic function W(q, E), or None if cannot solve
        """
        q = self.get_symbol(coordinate)
        p = self.get_symbol(f"p_{coordinate}")
        
        # Solve H(q, p) = E for p
        p_solution = sp.solve(hamiltonian - energy, p)
        
        if not p_solution:
            logger.warning("Could not solve H = E for momentum")
            return None
        
        # Take positive root (convention)
        p_expr = p_solution[0]
        
        # Integrate p dq
        W = sp.integrate(p_expr, q)
        
        return W
    
    def solve(self, hamiltonian: sp.Expr, 
              coordinates: List[str],
              energy: Optional[sp.Symbol] = None) -> sp.Expr:
        """
        Solve the Hamilton-Jacobi equation for the principal function S.
        
        For time-independent Hamiltonian H(q, p) = E:
            S(q, E) = W(q, E) - E*t
        where W is Hamilton's characteristic function.
        
        Args:
            hamiltonian: Hamiltonian H(q, p) - must use 'p' as momentum symbol
            coordinates: List of coordinate names
            energy: Energy symbol (constant of motion). Defaults to Symbol('E').
            
        Returns:
            Principal function S(q, E) for 1D, or characteristic function W for multi-D
        """
        if energy is None:
            energy = sp.Symbol('E', positive=True)
        
        if len(coordinates) == 1:
            # 1D case: solve directly
            coord = coordinates[0]
            q = self.get_symbol(coord)
            
            # Replace generic 'p' with coordinate-specific momentum
            p_generic = sp.Symbol('p')
            p_coord = self.get_symbol(f"p_{coord}")
            H_substituted = hamiltonian.subs(p_generic, p_coord)
            
            # Solve H(q, p) = E for p
            p_solution = sp.solve(H_substituted - energy, p_coord)
            
            if not p_solution:
                raise ValueError("Could not solve H = E for momentum")
            
            # Take positive root (physical convention for forward motion)
            p_expr = None
            for sol in p_solution:
                # Prefer the positive root
                if sp.simplify(sol).is_positive or len(p_solution) == 1:
                    p_expr = sol
                    break
            if p_expr is None:
                p_expr = p_solution[0]
            
            # Integrate p dq to get characteristic function W
            W = sp.integrate(p_expr, q)
            
            return sp.simplify(W)
        else:
            # Multi-D case: attempt separation of variables
            result = self.solve_separable(hamiltonian, coordinates)
            # Return sum of all W functions
            total_W = sp.S.Zero
            for coord, (W_func, alpha) in result.items():
                total_W += W_func
            return total_W

    def solve_separable(self, hamiltonian: sp.Expr,
                        coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Attempt to solve Hamilton-Jacobi equation by separation of variables.
        
        For separable systems:
            S(q₁, q₂, ..., t) = W₁(q₁) + W₂(q₂) + ... - Et
        
        Each Wᵢ satisfies a 1D HJ equation.
        
        Args:
            hamiltonian: Hamiltonian (should be separable)
            coordinates: List of coordinates
            
        Returns:
            Dictionary mapping coordinates to their W functions
        """
        # Check if Hamiltonian is separable (each term depends on only one q, p pair)
        terms = sp.Add.make_args(hamiltonian)
        
        W_functions = {}
        remaining_H = sp.S.Zero
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            p_sym = self.get_symbol(f"p_{q}")
            alpha_sym = sp.Symbol(f"alpha_{q}", real=True)  # Separation constant
            
            # Extract terms depending on this coordinate
            q_terms = sp.S.Zero
            for term in terms:
                term_symbols = term.free_symbols
                if q_sym in term_symbols or p_sym in term_symbols:
                    q_terms += term
            
            if q_terms == sp.S.Zero:
                continue
            
            # For this coordinate's part: H_q(q, p) = α
            p_solution = sp.solve(q_terms - alpha_sym, p_sym)
            if p_solution:
                W_q = sp.integrate(p_solution[0], q_sym)
                W_functions[q] = (W_q, alpha_sym)
            else:
                logger.warning(f"Could not separate coordinate {q}")
        
        return W_functions
