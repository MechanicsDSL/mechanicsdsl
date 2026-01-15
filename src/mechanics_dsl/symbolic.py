"""
Symbolic computation engine for MechanicsDSL
"""
import weakref
import sympy as sp
from typing import List, Dict, Tuple, Optional

from .utils import logger, config, profile_function, timeout, TimeoutError, LRUCache, _perf_monitor
from .parser import (
    Expression, NumberExpr, IdentExpr, GreekLetterExpr, BinaryOpExpr,
    UnaryOpExpr, FractionExpr, DerivativeVarExpr, DerivativeExpr,
    FunctionCallExpr, VectorExpr, VectorOpExpr
)

__all__ = ['SymbolicEngine']

class SymbolicEngine:
    """Enhanced symbolic mathematics engine with advanced caching and performance monitoring"""
    
    # Class-level weak reference registry for shared symbols across engines
    # This helps prevent memory leaks in long-running applications
    _global_symbol_registry: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    
    def __init__(self, use_weak_refs: bool = False):
        """
        Initialize the symbolic computation engine.
        
        Args:
            use_weak_refs: If True, use weak references for symbol storage.
                           Recommended for long-running applications to prevent
                           memory leaks. Default is False for compatibility.
        """
        self.sp = sp
        self._use_weak_refs = use_weak_refs
        
        # Symbol storage - either regular dict or weak value dict
        if use_weak_refs:
            self.symbol_map: Dict[str, sp.Symbol] = {}
            self._weak_symbol_map: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        else:
            self.symbol_map: Dict[str, sp.Symbol] = {}
            self._weak_symbol_map = None
            
        self.function_map: Dict[str, sp.Function] = {}
        self.time_symbol = sp.Symbol('t', real=True)
        self.assumptions: Dict[str, dict] = {}
        
        # v6.0: Advanced LRU cache
        if config.cache_symbolic_results:
            self._cache = LRUCache(
                maxsize=config.cache_max_size,
                max_memory_mb=config.cache_max_memory_mb
            )
        else:
            self._cache = None
        self._perf_monitor = _perf_monitor if config.enable_performance_monitoring else None

    def get_symbol(self, name: str, **assumptions) -> sp.Symbol:
        """Get or create a SymPy symbol with assumptions (cached)"""
        if name not in self.symbol_map:
            default_assumptions = {'real': True}
            default_assumptions.update(assumptions)
            self.symbol_map[name] = sp.Symbol(name, **default_assumptions)
            self.assumptions[name] = default_assumptions
            logger.debug(f"Created symbol: {name} with assumptions {default_assumptions}")
            
            # Also store in weak ref registry if using weak refs
            if self._use_weak_refs and self._weak_symbol_map is not None:
                self._weak_symbol_map[name] = self.symbol_map[name]
                
        return self.symbol_map[name]

    def clear_cache(self) -> int:
        """
        Clear all caches to free memory.
        
        Useful for long-running applications that process many different
        mechanical systems. Clears:
        - LRU expression cache
        - Symbol map (keeps time_symbol)
        - Function map
        
        Returns:
            Number of cached items cleared
            
        Example:
            >>> engine = SymbolicEngine()
            >>> # ... do lots of computation ...
            >>> cleared = engine.clear_cache()
            >>> print(f"Freed {cleared} cached items")
        """
        count = 0
        
        # Clear LRU cache
        if self._cache is not None:
            count += len(self._cache._cache) if hasattr(self._cache, '_cache') else 0
            self._cache.clear()
        
        # Clear symbol map (keep time symbol)
        count += len(self.symbol_map)
        self.symbol_map.clear()
        self.assumptions.clear()
        
        # Clear function map
        count += len(self.function_map)
        self.function_map.clear()
        
        # Clear weak refs
        if self._weak_symbol_map is not None:
            self._weak_symbol_map.clear()
        
        logger.info(f"Cleared {count} cached items from SymbolicEngine")
        return count

    def memory_stats(self) -> Dict[str, int]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with counts of cached items by category
        """
        stats = {
            'symbols': len(self.symbol_map),
            'functions': len(self.function_map),
            'assumptions': len(self.assumptions),
        }
        
        if self._cache is not None:
            stats['cache_entries'] = len(self._cache._cache) if hasattr(self._cache, '_cache') else 0
            stats['cache_hit_rate'] = self._cache.hit_rate
            
        if self._weak_symbol_map is not None:
            stats['weak_refs'] = len(self._weak_symbol_map)
            
        return stats

    def get_function(self, name: str) -> sp.Function:
        """Get or create a SymPy function (cached)"""
        if name not in self.function_map:
            self.function_map[name] = sp.Function(name, real=True)
            logger.debug(f"Created function: {name}")
        return self.function_map[name]

    @profile_function
    def ast_to_sympy(self, expr: Expression) -> sp.Expr:
        """
        Convert AST expression to SymPy with comprehensive support and caching
        
        Args:
            expr: AST expression node
            
        Returns:
            SymPy expression
        """
        # v6.0: Cache key generation
        cache_key = None
        if self._cache is not None:
            try:
                cache_key = str(hash(str(expr)))
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for expression: {expr}")
                    return cached
            except Exception as e:
                logger.debug(f"Cache key generation failed: {e}")
        
        if self._perf_monitor:
            self._perf_monitor.start_timer('ast_to_sympy')
        
        try:
            result = self._ast_to_sympy_impl(expr)
            
            # Cache result
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, result)
            
            if self._perf_monitor:
                self._perf_monitor.stop_timer('ast_to_sympy')
            
            return result
        except Exception as e:
            if self._perf_monitor:
                self._perf_monitor.stop_timer('ast_to_sympy')
            raise
    
    def _ast_to_sympy_impl(self, expr: Expression) -> sp.Expr:
        """Internal implementation of AST to SymPy conversion"""
        if isinstance(expr, NumberExpr):
            return sp.Float(expr.value)
            
        elif isinstance(expr, IdentExpr):
            # FIX: Map 't' to the canonical time symbol
            if expr.name == 't':
                return self.time_symbol
            return self.get_symbol(expr.name)
            
        elif isinstance(expr, GreekLetterExpr):
            return self.get_symbol(expr.letter)
            
        elif isinstance(expr, BinaryOpExpr):
            left = self._ast_to_sympy_impl(expr.left)
            right = self._ast_to_sympy_impl(expr.right)
            
            ops = {
                "+": lambda l, r: l + r,
                "-": lambda l, r: l - r,
                "*": lambda l, r: l * r,
                "/": lambda l, r: l / r,
                "^": lambda l, r: l ** r,
            }
            
            if expr.operator in ops:
                return ops[expr.operator](left, right)
            else:
                raise ValueError(f"Unknown operator: {expr.operator}")
                
        elif isinstance(expr, UnaryOpExpr):
            operand = self._ast_to_sympy_impl(expr.operand)
            if expr.operator == "-":
                return -operand
            elif expr.operator == "+":
                return operand
            else:
                raise ValueError(f"Unknown unary operator: {expr.operator}")
        
        elif isinstance(expr, FractionExpr):
            num = self._ast_to_sympy_impl(expr.numerator)
            denom = self._ast_to_sympy_impl(expr.denominator)
            return num / denom

        elif isinstance(expr, DerivativeVarExpr):
            if expr.order == 1:
                return self.get_symbol(f"{expr.var}_dot")
            elif expr.order == 2:
                return self.get_symbol(f"{expr.var}_ddot")
            else:
                raise ValueError(f"Derivative order {expr.order} not supported")
                
        elif isinstance(expr, DerivativeExpr):
            inner = self._ast_to_sympy_impl(expr.expr)
            var = self.get_symbol(expr.var)
            
            if expr.partial:
                return sp.diff(inner, var, expr.order)
            else:
                if expr.var == "t":
                    return sp.diff(inner, self.time_symbol, expr.order)
                else:
                    return sp.diff(inner, var, expr.order)
                    
        elif isinstance(expr, FunctionCallExpr):
            args = [self._ast_to_sympy_impl(arg) for arg in expr.args]
            
            builtin_funcs = {
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "exp": sp.exp, "log": sp.log, "ln": sp.log,
                "sqrt": sp.sqrt, "sinh": sp.sinh, "cosh": sp.cosh,
                "tanh": sp.tanh, "arcsin": sp.asin, "arccos": sp.acos,
                "arctan": sp.atan, "abs": sp.Abs,
            }
            
            if expr.name in builtin_funcs:
                return builtin_funcs[expr.name](*args)
            else:
                func = self.get_function(expr.name)
                return func(*args)
                
        elif isinstance(expr, VectorExpr):
            return sp.Matrix([self._ast_to_sympy_impl(comp) for comp in expr.components])
            
        elif isinstance(expr, VectorOpExpr):
            if expr.operation == "grad":
                if expr.left:
                    inner = self._ast_to_sympy_impl(expr.left)
                    vars_list = [self.get_symbol(v) for v in ['x', 'y', 'z']]
                    return sp.Matrix([sp.diff(inner, var) for var in vars_list])
                else:
                    return self.get_symbol('nabla')
            elif expr.operation == "dot":
                left_vec = self._ast_to_sympy_impl(expr.left)
                right_vec = self._ast_to_sympy_impl(expr.right)
                if isinstance(left_vec, sp.Matrix) and isinstance(right_vec, sp.Matrix):
                    return left_vec.dot(right_vec)
                else:
                    return left_vec * right_vec
            elif expr.operation == "cross":
                left_vec = self._ast_to_sympy_impl(expr.left)
                right_vec = self._ast_to_sympy_impl(expr.right)
                if isinstance(left_vec, sp.Matrix) and isinstance(right_vec, sp.Matrix):
                    return left_vec.cross(right_vec)
                else:
                    raise ValueError("Cross product requires vector arguments")
            elif expr.operation == "magnitude":
                vec = self._ast_to_sympy_impl(expr.left)
                if isinstance(vec, sp.Matrix):
                    return sp.sqrt(vec.dot(vec))
                else:
                    return sp.Abs(vec)
                    
        else:
            raise ValueError(f"Cannot convert {type(expr).__name__} to SymPy")

    @profile_function
    def derive_equations_of_motion(self, lagrangian: sp.Expr, 
                                   coordinates: List[str]) -> List[sp.Expr]:
        """
        Derive Euler-Lagrange equations from Lagrangian
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            List of equations of motion
            
        Note:
            The Euler-Lagrange equations are:
            d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = 0
            
            For coupled systems (e.g., double pendulum), ALL coordinates must
            be treated as functions of time simultaneously to correctly compute
            cross-coupling terms like ∂²θ₂/∂t² appearing in the θ₁ equation.
        """
        logger.info(f"Deriving equations of motion for {len(coordinates)} coordinates")
        
        # CRITICAL FIX: Create functions and symbols for ALL coordinates at once
        # This ensures cross-coupling terms are correctly derived
        coord_funcs = {}  # q -> q(t)
        coord_syms = {}   # q -> symbol
        coord_dots = {}   # q -> q_dot symbol
        coord_ddots = {}  # q -> q_ddot symbol
        
        for q in coordinates:
            coord_syms[q] = self.get_symbol(q)
            coord_dots[q] = self.get_symbol(f"{q}_dot")
            coord_ddots[q] = self.get_symbol(f"{q}_ddot")
            coord_funcs[q] = sp.Function(q)(self.time_symbol)
        
        # Substitute ALL coordinates and velocities as time-functions SIMULTANEOUSLY
        # This is critical for coupled systems where cos(theta1 - theta2) involves both
        L_with_funcs = lagrangian
        for q in coordinates:
            L_with_funcs = L_with_funcs.subs(coord_syms[q], coord_funcs[q])
        for q in coordinates:
            L_with_funcs = L_with_funcs.subs(coord_dots[q], sp.diff(coord_funcs[q], self.time_symbol))
        
        equations = []
        
        for q in coordinates:
            logger.debug(f"Processing coordinate: {q}")
            
            q_func = coord_funcs[q]
            dq_dt = sp.diff(q_func, self.time_symbol)
            d2q_dt2 = sp.diff(q_func, self.time_symbol, 2)
            
            # Euler-Lagrange: d/dt(∂L/∂q̇) - ∂L/∂q = 0
            dL_dq_dot = sp.diff(L_with_funcs, dq_dt)
            d_dt_dL_dq_dot = sp.diff(dL_dq_dot, self.time_symbol)
            dL_dq = sp.diff(L_with_funcs, q_func)
            
            equation = d_dt_dL_dq_dot - dL_dq
            
            # Back-substitute ALL coordinates' derivatives and functions
            # Order matters: most specific (second derivatives) first
            for coord in coordinates:
                d2coord_dt2 = sp.diff(coord_funcs[coord], self.time_symbol, 2)
                equation = equation.subs(d2coord_dt2, coord_ddots[coord])
            
            for coord in coordinates:
                dcoord_dt = sp.diff(coord_funcs[coord], self.time_symbol)
                equation = equation.subs(dcoord_dt, coord_dots[coord])
            
            for coord in coordinates:
                equation = equation.subs(coord_funcs[coord], coord_syms[coord])
            
            # Handle any remaining Derivative objects by pattern matching
            for term in equation.atoms(sp.Derivative):
                for coord in coordinates:
                    if hasattr(term, 'order'):
                        order = term.order if isinstance(term.order, int) else len(term.variables)
                    else:
                        order = len(term.variables) if hasattr(term, 'variables') else 0
                    
                    if order >= 2 and term.has(self.time_symbol):
                        try:
                            if hasattr(term.expr, 'func') and str(term.expr.func) == coord:
                                equation = equation.subs(term, coord_ddots[coord])
                        except:
                            if str(term).startswith(f"Derivative({coord}"):
                                equation = equation.subs(term, coord_ddots[coord])
                    elif order == 1 and term.has(self.time_symbol):
                        try:
                            if hasattr(term.expr, 'func') and str(term.expr.func) == coord:
                                equation = equation.subs(term, coord_dots[coord])
                        except:
                            if str(term).startswith(f"Derivative({coord}"):
                                equation = equation.subs(term, coord_dots[coord])

            # Simplify with timeout (after substitution to preserve acceleration terms)
            try:
                if config.simplification_timeout > 0:
                    with timeout(config.simplification_timeout):
                        equation = sp.simplify(equation)
                else:
                    equation = sp.simplify(equation)
            except TimeoutError:
                logger.warning(f"Simplification timeout for {q}, using unsimplified equation")
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Simplification error for {q}: {e}, using unsimplified equation")
            
            # Verify acceleration terms are present after simplification
            missing_accels = [coord for coord in coordinates if not equation.has(coord_ddots[coord])]
            if missing_accels:
                # For some simple systems, not all accelerations appear in all equations
                logger.debug(f"Accelerations {missing_accels} not in equation for {q}")
            
            equations.append(equation)
            logger.debug(f"Equation for {q}: {equation}")
            
        return equations

    def derive_equations_with_constraints(self, lagrangian: sp.Expr,
                                         coordinates: List[str],
                                         constraints: List[sp.Expr]) -> Tuple[List[sp.Expr], List[str]]:
        """
        Derive equations with holonomic constraints using Lagrange multipliers
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            constraints: List of constraint expressions
            
        Returns:
            Tuple of (augmented equations, extended coordinates including lambdas)
        """
        logger.info(f"Deriving constrained equations with {len(constraints)} constraints")
        
        # Create Lagrange multipliers
        lambdas = [self.get_symbol(f'lambda_{i}') for i in range(len(constraints))]
        
        # Augmented Lagrangian: L' = L + Σ(λ_i * g_i)
        L_augmented = lagrangian
        for lam, constraint in zip(lambdas, constraints):
            L_augmented += lam * constraint
        
        logger.debug(f"Augmented Lagrangian: {L_augmented}")
        
        # Derive augmented equations
        equations = self.derive_equations_of_motion(L_augmented, coordinates)
        
        # Add time derivatives of constraints as additional equations
        constraint_eqs = []
        for constraint in constraints:
            # First time derivative: dg/dt = 0
            constraint_dot = sp.diff(constraint, self.time_symbol)
            constraint_eqs.append(constraint_dot)
        
        extended_coords = coordinates + [str(lam) for lam in lambdas]
        all_equations = equations + constraint_eqs
        
        logger.info(f"Generated {len(all_equations)} constrained equations")
        return all_equations, extended_coords

    @profile_function
    def derive_hamiltonian_equations(self, hamiltonian: sp.Expr, 
                                    coordinates: List[str]) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """
        Derive Hamilton's equations from Hamiltonian
        
        Hamilton's equations:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        
        Args:
            hamiltonian: Hamiltonian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Tuple of (q_dot equations, p_dot equations)
        """
        logger.info(f"Deriving Hamiltonian equations for {len(coordinates)} coordinates")
        q_dot_equations = []
        p_dot_equations = []
        
        for q in coordinates:
            q_sym = self.get_symbol(q)
            p_sym = self.get_symbol(f"p_{q}")
            
            # dq/dt = ∂H/∂p
            q_dot = sp.diff(hamiltonian, p_sym)
            try:
                if config.simplification_timeout > 0:
                    with timeout(config.simplification_timeout):
                        q_dot = sp.simplify(q_dot)
                else:
                    q_dot = sp.simplify(q_dot)
            except TimeoutError:
                logger.debug(f"Simplification timeout for d{q}/dt, using unsimplified")
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Simplification error for d{q}/dt: {e}")
            q_dot_equations.append(q_dot)
            
            # dp/dt = -∂H/∂q
            p_dot = -sp.diff(hamiltonian, q_sym)
            try:
                if config.simplification_timeout > 0:
                    with timeout(config.simplification_timeout):
                        p_dot = sp.simplify(p_dot)
                else:
                    p_dot = sp.simplify(p_dot)
            except TimeoutError:
                logger.debug(f"Simplification timeout for dp_{q}/dt, using unsimplified")
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Simplification error for dp_{q}/dt: {e}")
            p_dot_equations.append(p_dot)
            
            logger.debug(f"Hamilton equations for {q}:")
            logger.debug(f"  d{q}/dt = {q_dot}")
            logger.debug(f"  dp_{q}/dt = {p_dot}")
            
        return q_dot_equations, p_dot_equations

    @profile_function
    def lagrangian_to_hamiltonian(self, lagrangian: sp.Expr, 
                                 coordinates: List[str]) -> sp.Expr:
        """
        Convert Lagrangian to Hamiltonian via Legendre transform
        
        H = Σ(p_i * q̇_i) - L
        where p_i = ∂L/∂q̇_i
        
        Args:
            lagrangian: Lagrangian expression
            coordinates: List of generalized coordinates
            
        Returns:
            Hamiltonian expression
        """
        logger.info("Converting Lagrangian to Hamiltonian")
        hamiltonian = sp.S.Zero
        
        for q in coordinates:
            q_dot_sym = self.get_symbol(f"{q}_dot")
            p_sym = self.get_symbol(f"p_{q}")
            
            # Calculate conjugate momentum p = ∂L/∂q̇
            momentum_def = sp.diff(lagrangian, q_dot_sym)
            logger.debug(f"Momentum for {q}: p_{q} = {momentum_def}")
            
            # Solve for q̇ in terms of p
            try:
                q_dot_solution = sp.solve(momentum_def - p_sym, q_dot_sym)
                if q_dot_solution:
                    q_dot_expr = q_dot_solution[0]
                    hamiltonian += p_sym * q_dot_expr
                    logger.debug(f"Solved for {q}_dot: {q_dot_expr}")
            except (ValueError, TypeError, NotImplementedError) as e:
                logger.warning(f"Could not solve for {q}_dot: {e}, using symbolic form")
                hamiltonian += p_sym * q_dot_sym
        
        # H = Σ(p_i * q̇_i) - L
        hamiltonian = hamiltonian - lagrangian
        
        # Substitute momentum definitions
        for q in coordinates:
            q_dot_sym = self.get_symbol(f"{q}_dot")
            p_sym = self.get_symbol(f"p_{q}")
            momentum_def = sp.diff(lagrangian, q_dot_sym)
            
            try:
                q_dot_solution = sp.solve(momentum_def - p_sym, q_dot_sym)
                if q_dot_solution:
                    hamiltonian = hamiltonian.subs(q_dot_sym, q_dot_solution[0])
            except (ValueError, TypeError, NotImplementedError):
                logger.debug(f"Could not substitute {q}_dot in Hamiltonian")
        
        # Simplify with timeout
        try:
            if config.simplification_timeout > 0:
                with timeout(config.simplification_timeout):
                    hamiltonian = sp.simplify(hamiltonian)
            else:
                hamiltonian = sp.simplify(hamiltonian)
        except TimeoutError:
            logger.warning("Hamiltonian simplification timeout, using unsimplified form")
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Hamiltonian simplification error: {e}, using unsimplified form")
        
        logger.info(f"Hamiltonian: {hamiltonian}")
        return hamiltonian

    def solve_for_accelerations(self, equations: List[sp.Expr], 
                               coordinates: List[str]) -> Dict[str, sp.Expr]:
        """
        Solve equations of motion for accelerations SIMULTANEOUSLY.
        
        For coupled systems like double pendulum, accelerations are interdependent:
        M * [q1_ddot, q2_ddot, ...]^T = F
        
        This function:
        1. Substitutes all derivative notations with symbols
        2. Extracts the mass matrix M and force vector F
        3. Solves the linear system M*a = F simultaneously
        4. Returns simplified acceleration expressions
        
        This is CRITICAL for coupled systems where accelerations appear in
        each other's equations.
        """
        logger.info("Solving for accelerations (Simultaneous Coupled System)")
        
        n = len(coordinates)
        if n == 0:
            return {}
        
        # --- Step 1: Create acceleration symbols ---
        accel_syms = []
        for q in coordinates:
            accel_key = f"{q}_ddot"
            accel_syms.append(self.get_symbol(accel_key))
        
        # --- Step 2: Substitute all derivatives with symbols ---
        processed_eqs = []
        for i, eq in enumerate(equations):
            # Clean up the equation - replace Derivative objects with symbols
            eq_clean = eq
            
            # Replace ALL second derivatives (not just for current coordinate)
            for j, q in enumerate(coordinates):
                accel_sym = accel_syms[j]
                vel_sym = self.get_symbol(f"{q}_dot")
                pos_sym = self.get_symbol(q)
                
                # Try pattern matching for derivatives
                try:
                    q_func = sp.Function(q)(self.time_symbol)
                    d2q_dt2 = sp.diff(q_func, self.time_symbol, 2)
                    dq_dt = sp.diff(q_func, self.time_symbol)
                    
                    eq_clean = eq_clean.subs(d2q_dt2, accel_sym)
                    eq_clean = eq_clean.subs(dq_dt, vel_sym)
                    eq_clean = eq_clean.subs(q_func, pos_sym)
                except Exception as e:
                    logger.debug(f"Pattern substitution warning for {q}: {e}")
                
                # Fallback: Iterate through Derivative atoms
                for term in eq_clean.atoms(sp.Derivative):
                    try:
                        term_str = str(term)
                        if f"Derivative({q}(t), (t, 2))" in term_str or f"Derivative({q}(t), t, t)" in term_str:
                            eq_clean = eq_clean.subs(term, accel_sym)
                        elif f"Derivative({q}(t), t)" in term_str:
                            eq_clean = eq_clean.subs(term, vel_sym)
                    except:
                        pass
                
                # Also clean up raw Function objects
                for term in eq_clean.atoms(sp.Function):
                    try:
                        if str(term.func) == q and term.args == (self.time_symbol,):
                            eq_clean = eq_clean.subs(term, pos_sym)
                    except:
                        pass
            
            processed_eqs.append(sp.expand(eq_clean))
            logger.debug(f"Processed equation {i}: {eq_clean}")
        
        # --- Step 3: For single coordinate, use direct extraction ---
        if n == 1:
            accel_sym = accel_syms[0]
            eq = processed_eqs[0]
            
            A = sp.diff(eq, accel_sym)
            B = eq.subs(accel_sym, 0)
            
            if A != 0:
                sol = sp.simplify(-B / A)
                accel_key = f"{coordinates[0]}_ddot"
                logger.info(f"Solved {accel_key} (single coordinate)")
                return {accel_key: sol}
            else:
                logger.error("Could not solve single-coordinate equation")
                return {f"{coordinates[0]}_ddot": sp.S.Zero}
        
        # --- Step 4: For multiple coordinates, solve SIMULTANEOUSLY ---
        # The equations are linear in accelerations: M * a + F = 0
        # Extract M (mass matrix) and F (force vector)
        
        try:
            # Use sympy's linear solver for the system
            solutions = sp.solve(processed_eqs, accel_syms, dict=True)
            
            if solutions:
                sol_dict = solutions[0] if isinstance(solutions, list) else solutions
                accelerations = {}
                for j, q in enumerate(coordinates):
                    accel_key = f"{q}_ddot"
                    accel_sym = accel_syms[j]
                    if accel_sym in sol_dict:
                        accelerations[accel_key] = sp.simplify(sol_dict[accel_sym])
                        logger.info(f"Solved {accel_key} via simultaneous solution")
                    else:
                        logger.warning(f"No solution found for {accel_key}")
                        accelerations[accel_key] = sp.S.Zero
                
                return accelerations
        except Exception as e:
            logger.warning(f"sp.solve failed: {e}, trying matrix method")
        
        # Fallback: Manual matrix extraction and solve
        M = sp.zeros(n, n)
        F = sp.zeros(n, 1)
        
        for i in range(n):
            eq = processed_eqs[i]
            for j in range(n):
                # Mass matrix entry M[i,j] = coefficient of accel_j in eq_i
                M[i, j] = sp.diff(eq, accel_syms[j])
            # Force vector F[i] = eq with all accelerations set to 0
            F[i, 0] = -eq.subs([(a, 0) for a in accel_syms])
        
        logger.debug(f"Mass matrix M:\n{M}")
        logger.debug(f"Force vector F:\n{F}")
        
        # Solve M * a = F
        try:
            if M.det() != 0:
                a_vec = M.solve(F)
                accelerations = {}
                for j, q in enumerate(coordinates):
                    accel_key = f"{q}_ddot"
                    accelerations[accel_key] = sp.simplify(a_vec[j])
                    logger.info(f"Solved {accel_key} via matrix inversion")
                return accelerations
            else:
                logger.error("Mass matrix is singular!")
        except Exception as e:
            logger.error(f"Matrix solve failed: {e}")
        
        # Last resort: return zeros with warning
        logger.error("CRITICAL: Could not solve for accelerations!")
        accelerations = {}
        for q in coordinates:
            accelerations[f"{q}_ddot"] = sp.S.Zero
        return accelerations

