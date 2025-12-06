"""
Symbolic computation engine for MechanicsDSL
"""
import sympy as sp
from typing import List, Dict, Tuple

from .utils import logger, config, profile_function, timeout, TimeoutError, LRUCache, _perf_monitor
from .parser import (
    Expression, NumberExpr, IdentExpr, GreekLetterExpr, BinaryOpExpr,
    UnaryOpExpr, FractionExpr, DerivativeVarExpr, DerivativeExpr,
    FunctionCallExpr, VectorExpr, VectorOpExpr
)

class SymbolicEngine:
    """Enhanced symbolic mathematics engine with advanced caching and performance monitoring"""
    
    def __init__(self):
        self.sp = sp
        self.symbol_map: Dict[str, sp.Symbol] = {}
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
        return self.symbol_map[name]

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
        """
        logger.info(f"Deriving equations of motion for {len(coordinates)} coordinates")
        equations = []
        
        for q in coordinates:
            logger.debug(f"Processing coordinate: {q}")
            q_sym = self.get_symbol(q)
            q_dot_sym = self.get_symbol(f"{q}_dot")
            q_ddot_sym = self.get_symbol(f"{q}_ddot")

            q_func = sp.Function(q)(self.time_symbol)

            L_with_funcs = lagrangian.subs(q_sym, q_func)
            L_with_funcs = L_with_funcs.subs(q_dot_sym, sp.diff(q_func, self.time_symbol))

            dL_dq_dot = sp.diff(L_with_funcs, sp.diff(q_func, self.time_symbol))
            d_dt_dL_dq_dot = sp.diff(dL_dq_dot, self.time_symbol)
            dL_dq = sp.diff(L_with_funcs, q_func)

            equation = d_dt_dL_dq_dot - dL_dq

            # CRITICAL: Replace derivatives BEFORE simplification to preserve structure
            # Replace second derivative first (most specific)
            d2q_dt2 = sp.diff(q_func, self.time_symbol, 2)
            equation = equation.subs(d2q_dt2, q_ddot_sym)
            
            # Replace first derivative
            dq_dt = sp.diff(q_func, self.time_symbol)
            equation = equation.subs(dq_dt, q_dot_sym)
            
            # Replace function
            equation = equation.subs(q_func, q_sym)
            
            # Also try to replace any remaining Derivative objects by pattern matching
            for term in equation.atoms(sp.Derivative):
                if term.order == 2 and term.has(self.time_symbol):
                    try:
                        if hasattr(term.expr, 'func') and str(term.expr.func) == q:
                            equation = equation.subs(term, q_ddot_sym)
                    except:
                        if str(term).startswith(f"Derivative({q}"):
                            equation = equation.subs(term, q_ddot_sym)
                elif term.order == 1 and term.has(self.time_symbol):
                    try:
                        if hasattr(term.expr, 'func') and str(term.expr.func) == q:
                            equation = equation.subs(term, q_dot_sym)
                    except:
                        if str(term).startswith(f"Derivative({q}"):
                            equation = equation.subs(term, q_dot_sym)

            # Simplify with timeout (after substitution to preserve acceleration term)
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
            
            # Verify acceleration term is present after simplification
            if not equation.has(q_ddot_sym):
                logger.warning(f"Acceleration term {q_ddot_sym} missing after simplification for {q}, equation: {equation}")
            
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
        Solve equations of motion using Atom Iteration + Linear Extraction.
        This ignores object identity issues by scanning the equation tree
        for any second derivatives and manually forcing them to be symbols.
        """
        logger.info("Solving for accelerations (Search & Destroy Mode)")
        accelerations = {}
        
        for i, q in enumerate(coordinates):
            accel_key = f"{q}_ddot"
            accel_sym = self.get_symbol(accel_key)
            
            # The Equation
            eq = equations[i]
            
            # Debug: log equation structure
            logger.debug(f"Equation for {q} (before processing): {eq}")
            
            # Check if acceleration symbol is already in the equation
            has_accel = False
            try:
                has_accel = eq.has(accel_sym) or str(accel_sym) in str(eq) or accel_sym in eq.free_symbols
            except:
                has_accel = str(accel_sym) in str(eq)
            
            if has_accel:
                # Equation already has acceleration symbol - use linear extraction directly
                eq_expanded = sp.expand(eq)
                A = sp.diff(eq_expanded, accel_sym)
                B = eq_expanded.subs(accel_sym, 0)
                if A != 0:
                    sol = -B / A
                    accelerations[accel_key] = sp.simplify(sol)
                    logger.info(f"Solved {accel_key} via direct symbol extraction")
                    continue
                else:
                    logger.warning(f"Acceleration symbol found but coefficient is zero for {accel_key}")
            
            # --- STEP 1: SEARCH AND DESTROY DERIVATIVES ---
            # Iterate through every atomic part of the equation
            # If it is a Derivative of order 2 matching our coordinate, replace it.
            
            # Approach 1: Direct Derivative objects
            for term in eq.atoms(sp.Derivative):
                # Check if it is a 2nd derivative with respect to time
                if hasattr(term, 'order') and term.order == 2:
                    if term.has(self.time_symbol):
                        # Check if the function name matches our coordinate
                        try:
                            if hasattr(term.expr, 'func') and str(term.expr.func) == q:
                                eq = eq.subs(term, accel_sym)
                        except:
                            # Try alternative matching
                            if str(term).startswith(f"Derivative({q}"):
                                eq = eq.subs(term, accel_sym)
            
            # Approach 2: Try to find derivatives by pattern matching
            try:
                q_func = sp.Function(q)(self.time_symbol)
                d2q_dt2 = sp.diff(q_func, self.time_symbol, 2)
                if eq.has(d2q_dt2):
                    eq = eq.subs(d2q_dt2, accel_sym)
            except:
                pass
            
            # Also clean up 1st derivatives (velocity) just in case
            vel_sym = self.get_symbol(f"{q}_dot")
            for term in eq.atoms(sp.Derivative):
                if hasattr(term, 'order') and term.order == 1:
                    if term.has(self.time_symbol):
                        try:
                            if hasattr(term.expr, 'func') and str(term.expr.func) == q:
                                eq = eq.subs(term, vel_sym)
                        except:
                            if str(term).startswith(f"Derivative({q}"):
                                eq = eq.subs(term, vel_sym)
            
            # Try pattern matching for first derivative
            try:
                q_func = sp.Function(q)(self.time_symbol)
                dq_dt = sp.diff(q_func, self.time_symbol)
                if eq.has(dq_dt):
                    eq = eq.subs(dq_dt, vel_sym)
            except:
                pass
                        
            # Clean up raw functions (position)
            pos_sym = self.get_symbol(q)
            for term in eq.atoms(sp.Function):
                try:
                    if hasattr(term, 'func') and str(term.func) == q:
                        eq = eq.subs(term, pos_sym)
                except:
                    pass

            # --- STEP 2: LINEAR EXTRACTION ---
            # Now the equation is guaranteed to be algebraic: A * accel + B = 0
            eq_expanded = sp.expand(eq)
            
            # Differentiating by the symbol isolates the mass matrix term (A)
            A = sp.diff(eq_expanded, accel_sym)
            
            # Setting symbol to 0 isolates the rest (B)
            B = eq_expanded.subs(accel_sym, 0)
            
            if A != 0:
                # Ax + B = 0  ->  x = -B/A
                sol = -B / A
                accelerations[accel_key] = sp.simplify(sol)
                logger.info(f"Solved {accel_key} via Search & Destroy")
            else:
                logger.error(f"CRITICAL: Acceleration term {accel_key} not found in equation!")
                logger.error(f"Equation (expanded): {eq_expanded}")
                logger.error(f"Looking for symbol: {accel_sym}")
                # Try one more time: check if the symbol name appears as a string
                if accel_key in str(eq):
                    logger.warning(f"Symbol name '{accel_key}' found in string representation, trying alternative extraction")
                    # Try to solve algebraically
                    try:
                        sol = sp.solve(eq, accel_sym)
                        if sol:
                            accelerations[accel_key] = sp.simplify(sol[0])
                            logger.info(f"Solved {accel_key} via algebraic solve")
                            continue
                    except:
                        pass
                accelerations[accel_key] = sp.S.Zero
                
        return accelerations
