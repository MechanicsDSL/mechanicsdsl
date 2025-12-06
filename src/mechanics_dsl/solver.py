"""
Numerical simulation engine for MechanicsDSL
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from typing import List, Dict, Optional, Tuple, Callable

from .utils import (
    logger, config, profile_function, timeout, TimeoutError,
    validate_array_safe, safe_float_conversion, safe_array_access,
    validate_finite, validate_time_span, _perf_monitor
)
from .symbolic import SymbolicEngine

class NumericalSimulator:
    """Enhanced numerical simulator with better stability and diagnostics"""
    
    def __init__(self, symbolic_engine: SymbolicEngine):
        self.symbolic = symbolic_engine
        self.equations: Dict[str, Callable] = {}
        self.parameters: Dict[str, float] = {}
        self.initial_conditions: Dict[str, float] = {}
        self.constraints: List[sp.Expr] = []
        self.state_vars: List[str] = []
        self.coordinates: List[str] = []
        self.use_hamiltonian: bool = False
        self.hamiltonian_equations: Optional[Dict[str, List[Tuple]]] = None

    def set_parameters(self, params: Dict[str, float]):
        """Set physical parameters"""
        self.parameters.update(params)
        logger.debug(f"Set parameters: {params}")

    def set_initial_conditions(self, conditions: Dict[str, float]):
        """Set initial conditions"""
        self.initial_conditions.update(conditions)
        logger.debug(f"Set initial conditions: {conditions}")
    
    def add_constraint(self, constraint_expr: sp.Expr):
        """Add a constraint equation"""
        self.constraints.append(constraint_expr)
        logger.debug(f"Added constraint: {constraint_expr}")

    @profile_function
    def compile_equations(self, accelerations: Dict[str, sp.Expr], coordinates: List[str]):
        """Compile symbolic equations to numerical functions"""
        
        logger.info(f"Compiling equations for {len(coordinates)} coordinates")
        
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"{q}_dot"])
            
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        compiled_equations = {}
        
        for q in coordinates:
            accel_key = f"{q}_ddot"
            if accel_key in accelerations:
                eq = accelerations[accel_key].subs(param_subs)
                
                # Attempt simplification with timeout
                try:
                    if config.simplification_timeout > 0:
                        with timeout(config.simplification_timeout):
                            eq = sp.simplify(eq)
                    else:
                        eq = sp.simplify(eq)
                except TimeoutError:
                    logger.debug(f"Simplification timeout for {accel_key}, skipping")
                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Simplification error for {accel_key}: {e}, skipping")

                eq = self._replace_derivatives(eq, coordinates)
                
                free_symbols = eq.free_symbols
                ordered_symbols = []
                symbol_indices = []
                has_time = False
                
                # Check if equation depends on time
                time_sym = self.symbolic.time_symbol
                if time_sym in free_symbols:
                    has_time = True
                    ordered_symbols.append(time_sym)
                    symbol_indices.append(-1)  # Special index for time
                
                for i, var_name in enumerate(state_vars):
                    sym = self.symbolic.get_symbol(var_name)
                    if sym in free_symbols:
                        ordered_symbols.append(sym)
                        symbol_indices.append(i)
                
                if ordered_symbols:
                    try:
                        func = sp.lambdify(ordered_symbols, eq, modules=['numpy', 'math'])
                        
                        def make_wrapper(func, indices, has_time_flag):
                            # Force capture by value
                            def wrapper(*args_with_time, _func=func, _indices=indices, _has_time=has_time_flag):
                                try:
                                    # FIX: Always split time from state vector
                                    # args_with_time comes from solve_ivp, so it is ALWAYS (t, y0, y1, ...)
                                    if len(args_with_time) < 1:
                                        return 0.0
                                    
                                    t_val = float(args_with_time[0])
                                    state_vector = args_with_time[1:]
                                    
                                    func_args = []
                                    for idx in _indices:
                                        if idx == -1:  # Time index
                                            func_args.append(t_val)
                                        elif idx >= 0:  # State variable index
                                            # Safe access to state vector
                                            if idx < len(state_vector):
                                                func_args.append(state_vector[idx])
                                            else:
                                                func_args.append(0.0)
                                    
                                    # Execute the lambda function
                                    if len(func_args) == len(_indices):
                                        result = _func(*func_args)
                                        # Ensure we return a float
                                        if isinstance(result, np.ndarray):
                                            result = float(result.item()) if result.size == 1 else float(result.flat[0])
                                        return float(result) if np.isfinite(result) else 0.0
                                    return 0.0
                                except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
                                    logger.debug(f"Evaluation error: {e}")
                                    return 0.0
                            return wrapper
                        
                        compiled_equations[accel_key] = make_wrapper(func, symbol_indices, has_time)
                        logger.debug(f"Compiled {accel_key}")
                        
                    except (ValueError, TypeError, AttributeError, 
                            NotImplementedError, SyntaxError) as e:
                        logger.error(f"Compilation failed for {accel_key}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0
                    except Exception as e:
                        logger.error(f"Unexpected compilation error for {accel_key}: {type(e).__name__}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0
                else:
                    try:
                        const_value = float(sp.N(eq))
                        compiled_equations[accel_key] = lambda *args: const_value
                        logger.debug(f"{accel_key} is constant: {const_value}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not evaluate constant for {accel_key}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0

        self.equations = compiled_equations
        self.state_vars = state_vars
        self.coordinates = coordinates
        logger.info("Equation compilation complete")

    def compile_hamiltonian_equations(self, q_dots: List[sp.Expr], p_dots: List[sp.Expr], 
                                     coordinates: List[str]):
        """Compile Hamiltonian equations"""
        logger.info("Compiling Hamiltonian equations")
        self.use_hamiltonian = True
        
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"p_{q}"])
        
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        
        self.hamiltonian_equations = {
            'q_dots': [],
            'p_dots': []
        }
        
        for i, q in enumerate(coordinates):
            q_dot_eq = q_dots[i].subs(param_subs)
            p_dot_eq = p_dots[i].subs(param_subs)
            
            # Compile q_dot
            free_syms = q_dot_eq.free_symbols
            ordered_syms = []
            indices = []
            for j, var_name in enumerate(state_vars):
                sym = self.symbolic.get_symbol(var_name)
                if sym in free_syms:
                    ordered_syms.append(sym)
                    indices.append(j)
            
            if ordered_syms:
                func = sp.lambdify(ordered_syms, q_dot_eq, modules=['numpy', 'math'])
                self.hamiltonian_equations['q_dots'].append((func, indices))
            else:
                const_val = float(sp.N(q_dot_eq))
                self.hamiltonian_equations['q_dots'].append((lambda *args, v=const_val: v, []))
            
            # Compile p_dot
            free_syms = p_dot_eq.free_symbols
            ordered_syms = []
            indices = []
            for j, var_name in enumerate(state_vars):
                sym = self.symbolic.get_symbol(var_name)
                if sym in free_syms:
                    ordered_syms.append(sym)
                    indices.append(j)
            
            if ordered_syms:
                func = sp.lambdify(ordered_syms, p_dot_eq, modules=['numpy', 'math'])
                self.hamiltonian_equations['p_dots'].append((func, indices))
            else:
                const_val = float(sp.N(p_dot_eq))
                self.hamiltonian_equations['p_dots'].append((lambda *args, v=const_val: v, []))
        
        self.state_vars = state_vars
        self.coordinates = coordinates
        logger.info("Hamiltonian compilation complete")

    def _replace_derivatives(self, expr: sp.Expr, coordinates: List[str]) -> sp.Expr:
        """Replace Derivative objects with corresponding symbols"""
        derivs = list(expr.atoms(sp.Derivative))
        for d in derivs:
            try:
                base = d.args[0]
                order = 1
                if len(d.args) >= 2:
                    arg2 = d.args[1]
                    if isinstance(arg2, tuple) and len(arg2) >= 2:
                        order = int(arg2[1])
                
                base_name = str(base)
                if base_name in coordinates:
                    if order == 1:
                        repl = self.symbolic.get_symbol(f"{base_name}_dot")
                    elif order == 2:
                        repl = self.symbolic.get_symbol(f"{base_name}_ddot")
                    else:
                        continue
                    expr = expr.xreplace({d: repl})
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug(f"Could not replace derivative: {e}")
                continue
        return expr

    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE system for numerical integration with comprehensive bounds checking and validation.
        
        Args:
            t: Current time
            y: State vector
            
        Returns:
            Derivative vector dydt
        """
        # Comprehensive input validation
        if not isinstance(t, (int, float)) or not np.isfinite(t):
            logger.error(f"equations_of_motion: invalid time t={t}, using 0.0")
            t = 0.0
        
        if y is None:
            logger.error("equations_of_motion: y is None, returning zeros")
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)
        
        if not isinstance(y, np.ndarray):
            logger.error(f"equations_of_motion: y is not numpy.ndarray, got {type(y).__name__}")
            try:
                y = np.array(y, dtype=float)
            except Exception as e:
                logger.error(f"equations_of_motion: cannot convert y to array: {e}")
                if self.coordinates:
                    return np.zeros(2 * len(self.coordinates))
                return np.zeros(1)
        
        if not validate_array_safe(y, "state_vector", min_size=1, check_finite=False):
            logger.warning("equations_of_motion: state vector validation failed, attempting recovery")
            # Try to fix non-finite values
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if self.use_hamiltonian:
            return self._hamiltonian_ode(t, y)
        
        # Validate expected size
        expected_size = 2 * len(self.coordinates) if self.coordinates else 1
        if y.size != expected_size:
            logger.warning(f"equations_of_motion: state vector size {y.size} != expected {expected_size}")
            # Try to pad or truncate
            if y.size < expected_size:
                y = np.pad(y, (0, expected_size - y.size), mode='constant', constant_values=0.0)
            else:
                y = y[:expected_size]
        
        try:
            dydt = np.zeros_like(y)

            # Position derivatives = velocities (with comprehensive bounds checking)
            for i in range(len(self.coordinates)):
                pos_idx = 2 * i
                vel_idx = 2 * i + 1

                if vel_idx < len(y) and pos_idx < len(dydt):
                    vel_value = safe_array_access(y, vel_idx, 0.0)
                    dydt[pos_idx] = vel_value
                elif pos_idx < len(dydt):
                    dydt[pos_idx] = 0.0

            for i, q in enumerate(self.coordinates):
                accel_key = f"{q}_ddot"
                vel_idx = 2 * i + 1

                if accel_key in self.equations and vel_idx < len(dydt):
                    try:
                        eq_func = self.equations.get(accel_key)
                        if eq_func is None:
                            logger.warning(f"equations_of_motion: equation function for {accel_key} is None")
                            dydt[vel_idx] = 0.0
                            continue

                        try:
                            # Pass time and state vector - wrapper handles whether time is needed
                            accel_value = eq_func(t, *y)
                            accel_value = safe_float_conversion(accel_value)
                            if np.isfinite(accel_value):
                                dydt[vel_idx] = accel_value
                            else:
                                dydt[vel_idx] = 0.0
                                logger.warning(f"Non-finite acceleration for {q} at t={t:.6f}")
                        except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                            dydt[vel_idx] = 0.0
                            logger.debug(f"Evaluation error for {q} at t={t:.6f}: {e}")
                    except Exception as e:
                        dydt[vel_idx] = 0.0
                        logger.error(f"Unexpected error evaluating {accel_key}: {e}", exc_info=True)
                elif vel_idx < len(dydt):
                    dydt[vel_idx] = 0.0

            # Final validation of result
            if not validate_array_safe(dydt, "dydt", check_finite=True):
                logger.warning("equations_of_motion: result validation failed, fixing non-finite values")
                dydt = np.nan_to_num(dydt, nan=0.0, posinf=1e10, neginf=-1e10)

            return dydt

        except Exception as e:
            logger.error(f"equations_of_motion: unexpected error: {e}", exc_info=True)
            # Return safe default
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)
    

    def _hamiltonian_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE system for Hamiltonian formulation with comprehensive validation.
        
        Args:
            t: Current time
            y: State vector
            
        Returns:
            Derivative vector dydt
        """
        # Input validation
        if not isinstance(t, (int, float)) or not np.isfinite(t):
            logger.error(f"_hamiltonian_ode: invalid time t={t}, using 0.0")
            t = 0.0
        
        if y is None:
            logger.error("_hamiltonian_ode: y is None")
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)
        
        if not isinstance(y, np.ndarray):
            logger.error(f"_hamiltonian_ode: y is not numpy.ndarray, got {type(y).__name__}")
            try:
                y = np.array(y, dtype=float)
            except Exception as e:
                logger.error(f"_hamiltonian_ode: cannot convert y to array: {e}")
                if self.coordinates:
                    return np.zeros(2 * len(self.coordinates))
                return np.zeros(1)
        
        if not validate_array_safe(y, "hamiltonian_state_vector", min_size=1, check_finite=False):
            logger.warning("_hamiltonian_ode: state vector validation failed, attempting recovery")
            y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
        
        try:
            dydt = np.zeros_like(y)

            if self.hamiltonian_equations is None:
                logger.error("_hamiltonian_ode: hamiltonian_equations is None, cannot compute ODE")
                return dydt

            if not isinstance(self.hamiltonian_equations, dict):
                logger.error(f"_hamiltonian_ode: hamiltonian_equations is not dict, got {type(self.hamiltonian_equations).__name__}")
                return dydt

            if 'q_dots' not in self.hamiltonian_equations or 'p_dots' not in self.hamiltonian_equations:
                logger.error("_hamiltonian_ode: hamiltonian_equations missing required keys")
                return dydt

            for i, q in enumerate(self.coordinates):
                if not isinstance(q, str):
                    logger.warning(f"_hamiltonian_ode: coordinate {i} is not string: {type(q).__name__}")
                    continue

                if i >= len(self.hamiltonian_equations['q_dots']):
                    logger.warning(f"_hamiltonian_ode: Index {i} out of range for q_dots (len={len(self.hamiltonian_equations['q_dots'])})")
                    continue

                try:
                    q_dot_data = self.hamiltonian_equations['q_dots'][i]
                    if not isinstance(q_dot_data, tuple) or len(q_dot_data) != 2:
                        logger.error(f"_hamiltonian_ode: invalid q_dot data structure at index {i}")
                        continue

                    func, indices = q_dot_data
                    if func is None:
                        logger.warning(f"_hamiltonian_ode: function is None for d{q}/dt")
                        continue

                    if not isinstance(indices, (list, tuple)):
                        logger.warning(f"_hamiltonian_ode: indices is not list/tuple for d{q}/dt")
                        indices = []

                    q_idx = 2 * i
                    if q_idx < len(dydt):
                        try:
                            args = [safe_array_access(y, j, 0.0) for j in indices if isinstance(j, int) and j >= 0]
                            if len(args) == len(indices):
                                result = func(*args)
                                dydt[q_idx] = safe_float_conversion(result)
                                if not np.isfinite(dydt[q_idx]):
                                    logger.warning(f"_hamiltonian_ode: non-finite d{q}/dt, setting to 0.0")
                                    dydt[q_idx] = 0.0
                            else:
                                dydt[q_idx] = 0.0
                                logger.warning(f"_hamiltonian_ode: Incomplete arguments for d{q}/dt (got {len(args)}, expected {len(indices)})")
                        except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                            dydt[q_idx] = 0.0
                            logger.debug(f"_hamiltonian_ode: Evaluation error for d{q}/dt: {e}")
                        except Exception as e:
                            dydt[q_idx] = 0.0
                            logger.error(f"_hamiltonian_ode: Unexpected error for d{q}/dt: {e}", exc_info=True)
                except (IndexError, TypeError, ValueError) as e:
                    logger.error(f"_hamiltonian_ode: Error accessing q_dots[{i}]: {e}")
                    continue

                if i >= len(self.hamiltonian_equations['p_dots']):
                    logger.warning(f"_hamiltonian_ode: Index {i} out of range for p_dots (len={len(self.hamiltonian_equations['p_dots'])})")
                    continue

                try:
                    p_dot_data = self.hamiltonian_equations['p_dots'][i]
                    if not isinstance(p_dot_data, tuple) or len(p_dot_data) != 2:
                        logger.error(f"_hamiltonian_ode: invalid p_dot data structure at index {i}")
                        continue

                    func, indices = p_dot_data
                    if func is None:
                        logger.warning(f"_hamiltonian_ode: function is None for dp_{q}/dt")
                        continue

                    if not isinstance(indices, (list, tuple)):
                        logger.warning(f"_hamiltonian_ode: indices is not list/tuple for dp_{q}/dt")
                        indices = []

                    p_idx = 2 * i + 1
                    if p_idx < len(dydt):
                        try:
                            args = [safe_array_access(y, j, 0.0) for j in indices if isinstance(j, int) and j >= 0]
                            if len(args) == len(indices):
                                result = func(*args)
                                dydt[p_idx] = safe_float_conversion(result)
                                if not np.isfinite(dydt[p_idx]):
                                    logger.warning(f"_hamiltonian_ode: non-finite dp_{q}/dt, setting to 0.0")
                                    dydt[p_idx] = 0.0
                            else:
                                dydt[p_idx] = 0.0
                                logger.warning(f"_hamiltonian_ode: Incomplete arguments for dp_{q}/dt (got {len(args)}, expected {len(indices)})")
                        except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                            dydt[p_idx] = 0.0
                            logger.debug(f"_hamiltonian_ode: Evaluation error for dp_{q}/dt: {e}")
                        except Exception as e:
                            dydt[p_idx] = 0.0
                            logger.error(f"_hamiltonian_ode: Unexpected error for dp_{q}/dt: {e}", exc_info=True)
                except (IndexError, TypeError, ValueError) as e:
                    logger.error(f"_hamiltonian_ode: Error accessing p_dots[{i}]: {e}")
                    continue

            if not validate_array_safe(dydt, "hamiltonian_dydt", check_finite=True):
                logger.warning("_hamiltonian_ode: result validation failed, fixing non-finite values")
                dydt = np.nan_to_num(dydt, nan=0.0, posinf=1e10, neginf=-1e10)

            return dydt

        except Exception as e:
            logger.error(f"_hamiltonian_ode: unexpected error: {e}", exc_info=True)
            if self.coordinates:
                return np.zeros(2 * len(self.coordinates))
            return np.zeros(1)

                        
    def _select_optimal_solver(self, t_span: Tuple[float, float], 
                              y0: np.ndarray) -> str:
        """v6.0: Intelligently select optimal solver based on system characteristics"""
        if not config.enable_adaptive_solver:
            # Default to LSODA for better stability in CI environments
            return 'LSODA'
        
        # Analyze system characteristics
        n_dof = len(self.coordinates)
        time_span = t_span[1] - t_span[0]
        
        # Large systems benefit from implicit methods
        if n_dof > 10:
            return 'LSODA'
        
        # Long time spans may need more stable methods
        if time_span > 100:
            return 'LSODA'
        
        # Small, simple systems can use fast explicit methods
        if n_dof <= 2 and time_span < 10:
            return 'RK45'
        
        # Default to LSODA for better stability (especially in CI)
        return 'LSODA'
                        
    @profile_function
    def simulate(self, t_span: Tuple[float, float], num_points: int = 1000,
                 method: str = None, rtol: float = None, atol: float = None,
                 detect_stiff: bool = True) -> dict:
        """
        Run numerical simulation with adaptive integration and diagnostics.
        
        Args:
            t_span: Time span (t_start, t_end) where t_start < t_end
            num_points: Number of output points (must be >= 2)
            method: Integration method ('RK45', 'LSODA', 'Radau', etc.)
            rtol: Relative tolerance (must be in (0, 1))
            atol: Absolute tolerance (must be positive)
            detect_stiff: Whether to detect stiff systems
            
        Returns:
            Dictionary with solution data and metadata, always contains 'success' key
            
        Raises:
            TypeError: If arguments have wrong types
            ValueError: If arguments are out of valid ranges
            
        Example:
            >>> solution = simulator.simulate((0, 10), num_points=1000)
            >>> if solution['success']:
            ...     t = solution['t']
            ...     y = solution['y']
        """
        # Comprehensive input validation
        validate_time_span(t_span)
        
        if not isinstance(num_points, int):
            raise TypeError(f"num_points must be int, got {type(num_points).__name__}")
        if num_points < 2:
            raise ValueError(f"num_points must be >= 2, got {num_points}")
        if num_points > 10_000_000:
            raise ValueError(f"num_points too large (>{10_000_000}), got {num_points}")
        
        # Validate method if provided, but don't set default yet (will be set by solver selection)
        if method is not None:
            if not isinstance(method, str):
                raise TypeError(f"method must be str, got {type(method).__name__}")
            valid_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
            if method not in valid_methods:
                raise ValueError(f"method must be one of {valid_methods}, got {method}")
        
        if rtol is not None:
            if not isinstance(rtol, (int, float)):
                raise TypeError(f"rtol must be numeric, got {type(rtol).__name__}")
            if rtol <= 0 or rtol >= 1:
                raise ValueError(f"rtol must be in (0, 1), got {rtol}")
        
        if atol is not None:
            if not isinstance(atol, (int, float)):
                raise TypeError(f"atol must be numeric, got {type(atol).__name__}")
            if atol <= 0:
                raise ValueError(f"atol must be positive, got {atol}")
        
        if not isinstance(detect_stiff, bool):
            raise TypeError(f"detect_stiff must be bool, got {type(detect_stiff).__name__}")
        
        # START PERFORMANCE TIMER
        if config.enable_performance_monitoring:
            _perf_monitor.start_timer('simulation')
        
        rtol = rtol or config.default_rtol
        atol = atol or config.default_atol
        
        # Build initial state vector first to get y0 for solver selection
        y0 = []
        for q in self.coordinates:
            if self.use_hamiltonian:
                pos_val = self.initial_conditions.get(q, 0.0)
                y0.append(pos_val)
                mom_key = f"p_{q}"
                mom_val = self.initial_conditions.get(mom_key, 0.0)
                y0.append(mom_val)
            else:
                pos_val = self.initial_conditions.get(q, 0.0)
                y0.append(pos_val)
                vel_key = f"{q}_dot"
                vel_val = self.initial_conditions.get(vel_key, 0.0)
                y0.append(vel_val)

        y0 = np.array(y0, dtype=float)
        
        # FIX: Actually select the optimal solver if none provided
        if method is None:
            method = self._select_optimal_solver(t_span, y0)
            logger.info(f"Adaptive solver selected: {method}")
        elif method == 'RK45' and config.enable_adaptive_solver:
            # Check if we should switch to a more stable method
            adaptive_method = True
        else:
            adaptive_method = False
        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        # Validate initial conditions
        if not validate_finite(y0, "Initial conditions"):
            return {
                'success': False,
                'error': 'Initial conditions contain non-finite values'
            }
        
        # Test initial evaluation
        try:
            dydt_test = self.equations_of_motion(t_span[0], y0)
            if not validate_finite(dydt_test, "Initial derivatives"):
                return {
                    'success': False,
                    'error': 'Initial derivatives contain non-finite values'
                }
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Initial evaluation failed: {e}")
            return {
                'success': False,
                'error': f'Initial evaluation failed: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error in initial evaluation: {type(e).__name__}: {e}")
            return {
                'success': False,
                'error': f'Initial evaluation failed: {str(e)}'
            }
        
        # Stiffness detection
        is_stiff = False
        if detect_stiff and method == 'RK45':
            try:
                test_sol = solve_ivp(
                    self.equations_of_motion,
                    (t_span[0], t_span[0] + 0.01),
                    y0,
                    method='RK45',
                    max_step=0.001
                )
                if not test_sol.success:
                    is_stiff = True
                    logger.warning("System may be stiff. Consider using 'LSODA' or 'Radau' method.")
            except (ValueError, RuntimeError, AttributeError) as e:
                logger.debug(f"Stiffness detection test failed: {e}")
        
        # Run integration
        try:
            solution = solve_ivp(
                self.equations_of_motion,
                t_span,
                y0,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=min(0.01, (t_span[1] - t_span[0]) / 100)
            )
            
            logger.info(f"Simulation complete: {solution.nfev} evaluations, "
                       f"status={'success' if solution.success else 'failed'}")
            
            # v6.0: Performance monitoring
            if config.enable_performance_monitoring:
                _perf_monitor.stop_timer('simulation')
                _perf_monitor.snapshot_memory("post_simulation")
            
            result = {
                'success': solution.success,
                't': solution.t,
                'y': solution.y,
                'coordinates': self.coordinates,
                'state_vars': self.state_vars,
                'message': solution.message if hasattr(solution, 'message') else '',
                'nfev': solution.nfev if hasattr(solution, 'nfev') else 0,
                'is_stiff': is_stiff,
                'use_hamiltonian': self.use_hamiltonian,
                'method_used': method,  # v6.0: Track which method was used
            }
            
            # v6.0: Add performance metrics if available
            if config.enable_performance_monitoring:
                sim_stats = _perf_monitor.get_stats('simulation')
                if sim_stats:
                    result['performance'] = sim_stats
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
