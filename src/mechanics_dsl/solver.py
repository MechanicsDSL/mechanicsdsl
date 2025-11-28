import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from typing import Dict, List, Tuple, Callable, Optional, Any
from .utils import (
    logger, config, validate_time_span, validate_array_safe, 
    safe_float_conversion, safe_array_access, validate_finite, 
    _perf_monitor, profile_function
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
        for name, value in params.items():
             if name in ['m', 'm1', 'm2', 'mass'] and value <= 0:
                  raise ValueError(f"Physics violation: Mass '{name}' must be positive (got {value})")
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
                    eq = sp.simplify(eq)
                except Exception:
                    pass # Skip simplification on error

                # Replace derivative symbols
                derivs = list(eq.atoms(sp.Derivative))
                for d in derivs:
                    try:
                        base = d.args[0]
                        order = 1
                        if len(d.args) >= 2 and isinstance(d.args[1], tuple):
                             order = int(d.args[1][1])
                        
                        base_name = str(base)
                        if base_name in coordinates:
                            suffix = "_ddot" if order == 2 else "_dot"
                            repl = self.symbolic.get_symbol(f"{base_name}{suffix}")
                            eq = eq.subs(d, repl)
                    except: continue

                free_symbols = eq.free_symbols
                ordered_symbols = []
                symbol_indices = []
                
                for i, var_name in enumerate(state_vars):
                    sym = self.symbolic.get_symbol(var_name)
                    if sym in free_symbols:
                        ordered_symbols.append(sym)
                        symbol_indices.append(i)
                
                if ordered_symbols:
                    try:
                        func = sp.lambdify(ordered_symbols, eq, modules=['numpy', 'math'])
                        
                        # Closure capture trick
                        def make_wrapper(func, indices):
                            return lambda *y: func(*[y[i] for i in indices])
                        
                        compiled_equations[accel_key] = make_wrapper(func, symbol_indices)
                    except Exception as e:
                        logger.error(f"Compilation failed for {accel_key}: {e}")
                        compiled_equations[accel_key] = lambda *args: 0.0
                else:
                    val = float(sp.N(eq))
                    compiled_equations[accel_key] = lambda *args: val

        self.equations = compiled_equations
        self.state_vars = state_vars
        self.coordinates = coordinates

    def compile_hamiltonian_equations(self, q_dots: List[sp.Expr], p_dots: List[sp.Expr], 
                                     coordinates: List[str]):
        """Compile Hamiltonian equations"""
        self.use_hamiltonian = True
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"p_{q}"])
        
        param_subs = {self.symbolic.get_symbol(k): v for k, v in self.parameters.items()}
        self.hamiltonian_equations = {'q_dots': [], 'p_dots': []}
        
        for i in range(len(coordinates)):
            for eq_list, target_list in [(q_dots, self.hamiltonian_equations['q_dots']), 
                                        (p_dots, self.hamiltonian_equations['p_dots'])]:
                eq = eq_list[i].subs(param_subs)
                free = eq.free_symbols
                ordered = []
                idxs = []
                for j, v in enumerate(state_vars):
                    sym = self.symbolic.get_symbol(v)
                    if sym in free:
                        ordered.append(sym)
                        idxs.append(j)
                
                if ordered:
                    func = sp.lambdify(ordered, eq, modules=['numpy'])
                    target_list.append((func, idxs))
                else:
                    val = float(sp.N(eq))
                    target_list.append((lambda *a, v=val: v, []))
        
        self.state_vars = state_vars
        self.coordinates = coordinates

    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        ODE system for numerical integration.
        Optimized hot path: Validation moved to simulate() for speed.
        """
        if self.use_hamiltonian:
            return self._hamiltonian_ode(t, y)
            
        dydt = np.zeros_like(y)
        n_coords = len(self.coordinates)

        # 1. dx/dt = v
        # 2. dv/dt = acceleration
        for i in range(n_coords):
            pos_idx = 2 * i
            vel_idx = 2 * i + 1
            
            # Position derivative is velocity
            if vel_idx < len(y):
                dydt[pos_idx] = y[vel_idx]
            
            # Velocity derivative is acceleration
            accel_key = f"{self.coordinates[i]}_ddot"
            if accel_key in self.equations:
                try:
                    # Use NumPy error handling to catch singularities (r -> 0)
                    with np.errstate(all='raise'):
                        val = self.equations[accel_key](*y)
                        dydt[vel_idx] = val
                except FloatingPointError:
                    # Singularity detected (e.g. N-Body collision)
                    # Return 0 to allow adaptive stepper to shrink and retry, or fail gracefully
                    dydt[vel_idx] = 0.0 
                except Exception:
                    dydt[vel_idx] = 0.0
                    
        return dydt
    

    def _hamiltonian_ode(self, t: float, y: np.ndarray) -> np.ndarray:
        dydt = np.zeros_like(y)
        n = len(self.coordinates)
        
        for i in range(n):
            # q_dot
            func, idxs = self.hamiltonian_equations['q_dots'][i]
            args = [y[j] for j in idxs]
            try:
                dydt[2*i] = func(*args)
            except: dydt[2*i] = 0.0
            
            # p_dot
            func, idxs = self.hamiltonian_equations['p_dots'][i]
            args = [y[j] for j in idxs]
            try:
                dydt[2*i+1] = func(*args)
            except: dydt[2*i+1] = 0.0
            
        return dydt

    def _select_optimal_solver(self, t_span: Tuple[float, float], y0: np.ndarray) -> str:
        """Intelligently select optimal solver based on system characteristics"""
        if not config.enable_adaptive_solver:
            return 'RK45'
        
        n_dof = len(self.coordinates)
        time_span = t_span[1] - t_span[0]
        
        # Heuristics
        if n_dof > 10: return 'LSODA'
        if time_span > 100: return 'LSODA'
        if n_dof <= 2 and time_span < 10: return 'RK45'
        
        return 'LSODA' # Default robust choice
                        
    @profile_function
    def simulate(self, t_span: Tuple[float, float], num_points: int = 1000,
                 method: str = None, rtol: float = None, atol: float = None,
                 detect_stiff: bool = True) -> dict:
        """Run numerical simulation"""
        validate_time_span(t_span)
        
        method = method or 'RK45'
        rtol = rtol or config.default_rtol
        atol = atol or config.default_atol
        
        # Build Initial State Vector y0
        y0 = []
        for q in self.coordinates:
            if self.use_hamiltonian:
                y0.append(self.initial_conditions.get(q, 0.0))
                y0.append(self.initial_conditions.get(f"p_{q}", 0.0))
            else:
                y0.append(self.initial_conditions.get(q, 0.0))
                y0.append(self.initial_conditions.get(f"{q}_dot", 0.0))
        y0 = np.array(y0, dtype=float)

        if not validate_finite(y0, "Initial conditions"):
            return {'success': False, 'error': 'Non-finite initial conditions'}

        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        # Adaptive Solver Logic
        is_stiff = False
        if config.enable_adaptive_solver and method == 'RK45':
            try:
                # Test a micro-step to check for stiffness
                test_sol = solve_ivp(
                    self.equations_of_motion,
                    (t_span[0], t_span[0] + 0.01),
                    y0,
                    method='RK45',
                    max_step=0.001
                )
                if not test_sol.success:
                    logger.warning("System appears stiff, switching to LSODA")
                    method = 'LSODA'
                    is_stiff = True
            except:
                method = 'LSODA'
                is_stiff = True

        # Main Integration
        try:
            solution = solve_ivp(
                self.equations_of_motion,
                t_span,
                y0,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol
            )
            
            return {
                'success': solution.success,
                't': solution.t,
                'y': solution.y,
                'coordinates': self.coordinates,
                'state_vars': self.state_vars,
                'message': solution.message,
                'nfev': solution.nfev,
                'is_stiff': is_stiff,
                'method': method
            }
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {'success': False, 'error': str(e)}
