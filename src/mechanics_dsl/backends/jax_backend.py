"""
JAX backend for MechanicsDSL.

Provides GPU acceleration, JIT compilation, and automatic differentiation
for physics simulations.

Requirements:
    pip install jax jaxlib diffrax

Example:
    from mechanics_dsl.backends import JAXBackend
    
    backend = JAXBackend(use_gpu=True)
    
    # Compile equations
    equations = backend.compile_equations(accelerations, coords, params)
    
    # Run simulation with JIT
    result = backend.simulate(equations, (0, 10), y0)
    
    # Compute gradients for optimization
    grads = backend.gradient(loss_fn, params)
"""
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import warnings

from .base import Backend, BackendCapabilities

# Attempt to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

# Attempt to import diffrax (JAX ODE solvers)
try:
    import diffrax
    DIFFRAX_AVAILABLE = True
except ImportError:
    DIFFRAX_AVAILABLE = False
    diffrax = None


@dataclass
class JAXConfig:
    """Configuration for JAX backend."""
    use_gpu: bool = True
    use_64bit: bool = True
    jit_compile: bool = True
    vectorize_batch: bool = True
    default_solver: str = "Tsit5"  # Tsitouras 5/4 (like RK45)
    max_steps: int = 100000
    rtol: float = 1e-6
    atol: float = 1e-8


class JAXSolver:
    """
    JAX-based ODE solver using diffrax.
    
    Provides JIT-compiled symplectic and non-symplectic integrators.
    """
    
    SOLVERS = {
        'euler': 'Euler',
        'heun': 'Heun',
        'tsit5': 'Tsit5',  # Tsitouras 5(4) - similar to RK45
        'dopri5': 'Dopri5',  # Dormand-Prince 5(4)
        'dopri8': 'Dopri8',  # Dormand-Prince 8(5)
        'bosh3': 'Bosh3',  # Bogacki-Shampine 3(2)
        'kvaerno3': 'Kvaerno3',  # Implicit for stiff
        'kvaerno4': 'Kvaerno4',
        'kvaerno5': 'Kvaerno5',
    }
    
    def __init__(self, config: Optional[JAXConfig] = None):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. Install with: pip install jax jaxlib"
            )
        if not DIFFRAX_AVAILABLE:
            raise ImportError(
                "diffrax is not installed. Install with: pip install diffrax"
            )
        
        self.config = config or JAXConfig()
        
        # Configure JAX
        if self.config.use_64bit:
            jax.config.update("jax_enable_x64", True)
        
        # Select device
        if self.config.use_gpu:
            try:
                devices = jax.devices('gpu')
                self.device = devices[0] if devices else jax.devices('cpu')[0]
            except RuntimeError:
                warnings.warn("GPU not available, falling back to CPU")
                self.device = jax.devices('cpu')[0]
        else:
            self.device = jax.devices('cpu')[0]
    
    def get_solver(self, name: str = None):
        """Get diffrax solver by name."""
        name = name or self.config.default_solver
        solver_name = self.SOLVERS.get(name.lower(), 'Tsit5')
        return getattr(diffrax, solver_name)()
    
    def solve(
        self,
        vector_field: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        solver: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve ODE using diffrax.
        
        Args:
            vector_field: Function (t, y, args) -> dy/dt
            t_span: (t_start, t_end)
            y0: Initial conditions
            t_eval: Evaluation times
            solver: Solver name
            
        Returns:
            Dictionary with 't', 'y', 'success' keys
        """
        t0, t1 = t_span
        
        # Convert to JAX arrays
        y0_jax = jnp.array(y0)
        
        if t_eval is not None:
            t_eval_jax = jnp.array(t_eval)
            saveat = diffrax.SaveAt(ts=t_eval_jax)
        else:
            saveat = diffrax.SaveAt(t1=True)
        
        # Create ODE term
        term = diffrax.ODETerm(vector_field)
        
        # Get solver
        solver_obj = self.get_solver(solver)
        
        # Step size controller
        stepsize_controller = diffrax.PIDController(
            rtol=self.config.rtol,
            atol=self.config.atol
        )
        
        # Solve
        solution = diffrax.diffeqsolve(
            term,
            solver_obj,
            t0=t0,
            t1=t1,
            dt0=None,  # Adaptive
            y0=y0_jax,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=self.config.max_steps,
        )
        
        return {
            'success': True,
            't': np.array(solution.ts),
            'y': np.array(solution.ys).T,  # Transpose to (state_dim, time)
            'nfev': int(solution.stats.get('num_steps', 0)),
        }


class JAXBackend(Backend):
    """
    JAX-accelerated simulation backend.
    
    Features:
    - JIT compilation for fast repeated simulations
    - GPU acceleration (if available)
    - Automatic differentiation for optimization
    - Vectorized batch simulations via vmap
    
    Example:
        backend = JAXBackend(use_gpu=True)
        
        # Compile and simulate
        eqns = backend.compile_equations(accels, coords, params)
        result = backend.simulate(eqns, (0, 10), y0)
        
        # Parameter optimization
        def loss(params):
            result = simulate_with_params(params)
            return jnp.mean((result - target)**2)
        
        grads = backend.gradient(loss, params)
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        use_64bit: bool = True,
        jit_compile: bool = True
    ):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. Install with:\n"
                "  pip install jax jaxlib  # CPU\n"
                "  pip install jax[cuda12]  # GPU with CUDA 12"
            )
        
        self.config = JAXConfig(
            use_gpu=use_gpu,
            use_64bit=use_64bit,
            jit_compile=jit_compile
        )
        
        self.solver = JAXSolver(self.config)
        self._compiled_cache: Dict[str, Callable] = {}
    
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="jax",
            supports_gpu=True,
            supports_autodiff=True,
            supports_jit=True,
            supports_batched=True,
            supports_stiff=True,
            max_batch_size=10000,
        )
    
    def compile_equations(
        self,
        accelerations: Dict[str, Any],
        coordinates: List[str],
        parameters: Dict[str, float]
    ) -> Callable:
        """
        Compile symbolic equations to JAX-compatible functions.
        
        Uses sympy2jax or manual conversion to create JAX-traceable
        functions that can be JIT compiled and differentiated.
        """
        import sympy as sp
        
        # Build state vector mapping
        state_vars = []
        for q in coordinates:
            state_vars.extend([q, f"{q}_dot"])
        
        n_state = len(state_vars)
        
        # Create numpy lambdified versions first
        t_sym = sp.Symbol('t')
        state_symbols = [sp.Symbol(s) for s in state_vars]
        
        accel_funcs = []
        for coord in coordinates:
            expr = accelerations[coord]
            # Substitute parameters
            for param, val in parameters.items():
                expr = expr.subs(sp.Symbol(param), val)
            
            # Lambdify with numpy (JAX-compatible for basic ops)
            func = sp.lambdify(
                [t_sym] + state_symbols,
                expr,
                modules=['numpy']
            )
            accel_funcs.append(func)
        
        def vector_field(t, y, args):
            """JAX-compatible vector field."""
            dydt = jnp.zeros_like(y)
            
            for i, accel_func in enumerate(accel_funcs):
                # Velocity
                dydt = dydt.at[2*i].set(y[2*i + 1])
                # Acceleration
                accel = accel_func(t, *y)
                dydt = dydt.at[2*i + 1].set(accel)
            
            return dydt
        
        # JIT compile if enabled
        if self.config.jit_compile:
            vector_field = jit(vector_field)
        
        return vector_field
    
    def simulate(
        self,
        equations: Callable,
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None,
        solver: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run JAX-accelerated simulation."""
        return self.solver.solve(
            equations,
            t_span,
            y0,
            t_eval=t_eval,
            solver=solver,
            **kwargs
        )
    
    def simulate_batch(
        self,
        equations: Callable,
        t_span: Tuple[float, float],
        y0_batch: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Run vectorized batch simulations using vmap.
        
        Much faster than sequential for many initial conditions.
        
        Args:
            equations: Compiled equations
            t_span: Time span
            y0_batch: (batch_size, state_dim) initial conditions
            
        Returns:
            (batch_size, state_dim, time_points) array of trajectories
        """
        def single_solve(y0):
            result = self.simulate(equations, t_span, y0, **kwargs)
            return result['y']
        
        # Vectorize over initial conditions
        batched_solve = vmap(single_solve)
        
        # Convert to JAX array
        y0_jax = jnp.array(y0_batch)
        
        return np.array(batched_solve(y0_jax))
    
    def gradient(
        self,
        loss_fn: Callable,
        params: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute gradient of loss function w.r.t. parameters.
        
        Uses JAX autodiff for efficient gradient computation.
        
        Args:
            loss_fn: Function (params_dict) -> scalar loss
            params: Dictionary of parameters
            
        Returns:
            Dictionary of parameter gradients
        """
        # Convert dict to array for grad
        param_names = list(params.keys())
        param_array = jnp.array([params[k] for k in param_names])
        
        def array_loss(param_arr):
            param_dict = {k: v for k, v in zip(param_names, param_arr)}
            return loss_fn(param_dict)
        
        grad_fn = grad(array_loss)
        grads = grad_fn(param_array)
        
        return {k: float(g) for k, g in zip(param_names, grads)}
    
    def hessian(
        self,
        loss_fn: Callable,
        params: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute Hessian matrix for second-order optimization.
        
        Args:
            loss_fn: Function (params_dict) -> scalar loss
            params: Dictionary of parameters
            
        Returns:
            (n_params, n_params) Hessian matrix
        """
        from jax import hessian as jax_hessian
        
        param_names = list(params.keys())
        param_array = jnp.array([params[k] for k in param_names])
        
        def array_loss(param_arr):
            param_dict = {k: v for k, v in zip(param_names, param_arr)}
            return loss_fn(param_dict)
        
        hess_fn = jax_hessian(array_loss)
        return np.array(hess_fn(param_array))
    
    def cleanup(self) -> None:
        """Clear compiled function cache."""
        self._compiled_cache.clear()


__all__ = [
    'JAXConfig',
    'JAXSolver',
    'JAXBackend',
    'JAX_AVAILABLE',
    'DIFFRAX_AVAILABLE',
]
