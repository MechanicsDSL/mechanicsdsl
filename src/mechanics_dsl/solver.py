import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from .utils import logger, safe_float_conversion

class NumericalSimulator:
    def __init__(self, symbolic_engine):
        self.symbolic = symbolic_engine
        self.lambdas = {}
        self.parameters = {}
        self.initial_conditions = {}
        self.coordinates = []

    def set_parameters(self, params):
        self.parameters = params

    def set_initial_conditions(self, ics):
        self.initial_conditions = ics

    def compile_equations(self, accel_exprs, coordinates):
        self.coordinates = coordinates
        # Create lambda functions for each acceleration
        # args: (t, y) where y contains [q1, q1_dot, q2, q2_dot, ...]
        
        syms = [self.symbolic.time_symbol]
        for q in coordinates:
            syms.append(self.symbolic.get_symbol(q))
            syms.append(self.symbolic.get_symbol(f"{q}_dot"))
            
        # Parameter substitution
        final_exprs = {k: v.subs(self.parameters) for k, v in accel_exprs.items()}
        
        self.lambdas = {}
        for q in coordinates:
            key = f"{q}_ddot"
            if key in final_exprs:
                self.lambdas[key] = sp.lambdify(syms, final_exprs[key], modules="numpy")

    def _ode_system(self, t, y):
        # Unpack state
        # y = [q1, v1, q2, v2...]
        dydt = np.zeros_like(y)
        
        # Prepare args for lambdify: t, q1, v1, q2, v2...
        args = [t] + list(y)
        
        for i, q in enumerate(self.coordinates):
            # dx/dt = v
            dydt[2*i] = y[2*i+1]
            
            # dv/dt = acceleration
            key = f"{q}_ddot"
            if key in self.lambdas:
                try:
                    accel = self.lambdas[key](*args)
                    dydt[2*i+1] = accel
                except Exception as e:
                    logger.error(f"Error evaluating {key}: {e}")
                    dydt[2*i+1] = 0.0
                    
        return dydt

    def simulate(self, t_span, num_points=1000, method='RK45', rtol=1e-6, atol=1e-8):
        y0 = []
        for q in self.coordinates:
            y0.append(self.initial_conditions.get(q, 0.0))
            y0.append(self.initial_conditions.get(f"{q}_dot", 0.0))
            
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        try:
            sol = solve_ivp(
                self._ode_system, 
                t_span, 
                y0, 
                t_eval=t_eval, 
                method=method,
                rtol=rtol,
                atol=atol
            )
            return {
                'success': sol.success,
                't': sol.t,
                'y': sol.y,
                'nfev': sol.nfev,
                'coordinates': self.coordinates
            }
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {'success': False, 'error': str(e)}
