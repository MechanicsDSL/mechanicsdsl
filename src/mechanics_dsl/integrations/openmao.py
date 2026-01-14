"""
OpenMDAO integration for MechanicsDSL.

Wraps MechanicsDSL simulations as OpenMDAO components for
multidisciplinary design optimization (MDO).

Requires: pip install openmdao

Example:
    import openmdao.api as om
    from mechanics_dsl.integrations.openmao import OpenMDAOMechanicsComponent
    
    prob = om.Problem()
    prob.model.add_subsystem('physics', OpenMDAOMechanicsComponent(
        dsl_code=dsl_code,
        params_as_inputs=['m', 'k'],
        outputs=['max_displacement', 'final_energy']
    ))
"""
from typing import Dict, List, Optional, Callable, Any
import numpy as np

try:
    import openmdao.api as om
    OPENMDAO_AVAILABLE = True
except ImportError:
    OPENMDAO_AVAILABLE = False
    om = None

try:
    from mechanics_dsl import PhysicsCompiler
except ImportError:
    PhysicsCompiler = None


class OpenMDAOMechanicsComponent:
    """
    OpenMDAO ExplicitComponent wrapping a MechanicsDSL simulation.
    
    Allows physics simulations to be optimized within OpenMDAO frameworks.
    
    Args:
        dsl_code: MechanicsDSL code defining the system
        params_as_inputs: Parameter names to expose as OpenMDAO inputs
        outputs: Output quantities to compute. Can be:
            - 'max_displacement': Maximum position excursion
            - 'final_energy': Energy at end of simulation
            - 'final_state': Full state vector at t_end
            - 'rms_velocity': RMS of velocities
            - Custom callable: func(solution) -> float
        t_span: Simulation time span
        num_points: Number of output points
        
    Example:
        comp = OpenMDAOMechanicsComponent(
            dsl_code=pendulum_dsl,
            params_as_inputs=['m', 'l', 'g'],
            outputs=['max_displacement'],
            t_span=(0, 10)
        )
        
        # In OpenMDAO problem
        prob = om.Problem()
        prob.model.add_subsystem('pendulum', comp)
        prob.model.add_design_var('pendulum.m', lower=0.1, upper=10)
        prob.model.add_objective('pendulum.max_displacement')
    """
    
    def __new__(cls, *args, **kwargs):
        if not OPENMDAO_AVAILABLE:
            raise ImportError(
                "OpenMDAO is not installed. Install with: pip install openmdao"
            )
        
        # Dynamically create the component class
        return cls._create_component(*args, **kwargs)
    
    @classmethod
    def _create_component(
        cls,
        dsl_code: str,
        params_as_inputs: List[str],
        outputs: List[str],
        t_span: tuple = (0, 10),
        num_points: int = 1000,
    ) -> 'om.ExplicitComponent':
        """Create the OpenMDAO component."""
        
        class _MechanicsComponent(om.ExplicitComponent):
            """Generated OpenMDAO component."""
            
            def initialize(self):
                self.options.declare('dsl_code', types=str)
                self.options.declare('t_span', types=tuple, default=(0, 10))
                self.options.declare('num_points', types=int, default=1000)
                self.options.declare('params_as_inputs', types=list)
                self.options.declare('output_names', types=list)
                
                # Store for later
                self._compiler = None
                self._base_params = {}
            
            def setup(self):
                # Compile DSL to get parameter info
                if PhysicsCompiler is None:
                    raise RuntimeError("PhysicsCompiler not available")
                
                self._compiler = PhysicsCompiler()
                result = self._compiler.compile_dsl(self.options['dsl_code'])
                
                if not result['success']:
                    raise ValueError(f"DSL compilation failed: {result.get('error')}")
                
                self._base_params = dict(self._compiler.simulator.parameters)
                
                # Add inputs for specified parameters
                for param in self.options['params_as_inputs']:
                    if param in self._base_params:
                        self.add_input(param, val=self._base_params[param])
                    else:
                        self.add_input(param, val=1.0)
                
                # Add outputs
                for out in self.options['output_names']:
                    if out == 'final_state':
                        n_coords = len(result.get('coordinates', []))
                        self.add_output(out, shape=(n_coords * 2,))
                    else:
                        self.add_output(out, val=0.0)
                
                # Declare partials (finite difference for now)
                self.declare_partials('*', '*', method='fd')
            
            def compute(self, inputs, outputs):
                # Update parameters from inputs
                params = self._base_params.copy()
                for param in self.options['params_as_inputs']:
                    if param in inputs:
                        params[param] = float(inputs[param])
                
                self._compiler.simulator.set_parameters(params)
                
                # Run simulation
                solution = self._compiler.simulate(
                    t_span=self.options['t_span'],
                    num_points=self.options['num_points']
                )
                
                if not solution['success']:
                    # Return large values on failure (for optimization)
                    for out in self.options['output_names']:
                        if out != 'final_state':
                            outputs[out] = 1e10
                    return
                
                # Compute outputs
                y = solution['y']
                t = solution['t']
                
                for out in self.options['output_names']:
                    if out == 'max_displacement':
                        # Max absolute value of first coordinate
                        outputs[out] = np.max(np.abs(y[0]))
                    
                    elif out == 'final_energy':
                        # Kinetic energy at end
                        n_coords = y.shape[0] // 2
                        ke = sum(0.5 * y[2*i + 1, -1]**2 for i in range(n_coords))
                        outputs[out] = ke
                    
                    elif out == 'final_state':
                        outputs[out] = y[:, -1]
                    
                    elif out == 'rms_velocity':
                        n_coords = y.shape[0] // 2
                        velocities = [y[2*i + 1] for i in range(n_coords)]
                        rms = np.sqrt(np.mean(np.sum([v**2 for v in velocities], axis=0)))
                        outputs[out] = rms
                    
                    elif out == 'settling_time':
                        # Time to reach 2% of initial value
                        threshold = 0.02 * np.abs(y[0, 0])
                        settled = np.where(np.abs(y[0]) < threshold)[0]
                        if len(settled) > 0:
                            outputs[out] = t[settled[0]]
                        else:
                            outputs[out] = t[-1]
                    
                    elif callable(out):
                        # Custom output function
                        outputs[out.__name__] = out(solution)
        
        # Create and return instance with options
        comp = _MechanicsComponent()
        comp.options['dsl_code'] = dsl_code
        comp.options['t_span'] = t_span
        comp.options['num_points'] = num_points
        comp.options['params_as_inputs'] = params_as_inputs
        comp.options['output_names'] = outputs
        
        return comp


def create_optimization_problem(
    dsl_code: str,
    objective: str,
    design_vars: Dict[str, tuple],  # name -> (lower, upper)
    constraints: Optional[Dict[str, tuple]] = None,
    t_span: tuple = (0, 10),
) -> 'om.Problem':
    """
    Create a complete OpenMDAO optimization problem.
    
    Args:
        dsl_code: MechanicsDSL code
        objective: Output to minimize
        design_vars: Dict of param -> (lower, upper) bounds
        constraints: Optional constraints
        t_span: Simulation time span
        
    Returns:
        Configured OpenMDAO Problem
    """
    if not OPENMDAO_AVAILABLE:
        raise ImportError("OpenMDAO not installed")
    
    prob = om.Problem()
    
    # Add component
    comp = OpenMDAOMechanicsComponent(
        dsl_code=dsl_code,
        params_as_inputs=list(design_vars.keys()),
        outputs=[objective] + (list(constraints.keys()) if constraints else []),
        t_span=t_span
    )
    prob.model.add_subsystem('physics', comp)
    
    # Design variables
    for param, (lower, upper) in design_vars.items():
        prob.model.add_design_var(f'physics.{param}', lower=lower, upper=upper)
    
    # Objective
    prob.model.add_objective(f'physics.{objective}')
    
    # Constraints
    if constraints:
        for name, (lower, upper) in constraints.items():
            prob.model.add_constraint(f'physics.{name}', lower=lower, upper=upper)
    
    # Driver
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    
    return prob


__all__ = [
    'OpenMDAOMechanicsComponent',
    'create_optimization_problem',
    'OPENMDAO_AVAILABLE',
]
