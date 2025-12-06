Advanced Topics
===============

This section covers advanced usage of MechanicsDSL for expert users and
developers looking to extend the framework.

.. contents:: Contents
   :local:
   :depth: 2

Performance Optimization
------------------------

Symbolic Simplification
~~~~~~~~~~~~~~~~~~~~~~~

Control simplification to balance speed vs. expression size:

.. code-block:: python

   from mechanics_dsl.utils import config
   
   # Increase timeout for complex systems
   config.simplification_timeout = 30.0
   
   # Or disable automatic simplification
   compiler = PhysicsCompiler()
   compiler.symbolic.auto_simplify = False
   
   # Manually simplify critical expressions
   simplified = compiler.symbolic.simplify(expr, timeout=60)

Numerical Precision
~~~~~~~~~~~~~~~~~~~

Adjust solver tolerances for accuracy vs. speed:

.. code-block:: python

   # High precision (slow)
   solution = compiler.simulate(
       (0, 100),
       rtol=1e-12,
       atol=1e-14,
       method='DOP853'
   )
   
   # Lower precision (fast, for exploration)
   solution = compiler.simulate(
       (0, 100),
       rtol=1e-6,
       atol=1e-8,
       method='RK45'
   )

Caching
~~~~~~~

Leverage the built-in caching system:

.. code-block:: python

   from mechanics_dsl.utils import LRUCache
   
   # Create a cache for expensive computations
   jacobian_cache = LRUCache(maxsize=1000)
   
   def cached_jacobian(expr, vars):
       key = (str(expr), tuple(str(v) for v in vars))
       if key in jacobian_cache:
           return jacobian_cache[key]
       
       result = compute_jacobian(expr, vars)
       jacobian_cache[key] = result
       return result


Stiff Systems
-------------

Detecting Stiffness
~~~~~~~~~~~~~~~~~~~

Some mechanical systems are numerically stiff:

- Systems with multiple timescales
- Near equilibrium points
- High spring constants

.. code-block:: python

   # Enable automatic stiffness detection
   solution = compiler.simulate(
       (0, 100),
       detect_stiff=True  # Default: True
   )
   
   # Check what method was selected
   print(f"Method used: {solution['method']}")

Implicit Solvers
~~~~~~~~~~~~~~~~

For stiff systems, use implicit methods:

.. code-block:: python

   # BDF (Backward Differentiation Formula)
   solution = compiler.simulate((0, 100), method='BDF')
   
   # Radau IIA (5th order implicit)
   solution = compiler.simulate((0, 100), method='Radau')
   
   # LSODA (automatic switching)
   solution = compiler.simulate((0, 100), method='LSODA')


Custom Solvers
--------------

Implementing Custom Integrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the solver framework:

.. code-block:: python

   from mechanics_dsl.core.solver import NumericalSimulator
   import numpy as np
   
   class SymplecticEuler(NumericalSimulator):
       """Symplectic Euler integrator for Hamiltonian systems."""
       
       def integrate(self, t_span, y0, dt):
           t = np.arange(t_span[0], t_span[1], dt)
           n = len(y0) // 2
           
           y = np.zeros((len(y0), len(t)))
           y[:, 0] = y0
           
           for i in range(len(t) - 1):
               q = y[:n, i]
               p = y[n:, i]
               
               # Symplectic update
               p_new = p + dt * self.dp_dt(q, p)
               q_new = q + dt * self.dq_dt(q, p_new)
               
               y[:n, i+1] = q_new
               y[n:, i+1] = p_new
           
           return {'t': t, 'y': y, 'success': True}


Extending the Parser
--------------------

Adding Custom Commands
~~~~~~~~~~~~~~~~~~~~~~

Extend the parser for domain-specific commands:

.. code-block:: python

   from mechanics_dsl.core.parser import MechanicsParser, ASTNode
   from dataclasses import dataclass
   
   @dataclass
   class CustomCommandNode(ASTNode):
       name: str
       args: list
   
   class ExtendedParser(MechanicsParser):
       def parse_statement(self):
           token = self.peek()
           
           if token.value == '\\mycommand':
               return self.parse_mycommand()
           
           return super().parse_statement()
       
       def parse_mycommand(self):
           self.expect('COMMAND')  # \mycommand
           self.expect('LBRACE')
           arg = self.parse_expression()
           self.expect('RBRACE')
           return CustomCommandNode('mycommand', [arg])


Adding New Physics Domains
--------------------------

Create a new physics domain by inheriting from ``PhysicsDomain``:

.. code-block:: python

   from mechanics_dsl.domains.base import PhysicsDomain
   import sympy as sp
   
   class ElectromagneticDomain(PhysicsDomain):
       """Electromagnetic field dynamics."""
       
       def __init__(self, name='em_system'):
           super().__init__(name)
           self._charge = None
           self._mass = None
           self._E_field = None
           self._B_field = None
       
       def set_fields(self, E, B):
           self._E_field = E
           self._B_field = B
       
       def define_lagrangian(self):
           # L = T - V + (q/c) * v · A
           q, m, c = sp.symbols('q m c', real=True)
           # ... implementation
           pass
       
       def define_hamiltonian(self):
           # H = (p - qA/c)² / 2m + qφ
           pass
       
       def derive_equations_of_motion(self):
           # Lorentz force: F = q(E + v × B)
           pass
       
       def get_state_variables(self):
           return ['x', 'y', 'z', 'px', 'py', 'pz']


Code Generation Backends
------------------------

Creating a New Backend
~~~~~~~~~~~~~~~~~~~~~~

Add support for a new target platform:

.. code-block:: python

   from mechanics_dsl.codegen.base import CodeGenerator
   
   class JuliaGenerator(CodeGenerator):
       """Generate Julia simulation code."""
       
       @property
       def target_name(self):
           return "julia"
       
       @property
       def file_extension(self):
           return ".jl"
       
       def generate(self, output_file):
           code = self._generate_julia_code()
           with open(output_file, 'w') as f:
               f.write(code)
           return output_file
       
       def generate_equations(self):
           # Convert SymPy to Julia syntax
           from sympy.printing.julia import julia_code
           
           lines = []
           for coord in self.coordinates:
               expr = self.equations.get(f"{coord}_ddot")
               julia_expr = julia_code(expr)
               lines.append(f"    dydt[{idx}] = {julia_expr}")
           
           return "\n".join(lines)


Parallel Execution
------------------

OpenMP Code Generation
~~~~~~~~~~~~~~~~~~~~~~

Generate parallelized C++ code:

.. code-block:: python

   # Generate OpenMP-parallelized code
   compiler.compile_to_cpp(
       "simulation_omp.cpp",
       target="openmp"
   )

GPU Acceleration
~~~~~~~~~~~~~~~~

For large particle systems (SPH, N-body), use GPU backends:

.. code-block:: python

   # Generate CUDA code (placeholder)
   compiler.compile_to_cpp(
       "simulation.cu",
       target="cuda"
   )


Testing and Validation
----------------------

Analytical Comparisons
~~~~~~~~~~~~~~~~~~~~~~

Validate against known solutions:

.. code-block:: python

   import numpy as np
   
   # Simple pendulum small-angle analytical solution
   def analytical_pendulum(t, theta0, l, g):
       omega = np.sqrt(g / l)
       return theta0 * np.cos(omega * t)
   
   # Compare
   numerical = solution['y'][0]
   analytical = analytical_pendulum(solution['t'], 0.1, 1.0, 9.81)
   
   error = np.max(np.abs(numerical - analytical))
   assert error < 1e-6, f"Error too large: {error}"

Energy Conservation
~~~~~~~~~~~~~~~~~~~

Validate symplectic behavior:

.. code-block:: python

   from mechanics_dsl.analysis import EnergyAnalyzer
   
   analyzer = EnergyAnalyzer()
   result = analyzer.check_conservation(
       solution, kinetic, potential,
       tolerance=1e-6
   )
   
   assert result['conserved'], \
       f"Energy not conserved: {result['max_relative_error']}"


Debugging
---------

Verbose Logging
~~~~~~~~~~~~~~~

Enable detailed logs:

.. code-block:: python

   from mechanics_dsl.utils import setup_logging
   
   setup_logging(level='DEBUG')

Inspecting Symbolic Expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

View derived equations:

.. code-block:: python

   result = compiler.compile(source)
   
   # Print equations of motion
   for coord, eom in compiler.symbolic.equations.items():
       print(f"{coord} = {eom}")
   
   # Pretty print with SymPy
   import sympy as sp
   sp.pprint(compiler.symbolic.lagrangian)

Step-by-Step Execution
~~~~~~~~~~~~~~~~~~~~~~

Debug simulation issues:

.. code-block:: python

   # Use small time steps
   solution = compiler.simulate(
       (0, 0.1),
       num_points=1000
   )
   
   # Check for NaN/Inf
   if not np.all(np.isfinite(solution['y'])):
       print("Simulation became unstable!")
       
       # Find where it failed
       bad_idx = np.where(~np.isfinite(solution['y']))[1][0]
       print(f"Failed at t = {solution['t'][bad_idx]}")
