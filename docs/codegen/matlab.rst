MATLAB/Octave Code Generation
=============================

Generate MATLAB/GNU Octave simulation scripts with built-in visualization.

Features
--------

- **Multiple ODE solvers**: ode45, ode23, ode15s, ode113, ode23s
- **Publication-quality plots**: Time histories, phase space, energy conservation
- **Nested function structure**: Parameters accessible without globals
- **CSV export**: Automatic results file with energy column
- **Octave compatible**: Works with GNU Octave 5.0+

Basic Usage
-----------

.. code-block:: python

   from mechanics_dsl.codegen.matlab import MatlabGenerator
   import sympy as sp

   theta, g, l = sp.symbols('theta g l')

   gen = MatlabGenerator(
       system_name="pendulum",
       coordinates=["theta"],
       parameters={"g": 9.81, "l": 1.0},
       initial_conditions={"theta": 0.5, "theta_dot": 0.0},
       equations={"theta_ddot": -g/l * sp.sin(theta)},
       solver="ode45"
   )
   gen.generate("pendulum.m")

Parameters
~~~~~~~~~~

- ``solver``: MATLAB ODE solver (default: ``ode45``)
- ``abstol``: Absolute tolerance (default: ``1e-8``)
- ``reltol``: Relative tolerance (default: ``1e-6``)
- ``generate_simulink``: Generate Simulink model (default: ``False``)

Supported solvers: ``ode45``, ``ode23``, ``ode15s``, ``ode113``, ``ode23s``

Running in MATLAB
-----------------

.. code-block:: matlab

   >> results = pendulum();
   >> plot(results.t, results.y(:,1));

Running in GNU Octave
---------------------

.. code-block:: bash

   octave --eval "pendulum"

Generated Code Structure
------------------------

The generated ``.m`` file contains:

1. **Main function**: Configuration, ODE solving, results packaging
2. **``equations_of_motion``**: Nested function with parameter access
3. **``compute_energy``**: Mechanical energy computation
4. **``plot_results``**: Time history subplots saved as PNG
5. **``plot_phase_space``**: Phase portrait with start/end markers
6. **``plot_energy``**: Energy conservation plot with drift annotation
7. **``export_csv``**: Results export with header

Output Files
------------

After running, the script produces:

- ``{system_name}_time_history.png`` — Position and velocity plots
- ``{system_name}_phase_space.png`` — Phase portrait
- ``{system_name}_energy.png`` — Energy conservation
- ``{system_name}_results.csv`` — Numerical results

Solver Selection Guide
----------------------

.. list-table::
   :header-rows: 1

   * - Solver
     - Best For
   * - ``ode45``
     - General non-stiff problems (default)
   * - ``ode23``
     - Crude tolerances, fast evaluation
   * - ``ode15s``
     - Stiff systems
   * - ``ode113``
     - High accuracy, smooth problems
   * - ``ode23s``
     - Stiff with constant mass matrix

See Also
--------

- :doc:`julia` - Julia with DifferentialEquations.jl
- :doc:`python` - NumPy-accelerated Python
- :doc:`overview` - All code generation targets
