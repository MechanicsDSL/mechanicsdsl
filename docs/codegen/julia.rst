Julia Code Generation
=====================

Generate Julia simulation scripts with DifferentialEquations.jl integration.

Features
--------

- **DifferentialEquations.jl**: Tsit5, Vern9, Rodas5, CVODE_BDF, RK4 solvers
- **Plots.jl visualization**: Time histories, phase space portraits
- **CSV/DataFrame export**: Via CSV.jl and DataFrames.jl
- **Energy tracking**: Conservation monitoring from Lagrangian
- **Adaptive tolerances**: Configurable ``abstol`` and ``reltol``

Basic Usage
-----------

.. code-block:: python

   from mechanics_dsl.codegen.julia import JuliaGenerator
   import sympy as sp

   theta, g, l = sp.symbols('theta g l')

   gen = JuliaGenerator(
       system_name="pendulum",
       coordinates=["theta"],
       parameters={"g": 9.81, "l": 1.0},
       initial_conditions={"theta": 0.5, "theta_dot": 0.0},
       equations={"theta_ddot": -g/l * sp.sin(theta)},
       solver="Tsit5",
       abstol=1e-8,
       reltol=1e-8
   )
   gen.generate("pendulum.jl")

Parameters
~~~~~~~~~~

- ``solver``: ODE solver name (default: ``Tsit5``)
- ``abstol``: Absolute tolerance (default: ``1e-8``)
- ``reltol``: Relative tolerance (default: ``1e-8``)

Supported solvers: ``Tsit5``, ``Vern9``, ``Rodas5``, ``CVODE_BDF``, ``RK4``

Generated Code Structure
------------------------

The generated Julia script includes:

1. **Package imports**: DifferentialEquations, Plots, CSV, DataFrames
2. **Physical constants**: Parameters as ``const`` globals
3. **``equations_of_motion!``**: In-place ODE function for DifferentialEquations.jl
4. **``simulate()``**: Configurable simulation with solver and tolerances
5. **``plot_results()``**: Time history subplots saved to PNG
6. **``plot_phase_space()``**: Phase portrait visualization
7. **``export_csv()``**: DataFrame-based CSV export
8. **Energy function**: ``compute_energy()`` from Lagrangian when available

Running the Generated Code
---------------------------

Install Julia dependencies first:

.. code-block:: julia

   using Pkg
   Pkg.add(["DifferentialEquations", "Plots", "CSV", "DataFrames"])

Then run:

.. code-block:: bash

   julia pendulum.jl

Solver Selection Guide
----------------------

.. list-table::
   :header-rows: 1

   * - Solver
     - Best For
     - Order
   * - ``Tsit5``
     - General non-stiff problems (default)
     - 5th
   * - ``Vern9``
     - High-accuracy requirements
     - 9th
   * - ``Rodas5``
     - Stiff systems
     - 5th
   * - ``CVODE_BDF``
     - Very stiff / large systems
     - Variable
   * - ``RK4``
     - Fixed-step, educational
     - 4th

See Also
--------

- :doc:`python` - NumPy-accelerated Python generation
- :doc:`matlab` - MATLAB/Octave generation
- :doc:`overview` - All code generation targets
