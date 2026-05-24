OpenMP Code Generation
======================

Generate multi-threaded C++ code with OpenMP for parallel simulations.

Features
--------

- **Thread-parallel integration**: Simulate many trajectories simultaneously
- **OpenMP reductions**: Parallel energy calculations
- **SIMD-friendly loops**: Optimized for modern CPUs
- **Auto-detect threads**: Uses all available cores by default
- **Performance timing**: Built-in ``omp_get_wtime()`` benchmarking

Basic Usage
-----------

Generate OpenMP code from a compiled system:

.. code-block:: python

   from mechanics_dsl import PhysicsCompiler

   compiler = PhysicsCompiler()
   compiler.compile_dsl(r'''
       \system{pendulum}
       \defvar{theta}{angle}{rad}
       \parameter{m}{1.0}{kg}
       \parameter{l}{1.0}{m}
       \parameter{g}{9.81}{m/s^2}
       \lagrangian{\frac{1}{2} m l^2 \dot{theta}^2 + m g l \cos{theta}}
       \initial{theta=0.5, theta_dot=0}
   ''')

   compiler.compile_to_cpp("pendulum_openmp.cpp", target="openmp")

Using the Generator Directly
-----------------------------

.. code-block:: python

   from mechanics_dsl.codegen.openmp import OpenMPGenerator
   import sympy as sp

   theta, g, l = sp.symbols('theta g l')

   gen = OpenMPGenerator(
       system_name="pendulum",
       coordinates=['theta'],
       parameters={'g': 9.81, 'l': 1.0},
       initial_conditions={'theta': 0.1, 'theta_dot': 0.0},
       equations={'theta_ddot': -g/l * sp.sin(theta)},
       num_threads=4,
       num_systems=1000
   )
   gen.generate("pendulum_openmp.cpp")

Parameters
~~~~~~~~~~

- ``num_threads``: Number of OpenMP threads (default: 0 = auto-detect)
- ``num_systems``: Number of parallel trajectories to simulate (default: 100)

Generated Code Structure
------------------------

The generated code includes:

1. **Parameters as constants**: Physical parameters inlined for optimization
2. **Derivatives function**: ``compute_derivatives()`` for a single system
3. **RK4 integrator**: ``rk4_step()`` with inline hints
4. **Parallel loop**: ``#pragma omp parallel for schedule(dynamic)`` over trajectories
5. **Performance timing**: Wall-clock time via ``omp_get_wtime()``
6. **CSV output**: Results for the first trajectory

Compilation
-----------

.. code-block:: bash

   # GCC
   g++ -fopenmp -O3 -o simulation simulation.cpp

   # Clang
   clang++ -fopenmp -O3 -o simulation simulation.cpp

   # Intel
   icpx -qopenmp -O3 -o simulation simulation.cpp

Thread Control
--------------

Set the number of threads at runtime:

.. code-block:: bash

   # Use 8 threads
   OMP_NUM_THREADS=8 ./simulation

   # Bind threads to cores for consistent performance
   OMP_PROC_BIND=close OMP_NUM_THREADS=4 ./simulation

Use Cases
---------

- **Parameter sweeps**: Run thousands of simulations with varied initial conditions
- **Monte Carlo**: Stochastic trajectory ensembles
- **Sensitivity analysis**: Parallel perturbation studies
- **SPH fluid dynamics**: Particle-parallel force computation

See Also
--------

- :doc:`cpp` - Single-threaded C++ generation
- :doc:`cuda` - GPU acceleration for massive parallelism
- :doc:`overview` - All code generation targets
