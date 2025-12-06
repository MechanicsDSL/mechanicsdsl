Performance Optimization
========================

Tips and techniques for maximizing simulation performance.

Profiling Your Simulation
-------------------------

Before optimizing, identify bottlenecks:

.. code-block:: python

   from mechanics_dsl import PhysicsCompiler
   from mechanics_dsl.utils.profiling import profile_simulation
   
   compiler = PhysicsCompiler()
   compiler.compile_dsl(source)
   
   # Profile the simulation
   with profile_simulation() as prof:
       solution = compiler.simulate(t_span=(0, 10))
   
   # Print timing breakdown
   prof.print_stats()

Typical output:

.. code-block:: text

   MechanicsDSL Profiling Report
   =============================
   Compilation:     0.234 s (12.3%)
   Simulation:      1.567 s (82.4%)
     - RHS evals:   1.234 s (64.9%)
     - Integration: 0.333 s (17.5%)
   Visualization:   0.100 s (5.3%)

Solver Selection
----------------

Choose the right integrator for your problem:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Solver
     - Best For
     - Notes
   * - **RK45**
     - General purpose
     - Default, adaptive step
   * - **DOP853**
     - High accuracy needs
     - 8th order, fewer evals
   * - **LSODA**
     - Unknown stiffness
     - Auto-switches methods
   * - **BDF**
     - Stiff systems
     - Implicit, stable
   * - **Radau**
     - Very stiff systems
     - Implicit, high order

Set solver in DSL:

.. code-block:: latex

   \solve{DOP853}

Or in Python:

.. code-block:: python

   solution = compiler.simulate(t_span=(0, 10), method='DOP853')

Tolerance Tuning
----------------

Balance accuracy vs speed with tolerances:

.. code-block:: python

   # Faster but less accurate
   solution = compiler.simulate(rtol=1e-3, atol=1e-6)
   
   # Slower but very accurate  
   solution = compiler.simulate(rtol=1e-12, atol=1e-14)

Rule of thumb:

- **Visualization only**: ``rtol=1e-3`` is fine
- **Conservation checks**: ``rtol=1e-6`` to ``1e-9``
- **Research quality**: ``rtol=1e-10`` or tighter

Symbolic Simplification
-----------------------

Complex Lagrangians generate complex equations. Simplify:

.. code-block:: python

   compiler = PhysicsCompiler(simplify=True)  # Default
   
   # For very complex systems, try aggressive simplification
   compiler = PhysicsCompiler(simplify='aggressive')

This calls SymPy's simplification routines which may take longer
but produce faster runtime code.

Caching
-------

Enable equation caching to avoid recompilation:

.. code-block:: python

   from mechanics_dsl.utils.caching import enable_cache
   
   enable_cache(max_size=100)  # Cache last 100 compilations
   
   # First call compiles
   compiler.compile_dsl(source)  # ~0.5s
   
   # Subsequent calls use cache
   compiler.compile_dsl(source)  # ~0.01s

C++ Code Generation
-------------------

For maximum performance, generate native code:

.. code-block:: python

   # Generate and compile C++
   compiler.compile_to_cpp("simulation.cpp", compile_binary=True)

Typical speedups: **10-100x faster** than Python.

See :doc:`codegen/cpp` for details.

Parallelization
---------------

For N-body or SPH simulations, use OpenMP:

.. code-block:: python

   compiler.compile_to_cpp("simulation.cpp", target="openmp")

Run with multiple threads:

.. code-block:: bash

   export OMP_NUM_THREADS=8
   ./simulation

Memory Efficiency
-----------------

For long simulations, avoid storing every time point:

.. code-block:: python

   # Store fewer points
   solution = compiler.simulate(
       t_span=(0, 1000),
       num_points=1000  # Instead of default 10000
   )

Or use dense output for interpolation:

.. code-block:: python

   solution = compiler.simulate(dense_output=True)
   
   # Evaluate at any time
   state_at_50 = solution.sol(50.0)

Common Performance Issues
-------------------------

**Issue**: Simulation slows down over time

- Check for energy divergence (numerical instability)
- Try smaller time step or implicit solver

**Issue**: Compilation takes too long

- Simplify Lagrangian if possible
- Enable caching
- Pre-compile equations and reuse

**Issue**: Memory usage grows

- Reduce ``num_points``
- Use streaming output for long simulations
- Process results in chunks

Benchmarks
----------

Reference performance on Intel i7-10700K:

.. list-table::
   :header-rows: 1

   * - System
     - Points
     - Python
     - C++
   * - Simple pendulum
     - 10,000
     - 50 ms
     - 1 ms
   * - Double pendulum  
     - 10,000
     - 210 ms
     - 8 ms
   * - 3-body problem
     - 10,000
     - 530 ms
     - 15 ms
   * - Figure-8 orbit
     - 10,000
     - 1.2 s
     - 35 ms
   * - SPH (1000 particles)
     - 2000 frames
     - N/A
     - 4.2 s
