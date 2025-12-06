C++ Code Generation
===================

Generate optimized C++ code for production deployment.

Features
--------

- **Zero-copy state vectors**: Eigen arrays with memory alignment
- **Compile-time parameters**: Constants inlined for optimization
- **Template metaprogramming**: Type-safe, efficient code
- **Multiple integrators**: RK4, RK45, Velocity Verlet

Basic C++ Generation
--------------------

.. code-block:: python

   compiler = PhysicsCompiler()
   compiler.compile_dsl(source)
   
   # Generate standard C++ (single-threaded)
   compiler.compile_to_cpp("simulation.cpp", target="standard")

The ``target`` parameter options:

- ``"standard"``: Single-threaded, portable C++17
- ``"openmp"``: Multi-threaded with OpenMP pragmas
- ``"simd"``: Explicit SIMD vectorization hints

OpenMP Parallelization
----------------------

For systems with many particles (SPH) or many bodies (N-body):

.. code-block:: python

   compiler.compile_to_cpp("simulation.cpp", target="openmp")

Generated code includes:

.. code-block:: cpp

   #pragma omp parallel for
   for (int i = 0; i < N; ++i) {
       compute_forces(particles[i]);
   }

Compile with OpenMP support:

.. code-block:: bash

   g++ -O3 -fopenmp simulation.cpp -o simulation

SPH Fluid Code
--------------

Fluid simulations generate specialized SPH code:

.. code-block:: python

   fluid_code = r'''
   \system{dam_break}
   \parameter{h}{0.04}{m}
   \fluid{water}
   \region{rectangle}{x=0..0.4, y=0..0.8}
   \boundary{walls}
   \region{line}{x=-0.05, y=0..1.5}
   '''
   
   compiler.compile_dsl(fluid_code)
   compiler.compile_to_cpp("dam_break.cpp", target="standard", compile_binary=True)

The generated SPH code includes:

1. **Particle structure**: Position, velocity, density, pressure
2. **Spatial hash grid**: O(N) neighbor finding
3. **Kernel functions**: Poly6, Spiky gradient
4. **Velocity Verlet**: Symplectic time integration
5. **CSV output**: Frame-by-frame particle data

Manual Compilation
------------------

If ``compile_binary=False``, compile manually:

.. code-block:: bash

   # Linux/macOS
   g++ -O3 -std=c++17 dam_break.cpp -o dam_break
   ./dam_break
   
   # Windows (MSVC)
   cl /O2 /std:c++17 dam_break.cpp
   dam_break.exe

The simulation outputs ``dam_break_sph.csv`` for visualization.

Customizing Generated Code
--------------------------

Override integrator settings:

.. code-block:: python

   compiler.compile_to_cpp(
       "simulation.cpp",
       target="standard",
       dt=0.001,           # Time step
       t_end=10.0,         # End time
       output_interval=100  # Write every N steps
   )

Benchmarks
----------

Typical speedups vs Python (scipy.integrate.solve_ivp):

.. list-table::
   :header-rows: 1

   * - System
     - Python (s)
     - C++ (s)
     - Speedup
   * - Simple pendulum (10s)
     - 0.05
     - 0.001
     - 50x
   * - Double pendulum (100s)
     - 2.1
     - 0.08
     - 26x
   * - 3-body (100s)
     - 5.3
     - 0.15
     - 35x
   * - SPH 1000 particles (2s)
     - N/A
     - 4.2
     - N/A

Troubleshooting
---------------

**Compilation errors**:

1. Ensure C++17 support: ``g++ --version`` (need 7+)
2. Check Eigen is installed or in include path
3. On Windows, use Developer Command Prompt for MSVC

**Runtime errors**:

1. Check for NaN in output (divergent simulation)
2. Reduce time step if unstable
3. Verify initial conditions are physical
