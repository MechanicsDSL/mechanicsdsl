Performance Optimization Guide
==============================

Optimize MechanicsDSL simulations for maximum performance.

Numba JIT Acceleration
----------------------

Use the Numba-accelerated solver for 5-10x speedups:

.. code-block:: python

    from mechanics_dsl.solver_numba import NumbaSimulator
    import sympy as sp
    
    # Define equations
    theta = sp.Symbol('theta')
    g, l = sp.Symbol('g'), sp.Symbol('l')
    
    accelerations = {'theta_ddot': -g/l * sp.sin(theta)}
    
    # Create Numba simulator
    sim = NumbaSimulator()
    sim.set_parameters({'g': 9.81, 'l': 1.0})
    sim.set_initial_conditions({'theta': 0.3, 'theta_dot': 0.0})
    sim.compile_equations(accelerations, ['theta'])
    
    # Run simulation (5-10x faster than SciPy)
    solution = sim.simulate_numba(
        t_span=(0, 100), 
        num_points=10000,
        method='rk4'  # 'euler', 'rk4', or 'rk45'
    )

Available Methods
~~~~~~~~~~~~~~~~~

- ``euler`` - Simple Euler (fastest, least accurate)
- ``rk4`` - 4th order Runge-Kutta (recommended)
- ``rk45`` - Adaptive Dormand-Prince (most accurate)

GPU Acceleration with CUDA
--------------------------

For massive parallelism on NVIDIA GPUs:

1. Generate CUDA code:

.. code-block:: python

    from mechanics_dsl.codegen import CudaGenerator
    gen = CudaGenerator(...)
    gen.generate("cuda_output/")

2. Compile with nvcc:

.. code-block:: bash

    cd cuda_output
    mkdir build && cd build
    cmake ..
    make

3. Run on GPU:

.. code-block:: bash

    ./simulation_cuda

CPU Fallback
~~~~~~~~~~~~

If no NVIDIA GPU is available, use the CPU version:

.. code-block:: bash

    ./simulation_cpu

Multi-Core Parallelism with OpenMP
----------------------------------

For multi-core CPU simulation:

.. code-block:: python

    from mechanics_dsl.codegen import OpenMPGenerator
    
    gen = OpenMPGenerator(
        ...,
        num_threads=8  # 0 = auto-detect
    )
    gen.generate("simulation_openmp.cpp")

Compile with:

.. code-block:: bash

    g++ -fopenmp -O3 -march=native -o simulation simulation_openmp.cpp

Memory Optimization
-------------------

For large particle simulations:

1. **Use float32** instead of float64 where precision allows
2. **Structure of Arrays (SoA)** layout for cache efficiency
3. **Spatial hashing** for O(n) neighbor search

Benchmarking
------------

Run the included benchmark:

.. code-block:: bash

    cd benchmarks
    python numba_performance.py

Expected output:

+----------+------------+------------+----------+
| Points   | Numba      | SciPy      | Speedup  |
+==========+============+============+==========+
| 1,000    | 5 ms       | 45 ms      | 9x       |
| 10,000   | 50 ms      | 450 ms     | 9x       |
| 100,000  | 500 ms     | 4500 ms    | 9x       |
+----------+------------+------------+----------+

Best Practices
--------------

1. **Start with Python** for debugging 
2. **Profile first** to identify bottlenecks
3. **Use Numba** for quick wins (no code changes)
4. **Generate C++/CUDA** for production
5. **Batch simulations** with OpenMP for parameter sweeps
