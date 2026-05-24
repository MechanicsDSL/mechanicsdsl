Fortran Code Generation
=======================

Generate Fortran 90+ simulation code for HPC and numerical computing.

Features
--------

- **Multiple precisions**: ``real(4)`` (single), ``real(8)`` (double), ``real(16)`` (quad)
- **RK4 and adaptive RK45**: Built-in Dormand-Prince integrator
- **OpenMP support**: Optional parallel parameter sweeps
- **MPI support**: Optional distributed computing
- **Module structure**: Clean Fortran 90 modules with ``contains``
- **Energy conservation**: Built-in energy tracking and drift reporting

Basic Usage
-----------

.. code-block:: python

   from mechanics_dsl.codegen.fortran import FortranGenerator
   import sympy as sp

   theta, g, l = sp.symbols('theta g l')

   gen = FortranGenerator(
       system_name="pendulum",
       coordinates=["theta"],
       parameters={"g": 9.81, "l": 1.0},
       initial_conditions={"theta": 0.5, "theta_dot": 0.0},
       equations={"theta_ddot": -g/l * sp.sin(theta)},
       precision=8,
       use_openmp=False
   )
   gen.generate("pendulum.f90")

Parameters
~~~~~~~~~~

- ``precision``: Fortran ``kind`` parameter — ``4`` (single), ``8`` (double), ``16`` (quad). Default: ``8``
- ``use_openmp``: Enable OpenMP parallelization (default: ``False``)
- ``use_mpi``: Enable MPI distributed computing (default: ``False``)

Generated Code Structure
------------------------

The generated Fortran program has two units:

**Module** (``{system_name}_physics``):

1. **Working precision constant**: ``WP`` parameter for consistent typing
2. **Physical parameters**: Module-level ``parameter`` constants
3. **``equations_of_motion``**: Subroutine computing state derivatives
4. **``rk4_step``**: Fixed-step RK4 integrator
5. **``rk45_step``**: Adaptive Dormand-Prince integrator with error control
6. **``compute_energy``**: Energy computation function

**Main program** (``{system_name}_simulation``):

1. Initial conditions and simulation parameters
2. CSV output with energy column
3. Energy drift reporting

Compilation
-----------

.. code-block:: bash

   # Double precision (default)
   gfortran -O3 -o pendulum pendulum.f90

   # With OpenMP
   gfortran -O3 -fopenmp -o pendulum pendulum.f90

   # Quad precision
   gfortran -O3 -fdefault-real-16 -o pendulum pendulum.f90

   # Intel Fortran
   ifx -O3 -o pendulum pendulum.f90

Precision Guide
---------------

.. list-table::
   :header-rows: 1

   * - Kind
     - Precision
     - Use Case
   * - ``4``
     - ~7 digits
     - Fast prototyping, embedded
   * - ``8``
     - ~15 digits
     - General simulations (default)
   * - ``16``
     - ~33 digits
     - High-precision celestial mechanics

See Also
--------

- :doc:`cpp` - C++ code generation
- :doc:`openmp` - OpenMP-parallel C++
- :doc:`overview` - All code generation targets
