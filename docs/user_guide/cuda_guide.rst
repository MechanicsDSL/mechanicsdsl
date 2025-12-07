CUDA Development Guide
======================

Complete guide for GPU-accelerated physics with MechanicsDSL CUDA backend.

Prerequisites
-------------

- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler

Installation Check
------------------

.. code-block:: bash

    nvcc --version
    nvidia-smi

Basic Usage
-----------

1. Generate CUDA Code
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from mechanics_dsl.codegen import CudaGenerator
    import sympy as sp
    
    theta = sp.Symbol('theta')
    g, l = sp.Symbol('g'), sp.Symbol('l')
    
    gen = CudaGenerator(
        system_name="pendulum",
        coordinates=['theta'],
        parameters={'g': 9.81, 'l': 1.0},
        initial_conditions={'theta': 0.3, 'theta_dot': 0.0},
        equations={'theta_ddot': -g/l * sp.sin(theta)},
        generate_cpu_fallback=True
    )
    
    gen.generate("cuda_pendulum/")

2. Build
~~~~~~~~

.. code-block:: bash

    cd cuda_pendulum
    mkdir build && cd build
    cmake ..
    make

3. Run
~~~~~~

.. code-block:: bash

    ./pendulum_cuda     # GPU version
    ./pendulum_cpu      # CPU fallback

Generated Files
---------------

+----------------------+---------------------------------------------+
| File                 | Description                                 |
+======================+=============================================+
| ``system.cu``        | CUDA source with kernels                    |
| ``system.h``         | Header file                                 |
| ``system_cpu.cpp``   | CPU fallback implementation                 |
| ``CMakeLists.txt``   | CMake build configuration                   |
+----------------------+---------------------------------------------+

CUDA SPH (Particle Simulations)
-------------------------------

For fluid/particle simulations:

.. code-block:: python

    from mechanics_dsl.codegen.cuda_sph import CudaSPHGenerator
    
    # Create particles
    fluid_particles = [
        {'x': i*0.04, 'y': j*0.04}
        for i in range(20) for j in range(20)
    ]
    
    gen = CudaSPHGenerator(
        system_name="dam_break",
        fluid_particles=fluid_particles,
        parameters={
            'h': 0.04,      # Smoothing length
            'rho0': 1000,   # Reference density
            'c0': 20        # Sound speed
        }
    )
    
    gen.generate("cuda_sph_output/")

The generated code includes:

- Density/pressure computation kernel
- Force computation kernel with Poly6 and Spiky kernels
- Symplectic Euler integration

Architecture Selection
----------------------

Specify GPU architecture in CMakeLists.txt:

.. code-block:: cmake

    target_compile_options(simulation PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>)

Common architectures:

- ``sm_60`` - Pascal (GTX 1000 series)
- ``sm_70`` - Volta (V100)
- ``sm_75`` - Turing (RTX 2000 series)
- ``sm_80`` - Ampere (RTX 3000 series)
- ``sm_86`` - Ampere (RTX 3000 mobile)
- ``sm_89`` - Ada Lovelace (RTX 4000 series)

Troubleshooting
---------------

**"No CUDA device found"**
    Install NVIDIA drivers and verify with ``nvidia-smi``

**Compilation errors**
    Ensure CUDA Toolkit is in PATH and cmake can find it

**Poor performance**
    - Increase block size (default 256)
    - Use shared memory for frequently accessed data
    - Coalesce memory accesses
