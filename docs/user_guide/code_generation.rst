Code Generation Guide
=====================

MechanicsDSL can generate simulation code for 12 different target platforms.

Available Backends
------------------

**Standard Languages**

- **C++** (`CppGenerator`) - High-performance native code
- **Python** (`PythonGenerator`) - NumPy/SciPy compatible
- **Julia** (`JuliaGenerator`) - DifferentialEquations.jl
- **Rust** (`RustGenerator`) - Memory-safe native code
- **MATLAB** (`MatlabGenerator`) - MATLAB/Octave compatible
- **Fortran** (`FortranGenerator`) - HPC scientific computing
- **JavaScript** (`JavaScriptGenerator`) - Browser/Node.js

**GPU Acceleration**

- **CUDA** (`CudaGenerator`) - NVIDIA GPU kernels
- **CUDA SPH** (`CudaSPHGenerator`) - GPU particle simulations
- **OpenMP** (`OpenMPGenerator`) - Multi-core parallelism

**Embedded & Web**

- **WebAssembly** (`WasmGenerator`) - Browser-based simulations
- **Arduino** (`ArduinoGenerator`) - Embedded microcontrollers

Quick Start
-----------

.. code-block:: python

    from mechanics_dsl.codegen import CppGenerator
    import sympy as sp
    
    # Define system
    theta = sp.Symbol('theta')
    g, l = sp.Symbol('g'), sp.Symbol('l')
    
    gen = CppGenerator(
        system_name="pendulum",
        coordinates=['theta'],
        parameters={'g': 9.81, 'l': 1.0},
        initial_conditions={'theta': 0.3, 'theta_dot': 0.0},
        equations={'theta_ddot': -g/l * sp.sin(theta)}
    )
    
    gen.generate("pendulum.cpp")

CUDA Backend
------------

For GPU-accelerated simulations:

.. code-block:: python

    from mechanics_dsl.codegen import CudaGenerator
    
    gen = CudaGenerator(
        system_name="pendulum",
        coordinates=['theta'],
        parameters={'g': 9.81, 'l': 1.0},
        initial_conditions={'theta': 0.3, 'theta_dot': 0.0},
        equations={'theta_ddot': -g/l * sp.sin(theta)},
        generate_cpu_fallback=True  # Also create CPU version
    )
    
    gen.generate("cuda_output/")

This generates:

- ``pendulum.cu`` - CUDA kernels
- ``pendulum.h`` - Header file
- ``pendulum_cpu.cpp`` - CPU fallback
- ``CMakeLists.txt`` - Build configuration

OpenMP Parallel Code
--------------------

For multi-core CPU parallelism:

.. code-block:: python

    from mechanics_dsl.codegen import OpenMPGenerator
    
    gen = OpenMPGenerator(
        system_name="pendulum",
        coordinates=['theta'],
        parameters={'g': 9.81, 'l': 1.0},
        initial_conditions={'theta': 0.3, 'theta_dot': 0.0},
        equations={'theta_ddot': -g/l * sp.sin(theta)},
        num_threads=8  # Or 0 for auto-detect
    )
    
    gen.generate("pendulum_openmp.cpp")

Compile with: ``g++ -fopenmp -O3 -o pendulum pendulum_openmp.cpp``

WebAssembly for Browsers
------------------------

.. code-block:: python

    from mechanics_dsl.codegen import WasmGenerator
    
    gen = WasmGenerator(
        system_name="pendulum",
        coordinates=['theta'],
        parameters={'g': 9.81, 'l': 1.0},
        initial_conditions={'theta': 0.3, 'theta_dot': 0.0},
        equations={'theta_ddot': -g/l * sp.sin(theta)}
    )
    
    gen.generate("wasm_output/")

This creates a complete web application with HTML canvas visualization.

Arduino Embedded
----------------

.. code-block:: python

    from mechanics_dsl.codegen import ArduinoGenerator
    
    gen = ArduinoGenerator(
        system_name="pendulum",
        coordinates=['theta'],
        parameters={'g': 9.81, 'l': 1.0},
        initial_conditions={'theta': 0.3, 'theta_dot': 0.0},
        equations={'theta_ddot': -g/l * sp.sin(theta)},
        servo_pin=9  # Optional: drive a servo
    )
    
    gen.generate("pendulum.ino")

Upload to Arduino and open Serial Plotter to visualize.

Comparison Table
----------------

+------------+------------+---------+------------+
| Backend    | Speed      | Memory  | Setup      |
+============+============+=========+============+
| C++        | Very Fast  | Low     | g++        |
| CUDA       | Fastest    | GPU     | CUDA       |
| OpenMP     | Fast       | Low     | g++/OpenMP |
| Numba      | Fast       | Medium  | pip        |
| Rust       | Very Fast  | Low     | rustc      |
| Julia      | Fast       | Medium  | Julia      |
| MATLAB     | Medium     | High    | MATLAB     |
| Python     | Slow       | High    | pip        |
| JavaScript | Medium     | Medium  | Node.js    |
| WASM       | Fast       | Low     | Emscripten |
| Arduino    | Slow       | Very Low| Arduino    |
| Fortran    | Very Fast  | Low     | gfortran   |
+------------+------------+---------+------------+
