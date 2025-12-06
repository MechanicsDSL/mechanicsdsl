Code Generation Overview
========================

MechanicsDSL can generate high-performance code in multiple target languages,
enabling deployment beyond Python for production applications.

Supported Targets
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Target
     - Use Case
     - Performance
   * - **C++**
     - Native applications, embedded systems
     - ~10-100x faster than Python
   * - **OpenMP**
     - Multi-core parallelization
     - Scales with CPU cores
   * - **WebAssembly**
     - Browser-based simulations
     - Near-native in browsers
   * - **CUDA** (planned)
     - GPU acceleration
     - Massive parallelism

Basic Usage
-----------

Generate C++ code from a compiled system:

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
   
   # Generate C++ source
   compiler.compile_to_cpp("pendulum.cpp", target="standard")
   
   # Or compile to binary directly
   compiler.compile_to_cpp("pendulum.cpp", target="standard", compile_binary=True)

Generated Code Structure
------------------------

The generated C++ code includes:

1. **Header with dependencies**: Eigen, standard library
2. **State vector definition**: Position and velocity arrays
3. **Derivatives function**: Right-hand side of ODEs
4. **Integrator**: RK4 or Velocity Verlet
5. **Main function**: I/O and time stepping

Example output structure:

.. code-block:: cpp

   #include <Eigen/Dense>
   #include <fstream>
   #include <cmath>
   
   using State = Eigen::VectorXd;
   
   // Parameters (inlined as constexpr)
   constexpr double m = 1.0;
   constexpr double l = 1.0;
   constexpr double g = 9.81;
   
   // Derivatives: returns d(state)/dt
   State derivatives(double t, const State& y) {
       State dydt(2);
       double theta = y[0];
       double theta_dot = y[1];
       
       // Derived equations of motion
       dydt[0] = theta_dot;
       dydt[1] = -(g/l) * sin(theta);
       
       return dydt;
   }
   
   // RK4 integrator
   State rk4_step(double t, const State& y, double dt) { ... }
   
   int main() { ... }

Compilation Requirements
------------------------

**C++ Standard**: Requires C++17 or later.

**Dependencies**:

- Eigen (header-only linear algebra)
- Standard library only (no external dependencies)

**Compiler options**:

.. code-block:: bash

   # GCC/Clang
   g++ -O3 -march=native -std=c++17 pendulum.cpp -o pendulum
   
   # MSVC
   cl /O2 /std:c++17 pendulum.cpp

Performance Tips
----------------

1. **Use -O3 optimization**: Essential for inlining and vectorization
2. **Enable -march=native**: Uses CPU-specific instructions
3. **Link-time optimization**: Add ``-flto`` for cross-file inlining
4. **Profile first**: Use profilers before micro-optimizing

See Also
--------

- :doc:`cpp` - Detailed C++ code generation
- :doc:`wasm` - WebAssembly for browsers
- :doc:`python` - NumPy-accelerated Python
