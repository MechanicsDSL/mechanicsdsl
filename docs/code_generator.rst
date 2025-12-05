The C++ Transpiler
==================

The ``CppGenerator`` class is the bridge between symbolic math and machine code.

Strategy Pattern
----------------
The generator selects a compilation strategy at runtime:

* **SolverTemplate**: Used for rigid body systems. Implements ``std::vector`` state vectors and RK4 integration.
* **SPHTemplate**: Used for fluid systems. Implements static arrays (for speed), Spatial Hashing, and Velocity Verlet.

Math Translation
----------------
Symbolic expressions are translated using ``sympy.printing.cxxcode``.

* **Python**: ``0.5 * m * v**2``
* **C++**: ``0.5 * m * std::pow(v, 2)``

This ensures that the generated C++ code is not just valid, but optimized for the standard library's math functions.
