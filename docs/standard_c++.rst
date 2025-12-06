Standard C++ Target
===================

The default compilation target is optimized for accuracy and benchmarking.

* **Compiler**: ``g++`` (Linux/Mac) or ``MSVC`` (Windows).
* **Flags**: ``-O3 -ffast-math -march=native``.
* **Integrator**: Explicit Runge-Kutta 4 (RK4).

This target generates a standalone executable that produces a CSV file. It is the most robust target for general mechanics.
