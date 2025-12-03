Changelog

All notable changes to MechanicsDSL will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

[1.1.0] - 2025-12-02

Added

C++ Transpiler: Added compile_to_cpp() method to generate standalone C++ simulations.

Multi-Target Support:

standard: Optimized C++ for benchmarking.

openmp: Parallel parameter sweeps for chaotic analysis.

raylib: Interactive real-time visualization executables.

arduino: Embedded C++ code for microcontrollers.

wasm: WebAssembly export for browser-based simulations.

python: C++ extensions via pybind11.

Codegen Module: New mechanics_dsl.codegen package for handling source-to-source translation.

Advanced Templates: Included C++ templates for all supported targets.

Changed

Compiler Architecture: Upgraded from pure Python execution to a dual-mode engine (Python/C++).

Performance: Large system simulations ($N > 100$ DOF) are now feasible via C++ export.

Documentation: Added tutorials for C++ export (examples/21_cpp_export.py) and advanced targets (examples/22_advanced_targets.py).
