.. MechanicsDSL documentation master file

MechanicsDSL: A Language for Computational Mechanics
====================================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/MechanicsDSL/mechanicsdsl/blob/main/LICENSE
   :alt: MIT License

.. image:: https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg
   :target: https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml
   :alt: CI Status


**MechanicsDSL** is a domain-specific language and multi-target compiler for computational mechanics. Write physics in a LaTeX-inspired syntax, and automatically generate optimized simulation code for multiple platforms.

The compiler implements a complete pipeline from symbolic mathematics to executable code:

1.  **Frontend**: A recursive descent parser converts LaTeX-style DSL into an Abstract Syntax Tree (no ``eval()`` - completely safe).
2.  **Symbolic Engine**: Automatically derives equations of motion using SymPy (Euler-Lagrange or Hamilton's equations with constraint handling via Lagrange multipliers).
3.  **Adaptive Solver**: Intelligently selects between RK45 and LSODA based on system stiffness detection.
4.  **Multi-Target Codegen**: Transpiles to multiple backends using strategy patterns:
    - **Standard C++**: RK4 integration with double precision
    - **WebAssembly**: Zero-copy browser simulations via Emscripten
    - **Arduino/Embedded**: Float precision with Euler integration for microcontrollers
    - **SPH Fluids**: Mesh-free Lagrangian fluid solver with spatial hashing

The system supports both rigid body dynamics (ODEs) and particle-based fluid dynamics (SPH), holonomic constraints, Hamiltonian formulations with symplectic integrators, and chaotic systems requiring long-term energy conservation.

Designed for physics education, research prototyping, robotics simulation, and computational mechanics where rapid iteration between mathematical formulation and high-performance simulation is critical.

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   tutorials
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Physics Theory

   physics_background
   lagrangian_mechanics
   hamiltonian_mechanics
   constraint_physics
   fluid_dynamics

.. toctree::
   :maxdepth: 2
   :caption: Compiler Internals

   compiler_architecture
   advanced
   code_generator

.. toctree::
   :maxdepth: 2
   :caption: Backend Targets

   standard_c++
   web_assembly
   embedded_(arduino)

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   mechanics_dsl

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
