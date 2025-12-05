.. MechanicsDSL documentation master file

MechanicsDSL: A Language for Computational Mechanics
====================================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/MechanicsDSL/mechanicsdsl/blob/main/LICENSE
   :alt: MIT License

.. image:: https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg
   :target: https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml
   :alt: CI Status


**MechanicsDSL** is a domain-specific language and compiler for modeling physical systems. It unifies symbolic derivation (Lagrangian/Hamiltonian mechanics) with high-performance numerical simulation (C++, SPH, Symplectic Integrators).

Instead of manually deriving equations of motion and hard-coding them into ODE solvers, users describe physical systems using a LaTeX-inspired syntax. The compiler automatically:

1.  **Parses** the physical description (Lagrangian, constraints, forces).
2.  **Derives** the equations of motion symbolically (Euler-Lagrange or Hamilton's equations).
3.  **Compiles** optimized numerical functions.
4.  **Simulates** the dynamics with adaptive stiffness detection.

Designed for physics education, research prototyping, and the study of chaotic or constrained systems.
MechanicsDSL Documentation
==========================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   tutorials
   user_guide
   contributing

.. toctree::
   :maxdepth: 2
   :caption: Physics Theory

   lagrangian
   hamiltonian
   fluid_dynamics
   constraints
   physics_background

.. toctree::
   :maxdepth: 2
   :caption: Compiler Internals

   architecture
   parser_logic
   symbolic_math
   transpiler

.. toctree::
   :maxdepth: 2
   :caption: Backend Targets

   standard_cpp
   openmp
   wasm
   arduino

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   mechanics_dsl

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
