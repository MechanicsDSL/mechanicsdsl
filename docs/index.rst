.. MechanicsDSL documentation master file

MechanicsDSL: A Language for Computational Mechanics
====================================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/MechanicsDSL/mechanicsdsl/blob/main/LICENSE
   :alt: MIT License

.. image:: https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg
   :target: https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml
   :alt: CI Status

**MechanicsDSL** is a domain-specific language and compiler designed to bridge the gap between symbolic derivation and numerical simulation in classical mechanics. 

Instead of manually deriving equations of motion and hard-coding them into ODE solvers, users describe physical systems using a LaTeX-inspired syntax. The compiler automatically:

1.  **Parses** the physical description (Lagrangian, constraints, forces).
2.  **Derives** the equations of motion symbolically (Euler-Lagrange or Hamilton's equations).
3.  **Compiles** optimized numerical functions.
4.  **Simulates** the dynamics with adaptive stiffness detection.

Designed for physics education, research prototyping, and the study of chaotic or constrained systems.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/getting_started
   guide/syntax
   guide/best_practices

.. toctree::
   :maxdepth: 2
   :caption: Physics Theory

   physics/lagrangian
   physics/hamiltonian
   physics/fluid_dynamics
   physics/constraints

.. toctree::
   :maxdepth: 2
   :caption: Compiler Internals

   internals/architecture
   internals/parser_logic
   internals/symbolic_math
   internals/transpiler

.. toctree::
   :maxdepth: 2
   :caption: Backend Targets

   backends/standard_cpp
   backends/openmp
   backends/wasm
   backends/arduino

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   mechanics_dsl

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
