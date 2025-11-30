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

It is designed for physics education, research prototyping, and the study of chaotic or constrained systems.

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   installation
   tutorials
   user_guide
   physics_background

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
