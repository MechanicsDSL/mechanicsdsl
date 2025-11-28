---
title: 'MechanicsDSL: A Domain-Specific Language for Symbolic and Numerical Analysis of Classical Mechanics'
tags:
  - python
  - physics
  - lagrangian mechanics
  - hamiltonian mechanics
  - dsl
  - symbolic computation
authors:
  - name: Noah Parsons
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 27 November 2025
bibliography: paper.bib
---

# Summary

`MechanicsDSL` is a Python library that implements a domain-specific language (DSL) for defining, analyzing, and simulating classical mechanical systems. It bridges the gap between symbolic derivation and numerical simulation by providing a unified pipeline that accepts LaTeX-inspired syntax (e.g., `\lagrangian{...}`) and automatically derives equations of motion, compiles them into optimized numerical functions, and performs time-domain simulations.

# Statement of Need

In computational physics education and research, students and scientists often manually derive equations of motion (using Lagrangian or Hamiltonian mechanics) before translating them into code for numerical solvers like `scipy.integrate.solve_ivp`. This manual translation is error-prone, especially for complex coupled systems like N-body problems or multi-link pendulums.

Existing tools usually focus either purely on symbolic algebra (e.g., SymPy) or numerical integration (e.g., SciPy), requiring users to write "glue code" to connect the two. `MechanicsDSL` automates this entire pipeline. It features:

1.  **Robust Parsing**: A custom recursive descent parser that safely converts DSL text into an Abstract Syntax Tree (AST), avoiding security risks associated with `eval()`.
2.  **Symbolic Engine**: Automatic derivation of Euler-Lagrange and Hamilton's equations, including support for non-holonomic constraints and non-conservative forces (friction, drive).
3.  **Adaptive Simulation**: An intelligent numerical engine that selects integration strategies (e.g., LSODA for stiff systems) based on system topology.
4.  **Verification**: Built-in energy conservation metrics to validate simulation stability.

The software has been verified against analytical solutions (Harmonic Oscillator) and strict conservation laws (Figure-8 three-body orbit), achieving energy drift as low as $10^{-9}\%$. It is designed for use in physics curriculum, prototyping mechanical controls, and investigating chaotic dynamics.
