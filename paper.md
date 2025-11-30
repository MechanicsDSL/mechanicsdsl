---
title: 'MechanicsDSL: A Domain-Specific Language for Symbolic and Numerical Analysis of Classical Mechanics'
tags:
  - python
  - physics
  - lagrangian mechanics
  - hamiltonian mechanics
  - dsl
  - symbolic computation
  - chaotic dynamics
authors:
  - name: Noah Parsons
    orcid: 0009-0000-7224-6040
    affiliation: 1
affiliations:
 - name: Independent Researcher, Newcastle, WY, USA
   index: 1
date: 30 November 2025
bibliography: paper.bib
---

# Summary

`MechanicsDSL` is a Python-based compiler and domain-specific language (DSL) designed to automate the modeling of classical mechanical systems. It allows researchers and educators to define systems using a high-level, LaTeX-inspired syntax, automatically derives the equations of motion using symbolic mathematics, and generates optimized numerical solvers for simulation. The software unifies the pipeline from symbolic derivation (Lagrangian/Hamiltonian mechanics) to numerical analysis (phase space, energy conservation, chaos), effectively bridging the gap between algebraic formalism and computational simulation.

# Statement of Need

In computational physics education and research, the transition from analytical derivation to numerical simulation is often a manual, error-prone process. While tools like `SymPy` [@sympy] provide symbolic capabilities and `SciPy` [@scipy] offers numerical integrators, "gluing" them together requires significant boilerplate code. This barrier is particularly high for complex coupled systems, such as N-body problems or multi-link pendulums, where deriving the mass matrix and forcing vector manually is intractable.

`MechanicsDSL` addresses this by providing a unified abstraction layer. Unlike `sympy.physics.mechanics`, which requires Python proficiency to define symbols and reference frames, `MechanicsDSL` uses a declarative syntax accessible to students familiar with standard physics notation. It is specifically designed for:

1.  **Physics Education**: Allowing students to focus on physical laws (Lagrangians) rather than numerical implementation details.
2.  **Rapid Prototyping**: Enabling researchers to test hypotheses by changing a single line of DSL (e.g., adding a damping term or changing a constraint) without rewriting solver code.
3.  **Complex Dynamics**: Facilitating the exploration of chaotic systems (e.g., double pendulums, Lorenz attractors) where numerical stability is critical.

# System Architecture

The software implements a multi-stage compiler pipeline:

1.  **Lexing & Parsing**: A custom recursive descent parser converts DSL text into an Abstract Syntax Tree (AST), ensuring input validation and preventing arbitrary code execution risks associated with `eval()`.
2.  **Symbolic Engine**: The AST is traversed to construct symbolic expressions. The engine automatically computes Euler-Lagrange equations [@goldstein]:
    $$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = Q_i$$
    It handles standard Lagrangians, non-conservative forces ($Q_i$), and holonomic constraints via Lagrange multipliers.
3.  **Numerical Compilation**: Symbolic equations are compiled into optimized Python functions using `numpy` [@numpy] backend.
4.  **Adaptive Simulation**: The simulation engine analyzes system characteristics (degrees of freedom, stiffness) to intelligently select between explicit (RK45) and implicit (LSODA) integration schemes [@lsoda].

# Example Usage

A damped driven harmonic oscillator can be defined and simulated with the following DSL code:

```latex
\system{damped_oscillator}
\defvar{x}{Position}{m}
\parameter{m}{1.0}{kg} \parameter{k}{10.0}{N/m} \parameter{b}{0.5}{N*s/m}

% Define Lagrangian T - V
\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}

% Add non-conservative damping
\force{-b * \dot{x}}

\initial{x=1.0, x_dot=0.0}
```
The software automatically derives the equation of motion and performs the time-domain simulation.

# Validation

The software has been rigorously verified against analytical solutions (Harmonic Oscillator) and strict conservation laws. notably, it successfully simulates the Chenciner & Montgomery "Figure-8" three-body orbit, maintaining a periodicity error of 6.31×10−6 over 100 orbital periods by automatically adapting the solver strategy to handle the system's stiff gradients.

# Acknowledgements

This project was developed as an independent research initiative. The author thanks the open-source community for maintaining the foundational libraries SymPy, NumPy, and SciPy that make this work possible.
