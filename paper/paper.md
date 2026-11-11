---
title: 'MechanicsDSL: from a Lagrangian to bare-metal C in one pipeline'
tags:
  - Python
  - classical mechanics
  - symbolic computation
  - code generation
  - embedded systems
  - domain-specific language
authors:
  - name: Noah Parsons
    orcid: 0009-0000-7224-6040
    affiliation: 1
affiliations:
  - index: 1
    name: Independent Researcher, United States
date: 10 November 2026
bibliography: paper.bib
---

# Summary

MechanicsDSL is a Python package that turns a description of a mechanical system into working simulation code. The user writes a Lagrangian or Hamiltonian in a LaTeX-inspired notation, declaring coordinates, parameters, constraints and initial conditions. A symbolic engine built on SymPy [@meurer2017sympy] derives the equations of motion, a compiler turns them into a numerical system that can be integrated with SciPy [@virtanen2020scipy], and a set of code generators exports the same system to twelve targets: C++, Python, Rust, Julia, CUDA, Fortran, MATLAB, JavaScript, OpenMP, WebAssembly, Arduino and ARM. The ARM target produces C with no dependency on the C mathematics library, which cross-compiles for an ARM Cortex-M4 microcontroller. A JAX [@jax2018github] backend provides just-in-time compilation and automatic differentiation for parameter estimation and sensitivity analysis.

The package is distributed on PyPI as `mechanicsdsl-core`, is released under the MIT licence, and is archived at doi:10.5281/zenodo.17771040.

# Statement of need

Deriving equations of motion by hand is slow and error-prone, and translating them into code for a particular platform is slower still. In teaching, a student who has written a Lagrangian on paper usually cannot run it without first learning a simulation library. In research and engineering, a model that has been derived and tested in Python must often be rewritten by hand in C for a controller or in CUDA for a parameter sweep, and each rewrite is a new opportunity to introduce an error that compiles cleanly and simulates a different system.

MechanicsDSL addresses both problems with a single source of truth. The physics is stated once, in notation close to the textbook, and every target is generated from the same derived equations. Its intended users are students and instructors of upper-division classical mechanics, researchers who need a quick, checkable model of a mechanical system, and engineers who need that model on embedded hardware without a manual port.

# State of the field

Several mature tools derive or evaluate multibody dynamics. `sympy.physics.mechanics` and the PyDy project [@gede2013pydy] derive equations of motion symbolically from a description of bodies, frames and forces. Drake [@drake] and Pinocchio [@carpentier2019pinocchio] provide fast numerical rigid-body dynamics for robotics, built around articulated-body descriptions. ModelingToolkit [@ma2021modelingtoolkit] offers a symbolic-numeric modelling framework with code generation in Julia. RobCoGen [@frigerio2016robcogen] is the closest in spirit: a domain-specific language for articulated robots from which efficient C++ kinematics and dynamics are generated.

MechanicsDSL differs in its starting point and its end point. It starts from an arbitrary Lagrangian or Hamiltonian rather than from a robot or body-and-joint description, so it accepts systems that are not articulated trees, such as orbital problems, coupled oscillators and constrained particles. And it ends, if required, in bare-metal C for a microcontroller, with the same derived equations also available in eleven other languages. It is not intended to replace the robotics libraries for large articulated systems, where recursive algorithms are far faster than symbolic derivation.

# Software design

The package is organised as a pipeline of independent stages. A parser reads the notation into an abstract syntax tree. The symbolic engine applies the Euler–Lagrange or Hamilton equations, including holonomic constraints through Lagrange multipliers and Rayleigh dissipation. The compiler turns the symbolic system into numerical functions and integrates them. Each code generator is a separate class registered with the compiler, and a plugin architecture allows new physics domains and solvers to be added without modifying the core.

Several design decisions follow from correctness rather than speed. A simulation reports success only when the numerical integration has produced no warning, so that a result cannot be reported as successful while its solver has signalled trouble. The compiler checks the conditioning of the mass matrix and tightens the integration tolerance as conditioning worsens. For systems with four or more coordinates, inversion of the mass matrix is deferred from the symbolic stage to numerical time, which avoids the expression swell that makes symbolic inversion impractical. In the embedded generator, small integer powers are emitted as repeated multiplication, trigonometric functions are routed to self-contained helpers, and any function that would still require the mathematics library produces an explicit compile-time error naming it, rather than code that fails at link time.

The test suite contains over 2,000 tests and runs in continuous integration. Separately, an adversarial stress suite of 55 cases across six axes (number of degrees of freedom, closed kinematic loops, redundant constraints, near-singular mass matrices, extreme mass ratios and deep symbolic nesting) was frozen before measurement and is distributed with the package.

# Research impact statement

MechanicsDSL was one of three engines in a study of silent failures in rigid-body dynamics software, in which each engine was adjudicated against an independent closed-form reference sharing no code with any engine [@parsons2026adjudication]. In that study MechanicsDSL agreed with the reference to within $5.5\times10^{-15}$ relative error on every case both could compute. The study also led to a correctness release (2.1.3) fixing several defects that had produced wrong results without warning, including an embedded generator that ignored the supplied equations. Generated C for five representative systems now cross-compiles for the Cortex-M4 without warnings under `-Wall -Wextra`.

Outside its author's own research, the evidence of use is so far limited to installations. Between November 2025 and November 2026, PyPI recorded 947 pip and uv installs of the six MechanicsDSL packages, excluding continuous-integration traffic. Removing automated traffic, defined as any client that recorded more than 25 installs in a single day or installed five or more distinct versions within seven days, leaves 373 installs from 19 countries, 119 of them from outside the United States. Installs cluster around releases rather than showing sustained growth, and the project has not yet received code contributions from others. MechanicsDSL Studio, a desktop worksheet application for Windows and macOS built on the library, extends the software to teaching use in undergraduate classical mechanics.

# AI usage disclosure

Generative AI was used in the development of this software and in the preparation of this paper. Claude (Anthropic) is credited as co-author on five commits to the repository: three in March 2026 using Claude Opus 4.6, and two in the week beginning 30 August 2026 using Claude Opus 5. Claude was also used to draft and edit the text of this paper. The author made the core design decisions, reviewed and tested every AI-assisted change, checked all technical claims in this paper against the software and its test suite, and takes full responsibility for both.

# Acknowledgements

This work received no specific funding.

# References
