[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-prototype%20%2F%20research-orange)](https://github.com/MechanicsDSL/mechanics-dsl)
# MechanicsDSL

A Domain-Specific Language for Classical Mechanics.

> "Computational tools for physics education often have a fundamental problem: they prioritize power over pedagogy. MechanicsDSL makes Lagrangian mechanics computationally accessible without sacrificing rigor."

MechanicsDSL is a Python framework and custom language compiler that converts LaTeX-inspired descriptions of physical systems into symbolic equations of motion, numerical simulations, and interactive visualizations. It is designed to lower the cognitive load of computational physics, allowing students and researchers to focus on the physics, not the boilerplate.

---

## Key Features

- **Pedagogy-First Syntax**: Define systems using familiar LaTeX-style commands (`\system`, `\lagrangian`, `\initial`).
- **Symbolic Power**: Automated derivation of Euler-Lagrange and Hamiltonian equations using SymPy.
- **Numerical Rigor**: Production-grade solvers (RK45, LSODA) with stiffness detection.
- **True 3D Motion**: Built-in support for Euler angles, quaternions, and 3D visualizations.
- **Safety**: AST-based parsing (no eval()) ensuring secure execution of DSL code.

---

## Installation

```
pip install mechanics-dsl
```

---

## Quick Start

Define a Simple Pendulum in 5 lines of DSL code:

```
\system{simple_pendulum}
\defvar{theta}{Angle}{rad}
\defvar{l}{Length}{m}
\defvar{g}{Acceleration}{m/s^2}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

% The Physics
\lagrangian{ \frac{1}{2} * l^2 * \dot{theta}^2 - g * l * (1 - \cos{theta}) }

\initial{theta=0.5, theta_dot=0.0}
\solve{RK45}
\animate{pendulum}
```

Run it from the command line:

```
mechanics-dsl --file pendulum.dsl
```

---

## Why MechanicsDSL?

- Existing tools like Mathematica or MATLAB are powerful but impose a steep learning curve. MechanicsDSL bridges the gap between handwritten derivation and computational simulation.
- **Write physics, not code:** The DSL mirrors the mathematical formulation found in textbooks.
- **Immediate Feedback:** Go from abstract Lagrangian to animated simulation in seconds.
- **Bridge to Python:** Export generated systems to pure Python code for advanced analysis.

---

## Roadmap & Contributing

We are actively developing v1.0 for Fall 2026. Current focus areas:

- [ ] Quantum Mechanics module
- [ ] Plugin architecture for custom forces
- [ ] Jupyter Notebook integration

See [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for how to get involved.

---

## License

MIT License. Copyright (©) 2025 Noah Parsons.
```

