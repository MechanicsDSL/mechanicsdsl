# MechanicsDSL

[![Python CI](https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg)](https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17771040.svg)](https://zenodo.org/badge/DOI/10.5281/zenodo.17771040)

A Domain-Specific Language for Classical Mechanics - A comprehensive framework for symbolic and numerical analysis of classical mechanical systems using LaTeX-inspired notation.

## Features

- **Symbolic Computation**: Automatic derivation of equations of motion from Lagrangians and Hamiltonians
- **Numerical Simulation**: Advanced ODE solvers with adaptive step sizing
- **Visualization**: Interactive animations and phase space plots
- **Unit System**: Comprehensive dimensional analysis and unit checking
- **Constraint Handling**: Support for holonomic and non-holonomic constraints
- **Performance Monitoring**: Built-in profiling and optimization tools

## Installation
 
```bash
pip install mechanicsdsl-core
```

## Quick Start

```python
from mechanics_dsl import PhysicsCompiler

# Define a simple pendulum system
dsl_code = r"""
\system{pendulum}
\var{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{L}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{0.5 * m * L^2 * \dot{theta}^2 - m * g * L * (1 - \cos{theta})}
\initial{theta=0.1, theta_dot=0.0}
"""

compiler = PhysicsCompiler()
result = compiler.compile_dsl(dsl_code)
solution = compiler.simulate(t_span=(0, 10))
compiler.animate(solution)
```

## Documentation

Full documentation is available at [https://mechanicsdsl.readthedocs.io/en/latest/index.html](https://mechanicsdsl.readthedocs.io/en/latest/index.html)

## License

MIT License - see LICENSE file for details.
