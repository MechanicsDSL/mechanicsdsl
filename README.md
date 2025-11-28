[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-prototype%20%2F%20research-orange)](https://github.com/MechanicsDSL/mechanics-dsl)
# MechanicsDSL

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
pip install mechanics-dsl
```

Or install from source:

```bash
git clone https://github.com/yourusername/mechanics_dsl.git
cd mechanics_dsl
pip install -e .
```

## Quick Start

```python
from mechanics_dsl import PhysicsCompiler

# Define a simple pendulum system
dsl_code = """
system pendulum
var theta: angle [rad]
parameter g = 9.81 [m/s^2]
parameter L = 1.0 [m]
lagrangian = 0.5 * m * L^2 * \\dot{theta}^2 - m * g * L * (1 - \\cos{theta})
initial theta = 0.1, \\dot{theta} = 0
solve method=rk45 t_span=[0, 10]
animate
"""

# Compile and run
compiler = PhysicsCompiler()
result = compiler.compile_dsl(dsl_code)
solution = compiler.simulate(t_span=(0, 10))
compiler.animate(solution)
```

## Documentation

Full documentation is available at [https://mechanics-dsl.readthedocs.io](https://mechanics-dsl.readthedocs.io)

## License

MIT License - see LICENSE file for details.

## Citation

If you use MechanicsDSL in your research, please cite:

```bibtex
@software{mechanics_dsl,
  author = {Parsons, Noah},
  title = {MechanicsDSL: A Domain-Specific Language for Classical Mechanics},
  year = {2025},
  url = {https://github.com/yourusername/mechanics_dsl}
}
```
