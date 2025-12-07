<p align="center">
  <img src="docs/images/logo.png" alt="MechanicsDSL Logo" width="400">
</p>

<h1 align="center">MechanicsDSL</h1>

<p align="center">
  <a href="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml"><img src="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg" alt="Python CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://doi.org/10.5281/zenodo.17771040"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17771040.svg" alt="DOI"></a>
  <a href="https://mechanicsdsl.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mechanicsdsl/badge/?version=latest" alt="Documentation Status"></a>
</p>

---

**MechanicsDSL** is a computational physics framework that lets you define physical systems in a natural, LaTeX-inspired syntax and automatically generates high-performance simulations. From pendulums to planetary orbits, from Lagrangian mechanics to fluid dynamics—describe it once, simulate it anywhere.

## Why MechanicsDSL?

| Feature | Description |
|---------|-------------|
| **Symbolic Engine** | Automatically derives equations of motion from Lagrangians or Hamiltonians |
| **Fluid Dynamics** | Built-in SPH solver for dam breaks, waves, and liquid physics |
| **High Performance** | Generates optimized C++, OpenMP, and WebAssembly code |
| **Rich Visualization** | Phase space plots, energy analysis, and smooth animations |
| **Research Ready** | Validated against analytical solutions and conservation laws |

---

## Installation

```bash
pip install mechanicsdsl-core
```

**Requirements:** Python 3.8+ with NumPy, SciPy, SymPy, and Matplotlib (installed automatically).

---

## Quick Start

### The Famous Figure-8 Three-Body Orbit

Define a gravitational three-body system and watch it trace the celebrated Figure-8 periodic orbit:

```python
from mechanics_dsl import PhysicsCompiler

# Define the system using LaTeX-inspired DSL
figure8_code = r"""
\system{figure8_orbit}
\defvar{x1}{Position}{m} \defvar{y1}{Position}{m}
\defvar{x2}{Position}{m} \defvar{y2}{Position}{m}
\defvar{x3}{Position}{m} \defvar{y3}{Position}{m}
\defvar{m}{Mass}{kg} \defvar{G}{Grav}{1}

\parameter{m}{1.0}{kg} \parameter{G}{1.0}{1}

\lagrangian{
    0.5 * m * (\dot{x1}^2 + \dot{y1}^2 + \dot{x2}^2 + \dot{y2}^2 + \dot{x3}^2 + \dot{y3}^2)
    + G*m^2/\sqrt{(x1-x2)^2 + (y1-y2)^2}
    + G*m^2/\sqrt{(x2-x3)^2 + (y2-y3)^2}
    + G*m^2/\sqrt{(x1-x3)^2 + (y1-y3)^2}
}
"""

# Compile and simulate
compiler = PhysicsCompiler()
compiler.compile_dsl(figure8_code)
compiler.simulator.set_initial_conditions({
    'x1': 0.97000436,  'y1': -0.24308753, 'x1_dot': 0.466203685, 'y1_dot': 0.43236573,
    'x2': -0.97000436, 'y2': 0.24308753,  'x2_dot': 0.466203685, 'y2_dot': 0.43236573,
    'x3': 0.0,         'y3': 0.0,         'x3_dot': -0.93240737, 'y3_dot': -0.86473146
})
solution = compiler.simulate(t_span=(0, 6.326), num_points=2000)
```

### Dam Break Fluid Simulation

Simulate fluid dynamics with the integrated SPH solver:

```python
from mechanics_dsl import PhysicsCompiler

fluid_code = r"""
\system{dam_break}

\parameter{h}{0.04}{m}
\parameter{g}{9.81}{m/s^2}

\fluid{water}
\region{rectangle}{x=0.0 .. 0.4, y=0.0 .. 0.8}
\particle_mass{0.02}
\equation_of_state{tait}

\boundary{walls}
\region{line}{x=-0.05, y=0.0 .. 1.5}
\region{line}{x=1.5, y=0.0 .. 1.5}
\region{line}{x=-0.05 .. 1.5, y=-0.05}
"""

compiler = PhysicsCompiler()
compiler.compile_dsl(fluid_code)
compiler.compile_to_cpp("dam_break.cpp", target="standard", compile_binary=True)
```

## Core Capabilities

### Classical Mechanics (17 Modules)
- **Lagrangian & Hamiltonian** formulations with automatic EOM derivation
- **Constraints**: Holonomic, non-holonomic, rolling, knife-edge (Baumgarte stabilization)
- **Dissipation**: Rayleigh function, viscous/Coulomb/Stribeck friction
- **Stability Analysis**: Equilibrium points, linearization, eigenvalue analysis
- **Noether's Theorem**: Symmetry detection, conservation laws, cyclic coordinates
- **Central Forces**: Effective potential, Kepler problem, orbital mechanics
- **Canonical Transformations**: Generating functions, action-angle, Hamilton-Jacobi
- **Normal Modes**: Mass/stiffness matrices, coupled oscillators, modal decomposition
- **Rigid Body**: Euler angles, quaternions, gyroscopes, symmetric top
- **Perturbation Theory**: Lindstedt-Poincaré, averaging, multi-scale analysis
- **Collisions**: Elastic/inelastic, impulse, center of mass frame
- **Scattering**: Rutherford, cross-sections, impact parameter
- **Variable Mass**: Tsiolkovsky rocket equation, conveyor belts
- **Continuous Systems**: Vibrating strings, membranes, field equations

### Code Generation
- **Native C++** with optimized numerics
- **OpenMP** for parallel simulations
- **WebAssembly** for browser-based physics

### Analysis Tools
- **Energy conservation** monitoring
- **Phase space** visualization & Poincaré sections
- **534 passing tests** for reliability

---

## Examples

The [`examples/`](examples/) directory contains 30 progressive tutorials:

| Level | Examples |
|-------|----------|
| **Beginner** | Harmonic oscillator, Simple pendulum, Plotting basics |
| **Intermediate** | Double pendulum, Coupled oscillators, 2D motion, Damping |
| **Advanced** | 3D gyroscope, Hamiltonian formulation, Phase space, Energy analysis |
| **Expert** | C++ export, WebAssembly targets, SPH fluid dynamics |

---

## Documentation

Full documentation with tutorials, API reference, and DSL syntax guide:

**[Read the Docs](https://mechanicsdsl.readthedocs.io/en/latest/index.html)**

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built with ❤️ for physicists, engineers, and curious minds.</em>
</p>
