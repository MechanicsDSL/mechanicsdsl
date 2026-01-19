<p align="center">
  <img src="docs/images/logo.png" alt="MechanicsDSL Logo" width="400">
</p>

<h1 align="center">MechanicsDSL</h1>

<p align="center">
  <a href="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml"><img src="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg" alt="Python CI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+ Badge">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://doi.org/10.5281/zenodo.17771040"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17771040.svg" alt="DOI"></a>
  <a href="https://mechanicsdsl.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mechanicsdsl/badge/?version=latest" alt="Documentation Status"></a>
 <img src="https://codecov.io/github/MechanicsDSL/mechanicsdsl/graph/badge.svg"/> 
</a>
<a href="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/codeql.yml"> 
 <img src="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/codeql.yml/badge.svg" alt="CodeQL Advanced"> 
</a>
<a href="https://mybinder.org/v2/gh/MechanicsDSL/mechanicsdsl/main?filepath=tutorials">
 <img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder">
</a>

</p>

---

**MechanicsDSL** is a computational physics framework that lets you define physical systems in a natural, LaTeX-inspired syntax and automatically generates high-performance simulations. From pendulums to planetary orbits, from Lagrangian mechanics to fluid dynamics—describe it once, simulate it anywhere.

## ✨ Why MechanicsDSL?

| Feature | Description |
|---------|-------------|
| **Symbolic Engine** | Automatically derives equations of motion from Lagrangians or Hamiltonians |
| **12+ Code Generators** | C++, Rust, Julia, CUDA, WebAssembly, Unity, Unreal, Modelica, and more |
| **GPU Acceleration** | JAX backend with JIT compilation and automatic differentiation |
| **Inverse Problems** | Parameter estimation, sensitivity analysis, MCMC uncertainty |
| **Jupyter Native** | `%%mechanicsdsl` magic commands for notebooks |
| **Real-time API** | FastAPI server with WebSocket streaming |
| **IDE Support** | LSP server for VS Code with autocomplete and diagnostics |
| **Plugin Architecture** | Extensible with custom physics domains and solvers |

---

## 📦 Installation

```bash
pip install mechanicsdsl-core
```

**With optional features:**

```bash
pip install mechanicsdsl-core[jax]      # GPU acceleration + autodiff
pip install mechanicsdsl-core[server]   # FastAPI real-time server
pip install mechanicsdsl-core[jupyter]  # Notebook magic commands
pip install mechanicsdsl-core[lsp]      # VS Code language server
pip install mechanicsdsl-core[embedded] # Raspberry Pi / ARM support
pip install mechanicsdsl-core[all]      # Everything
```

**Docker deployment:**

```bash
# CPU version
docker pull ghcr.io/mechanicsdsl/mechanicsdsl:latest
docker run -it ghcr.io/mechanicsdsl/mechanicsdsl:latest

# GPU version (requires nvidia-docker)
docker pull ghcr.io/mechanicsdsl/mechanicsdsl:gpu
docker run --gpus all -it ghcr.io/mechanicsdsl/mechanicsdsl:gpu
```

**Requirements:** Python 3.9+ with NumPy, SciPy, SymPy, and Matplotlib (installed automatically).

---

## 🚀 What's New in v2.0.0

**Released January 17, 2026** — Now deployed in **19 countries** across enterprise, research, and embedded platforms.

### Enterprise Deployment
- **Docker Support** — Production-ready multi-stage containers for CPU and GPU
- **docker-compose** — API server, Jupyter, and worker service orchestration
- **Kubernetes Ready** — Enterprise deployment guide with security best practices

### ARM & Embedded Platforms
- **Raspberry Pi Examples** — Real-time pendulum simulation with C++ export
- **IMU Integration** — MPU6050 sensor fusion examples
- **ARM Optimization** — NEON detection and cross-compilation support

### Enhanced Code Generation
- **C++ CMake Projects** — `generate_cmake()` and `generate_project()` methods
- **Rust Cargo Projects** — Full project scaffolding with `no_std` embedded option
- **11 Target Platforms** — C++, CUDA, Rust, Julia, Fortran, MATLAB, JavaScript, WebAssembly, Python, Arduino, OpenMP

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

---

## 🆕 New in v1.6.0

### Jupyter Magic Commands

```python
%load_ext mechanics_dsl.jupyter

%%mechanicsdsl --animate --t_span=0,20
\system{pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 - m*g*l*(1-\cos{theta})}
\initial{theta=2.5, theta_dot=0.0}
```

### Parameter Estimation

```python
from mechanics_dsl.inverse import ParameterEstimator

estimator = ParameterEstimator(compiler)
result = estimator.fit(observations, t_obs, ['m', 'k'])
print(f"Fitted: m={result.parameters['m']:.3f}, k={result.parameters['k']:.3f}")
```

### Real-time API Server

```bash
python -m mechanics_dsl.server
# -> http://localhost:8000/docs
```

### External Integrations

| Platform | Module | Purpose |
|----------|--------|---------|
| **OpenMDAO** | `integrations.openmao` | Multidisciplinary optimization |
| **ROS2** | `integrations.ros2` | Robotics simulation |
| **Unity** | `integrations.unity` | Game engine (C#) |
| **Unreal** | `integrations.unreal` | Game engine (C++) |
| **Modelica** | `integrations.modelica` | Standards-based simulation |

---

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

### Quantum Mechanics
- **Bound States**: Infinite well, finite square well, hydrogen atom
- **Scattering**: Step potential, delta barriers, transmission/reflection coefficients
- **Quantum Tunneling**: Rectangular barriers, WKB approximation, Gamow factor
- **Semiclassical**: WKB wavefunctions, Bohr-Sommerfeld quantization
- **Hydrogen Atom**: Energy levels, Bohr radius, spectral series (Lyman, Balmer, etc.)
- **Ehrenfest Theorem**: Quantum-classical correspondence

### Electromagnetism
- **Charged Particles**: Lorentz force, cyclotron motion, Larmor radius
- **Waves**: Plane waves, Poynting vector, radiation pressure
- **Antennas**: Hertzian dipole, λ/2 dipole, radiation resistance
- **Waveguides**: TE/TM modes, cutoff frequencies, group velocity
- **Traps**: Penning trap, magnetic dipole traps, gradient/curvature drift

### Special Relativity
- **Kinematics**: Lorentz boosts, velocity addition, time dilation, length contraction
- **Four-Vectors**: Spacetime intervals, invariants, metric signature (+,-,-,-)
- **Doppler Effect**: Longitudinal, transverse, cosmological redshift
- **Radiation**: Synchrotron radiation, Thomas precession, twin paradox

### General Relativity
- **Black Holes**: Schwarzschild metric, Kerr (rotating), ergosphere
- **Geodesics**: Light bending, ISCO, photon sphere
- **Lensing**: Deflection angle, Einstein radius, magnification
- **Cosmology**: FLRW metric, Hubble law, comoving distance

### Statistical Mechanics
- **Ensembles**: Microcanonical, canonical, grand canonical
- **Distributions**: Boltzmann, Fermi-Dirac, Bose-Einstein
- **Models**: Ising model, ideal gas, quantum harmonic oscillator
- **Thermodynamic Quantities**: Partition functions, entropy, free energy

### Thermodynamics
- **Heat Engines**: Carnot, Otto, Diesel cycles
- **Equations of State**: Ideal gas, van der Waals
- **Phase Transitions**: Clausius-Clapeyron, latent heat
- **Heat Capacity**: Debye, Einstein models

### Fluid Dynamics
- **SPH Solver**: Smoothed Particle Hydrodynamics for incompressible fluids
- **Kernels**: Poly6, Spiky, Viscosity with Tait equation of state
- **Boundaries**: No-slip, periodic, reflective conditions


---

## 📚 Examples & Tutorials

### Interactive Tutorials (Jupyter)

| # | Tutorial | Topics |
|---|----------|--------|
| 1 | [Getting Started](tutorials/01_getting_started.ipynb) | DSL basics, simple pendulum, export |
| 2 | [Double Pendulum](tutorials/02_double_pendulum.ipynb) | Chaos, sensitivity, phase space |
| 3 | [Parameter Estimation](tutorials/03_parameter_estimation.ipynb) | Inverse problems, Sobol analysis |

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MechanicsDSL/mechanicsdsl/main?labpath=tutorials)

### Example Scripts

The [`examples/`](examples/) directory contains 30+ progressive examples:

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
