<p align="center">
  <img src="docs/images/logo.png" alt="MechanicsDSL Logo" width="400">
</p>

<h1 align="center">MechanicsDSL</h1>

<p align="center">
  <a href="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml"><img src="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg" alt="Python CI"></a>
  <a href="https://pepy.tech/projects/mechanicsdsl-core"><img src="https://static.pepy.tech/personalized-badge/mechanicsdsl-core?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Downloads" alt="PyPI Downloads"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://doi.org/10.5281/zenodo.17771040"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17771040.svg" alt="DOI"></a>
  <a href="https://mechanicsdsl.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mechanicsdsl/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://codecov.io/github/MechanicsDSL/mechanicsdsl"><img src="https://codecov.io/github/MechanicsDSL/mechanicsdsl/graph/badge.svg" alt="Code Coverage"></a>
  <a href="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/codeql.yml"><img src="https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/codeql.yml/badge.svg" alt="CodeQL Advanced"></a>
  <a href="https://mybinder.org/v2/gh/MechanicsDSL/mechanicsdsl/main?filepath=tutorials"><img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"></a>
</p>

<p align="center"><strong>Define physics in LaTeX. Simulate anywhere.</strong></p>

**MechanicsDSL** is a computational physics framework that lets you describe physical systems in a natural, LaTeX-inspired syntax and automatically generates high-performance simulations. Write your Lagrangian once, compile to 15 target platforms.

---

## Why MechanicsDSL?

| Feature | Description |
|---------|-------------|
| **Symbolic Engine** | Automatically derives equations of motion from Lagrangians or Hamiltonians |
| **15 Code Generators** | C++, Python, Rust, Julia, CUDA, Fortran, MATLAB, JavaScript, OpenMP, WebAssembly, Arduino, ARM, Unity, Unreal, Modelica |
| **GPU Acceleration** | JAX backend with JIT compilation and automatic differentiation |
| **Inverse Problems** | Parameter estimation, sensitivity analysis, MCMC uncertainty |
| **Jupyter Native** | `%%mechanicsdsl` magic commands for notebooks |
| **Real-time API** | FastAPI server with WebSocket streaming |
| **IDE Support** | LSP server for VS Code with autocomplete and diagnostics |
| **Plugin Architecture** | Extensible with custom physics domains and solvers |

---

## Installation

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

**Docker:**

```bash
docker pull ghcr.io/mechanicsdsl/mechanicsdsl:latest
docker run -it ghcr.io/mechanicsdsl/mechanicsdsl:latest

# GPU (requires nvidia-docker)
docker pull ghcr.io/mechanicsdsl/mechanicsdsl:gpu
docker run --gpus all -it ghcr.io/mechanicsdsl/mechanicsdsl:gpu
```

**Requirements:** Python 3.9+ with NumPy, SciPy, SymPy, and Matplotlib (installed automatically).

---

## Quick Start

### Simple Pendulum

```python
from mechanics_dsl import PhysicsCompiler

compiler = PhysicsCompiler()
compiler.compile_dsl(r"""
\system{pendulum}
\defvar{theta}{Angle}{rad}
\defvar{m}{Mass}{kg}
\defvar{l}{Constant}{m}
\defvar{g}{Acceleration}{m/s^2}

\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{\frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
\initial{theta=0.5, theta_dot=0.0}
""")

solution = compiler.simulate(t_span=(0, 10), num_points=1000)
compiler.plot(solution)
```

### Figure-8 Three-Body Orbit

Define a gravitational three-body system and watch it trace the celebrated Figure-8 periodic orbit:

```python
from mechanics_dsl import PhysicsCompiler

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

```python
from mechanics_dsl import PhysicsCompiler

compiler = PhysicsCompiler()
compiler.compile_dsl(r"""
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
""")

compiler.compile_to_cpp("dam_break.cpp", target="standard", compile_binary=True)
```

---

## Code Generation

Export any system to optimized, standalone code for 15 targets:

| Target | Generator Class | Output |
|--------|----------------|--------|
| **C++** | `CppGenerator` | CMake project with solver |
| **Python** | `PythonGenerator` | NumPy/SciPy standalone script |
| **Rust** | `RustGenerator` | Cargo project, `no_std` embedded option |
| **Julia** | `JuliaGenerator` | DifferentialEquations.jl integration |
| **CUDA** | `CudaGenerator` | GPU-accelerated parallel solver |
| **Fortran** | `FortranGenerator` | F90 with LAPACK support |
| **MATLAB** | `MatlabGenerator` | .m script with ode45 |
| **JavaScript** | `JavaScriptGenerator` | Browser or Node.js |
| **OpenMP** | `OpenMPGenerator` | Multi-threaded C++ |
| **WebAssembly** | `WasmGenerator` | Emscripten-compiled WASM |
| **Arduino** | `ArduinoGenerator` | .ino sketch for embedded |
| **ARM** | `ARMGenerator` | NEON-optimized for Raspberry Pi |
| **Unity** | via `CppGenerator` | C# MonoBehaviour |
| **Unreal** | via `CppGenerator` | UE Actor component |
| **Modelica** | via integrations | Standards-based FMU |

```python
from mechanics_dsl.codegen.rust import RustGenerator

gen = RustGenerator(
    system_name="pendulum",
    coordinates=compiler.get_coordinates(),
    parameters=compiler.simulator.parameters,
    initial_conditions=compiler.initial_conditions,
    equations=compiler.equations,
)
gen.generate("pendulum.rs")
```

---

## Physics Domains

### Classical Mechanics
**Lagrangian & Hamiltonian** formulations with automatic EOM derivation. **Constraints**: holonomic, non-holonomic, rolling, knife-edge (Baumgarte stabilization). **Dissipation**: Rayleigh function, viscous/Coulomb/Stribeck friction. **Stability analysis**: equilibrium points, linearization, eigenvalue analysis. **Noether's theorem**: symmetry detection, conservation laws, cyclic coordinates. **Central forces**: effective potential, Kepler problem, orbital mechanics. **Canonical transformations**: generating functions, action-angle, Hamilton-Jacobi. **Normal modes**: mass/stiffness matrices, coupled oscillators, modal decomposition. **Rigid body**: Euler angles, quaternions, gyroscopes, symmetric top. **Perturbation theory**: Lindstedt-Poincare, averaging, multi-scale analysis. **Collisions**: elastic/inelastic, impulse, center of mass frame. **Scattering**: Rutherford, cross-sections, impact parameter. **Variable mass**: Tsiolkovsky rocket equation, conveyor belts. **Continuous systems**: vibrating strings, membranes, field equations.

### Quantum Mechanics
**Bound states**: infinite well, finite square well, hydrogen atom. **Scattering**: step potential, delta barriers, transmission/reflection coefficients. **Tunneling**: rectangular barriers, WKB approximation, Gamow factor. **Semiclassical**: WKB wavefunctions, Bohr-Sommerfeld quantization. **Hydrogen atom**: energy levels, Bohr radius, spectral series. **Ehrenfest theorem**: quantum-classical correspondence.

### Electromagnetism
**Charged particles**: Lorentz force, cyclotron motion, Larmor radius. **Waves**: plane waves, Poynting vector, radiation pressure. **Antennas**: Hertzian dipole, half-wave dipole, radiation resistance. **Waveguides**: TE/TM modes, cutoff frequencies, group velocity. **Traps**: Penning trap, magnetic dipole traps, gradient/curvature drift.

### Special Relativity
**Kinematics**: Lorentz boosts, velocity addition, time dilation, length contraction. **Four-vectors**: spacetime intervals, invariants, metric signature. **Doppler effect**: longitudinal, transverse, cosmological redshift. **Radiation**: synchrotron radiation, Thomas precession, twin paradox.

### General Relativity
**Black holes**: Schwarzschild metric, Kerr (rotating), ergosphere. **Geodesics**: light bending, ISCO, photon sphere. **Lensing**: deflection angle, Einstein radius, magnification. **Cosmology**: FLRW metric, Hubble law, comoving distance.

### Statistical Mechanics & Thermodynamics
**Ensembles**: microcanonical, canonical, grand canonical. **Distributions**: Boltzmann, Fermi-Dirac, Bose-Einstein. **Models**: Ising model, ideal gas, quantum harmonic oscillator. **Thermodynamic quantities**: partition functions, entropy, free energy. **Heat engines**: Carnot, Otto, Diesel cycles. **Equations of state**: ideal gas, van der Waals. **Phase transitions**: Clausius-Clapeyron, latent heat. **Heat capacity**: Debye, Einstein models.

### Fluid Dynamics
**SPH solver**: smoothed particle hydrodynamics for incompressible fluids. **Kernels**: Poly6, Spiky, Viscosity with Tait equation of state. **Boundaries**: no-slip, periodic, reflective conditions.

---

## Key Features

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

## Examples & Tutorials

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

**[mechanicsdsl.readthedocs.io](https://mechanicsdsl.readthedocs.io/en/latest/index.html)**

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <em>Built with care for physicists, engineers, and curious minds.</em>
</p>
