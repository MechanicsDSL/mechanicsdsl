<p align="center">
  <img src="docs/images/logo.png" alt="MechanicsDSL Logo" width="400">
</p>

<h1 align="center">MechanicsDSL</h1>

<p align="center">
  <a href="https://github.com"><img src="https://github.com/badge.svg" alt="Python CI"></a>
  <a href="https://pepy.tech"><img src="https://pepy.tech" alt="PyPI Downloads"></a>
  <img src="https://shields.io" alt="Python 3.9+">
  <a href="https://opensource.org"><img src="https://shields.io" alt="License: MIT"></a>
  <a href="https://doi.org"><img src="https://zenodo.org" alt="DOI"></a>
  <a href="https://readthedocs.io"><img src="https://readthedocs.org" alt="Documentation Status"></a>
  <a href="https://codecov.io"><img src="https://codecov.io/graph/badge.svg" alt="Code Coverage"></a>
  <a href="https://github.com"><img src="https://github.com/badge.svg" alt="CodeQL Advanced"></a>
  <a href="https://mybinder.org"><img src="https://mybinder.org" alt="Launch Binder"></a>
</p>

<p align="center"><strong>Write a Lagrangian. Get a simulation.</strong></p>

**MechanicsDSL** is an educational physics framework that lets you describe mechanical systems in LaTeX-like syntax and instantly simulate them. It's built for students and researchers who want to go from textbook equations to working simulations without writing boilerplate ODE solvers.

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

## What It Does


| Feature | Description |
| :--- | :--- |
| **Symbolic Engine** | Derives equations of motion from Lagrangians or Hamiltonians automatically |
| **Code Generation** | Export simulations to C++, Rust, Julia, CUDA, Fortran, MATLAB, JavaScript, and more |
| **Visualization** | Built-in plotting, animation, and phase space diagrams |
| **Jupyter Integration** | `%%mechanicsdsl` magic commands for interactive notebooks |
| **Inverse Problems** | Parameter estimation, sensitivity analysis, MCMC uncertainty quantification |
| **Plugin Architecture** | Extensible with custom physics domains and solvers |

> [!NOTE]
> MechanicsDSL is a learning and prototyping tool. The code generators produce working starter code, not production-optimized binaries. For mission-critical applications, use the generated code as a reference implementation.

## Installation

```bash
pip install mechanicsdsl-core
```

### Optional extras:

```bash
pip install mechanicsdsl-core[jax]      # GPU acceleration + autodiff
pip install mechanicsdsl-core[jupyter]  # Notebook magic commands
pip install mechanicsdsl-core[all]      # Everything
```

**Requirements:** Python 3.9+ with NumPy, SciPy, SymPy, and Matplotlib (installed automatically).

## Examples

### Figure-8 Three-Body Orbit

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

### Jupyter Magic Commands

```python
%load_ext mechanics_dsl.jupyter
```

```latex
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

The `examples/` directory has 30+ progressive examples from harmonic oscillators to SPH fluid dynamics.

_Show Image_

## Code Generation

Export any system to standalone code for 13 target languages:


| Target | Generator | Output |
| :--- | :--- | :--- |
| **C++** | CppGenerator | CMake project with solver |
| **Python** | PythonGenerator | NumPy/SciPy standalone script |
| **Rust** | RustGenerator | Cargo project, no_std option |
| **Julia** | JuliaGenerator | DifferentialEquations.jl |
| **CUDA** | CudaGenerator | GPU-parallel solver |
| **Fortran** | FortranGenerator | F90 with LAPACK |
| **MATLAB** | MatlabGenerator | .m script with ode45 |
| **JavaScript** | JavaScriptGenerator | Browser or Node.js |
| **OpenMP** | OpenMPGenerator | Multi-threaded C++ |
| **WebAssembly** | WasmGenerator | Emscripten WASM |
| **Arduino** | ArduinoGenerator | .ino embedded sketch |
| **ARM** | ARMGenerator | Raspberry Pi / NEON |
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

## Physics Coverage

*   **Classical Mechanics** — Lagrangian & Hamiltonian formulations, constraints (holonomic, non-holonomic, rolling), Rayleigh dissipation, stability analysis, Noether's theorem, central forces, canonical transformations, normal modes, rigid body dynamics, perturbation theory, collisions, scattering, variable mass systems, continuous media.
*   **Quantum Mechanics** — Bound states, scattering, tunneling, WKB approximation, hydrogen atom, Ehrenfest theorem.
*   **Electromagnetism** — Lorentz force, cyclotron motion, plane waves, antennas, waveguides, Penning traps.
*   **Relativity** — Special: Lorentz boosts, four-vectors, Doppler effect. General: Schwarzschild & Kerr metrics, geodesics, gravitational lensing, FLRW cosmology.
*   **Statistical Mechanics & Thermodynamics** — Microcanonical/canonical/grand canonical ensembles, Boltzmann/Fermi-Dirac/Bose-Einstein distributions, Ising model, heat engines, phase transitions.
*   **Fluid Dynamics** — SPH solver with Poly6/Spiky/Viscosity kernels, Tait equation of state, boundary conditions.

## Documentation

[mechanicsdsl.readthedocs.io](https://readthedocs.io)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
