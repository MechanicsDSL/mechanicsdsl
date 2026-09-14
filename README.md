![MechanicsDSL Logo](https://raw.githubusercontent.com/MechanicsDSL/mechanicsdsl/main/docs/images/logo.png)

# MechanicsDSL

[![Python CI](https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg)](https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17771040.svg)](https://doi.org/10.5281/zenodo.17771040)
[![Documentation Status](https://readthedocs.org/projects/mechanicsdsl/badge/?version=latest)](https://mechanicsdsl.readthedocs.io/en/latest/?badge=latest)

*Write a Lagrangian. Get a simulation.*

---

MechanicsDSL takes you from a Lagrangian to C that runs on a bare-metal microcontroller in one pipeline. You write the Lagrangian or Hamiltonian in a LaTeX-inspired syntax, the symbolic engine (built on SymPy) derives the equations of motion, and the compiler emits standalone code. That code can be a Python script, or it can be libm-free C that cross-compiles cleanly for an ARM Cortex-M4. The whole path from textbook physics to firmware is a single step.

The same compiled system exports to twelve targets in all, including C++, CUDA, Rust, and WebAssembly (see [Code generation](#code-generation)).

```python
from mechanics_dsl import PhysicsCompiler

compiler = PhysicsCompiler()
compiler.compile_dsl(r"""
\system{pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{\frac{1}{2} * m * l^2 * \dot{theta}^2 - m * g * l * (1 - \cos{theta})}
\initial{theta=0.5, theta_dot=0.0}
""")

solution = compiler.simulate(t_span=(0, 10), num_points=1000)
compiler.plot(solution)
```

## What's in the box

| Component | Description |
|-----------|-------------|
| **Symbolic engine** | Derives equations of motion from Lagrangians or Hamiltonians, built on SymPy |
| **Embedded C** | Bare-metal Cortex-M output with no libm dependency, compile-verified under `-Wall -Wextra` |
| **Code generation** | Twelve targets: C++, Python, Rust, Julia, CUDA, Fortran, MATLAB, JavaScript, OpenMP, WebAssembly, Arduino, ARM |
| **JAX backend** | GPU acceleration with JIT compilation and automatic differentiation |
| **Inverse problems** | Parameter estimation, sensitivity analysis, MCMC uncertainty quantification |
| **Jupyter integration** | `%%mechanicsdsl` magic commands for interactive notebooks |
| **Plugin architecture** | Custom physics domains and solvers without modifying the core |

> **What is verified in generated code.** The ARM and embedded C paths are compile-verified for Cortex-M4: five representative systems (1-DOF oscillator, pendulum, 2-DOF coupled, 3-DOF attitude, power-in-denominator) cross-compile with `arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -Os -Wall -Wextra` and produce zero warnings. If a system needs a libm function that bare metal can't provide, the generator emits an explicit `#error` naming that function instead of producing code that fails at link time. Generated code is not performance-tuned.

## Installation

```bash
pip install mechanicsdsl-core
```

Optional extras:

```bash
pip install mechanicsdsl-core[jax]      # GPU + autodiff
pip install mechanicsdsl-core[jupyter]  # Notebook magic
pip install mechanicsdsl-core[all]      # Everything
```

Requires Python 3.9+. NumPy, SciPy, SymPy, and Matplotlib are installed automatically.

## Example: Figure-8 three-body orbit

```python
from mechanics_dsl import PhysicsCompiler

code = r"""
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
compiler.compile_dsl(code)
compiler.simulator.set_initial_conditions({
    'x1': 0.97000436,  'y1': -0.24308753, 'x1_dot': 0.466203685, 'y1_dot': 0.43236573,
    'x2': -0.97000436, 'y2': 0.24308753,  'x2_dot': 0.466203685, 'y2_dot': 0.43236573,
    'x3': 0.0,         'y3': 0.0,         'x3_dot': -0.93240737, 'y3_dot': -0.86473146
})
solution = compiler.simulate(t_span=(0, 6.326), num_points=2000)
```

The [`examples/`](https://github.com/MechanicsDSL/mechanicsdsl/tree/main/examples) directory contains 40+ progressive examples, from harmonic oscillators to SPH fluid dynamics.

## Code generation

Any compiled system can be exported as standalone code in any of the supported targets:

| Target | Output |
|--------|--------|
| ARM | Bare-metal Cortex-M (compile-verified on Cortex-M4), Raspberry Pi / NEON |
| Arduino | `.ino` embedded sketch |
| C++ | CMake project with solver |
| Python | NumPy/SciPy standalone script |
| Rust | Cargo project, `no_std` option |
| Julia | DifferentialEquations.jl |
| CUDA | GPU-parallel solver |
| Fortran | F90 with LAPACK |
| MATLAB | `.m` script with `ode45` |
| JavaScript | Browser or Node.js |
| OpenMP | Multi-threaded C++ |
| WebAssembly | Emscripten WASM |

That makes twelve targets. Modelica isn't one of them. It's an *integration*
(`mechanics_dsl.integrations.modelica`) that writes a `.mo` model for an external
Modelica tool to compile, so it isn't a code generator registered with
`PhysicsCompiler.export()`.

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

## Validation

- **Test suite.** 2,194 tests pass at the 2.1.3 release (11 skipped).
- **Adversarial stress suite.** 55 cases across six stress axes (degrees of freedom, closed loops, redundant constraints, near-singular mass matrices, extreme mass ratios, deep symbolic nesting), frozen before measurement. The result was zero silent failures: no case returned a wrong answer while reporting success.
- **Cross-engine comparison.** The engine was measured against Drake and SymPy and adjudicated by an independent closed-form reference written in NumPy alone, which shares no library with any engine under test. MechanicsDSL agrees with that reference to machine precision (≤ 5.5 × 10⁻¹⁵ relative) on every case both could compute. Independent adjudication covers 31 of the 55 cases.
- **Scope.** This validation work covers classical mechanics: Lagrangian, Hamiltonian, and constrained formulations.

The scripts, results, and stated limits are in [`stress_suite/`](https://github.com/MechanicsDSL/mechanicsdsl/tree/main/stress_suite). The study artefacts are archived at [doi:10.5281/zenodo.22315172](https://doi.org/10.5281/zenodo.22315172).

## Physics coverage

**Classical mechanics** is the core of the project, and it's the domain the validation above covers: Lagrangian and Hamiltonian formulations; holonomic, non-holonomic, and rolling constraints; Rayleigh dissipation; stability analysis; Noether's theorem; central forces; canonical transformations; normal modes; rigid body dynamics; perturbation theory; collisions; scattering; variable-mass systems; continuous media.

<details>
<summary>Other domains (not covered by the stress suite)</summary>

- **Quantum mechanics**: bound states, scattering, tunneling, WKB approximation, hydrogen atom, Ehrenfest theorem.
- **Electromagnetism**: Lorentz force, cyclotron motion, plane waves, antennas, waveguides, Penning traps.
- **Relativity**: special (Lorentz boosts, four-vectors, Doppler effect) and general (Schwarzschild and Kerr metrics, geodesics, gravitational lensing, FLRW cosmology).
- **Statistical mechanics and thermodynamics**: microcanonical, canonical, and grand canonical ensembles; Boltzmann, Fermi-Dirac, and Bose-Einstein distributions; Ising model; heat engines; phase transitions.
- **Fluid dynamics**: SPH solver with Poly6, Spiky, and viscosity kernels; Tait equation of state; boundary conditions.

</details>

## Project status

MechanicsDSL is under active development. **2.1.x is the current line.** Release 2.1.3 fixed correctness defects in earlier releases, several of which returned wrong answers while reporting success: a Hamiltonian pathway that silently froze on coupled momenta (a two-link pendulum reported success and never moved), constrained Lagrangian systems that froze, a Modelica export that emitted a failure placeholder as its equation of motion, and a CLI where 10 of 11 targets raised `AttributeError`. If you are on an earlier version, upgrade. The [CHANGELOG](https://github.com/MechanicsDSL/mechanicsdsl/blob/main/CHANGELOG.md) has the full list.

Issues, pull requests, and use-case reports are all welcome.

## Documentation

Full documentation, tutorials, and DSL reference at **[mechanicsdsl.readthedocs.io](https://mechanicsdsl.readthedocs.io/)**.

## Citing

If you use MechanicsDSL in your work, please cite the software:

> Parsons, N. *MechanicsDSL: A Domain-Specific Language for Classical Mechanics* (v2.1.3). Zenodo. [doi:10.5281/zenodo.17771040](https://doi.org/10.5281/zenodo.17771040)

Citation metadata is also in [`CITATION.cff`](https://github.com/MechanicsDSL/mechanicsdsl/blob/main/CITATION.cff).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/MechanicsDSL/mechanicsdsl/blob/main/CONTRIBUTING.md) for guidelines.

## License

MIT; see [LICENSE](https://github.com/MechanicsDSL/mechanicsdsl/blob/main/LICENSE).
