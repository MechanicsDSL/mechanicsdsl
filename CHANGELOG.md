# Changelog

All notable changes to MechanicsDSL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1] - 2026-05-29

### 🔬 Quality & Validation Release

Backs up v2.1.0's correctness fixes with verification. Adds physics
validation against textbook closed-form answers, smoke tests for every
codegen backend, and end-to-end tests for the plugin, inverse, and JAX
features. Each batch of new tests surfaced a real bug, all fixed here.

### Added

- **Physics validation suite** (`tests/physics/test_domain_validation.py`,
  11 tests) — every domain module is now spot-checked against a textbook
  closed-form answer: small-angle pendulum period, projectile range,
  Lorentz factor, Schwarzschild radius, hydrogen ground state, harmonic
  oscillator levels, Carnot efficiency, ideal gas law, SPH Poly6 kernel
  normalization, cantilever tip deflection.
- **Codegen smoke tests** (`tests/integration/test_codegen_smoke.py`,
  14 tests + skips) — every backend generates non-trivial output for a
  pendulum, with optional `node --check` and `g++ -fsyntax-only` passes
  where the host tooling is available.
- **Feature end-to-end tests** (`tests/integration/test_features_e2e.py`)
  — plugin file load + invoke, parameter estimator recovery of a known
  ground-truth length, and a JAX-backend pendulum run (skipped when JAX
  isn't installed).
- ``MECHANICSDSL_CODEGEN_FAILED`` sentinel — the new placeholder emitted
  by Unity, Unreal, Modelica, and WASM when a sympy → target-language
  conversion fails. The string deliberately doesn't compile in any of
  those targets so the failure surfaces at build time instead of
  silently producing a `0.0` literal.

### Fixed

- **`ParameterEstimator` actually fits now.** It was setting new parameter
  values on `compiler.simulator.parameters` but never recompiling the
  lambdified equations — the optimizer's loss surface was flat because
  the simulated trajectory always used the original baked-in constants.
  The fit appeared to succeed (`result.success is True`) while returning
  the initial guess unchanged. Now the simulator recompiles its
  equations on every objective evaluation.
- **`CudaGenerator` accepts `hamiltonian=`** like every other generator.
  The new `PhysicsCompiler.export()` passes `hamiltonian=` to all
  backends; CUDA was the only one that rejected it.
- **CUDA and WASM generators write to a file path** (matching
  `PhysicsCompiler.export()`'s contract) in addition to the legacy
  "write to a directory" mode. Calling `.generate("foo.cu")` no longer
  raises `FileExistsError` on Windows.
- **JavaScript output no longer embeds a Python `# noqa: E501` comment**
  inside the generated `console.log(...)` block; Node was rejecting the
  generated file with `SyntaxError`.
- **Domain methods that masked failures as numeric defaults** — silent
  `return 0.0` / `return float("inf")` paths in
  `canonical.compute_action_variable`, `central_forces` precession and
  period integrations, and `scattering` cross-section now return
  `float("nan")` with an ERROR-level log. NaN propagates obviously
  downstream; the old defaults looked like physically valid results.
- **Unity / Unreal / Modelica / WASM** integration codegen no longer emits
  `0.0f /* Error */` (which compiled to silent-zero output in the target
  language). They now emit `MECHANICSDSL_CODEGEN_FAILED("...")` plus an
  ERROR log; the generated target source fails to compile, surfacing
  the conversion problem at the right layer.
- `compile_dsl` initializes `equations` before the formulation branch so
  a DSL with no Lagrangian / Hamiltonian / fluid produces a clear error
  instead of `UnboundLocalError`.
- `tests/unit/test_server_comprehensive.py` previously used non-existent
  `\coordinates` / `\parameters` (plural) DSL keywords; replaced with
  real `\defvar` / `\parameter` syntax. These tests were silently
  skipped before fastapi was installed; now they actually run.
- `tests/test_cli.py::test_cli_version` no longer hardcodes `"2.0"`;
  it now asserts the CLI version tracks the package version.

### Changed

- **`validators.py` fully migrated to Pydantic v2** — `@validator` →
  `@field_validator` (+`@classmethod`), `@root_validator` →
  `@model_validator(mode="after")`, `class Config` →
  `model_config = ConfigDict(...)`. Removes deprecation warnings on
  Pydantic 2.x.
- **Compiler version is single-source** — `compiler.py` no longer carries
  a stale `__version__ = "1.5.0"` fallback; it lazy-loads the real
  version from the package root.

### Tests

- 2182 passing, 9 skipped, 0 failing (was 2155). +27 net tests: 11 physics
  validation, 14 codegen smoke, 3 feature e2e (one skipped).

---

## [2.1.0] - 2026-05-29

### 🛡️ Correctness, Security & Consolidation Release

A focused pass over the compile pipeline, the server, and the package
layout. Most users should upgrade; two behaviors changed in ways callers
may rely on - see **Breaking changes** below.

### Added

- **`\force{coord}{expr}`** — generalized forces can now target a named
  coordinate. The legacy `\force{expr}` form still works but applies
  positionally and now warns when used with a multi-DOF system.
- **`PhysicsCompiler.export(target, filename)`** — public method that
  dispatches to any of the code generators. The server `/export` route
  (previously broken — see Fixed) now uses it.
- **`result["warnings"]`** — `compile_dsl()` now surfaces diagnostics from
  the symbolic solver and the equation compiler instead of silently
  returning zero accelerations. A successful compile with non-empty
  warnings is no longer indistinguishable from a clean one.
- **Public exports for complementary modules** — `NumbaSimulator`,
  `is_numba_available`, `EnergyAnalyzer`, and `StabilityAnalyzer` are now
  importable from `mechanics_dsl` directly.
- **`MAX_SESSIONS = 256`** bound on the server's in-memory session store
  with LRU eviction and a thread lock. Anonymous (no `session_id`)
  requests now get a fresh per-request compiler.
- Regression test suite (`tests/unit/test_v2_1_0_regressions.py`, 24
  tests) and FastAPI end-to-end tests (`tests/integration/
  test_server_endpoints.py`, 10 tests) pinning every fix in this release.

### Changed

- **Visualization package layout** — `src/mechanics_dsl/visualization.py`
  moved into the `visualization/` package as `_legacy.py`. The previous
  `__init__.py` re-loaded the colliding module file via
  `importlib.util.spec_from_file_location`; that hack is gone.
  `MechanicsVisualizer` still re-exports unchanged.
- `requirements.txt` is now a thin pointer at `pyproject.toml` listing
  only the real runtime deps. The `numba`, `pytest`, `black`, and
  `flake8` entries are gone (none of those are runtime dependencies).
- `mypy` `attr-defined` warnings are no longer suppressed — re-enabling
  that check would have caught the broken `/export` route.

### Fixed

- **`compile_dsl()` no longer leaks state between calls.** A reused
  `PhysicsCompiler` previously inherited variables, constraints, forces,
  initial conditions, and simulator parameters from the prior compile;
  the server's reused per-session compiler made this silently corrupt
  subsequent systems. The compiler now resets all per-system state at
  the start of every `compile_dsl()` call.
- **`/export` endpoint** called `compiler.export(target, filename)` —
  a method that did not exist on `PhysicsCompiler`. Now the method
  exists and the route works.
- **Generalized forces are matched by coordinate**, not by list index.
  Previously the i-th `\force` was bound to the i-th coordinate, which
  silently mis-assigned in any multi-DOF system.
- **Symbolic solver diagnostics surface** through `result["warnings"]`
  instead of being swallowed. `solve_for_accelerations` previously
  returned `0` on every failure path and the compile still reported
  `success: True`.
- **`compile_to_cpp` argv hardening** — the generated source path is
  normalized to absolute so a filename starting with `-` cannot be
  reinterpreted as a g++ flag.
- **Symbolic cache key** is now `str(expr)` instead of
  `str(hash(str(expr)))`. The previous hash-of-string keys could
  collide and return wrong cached SymPy expressions.
- `compile_dsl` initialises `equations` before the formulation branch so
  a Lagrangian-less / Hamiltonian-less / fluid-less input gets a clear
  error rather than `UnboundLocalError`.

### Security

- **Pickle deserialization is opt-in.**
  `SystemSerializer.load_pickle`, `deserialize_solution`,
  `SystemSerializer.import_system`, and
  `PhysicsCompiler.import_system` now refuse `.pkl` / `.pickle` files
  by default and require an explicit `allow_pickle=True` argument.
  Pickle can execute arbitrary code; the previous default was unsafe
  for any file the user did not produce themselves.
- **Server session DoS** mitigated — the session store is bounded and
  the default session is no longer a shared mutable compiler across
  anonymous requests.
- **Rate limiter** keys anonymous traffic under `"anonymous"` so it
  can't be bypassed by simply omitting `session_id`.
- **Tokenizer rejects unrecognized characters** with a precise position
  (was silently dropping them). Malformed DSL is now an error at the
  earliest stage, not surprise behavior downstream.

### Removed

- `src/mechanics_dsl/compiler_pkg/` — zero references anywhere in the
  codebase. Dead.
- `src/mechanics_dsl/utils/units.py` — a verbatim duplicate of the
  top-level `units.py`. Only its own test imported it.
- `src/mechanics_dsl/error_handling.py` and its test — imported only
  by the test itself, never wired into the package. Use the
  `exceptions` module instead.

### Breaking changes

- **Pickle is no longer loaded by default.** Code that did
  `import_system("system.pkl")` must now pass `allow_pickle=True`. JSON
  loading is unaffected.
- **The tokenizer rejects unknown characters.** DSL sources that
  previously parsed by silently dropping `@`, `&`, etc. now raise a
  `ValueError` at compile time.

### Migration

```python
# Before
state = PhysicsCompiler.import_system("system.pkl")

# After (only if you produced the file yourself)
state = PhysicsCompiler.import_system("system.pkl", allow_pickle=True)
```

```latex
% Before - works on 1-DOF, silently wrong on multi-DOF
\force{F_drive}

% After - explicit target, multi-DOF safe
\force{theta}{F_drive}
```

---

## [2.0.7] - 2026-01-30

### Changed

- Version bumped to 2.0.7
- Internal code quality improvements

---

## [2.0.6] - 2026-01-18

### 🔧 Integration & Infrastructure Release

This release fixes critical issues with code generation integrations and Docker builds.

### Fixed

#### Game Engine Integrations
- **Unity Generator** — Now converts sympy equations to real C# code using `sympy_to_csharp()`
- **Unreal Generator** — Now converts sympy equations to real C++ code with FMath functions
- **Modelica Generator** — Now converts sympy equations to Modelica syntax

#### Docker Build
- Fixed `repository name must be lowercase` error in GPU image builds
- Fixed `Unable to get ACTIONS_ID_TOKEN_REQUEST_URL` attestation error
- Added `id-token: write` permission for provenance attestation

### Added

- **DSL File Imports** — `\import{filename}` directive for composable physics systems
- **CPU SPH Fallback** — Full CPU implementation when CUDA is unavailable (~200 lines C++)
- **Integration Tests** — New `tests/test_integrations.py` with 12 tests

### Changed

- Improved type hints in `repl.py` and `presets.py`
- Version bumped to 2.0.6

---

## [2.0.5] - 2026-01-17

### 🎮 Interactive Mode & Presets Release

This release adds an interactive REPL, built-in physics presets, and improved test infrastructure.

### Added

#### Interactive REPL
- **`mechanicsdsl repl`** — Interactive shell for experimenting with DSL
  - `:load <file>` — Load DSL from file
  - `:preset <name>` — Load built-in preset (pendulum, orbit, etc.)
  - `:compile` — Compile current buffer
  - `:run [t]` — Run simulation
  - `:plot [var]` — Plot results
  - `:export <fmt>` — Export to JSON/CSV/NumPy
  - Command history with readline support

#### Presets Library (`src/mechanics_dsl/presets.py`)
- **Classical Mechanics**: pendulum, double_pendulum, spring_mass, damped_oscillator, projectile, coupled_oscillators
- **Celestial Mechanics**: kepler_orbit, three_body_figure8
- **`mechanicsdsl presets`** — List all available presets

### Fixed

- **CLI tests** — Rewrote to use direct function imports instead of subprocess, fixing CI on all Python versions
- **Version consistency** — All version strings now synchronized at 2.0.5

### Changed

- Bumped version to 2.0.5
- CLI tests now use `unittest.mock` for reliability

---

## [2.0.1] - 2026-01-17

### 🛠️ Developer Experience & Documentation Release

This release adds a command-line interface, expanded tutorials, internationalization infrastructure, and quality-of-life improvements following the v2.0.0 enterprise release.

### Added

#### Command-Line Interface
- **`mechanicsdsl` CLI** (`src/mechanics_dsl/cli.py`)
  - `mechanicsdsl compile <input> --target <lang>` — Generate code in 11 languages
  - `mechanicsdsl run <input> --t-span 0,10 --animate` — Run simulations
  - `mechanicsdsl export <input> --format csv` — Export results (JSON, CSV, NumPy)
  - `mechanicsdsl validate <input>` — Validate DSL files without running
  - `mechanicsdsl info` — Show version and system capabilities
- **Shell Completion** — Bash, Zsh, and Fish autocomplete scripts

#### New Tutorials
- **Quantum Mechanics** (`tutorials/04_quantum_mechanics.ipynb`) — Tunneling, wavefunctions, hydrogen atom, WKB approximation
- **General Relativity** (`tutorials/05_general_relativity.ipynb`) — Schwarzschild geodesics, light bending, Einstein rings
- **Statistical Mechanics** (`tutorials/06_statistical_mechanics.ipynb`) — Ensembles, partition functions, Fermi-Dirac/Bose-Einstein, Ising model

#### Documentation
- **Examples Gallery** (`GALLERY.md`) — Curated showcase of all 9 physics domains with code examples
- **DEMO.md** — Quick one-liner examples for demos and presentations
- **Internationalization** (`docs/translations/`) — README stubs in Chinese, Japanese, German, Russian

#### Example Files
- **Standalone DSL files** (`examples/dsl/`) — Ready-to-run `.mdsl` files for CLI
  - `pendulum.mdsl`, `double_pendulum.mdsl`, `kepler_orbit.mdsl`, `damped_oscillator.mdsl`

#### Developer Tools
- **Benchmark Script** (`benchmarks/quick_benchmark.py`) — One-liner performance demonstration
- **CLI Unit Tests** (`tests/test_cli.py`) — Comprehensive CLI test coverage
- **VS Code Snippets** — New snippets for quantum, GR, fluids, statistical mechanics
- **Progress Bar** — tqdm integration for long simulations

### Changed

- Bumped version to 2.0.1
- Added Python 3.14 classifier to `pyproject.toml`
- Updated web demo stats to reflect 19 countries adoption
- Updated tutorials README with new tutorials

### Adoption Metrics

First 3 hours after v2.0.0 release:
- **17 countries** downloading (9 new countries)
- **6 enterprise mirrors** syncing (including new Philippine organization)
- **Real pip installs** on Debian and Ubuntu workstations

---

## [2.0.0] - 2026-01-16

### 🚀 Major Release: Enterprise & Embedded Support

This release adds comprehensive support for enterprise deployment, ARM/embedded platforms, and enhanced code generation based on analysis of real-world user adoption patterns.

### Added

#### Enterprise Deployment
- **Docker Support** - Multi-stage Dockerfile with CPU and GPU variants
  - `Dockerfile` - Production-ready container image
  - `docker-compose.yml` - API server, Jupyter, and worker services
  - `docker-compose.gpu.yml` - NVIDIA GPU override for CUDA acceleration
- **Enterprise Deployment Guide** (`docs/enterprise_deployment.md`)
  - Docker and Kubernetes deployment instructions
  - Internal PyPI mirror setup (Nexus, bandersnatch)
  - Security sandboxing and monitoring configuration

#### ARM & Embedded Platform Support
- **Raspberry Pi Examples** (`examples/embedded/`)
  - `raspberry_pi_pendulum.py` - Complete demo with C++ export
  - `raspberry_pi_imu.py` - MPU6050 IMU integration with sensor fusion
- **ARM Optimization Guide** (`docs/arm_optimization.md`)
  - Platform compatibility matrix
  - Performance tuning for Pi 3/4/5
  - Real-time control examples
- **Optional embedded dependencies** - `pip install mechanicsdsl-core[embedded]`

#### Enhanced Code Generation
- **C++ CMake Support** (`codegen/cpp.py`)
  - New `generate_cmake()` method for CMakeLists.txt generation
  - New `generate_project()` for complete project scaffolding
  - ARM/NEON optimization detection
  - Cross-compilation hints for Raspberry Pi
- **Rust Cargo Support** (`codegen/rust.py`)
  - New `generate_cargo_toml()` for Rust project setup
  - New `generate_project()` with embedded (`no_std`) option
  - ARM cross-compilation support

#### CI/CD Improvements
- **Python 3.14 Testing** - Added Python 3.14-dev to CI matrix (experimental)
- **Extended Python Classifiers** - Added 3.8-3.13 individual classifiers
- **Comprehensive Keywords** - Added `embedded`, `raspberry-pi`, `arm`, `gpu`, `cuda`

### Changed

- Bumped version to 2.0.0
- Updated `pyproject.toml` with new classifiers and keywords
- Black configuration extended to Python 3.13

### User Impact Analysis

This release specifically addresses needs identified from our global enterprise adoption:
- 🇸🇬 **Singapore** (Rust developer, 42 downloads) → Enhanced Rust/C++ codegen
- 🇬🇧 **UK** (Raspberry Pi user) → ARM examples and optimization guide
- 🇨🇳 **China** (regulated industry) → Python 3.8 support maintained
- 🇯🇵 **Japan** (Python 3.14 tester) → CI support for bleeding-edge Python
- 🇰🇷 **Korea** (enterprise deployment) → Docker and enterprise docs



## [1.5.1] - 2026-01-11

### Added

#### Architecture Improvements
- **Variable Registry Module** (`utils/registry.py`) - Centralized variable type classification
  - Replaces hardcoded coordinate/constant detection logic
  - Extensible `VariableCategory` enum and classification functions
  - `is_likely_coordinate()`, `is_constant_type()`, `classify_variable()` utilities

- **Solver Package Migration** - Created modular `solver/` package
  - `solver/core.py` - Main NumericalSimulator class (725 lines)
  - `solver/__init__.py` - Clean public API
  - Deleted legacy `solver.py` module

- **Protocol-based Typing** (`protocols.py`) - Structural typing support
  - `SimulatableProtocol`, `CompilableProtocol`, `VisualizableProtocol`
  - `CodeGeneratorProtocol`, `PhysicsDomainProtocol`
  - Runtime-checkable for duck-typing validation

- **Actionable Exception Classes** (`exceptions.py`) - 14 new exception types
  - `MechanicsDSLError` base class with suggestions and docs links
  - `NoLagrangianError`, `NoCoordinatesError`, `IntegrationError`
  - Each exception provides actionable suggestions for fixing errors

- **Expanded Constants** - 50+ magic numbers moved to `utils/config.py`
  - Physical constants: `STANDARD_GRAVITY`, `SPEED_OF_LIGHT`, `HBAR`
  - Numerical constants: `DEFAULT_NUM_POINTS`, `SINGULARITY_THRESHOLD`
  - Visualization: `DEFAULT_DPI`, `ACCENT_COLOR`, `WARNING_COLOR`
  - Cache/file limits: `MAX_PATH_LENGTH`, `MAX_FILE_SIZE`

#### Security Enhancements
- **Pickle Security Warning** - Added explicit warning when loading pickle files
  - Warns users about arbitrary code execution risks
  - Recommends JSON format for untrusted data
  
- **Enhanced File Path Validation** - Comprehensive security checks
  - Null byte injection prevention
  - Path traversal (`..`) detection
  - Special character filtering
  - Symlink attack prevention (optional)
  - Excessive path length check

#### Developer Experience
- **VS Code Extension Enhancements**
  - 15+ snippets for common physics systems (pendulum, oscillator, fluid)
  - Version sync with package (now v1.5.0)
  - Added snippet support and better metadata

- **Pre-commit Hooks** - Enhanced code quality automation
  - `isort` - Import sorting
  - `bandit` - Security vulnerability scanning
  - `docformatter` - Consistent docstring formatting
  - Additional pre-commit-hooks (check-ast, check-docstring-first)

- **CI/CD Pipeline Improvements**
  - `mypy` runs on all Python versions (was only 3.11)
  - Benchmark tests with `pytest-benchmark`
  - Lint job with `black`, `isort`, `flake8`
  - Security scanning job with `bandit`

- **Pytest Markers** - Test categorization
  - `@pytest.mark.slow` - Long-running tests
  - `@pytest.mark.gpu_required` - GPU-dependent tests
  - `@pytest.mark.physics_validation` - Physics correctness checks
  - `@pytest.mark.performance` - Benchmark tests

#### Documentation
- **Expanded Fluid Dynamics Docs** - 38 → 250+ lines
  - Comprehensive SPH theory
  - Code examples for all features
  - Kernel descriptions and boundary handling

- **Updated Binder Environment**
  - JupyterLab support
  - Flexible Python version (3.9-3.12)
  - Additional packages (ipywidgets, hypothesis)

- **Updated Conda Recipe**
  - Version 1.5.0
  - Expanded test suite
  - Better feature description

### Changed

#### Code Organization
- **Parser Package Migration Complete** - Deleted legacy `parser.py` (1,030 lines)
  - All imports now use the modular `parser/` package structure
  - Backward compatible - no import changes required
  
- **Solver Package Migration** - Deleted legacy `solver.py` (700+ lines)
  - All functionality now in `solver/` package

- **Module Exports** - Added `__all__` to core modules
  - `symbolic.py`, `solver.py`, `visualization.py`
  - Clearer public API definition
   
#### Configuration
- **Config.to_dict()** - Now exports all v6.0 properties (was missing 11 fields)
- **Config.from_dict()** - Now uses property setters for proper validation

### Fixed
- Version string mismatch between `compiler.py` (was 1.4.0) and package (1.5.0)

### Improved
- Enhanced docstrings throughout with `Warning:` sections for security-sensitive APIs
- Improved error messages to be more actionable with suggestions
- All 1,052 unit tests passing

---

## [1.4.0] - 2026-01-04

### Added

#### New Physics Domains
- **9 New Physics Domains** with rigorous implementations:
  - `mechanics_dsl.domains.quantum/` - Quantum mechanics (tunneling, finite well, hydrogen atom)
  - `mechanics_dsl.domains.electromagnetic/` - Electromagnetic field theory
  - `mechanics_dsl.domains.relativistic/` - Special and general relativity
  - `mechanics_dsl.domains.statistical_mechanics/` - Statistical mechanics
  - `mechanics_dsl.domains.thermodynamics/` - Thermodynamics

#### Testing
- **1,465+ unit tests** with comprehensive coverage
- Property-based testing with Hypothesis
- Graceful test skipping when optional dependencies not installed

#### Tooling
- **VS Code Extension** for MechanicsDSL syntax highlighting
- **Conda Recipe** for conda-forge distribution
- **py.typed** marker for PEP 561 compliance
- **MyPy CI** for static type checking
- **Binder support** for interactive notebooks

### Fixed
- Improved error handling in `animate_fluid` for file validation and missing columns
- Visualization tests updated to match new Animator API
- DSL syntax fixes in property tests (curly braces for functions)

### Changed
- Enhanced test infrastructure with 270+ new unit tests
- Improved code coverage across all modules

---

## [1.2.3] - 2024-12-06

### Added
- `HamiltonJacobi.solve()` method for solving Hamilton-Jacobi equations with an intuitive API

---

## [1.2.2] - 2024-12-06

### Added

#### Classical Mechanics Modules
- **13 New Physics Modules** in `mechanics_dsl.classical/`:
  - `central_force.py` - Central force problems (orbits, scattering)
  - `coupled_oscillators.py` - Coupled harmonic oscillators
  - `damped_driven.py` - Damped and driven oscillators
  - `elastic_collision.py` - Elastic collision mechanics
  - `gravitational.py` - N-body gravitational systems
  - `gyroscope.py` - Gyroscopic motion and precession
  - `noninertial.py` - Non-inertial reference frames
  - `projectile.py` - Projectile motion with drag
  - `rigid_body_dynamics.py` - Rigid body motion
  - `rotating_frame.py` - Rotating reference frames
  - `small_oscillations.py` - Small oscillation analysis
  - `variational.py` - Variational methods
  - `virtual_work.py` - Virtual work principles

#### Code Generation Backends
- **5 New Code Generators** in `mechanics_dsl.codegen/`:
  - `julia.py` - Julia code generation (DifferentialEquations.jl)
  - `rust.py` - Rust code generation (nalgebra/ode_solvers)
  - `matlab.py` - MATLAB/Octave code generation
  - `fortran.py` - Modern Fortran code generation
  - `javascript.py` - JavaScript/Node.js code generation

#### Jupyter Notebooks
- **30+ Interactive Notebooks** in `examples/notebooks/`:
  - Beginner tutorials (pendulum, oscillator, projectile)
  - Intermediate examples (double pendulum, coupled systems)
  - Advanced topics (Hamiltonian, Lagrangian, rigid body)
  - Tools demonstrations (visualization, code generation)
  - Fluids and celestial mechanics examples

#### Documentation
- Comprehensive documentation for all 13 new classical mechanics modules
- API references with theory explanations and usage examples
- Updated `docs/index.rst` with new module documentation
- Updated `README.md` with expanded feature list

### Fixed
- Test suite now passes 100% (47/47 tests)
- Fixed DSL syntax issues in `test_compiler.py`
- Fixed acceleration key format in `test_solver.py`
- Corrected method names in `test_visualization.py`
- Fixed validation behavior in `test_energy.py`
- Fixed `Config` attribute error (`default_trail_length`)

### Changed
- Improved test coverage across all modules
- Enhanced backward compatibility imports

---

## [1.2.0] - 2025-12-31

### Added

#### New Package Structure
- **`mechanics_dsl.utils/`** - Modular utilities package
  - `logging.py` - Centralized logging configuration
  - `config.py` - Configuration management and constants
  - `caching.py` - LRU cache implementation
  - `profiling.py` - Performance monitoring and timeouts
  - `validation.py` - Input validation functions
  - `units.py` - Physical units system

- **`mechanics_dsl.core/`** - Core compiler infrastructure
  - Reorganized `compiler.py`, `parser.py`, `symbolic.py`, `solver.py`
  - Updated imports for new package structure

- **`mechanics_dsl.domains/`** - Physics domain implementations
  - `base.py` - Abstract `PhysicsDomain` class
  - `classical/` - Lagrangian, Hamiltonian, constraints, rigid body
  - `fluids/` - SPH simulation, boundary conditions

- **`mechanics_dsl.visualization/`** - Modular visualization
  - `animator.py` - Animation classes
  - `plotter.py` - General plotting utilities
  - `phase_space.py` - Phase space and Poincaré sections

- **`mechanics_dsl.analysis/`** - Analysis tools
  - `energy.py` - Energy conservation analysis
  - `stability.py` - Fixed point and eigenvalue analysis

- **`mechanics_dsl.codegen/`** - Code generation backends
  - `base.py` - Abstract `CodeGenerator` class
  - `python.py` - Python/NumPy code generation
  - Improved C++ templates

- **`mechanics_dsl.io/`** - I/O utilities
  - `serialization.py` - JSON/pickle serialization
  - `export.py` - CSV and JSON exporters

#### Documentation
- Complete rewrite of documentation with extensive API reference
- 8 detailed tutorials covering various physics scenarios
- DSL syntax reference with all commands
- Advanced topics guide

### Changed
- Reorganized package structure for better modularity
- Improved backward compatibility with legacy imports
- Enhanced error messages throughout

### Fixed
- Various import path issues
- Profiling conflicts with nested decorators

### Deprecated
- Direct imports from `mechanics_dsl.utils` (monolithic) - use submodules
- `MechanicsVisualizer` - use modular `Animator`, `Plotter`, `PhaseSpaceVisualizer`

---

## [1.1.0] - 2024-09-01

### Added
- SPH fluid dynamics support
- WebAssembly code generation
- Hamiltonian formulation support
- Non-holonomic constraints
- Performance monitoring utilities

### Changed
- Improved symbolic simplification
- Better stiffness detection
- Enhanced visualization

---

## [1.0.0] - 2024-06-15

### Added
- Initial release
- LaTeX-inspired DSL for classical mechanics
- Lagrangian mechanics support
- Automatic Euler-Lagrange equation derivation
- Numerical simulation with SciPy
- 3D visualization with Matplotlib
- C++ code generation
- Energy conservation analysis
- Phase space visualization

### Features
- Simple pendulum, double pendulum examples
- Harmonic oscillator support
- Basic constraint handling
- CSV export
