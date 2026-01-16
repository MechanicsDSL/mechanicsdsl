# Changelog

All notable changes to MechanicsDSL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
