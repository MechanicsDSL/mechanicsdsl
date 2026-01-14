# MechanicsDSL: Comprehensive Codebase Analysis

**Generated:** 2025-01-XX  
**Version Analyzed:** 1.5.1  
**Total Python Files:** ~92+ source files, 1569+ test functions  
**Lines of Code:** Estimated 20,000+ lines  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Architecture & Design](#architecture--design)
4. [Complete Directory Structure](#complete-directory-structure)
5. [Core Components - Detailed Analysis](#core-components---detailed-analysis)
6. [Physics Domains](#physics-domains)
7. [Code Generation System](#code-generation-system)
8. [Visualization System](#visualization-system)
9. [Utilities & Infrastructure](#utilities--infrastructure)
10. [Testing Infrastructure](#testing-infrastructure)
11. [Examples & Demos](#examples--demos)
12. [Documentation System](#documentation-system)
13. [Build System & Configuration](#build-system--configuration)
14. [Dependencies & Requirements](#dependencies--requirements)
15. [Development Tools](#development-tools)
16. [Version History & Changelog](#version-history--changelog)
17. [Performance Characteristics](#performance-characteristics)
18. [Security Features](#security-features)
19. [API Reference Summary](#api-reference-summary)
20. [Future Extensibility](#future-extensibility)

---

## Executive Summary

**MechanicsDSL** is a sophisticated Domain-Specific Language (DSL) and compiler framework for computational physics, specifically designed for classical mechanics and related physics domains. The project implements a complete multi-stage compiler pipeline that transforms LaTeX-inspired DSL code into optimized numerical simulations.

### Key Statistics
- **Primary Language:** Python 3.8+
- **Core Dependencies:** NumPy, SciPy, SymPy, Matplotlib
- **Architecture:** Multi-stage compiler (Lexer → Parser → Semantic Analysis → Symbolic Engine → Numerical Solver → Code Generation)
- **Physics Coverage:** Classical mechanics, quantum mechanics, electromagnetism, relativity, thermodynamics, statistical mechanics, fluid dynamics
- **Code Generation Targets:** C++, CUDA, WebAssembly, Python, Julia, Rust, MATLAB, Fortran, JavaScript, Arduino, OpenMP
- **Test Coverage:** 1569+ test functions across 74+ test files
- **Documentation:** 60+ reStructuredText files in comprehensive Sphinx documentation

### Design Philosophy
The codebase follows enterprise-grade design principles:
- **No eval()** - Safe AST-based parsing prevents code injection
- **Modular Architecture** - Clear separation of concerns across packages
- **Type Safety** - Extensive type hints throughout (PEP 561 compliant)
- **Error Recovery** - Comprehensive exception handling with actionable error messages
- **Performance Monitoring** - Built-in profiling and memory monitoring
- **Security Hardening** - Input validation, path traversal protection, pickle warnings

---

## Project Overview

### Purpose
MechanicsDSL bridges the gap between symbolic physics (Lagrangian/Hamiltonian mechanics) and numerical simulation, allowing researchers and educators to:
1. Define physical systems using intuitive LaTeX-inspired syntax
2. Automatically derive equations of motion (Euler-Lagrange, Hamilton's equations)
3. Generate optimized numerical simulations
4. Visualize results with phase space plots, animations, and energy analysis
5. Export to multiple target languages for deployment

### Target Users
- **Physics Educators:** Students learning Lagrangian mechanics
- **Researchers:** Rapid prototyping of complex systems
- **Engineers:** Numerical simulation generation
- **Students:** Interactive physics exploration

### Core Value Proposition
Instead of manually deriving equations and writing numerical solvers, users write DSL code like:
```latex
\system{pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg} \parameter{l}{1.0}{m} \parameter{g}{9.81}{m/s^2}
\lagrangian{\frac{1}{2}*m*l^2*\dot{theta}^2 + m*g*l*cos(theta)}
\initial{theta=0.1, theta_dot=0.0}
```

The system automatically:
1. Parses the DSL
2. Derives the equation of motion: `theta_ddot = -(g/l)*sin(theta)`
3. Compiles to numerical functions
4. Integrates using adaptive ODE solvers
5. Provides visualization

---

## Architecture & Design

### Compiler Pipeline

The MechanicsDSL compiler implements a multi-stage pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    DSL Source Code                      │
│  (LaTeX-inspired syntax with \commands)                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  TOKENIZER (parser/tokens.py)                           │
│  - Break source into tokens (COMMAND, IDENT, NUMBER...) │
│  - Handle LaTeX commands, Greek letters, operators      │
│  - Position tracking for error messages                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  PARSER (parser/core.py)                                │
│  - Recursive descent parser                             │
│  - Build Abstract Syntax Tree (AST)                     │
│  - Operator precedence (^ > * / > + -)                  │
│  - Error recovery and reporting                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  SEMANTIC ANALYSIS (compiler.py:analyze_semantics)      │
│  - Extract variables, parameters, Lagrangians           │
│  - Build symbol tables                                  │
│  - Detect coordinate types vs constants                 │
│  - Process fluid/particle definitions                   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  SYMBOLIC ENGINE (symbolic.py)                          │
│  - Convert AST to SymPy expressions                     │
│  - Derive Euler-Lagrange equations                      │
│  - Handle constraints (Lagrange multipliers)            │
│  - Lagrangian → Hamiltonian conversion                  │
│  - Common subexpression elimination                     │
│  - Symbolic simplification with timeouts                │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  NUMERICAL COMPILATION (solver/core.py)                 │
│  - Convert SymPy to NumPy functions (lambdify)          │
│  - Build ODE system (state vector)                      │
│  - Substitute parameters                                │
│  - Create wrapper functions for solve_ivp               │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  SIMULATION (solver/core.py:NumericalSimulator)         │
│  - Adaptive ODE solver (RK45, LSODA, Radau)             │
│  - Stiffness detection                                  │
│  - Time step adaptation                                 │
│  - Generate solution arrays                             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  VISUALIZATION (visualization/)                         │
│  - Animations (matplotlib.animation)                    │
│  - Phase space plots                                    │
│  - Energy analysis                                      │
│  - Trajectory plots                                     │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  CODE GENERATION (codegen/)                             │
│  - C++, CUDA, WebAssembly, Python, etc.                 │
│  - Template-based generation                            │
│  - Target-specific optimizations                        │
└─────────────────────────────────────────────────────────┘
```

### Design Patterns

1. **Visitor Pattern:** AST traversal in symbolic engine
2. **Strategy Pattern:** Multiple code generation backends
3. **Template Method:** CodeGenerator base class
4. **Factory Pattern:** Symbol creation in SymbolicEngine
5. **Singleton Pattern:** Config and logger (module-level)
6. **Registry Pattern:** Variable type classification (utils/registry.py)

### Module Organization

The codebase is organized into logical packages:

```
mechanics_dsl/
├── __init__.py              # Public API exports
├── compiler.py              # Main PhysicsCompiler class
├── symbolic.py              # SymbolicEngine for SymPy operations
├── solver/                  # Numerical simulation (package)
├── parser/                  # Lexer and parser (package)
├── codegen/                 # Code generation backends (package)
├── domains/                 # Physics domain implementations (package)
├── visualization/           # Plotting and animation (package)
├── analysis/                # Energy, stability analysis (package)
├── utils/                   # Utilities (package)
├── io/                      # I/O operations (package)
└── exceptions.py            # Custom exception classes
```

---

## Complete Directory Structure

### Root Level
```
mechanicsdsl-main/
├── src/mechanics_dsl/          # Main source code (92+ Python files)
├── tests/                       # Test suite (74+ test files, 1569+ tests)
├── examples/                    # 30+ example scripts + 30+ notebooks
├── docs/                        # Sphinx documentation (60+ .rst files)
├── benchmarks/                  # Performance benchmarks
├── demos/                       # Language-specific demos (C++, CUDA, etc.)
├── web-demo/                    # WebAssembly demo
├── vscode-extension/            # VS Code syntax highlighting
├── binder/                      # Binder environment config
├── conda-recipe/                # Conda package recipe
├── dist/                        # Built distributions
├── README.md                    # Main project README
├── pyproject.toml               # Modern Python build config
├── requirements.txt             # Core dependencies
├── LICENSE                      # MIT License
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── CODE_OF_CONDUCT.md           # Community standards
├── SECURITY.md                  # Security policy
├── CITATION.cff                 # Citation metadata
└── paper.md                     # Academic paper
```

### Source Code Structure (`src/mechanics_dsl/`)

#### Core Modules
- **`compiler.py`** (875 lines): Main `PhysicsCompiler` class, compilation pipeline, system serialization
- **`symbolic.py`** (~600 lines): `SymbolicEngine` for SymPy operations, equation derivation
- **`exceptions.py`** (~200 lines): 14 custom exception classes with actionable error messages
- **`protocols.py`**: Protocol-based typing (SimulatableProtocol, CompilableProtocol, etc.)
- **`units.py`**: Unit system implementation
- **`energy.py`**: Energy analysis utilities

#### Parser Package (`parser/`)
- **`tokens.py`** (~210 lines): Token definitions, tokenization function
  - 40+ token types (COMMAND, IDENT, NUMBER, GREEK_LETTER, etc.)
  - Position tracking for error messages
  - Regex-based tokenization
- **`ast_nodes.py`** (~800 lines): 33+ AST node classes
  - Expression nodes: NumberExpr, IdentExpr, BinaryOpExpr, FunctionCallExpr, etc.
  - Statement nodes: SystemDef, VarDef, LagrangianDef, ConstraintDef, etc.
  - SPH nodes: FluidDef, BoundaryDef, RegionDef
- **`core.py`** (~800 lines): `MechanicsParser` recursive descent parser
  - Operator precedence parsing
  - Error recovery
  - AST construction

#### Solver Package (`solver/`)
- **`core.py`** (~725 lines): `NumericalSimulator` class
  - ODE compilation (SymPy → NumPy lambdify)
  - Adaptive solver selection (RK45, LSODA, Radau)
  - Stiffness detection
  - State vector management
- **`__init__.py`**: Public API exports
- **`solver_numba.py`**: Numba JIT-accelerated solver (optional)

#### Code Generation Package (`codegen/`)
- **`base.py`**: Abstract `CodeGenerator` base class
- **`cpp.py`**: C++ code generator with multiple templates
- **`cuda.py`**: CUDA GPU code generation
- **`cuda_sph.py`**: CUDA SPH fluid simulation generator
- **`wasm.py`**: WebAssembly code generation (Emscripten)
- **`python.py`**: Python code generation
- **`julia.py`**: Julia code generation
- **`rust.py`**: Rust code generation
- **`matlab.py`**: MATLAB/Octave code generation
- **`fortran.py`**: Modern Fortran code generation
- **`javascript.py`**: JavaScript/Node.js code generation
- **`openmp.py`**: OpenMP parallel code generation
- **`arduino.py`**: Arduino sketch generation
- **`templates/`**: 12 template files (.cpp, .f90, .ino, .jl, .rs, .m, .js, .wasm)

#### Physics Domains Package (`domains/`)

##### Classical Mechanics (`domains/classical/`)
17 modules covering:
- **`lagrangian.py`**: LagrangianMechanics class
- **`hamiltonian.py`**: HamiltonianMechanics class
- **`constraints.py`**: ConstraintHandler, BaumgarteStabilization, ConstrainedLagrangianSystem
- **`rigidbody.py`**: RigidBodyDynamics, EulerAngles, Quaternion, SymmetricTop, Gyroscope
- **`dissipation.py`**: RayleighDissipation, FrictionModel (viscous, Coulomb, Stribeck)
- **`stability.py`**: StabilityAnalyzer, equilibrium point finding, eigenvalue analysis
- **`symmetry.py`**: NoetherAnalyzer, conserved quantity detection, cyclic coordinates
- **`central_forces.py`**: CentralForceAnalyzer, EffectivePotential, KeplerProblem, OrbitalElements
- **`canonical.py`**: CanonicalTransformation, GeneratingFunction, ActionAngleVariables, HamiltonJacobi
- **`oscillations.py`**: NormalModeAnalyzer, coupled oscillators, modal decomposition
- **`perturbation.py`**: Perturbation theory implementations
- **`collisions.py`**: Elastic/inelastic collision mechanics
- **`scattering.py`**: Rutherford scattering, cross-sections
- **`nonholonomic.py`**: Non-holonomic constraint handling
- **`variable_mass.py`**: Rocket equations, variable mass systems
- **`continuum.py`**: Continuous systems (strings, membranes)

##### Other Physics Domains
- **`fluids/`**: SPH (Smoothed Particle Hydrodynamics) implementation
  - `sph.py`: SPHFluid class with Poly6, Spiky, Viscosity kernels
  - `boundary.py`: BoundaryConditions (no-slip, periodic, reflective)
- **`quantum/`**: Quantum mechanics (tunneling, finite well, hydrogen atom)
- **`electromagnetic/`**: Electromagnetic field theory, charged particles
- **`relativistic/`**: Special relativity (Lorentz transformations, four-vectors)
- **`general_relativity/`**: General relativity (black holes, geodesics, lensing)
- **`statistical/`**: Statistical mechanics (ensembles, distributions)
- **`thermodynamics/`**: Thermodynamics (heat engines, equations of state)
- **`kinematics/`**: Kinematics (1D/2D motion, projectiles, relative motion)
- **`base.py`**: Abstract PhysicsDomain base class

#### Visualization Package (`visualization/`)
- **`animator.py`**: Animator class for mechanical system animations
- **`plotter.py`**: Plotter class for time series, trajectories, energy plots
- **`phase_space.py`**: PhaseSpaceVisualizer for phase portraits, Poincaré sections
- **`__init__.py`**: Package exports with backward compatibility
- **`visualization.py`**: Legacy MechanicsVisualizer (backward compatibility)

#### Analysis Package (`analysis/`)
- **`energy.py`**: PotentialEnergyCalculator, energy conservation analysis
- **`stability.py`**: Stability analysis tools

#### Utilities Package (`utils/`)
- **`config.py`** (~430 lines): Config class with 50+ constants
  - Physical constants (gravity, speed of light, Planck constant)
  - Numerical constants (tolerances, timeouts, limits)
  - Visualization constants (colors, DPI, animation settings)
  - Cache and memory settings
- **`logging.py`**: Centralized logging configuration
- **`caching.py`**: LRUCache implementation with memory limits
- **`profiling.py`**: Performance monitoring, timeouts, memory snapshots
- **`validation.py`**: Input validation functions (file paths, types, ranges)
- **`registry.py`**: Variable type classification (coordinate vs constant detection)
- **`units.py`**: Unit system utilities

#### I/O Package (`io/`)
- **`serialization.py`**: JSON/pickle serialization with security warnings
- **`export.py`**: CSV, JSON exporters

#### Compiler Package (`compiler_pkg/`)
- **`serializer.py`**: SystemSerializer for export/import
- **`particles.py`**: Particle generation utilities

### Test Structure (`tests/`)

#### Test Categories
- **`unit/`** (32 files): Unit tests for individual modules
  - `test_compiler.py`, `test_parser.py`, `test_symbolic.py`, `test_solver.py`
  - `test_codegen.py`, `test_visualization.py`, `test_utils.py`
  - Coverage tests, extended tests
- **`physics/`** (36 files): Physics correctness tests
  - Classical mechanics: `test_hamiltonian_formulation.py`, `test_constrained_systems.py`
  - Advanced: `test_rigidbody_quaternions.py`, `test_chaotic_systems.py`
  - Quantum: `test_quantum.py`, `test_tunneling.py`
  - Relativity: `test_relativistic.py`, `test_general_relativity.py`
  - Fluids: `test_fluids.py`
  - And 30+ more specialized physics tests
- **`property/`** (3 files): Property-based testing with Hypothesis
- **`performance/`** (2 files): Performance benchmarks
- **`integration/`** (1 file): End-to-end integration tests
- **`test_backends/`** (2 files): Code generation backend tests
- **`conftest.py`**: Pytest fixtures and configuration

**Total:** 1569+ test functions across 74+ test files

### Examples Structure (`examples/`)

#### Python Examples (30+ files)
- **`beginner/`** (7 files): Getting started, harmonic oscillator, pendulum, plotting
- **`intermediate/`** (5 files): Double pendulum, coupled oscillators, damping, forcing
- **`advanced/`** (8 files): 3D gyroscope, spherical pendulum, constraints, Hamiltonian, chaos, energy, phase space, quaternions
- **`tools/`** (4 files): Custom visualizations, export/import, performance tuning, units
- **`codegen/`** (2 files): C++ export, advanced targets
- **`fluids/`** (4 files): SPH introduction, wave tank, droplet, sloshing
- **`celestial/`** (4 files): Elastic pendulum, N-body gravity, orbital mechanics, figure-8 orbit

#### Jupyter Notebooks (`notebooks/`)
30+ interactive notebooks mirroring the Python examples, organized by category

### Documentation Structure (`docs/`)

60+ reStructuredText files organized into:

#### Getting Started
- `getting_started/installation.rst`
- `getting_started/quickstart.rst`
- `getting_started/tutorials.rst`

#### User Guide
- `user_guide/guide.rst`
- `user_guide/dsl_syntax.rst`
- `user_guide/code_generation.rst`
- `user_guide/cuda_guide.rst`
- `user_guide/performance_optimization.rst`
- `user_guide/physics_background.rst`

#### API Reference (`api/`)
- `api/core.rst`
- `api/codegen.rst`
- `api/domains.rst`
- `api/analysis.rst`
- `api/visualization.rst`
- `api/utils.rst`
- `api/io.rst`

#### Physics Documentation (`physics/`)
- `physics/lagrangian_mechanics.rst`
- `physics/hamiltonian_mechanics.rst`
- `physics/constraint_physics.rst`
- `physics/rigid_body.rst`
- `physics/oscillations.rst`
- `physics/central_forces.rst`
- `physics/collisions.rst`
- `physics/scattering.rst`
- `physics/dissipation.rst`
- `physics/stability.rst`
- `physics/symmetry.rst`
- `physics/canonical.rst`
- `physics/perturbation.rst`
- `physics/nonholonomic.rst`
- `physics/variable_mass.rst`
- `physics/continuum.rst`
- `physics/fluid_dynamics.rst`
- `physics/quantum.rst`
- `physics/electromagnetic.rst`
- `physics/relativistic.rst`
- `physics/general_relativity.rst`
- `physics/statistical.rst`
- `physics/thermodynamics.rst`
- `physics/kinematics.rst`
- `physics/multiphysics.rst`

#### Code Generation (`codegen/`)
- `codegen/overview.rst`
- `codegen/cpp.rst`
- `codegen/cuda.rst`
- `codegen/wasm.rst`
- `codegen/python.rst`

#### Advanced Topics (`advanced_topics/`)
- `advanced_topics/architecture.rst`
- `advanced_topics/advanced.rst`
- `advanced_topics/extending.rst`
- `advanced_topics/performance.rst`

#### Project (`project/`)
- `project/changelog.rst`
- `project/contributing.rst`
- `project/license.rst`

#### Other
- `compiler_architecture.rst`
- `code_generator.rst`
- `embedded_(arduino).rst`
- `web_assembly.rst`
- `standard_c++.rst`
- `mechanics_dsl.rst`
- `index.rst`

### Supporting Files

#### Benchmarks (`benchmarks/`)
- `comparison_matrix.py`: Performance comparison
- `cuda_performance.py`: CUDA benchmarks
- `numba_performance.py`: Numba benchmarks

#### Demos (`demos/`)
Language-specific demonstrations:
- `cpp_pendulum/`: C++ code generation demo
- `cuda_pendulum/`: CUDA code generation demo
- `fortran_pendulum/`: Fortran code generation demo
- `javascript_pendulum/`: JavaScript demo
- `julia_pendulum/`: Julia demo
- `matlab_pendulum/`: MATLAB demo
- `openmp_pendulum/`: OpenMP parallel demo
- `python_pendulum/`: Python demo
- `rust_pendulum/`: Rust demo
- `wasm_pendulum/`: WebAssembly demo

#### Development Tools
- `vscode-extension/`: VS Code syntax highlighting and snippets
- `binder/`: Jupyter Binder environment configuration
- `conda-recipe/`: Conda package recipe
- `web-demo/`: WebAssembly web demo

---

## Core Components - Detailed Analysis

### 1. PhysicsCompiler (`compiler.py`)

The main entry point and orchestrator class.

#### Key Responsibilities
1. **DSL Compilation Pipeline:** Orchestrates tokenization → parsing → semantic analysis → equation derivation → simulation setup
2. **System State Management:** Maintains variables, parameters, Lagrangians, constraints, initial conditions
3. **Equation Derivation:** Coordinates with SymbolicEngine to derive equations of motion
4. **Simulation Management:** Sets up and runs numerical simulations
5. **Code Generation:** Coordinates code generation to various targets
6. **System Serialization:** Export/import system state (JSON/pickle)

#### Key Attributes
```python
# AST and System Definition
self.ast: List[ASTNode]              # Abstract syntax tree
self.variables: Dict[str, dict]      # Variable definitions
self.parameters_def: Dict[str, dict] # Parameter definitions
self.system_name: str                # System identifier
self.lagrangian: Optional[Expression] # Lagrangian AST
self.hamiltonian: Optional[Expression] # Hamiltonian AST

# Constraints and Forces
self.constraints: List[Expression]   # Holonomic constraints
self.non_holonomic_constraints: List[Expression]
self.forces: List[Expression]        # Non-conservative forces
self.damping_forces: List[Expression]

# Fluid Dynamics
self.fluid_particles: List[Dict]     # SPH fluid particles
self.boundary_particles: List[Dict]  # SPH boundary particles
self.smoothing_length: float         # SPH kernel length

# Core Engines
self.symbolic: SymbolicEngine        # Symbolic computation
self.simulator: NumericalSimulator   # Numerical solver
self.visualizer: MechanicsVisualizer # Visualization
self.unit_system: UnitSystem         # Unit handling

# Compilation State
self.equations: Optional[Dict]       # Derived equations
self.use_hamiltonian_formulation: bool
self.compilation_time: Optional[float]
```

#### Key Methods

##### `compile_dsl(dsl_source: str, use_hamiltonian: bool = False, use_constraints: bool = True) -> dict`
Main compilation method. Performs:
1. Input validation (type checking, size limits, security checks)
2. Tokenization via `tokenize()`
3. Parsing via `MechanicsParser`
4. Semantic analysis via `analyze_semantics()`
5. Fluid processing via `process_fluids()`
6. Equation derivation (Lagrangian or Hamiltonian)
7. Simulation setup via `setup_simulation()`
8. Returns compilation result dictionary with success status, timing, equations, etc.

##### `analyze_semantics()`
Extracts system information from AST:
- System name from `SystemDef`
- Variables from `VarDef`
- Parameters from `ParameterDef`
- Lagrangians/Hamiltonians from `LagrangianDef`/`HamiltonianDef`
- Constraints from `ConstraintDef`/`NonHolonomicConstraintDef`
- Forces from `ForceDef`
- Initial conditions from `InitialCondition`

##### `derive_equations() -> Dict[str, sp.Expr]`
Derives Euler-Lagrange equations:
1. Converts Lagrangian AST to SymPy
2. Calls `symbolic.derive_equations_of_motion()`
3. Applies non-conservative forces
4. Solves for accelerations
5. Returns dictionary mapping `{coord}_ddot` to acceleration expressions

##### `derive_constrained_equations() -> Dict[str, sp.Expr]`
Uses Lagrange multipliers for constrained systems:
1. Builds extended coordinate list (includes multipliers)
2. Derives constrained equations
3. Solves for accelerations and multipliers
4. Filters out multiplier equations

##### `derive_hamiltonian_equations() -> Tuple[List[sp.Expr], List[sp.Expr]]`
Derives Hamilton's equations:
1. Converts Hamiltonian to SymPy
2. Calls `symbolic.derive_hamiltonian_equations()`
3. Returns (q_dots, p_dots) lists

##### `setup_simulation(equations)`
Configures NumericalSimulator:
1. Collects parameters
2. Sets initial conditions
3. Compiles equations (Lagrangian or Hamiltonian formulation)

##### `simulate(t_span, num_points, **kwargs) -> dict`
Runs numerical simulation via `self.simulator.simulate()`

##### `compile_to_cpp(filename, target, compile_binary) -> bool`
Generates C++ code using `CppGenerator`:
- Targets: 'standard', 'raylib', 'arduino', 'wasm', 'openmp', 'python'
- Optionally compiles binary (g++, emcc, etc.)

#### Security Features
- Input size limits (1MB max)
- Dangerous pattern detection (`__import__`, `eval()`, `exec()`, `compile()`)
- Comprehensive type validation
- File path validation (via `utils.validation`)

#### Performance Features
- Performance monitoring hooks
- Memory monitoring
- Compilation time tracking
- Garbage collection control

### 2. SymbolicEngine (`symbolic.py`)

Handles all symbolic mathematics using SymPy.

#### Key Responsibilities
1. **AST → SymPy Conversion:** Converts AST expressions to SymPy
2. **Equation Derivation:** Euler-Lagrange, Hamilton's equations
3. **Constraint Handling:** Lagrange multipliers
4. **Symbolic Simplification:** Common subexpression elimination, simplification
5. **Lagrangian → Hamiltonian:** Legendre transformation

#### Key Attributes
```python
self.symbol_map: Dict[str, sp.Symbol]      # Cached symbols
self.function_map: Dict[str, sp.Function]  # Cached functions
self.time_symbol: sp.Symbol                # Canonical 't' symbol
self.assumptions: Dict[str, dict]          # Symbol assumptions
self._cache: Optional[LRUCache]            # Expression cache
```

#### Key Methods

##### `ast_to_sympy(expr: Expression) -> sp.Expr`
Converts AST to SymPy with caching:
- Handles NumberExpr, IdentExpr, GreekLetterExpr
- BinaryOpExpr (+, -, *, /, ^)
- UnaryOpExpr (+, -)
- FunctionCallExpr (sin, cos, exp, log, sqrt, etc.)
- DerivativeVarExpr (\\dot{x} → x_dot symbol)
- DerivativeExpr (symbolic differentiation)
- VectorExpr, VectorOpExpr (dot, cross, grad)
- Uses LRU cache for performance

##### `derive_equations_of_motion(L_sympy, coordinates) -> List[sp.Expr]`
Derives Euler-Lagrange equations:
1. For each coordinate q:
   - Computes ∂L/∂q (partial derivative)
   - Computes ∂L/∂q_dot (partial derivative)
   - Computes d/dt(∂L/∂q_dot) (time derivative)
   - Forms: d/dt(∂L/∂q_dot) - ∂L/∂q = 0
2. Returns list of equations

##### `derive_equations_with_constraints(L, coordinates, constraints) -> Tuple[List[sp.Expr], List[str]]`
Uses Lagrange multipliers:
1. Creates multiplier symbols (lambda_i)
2. Forms extended Lagrangian: L' = L + Σ λ_i * constraint_i
3. Derives equations including multipliers
4. Returns (equations, extended_coordinates)

##### `derive_hamiltonian_equations(H, coordinates) -> Tuple[List[sp.Expr], List[sp.Expr]]`
Derives Hamilton's equations:
1. For each coordinate q:
   - q_dot = ∂H/∂p_q
   - p_dot = -∂H/∂q
2. Returns (q_dots, p_dots)

##### `lagrangian_to_hamiltonian(L, coordinates) -> sp.Expr`
Performs Legendre transformation:
1. Computes conjugate momenta: p_i = ∂L/∂q_i_dot
2. Inverts to get q_dot in terms of p
3. Forms H = Σ p_i * q_i_dot - L
4. Returns Hamiltonian expression

##### `solve_for_accelerations(eq_list, coordinates) -> Dict[str, sp.Expr]`
Solves equations for accelerations:
1. Replaces derivatives with acceleration symbols
2. Solves linear system
3. Returns dictionary mapping `{coord}_ddot` to expressions

#### Performance Optimizations
- LRU cache for expression conversion
- Common subexpression elimination
- Symbolic simplification with timeouts
- Performance monitoring hooks

### 3. NumericalSimulator (`solver/core.py`)

Compiles symbolic equations to numerical functions and runs simulations.

#### Key Responsibilities
1. **Equation Compilation:** SymPy → NumPy functions (lambdify)
2. **ODE System Construction:** Builds state vector and derivative function
3. **Solver Selection:** Chooses appropriate ODE solver (RK45, LSODA, Radau)
4. **Adaptive Integration:** Time step adaptation, stiffness detection
5. **Result Formatting:** Converts solution to structured format

#### Key Attributes
```python
self.symbolic: SymbolicEngine        # Symbolic engine reference
self.equations: Dict[str, Callable]  # Compiled equation functions
self.parameters: Dict[str, float]    # Physical parameters
self.initial_conditions: Dict[str, float]
self.constraints: List[sp.Expr]
self.state_vars: List[str]           # State variable names
self.coordinates: List[str]
self.use_hamiltonian: bool
self.hamiltonian_equations: Optional[Dict]
```

#### Key Methods

##### `compile_equations(accelerations: Dict[str, sp.Expr], coordinates: List[str])`
Compiles symbolic equations to numerical functions:
1. Builds state variable list: [q1, q1_dot, q2, q2_dot, ...]
2. Substitutes parameters into equations
3. Replaces derivatives with state variables
4. Uses `sp.lambdify()` to create NumPy functions
5. Creates wrapper functions for `solve_ivp` interface
6. Stores compiled functions in `self.equations`

##### `compile_hamiltonian_equations(q_dots, p_dots, coordinates)`
Compiles Hamiltonian equations similarly

##### `simulate(t_span, num_points, method='auto', **kwargs) -> dict`
Runs numerical simulation:
1. Builds initial state vector from initial conditions
2. Creates derivative function for ODE system
3. Selects solver method:
   - 'auto': Detects stiffness, chooses RK45 or LSODA
   - 'RK45': Explicit Runge-Kutta (default)
   - 'LSODA': Adaptive BDF/Adams (stiff systems)
   - 'Radau': Implicit Radau (very stiff)
4. Calls `scipy.integrate.solve_ivp()`
5. Formats solution dictionary:
   ```python
   {
       'success': bool,
       't': np.ndarray,      # Time points
       'y': np.ndarray,      # State vector [N_points, N_states]
       'coordinates': List[str],
       'message': str,
       'stats': dict
   }
   ```

#### Solver Selection Logic
- **RK45 (Dormand-Prince):** Default for non-stiff systems
- **LSODA:** Automatically switches between Adams and BDF methods
- **Radau:** For very stiff systems
- **Stiffness Detection:** Tests system behavior, selects accordingly

#### Error Handling
- Integration error handling
- Singularity detection
- Timeout protection
- Validation of inputs/outputs

### 4. Parser System (`parser/`)

#### Tokenizer (`parser/tokens.py`)

##### Token Types (40+)
- **Commands:** SYSTEM, DEFVAR, LAGRANGIAN, HAMILTONIAN, CONSTRAINT, FORCE, etc.
- **Math:** DOT_NOTATION (\\dot, \\ddot), FRAC, PARTIAL, INTEGRAL, SUM, LIMIT
- **Vectors:** VEC, HAT, MAGNITUDE, VECTOR_DOT, VECTOR_CROSS, GRADIENT, DIVERGENCE, CURL, LAPLACIAN
- **Greek Letters:** GREEK_LETTER (comprehensive: alpha, beta, gamma, ..., omega)
- **SPH:** FLUID, BOUNDARY, REGION, PARTICLE_MASS, EOS
- **Basic:** NUMBER, IDENT, WHITESPACE, COMMENT
- **Operators:** PLUS, MINUS, MULTIPLY, DIVIDE, POWER (^)
- **Punctuation:** LBRACE, RBRACE, LPAREN, RPAREN, COMMA, SEMICOLON, EQUALS

##### Token Class
```python
@dataclass
class Token:
    type: str           # Token type
    value: str          # Matched string
    position: int       # Character position
    line: int           # Line number (1-indexed)
    column: int         # Column number (1-indexed)
```

##### `tokenize(source: str) -> List[Token]`
- Uses regex pattern matching
- Tracks line/column for error messages
- Handles comments (% ...)
- Returns list of tokens

#### Parser (`parser/core.py`)

##### MechanicsParser Class
Recursive descent parser with operator precedence.

**Operator Precedence:**
1. Primary: numbers, identifiers, parentheses, function calls
2. Unary: +, -
3. Power: ^ (right-associative)
4. Multiplicative: *, /, dot, cross
5. Additive: +, -

**Key Methods:**
- `parse() -> List[ASTNode]`: Main parsing method
- `parse_expression() -> Expression`: Expression parsing
- `parse_command() -> ASTNode`: Command parsing (\\system, \\lagrangian, etc.)
- `parse_term()`, `parse_factor()`, `parse_power()`: Precedence-based parsing
- `peek(offset) -> Token`: Lookahead
- `match(*types) -> Token`: Match and consume
- `expect(type) -> Token`: Require specific token

**Error Recovery:**
- Collects errors in `self.errors` list
- Continues parsing when possible
- Provides position information in errors

#### AST Nodes (`parser/ast_nodes.py`)

33+ node classes organized into:

**Base Classes:**
- `ASTNode`: Base class for all nodes
- `Expression`: Base class for expressions

**Expression Nodes:**
- `NumberExpr`: Numeric literals
- `IdentExpr`: Identifiers
- `GreekLetterExpr`: Greek letters
- `DerivativeVarExpr`: \\dot{x}, \\ddot{x}
- `BinaryOpExpr`: Binary operations
- `UnaryOpExpr`: Unary operations
- `FunctionCallExpr`: Function calls
- `FractionExpr`: \\frac{numerator}{denominator}
- `DerivativeExpr`: Symbolic derivatives
- `IntegralExpr`: Integrals
- `VectorExpr`: Vector literals
- `VectorOpExpr`: Vector operations

**Statement Nodes:**
- `SystemDef`: \\system{name}
- `VarDef`: \\defvar{name}{type}{unit}
- `ParameterDef`: \\parameter{name}{value}{unit}
- `DefineDef`: \\define{name}{args}{body}
- `LagrangianDef`: \\lagrangian{expr}
- `HamiltonianDef`: \\hamiltonian{expr}
- `TransformDef`: \\transform{var}{type}{expr}
- `ConstraintDef`: \\constraint{expr}
- `NonHolonomicConstraintDef`: \\nonholonomic{expr}
- `ForceDef`: \\force{expr}
- `DampingDef`: \\damping{expr}
- `InitialCondition`: \\initial{conditions}
- `FluidDef`: \\fluid{name} \\region{...}
- `BoundaryDef`: \\boundary{name} \\region{...}
- `RegionDef`: \\region{shape}{constraints}

---

## Physics Domains

### Domain Architecture

All physics domains inherit from `PhysicsDomain` base class (`domains/base.py`):

```python
class PhysicsDomain(ABC):
    @abstractmethod
    def define_lagrangian(self) -> sp.Expr: ...
    @abstractmethod
    def define_hamiltonian(self) -> sp.Expr: ...
    @abstractmethod
    def derive_equations_of_motion(self) -> Dict[str, sp.Expr]: ...
    @abstractmethod
    def get_state_variables(self) -> List[str]: ...
```

### Classical Mechanics (`domains/classical/`)

The most comprehensive domain with 17 modules.

#### Lagrangian Mechanics (`lagrangian.py`)
- `LagrangianMechanics` class
- Kinetic energy T = (1/2) * m * v²
- Potential energy V(r)
- L = T - V

#### Hamiltonian Mechanics (`hamiltonian.py`)
- `HamiltonianMechanics` class
- Legendre transformation
- H = T + V (for conservative systems)
- Hamilton's equations: q_dot = ∂H/∂p, p_dot = -∂H/∂q

#### Constraints (`constraints.py`)
- `ConstraintHandler`: Holonomic constraint management
- `BaumgarteStabilization`: Constraint stabilization method
- `ConstrainedLagrangianSystem`: Lagrange multiplier implementation

#### Rigid Body Dynamics (`rigidbody.py`)
- `RigidBodyDynamics`: General rigid body motion
- `EulerAngles`: Euler angle representation (φ, θ, ψ)
- `Quaternion`: Quaternion representation (w, x, y, z)
- `SymmetricTop`: Symmetric top dynamics
- `Gyroscope`: Gyroscopic motion

#### Dissipation (`dissipation.py`)
- `RayleighDissipation`: Rayleigh dissipation function
- `FrictionModel`: Friction models
  - `FrictionType`: Viscous, Coulomb, Stribeck
- `GeneralizedForce`: Non-conservative forces
- `DissipativeLagrangianMechanics`: Lagrangian with dissipation

#### Stability Analysis (`stability.py`)
- `StabilityAnalyzer`: Stability analysis tools
- `EquilibriumPoint`: Equilibrium point representation
- `StabilityType`: Stable, unstable, saddle, center
- `find_equilibria()`: Find equilibrium points
- `analyze_stability()`: Linearization and eigenvalue analysis

#### Symmetry & Conservation (`symmetry.py`)
- `NoetherAnalyzer`: Noether's theorem implementation
- `ConservedQuantity`: Energy, momentum, angular momentum
- `SymmetryType`: Translation, rotation, time translation
- `detect_cyclic_coordinates()`: Find cyclic coordinates
- `get_conserved_quantities()`: Extract conserved quantities

#### Central Forces (`central_forces.py`)
- `CentralForceAnalyzer`: Central force analysis
- `EffectivePotential`: Effective potential construction
- `KeplerProblem`: Kepler orbit mechanics
- `OrbitalElements`: Orbital element calculation
- `OrbitType`: Elliptic, parabolic, hyperbolic
- `TurningPoints`: Apsidal distances

#### Canonical Transformations (`canonical.py`)
- `CanonicalTransformation`: Canonical transformation framework
- `GeneratingFunction`: Generating function types
- `ActionAngleVariables`: Action-angle variable transformation
- `HamiltonJacobi`: Hamilton-Jacobi equation solving

#### Oscillations (`oscillations.py`)
- `NormalModeAnalyzer`: Normal mode analysis
- `NormalMode`: Normal mode representation
- Coupled oscillator systems
- Modal decomposition

#### Other Classical Modules
- **`perturbation.py`**: Perturbation theory
- **`collisions.py`**: Elastic/inelastic collisions
- **`scattering.py`**: Rutherford scattering, cross-sections
- **`nonholonomic.py`**: Non-holonomic constraints
- **`variable_mass.py`**: Rocket equations, variable mass
- **`continuum.py`**: Continuous systems (strings, membranes)

### Fluid Dynamics (`domains/fluids/`)

#### SPH Implementation (`sph.py`)
**SPHFluid Class:**
- Particle-based fluid simulation
- Kernels:
  - `kernel_poly6()`: Density kernel
  - `kernel_spiky_grad()`: Pressure gradient kernel
  - `kernel_viscosity_laplacian()`: Viscosity kernel
- Tait equation of state: P = k(ρ - ρ₀)
- Forces:
  - Pressure forces
  - Viscosity forces
  - Gravity
- Integration: Semi-implicit Euler

**Methods:**
- `add_particle()`: Add fluid or boundary particle
- `compute_density_pressure()`: SPH density and pressure calculation
- `compute_forces()`: Pressure, viscosity, gravity forces
- `step(dt)`: Advance simulation one timestep
- `get_positions()`: Extract particle positions

#### Boundary Conditions (`boundary.py`)
**BoundaryConditions Class:**
- Wall types: no-slip, free-slip, open
- `add_wall()`: Add wall boundary
- `enforce_box_boundary()`: Box boundary enforcement
- `enforce_periodic()`: Periodic boundary conditions
- `generate_boundary_particles()`: Generate boundary particles

### Other Physics Domains

#### Quantum Mechanics (`domains/quantum/`)
- Quantum tunneling
- Finite square well
- Hydrogen atom energy levels
- Quantum state analysis

#### Electromagnetic (`domains/electromagnetic/`)
- Charged particle motion
- Lorentz force
- Electromagnetic fields
- Cyclotron motion

#### Relativistic (`domains/relativistic/`)
- Special relativity
- Lorentz transformations
- Relativistic momentum/energy
- Four-vectors

#### General Relativity (`domains/general_relativity/`)
- Black hole metrics (Schwarzschild, Kerr)
- Geodesics
- Gravitational lensing
- Cosmology (FLRW metric)

#### Statistical Mechanics (`domains/statistical/`)
- Statistical ensembles
- Distribution functions
- Partition functions

#### Thermodynamics (`domains/thermodynamics/`)
- Heat engines (Carnot, Otto, Diesel)
- Equations of state
- Phase transitions

#### Kinematics (`domains/kinematics/`)
- 1D motion
- 2D motion
- Projectile motion
- Relative motion

---

## Code Generation System

### Architecture

All code generators inherit from `CodeGenerator` base class (`codegen/base.py`):

```python
class CodeGenerator(ABC):
    @abstractmethod
    def generate(self, output_file: str) -> str: ...
    @abstractmethod
    def generate_equations(self) -> str: ...
```

### Code Generation Targets

#### 1. C++ (`codegen/cpp.py`)
**CppGenerator Class:**
- Generates standard C++ code
- Multiple templates:
  - `solver_template.cpp`: Standard ODE solver
  - `sph_template.cpp`: SPH fluid simulation
  - `raylib_template.cpp`: Raylib visualization
  - `pybind_template.cpp`: Python extension
- Features:
  - RK4 integration
  - Optimized numerical functions
  - Parameter substitution
  - Initial condition setup

**Targets:**
- 'standard': Basic C++ solver
- 'raylib': Raylib visualization
- 'arduino': Arduino sketch
- 'wasm': WebAssembly (via Emscripten)
- 'openmp': OpenMP parallelization
- 'python': Python extension (pybind11)

#### 2. CUDA (`codegen/cuda.py`, `codegen/cuda_sph.py`)
**CudaGenerator Class:**
- GPU kernel generation
- RK4 integration on GPU
- CPU fallback support
- CMakeLists.txt generation
- SPH support via `CudaSPHGenerator`

#### 3. WebAssembly (`codegen/wasm.py`)
**WasmGenerator Class:**
- Emscripten-compatible C code
- HTML canvas visualization
- JavaScript interop functions
- Browser-based simulation

#### 4. Python (`codegen/python.py`)
**PythonGenerator Class:**
- Pure Python code generation
- NumPy/SciPy integration
- Standalone simulation scripts

#### 5. Julia (`codegen/julia.py`)
**JuliaGenerator Class:**
- Julia code generation
- DifferentialEquations.jl integration
- High-performance numerical methods

#### 6. Rust (`codegen/rust.py`)
**RustGenerator Class:**
- Rust code generation
- nalgebra integration
- ODE solver libraries

#### 7. MATLAB (`codegen/matlab.py`)
**MatlabGenerator Class:**
- MATLAB/Octave code generation
- ODE45 integration

#### 8. Fortran (`codegen/fortran.py`)
**FortranGenerator Class:**
- Modern Fortran code generation
- ODE solver integration

#### 9. JavaScript (`codegen/javascript.py`)
**JavaScriptGenerator Class:**
- Node.js code generation
- Numerical.js integration

#### 10. OpenMP (`codegen/openmp.py`)
**OpenMPGenerator Class:**
- OpenMP parallelization
- Multi-threaded simulation

#### 11. Arduino (`codegen/arduino.py`)
**ArduinoGenerator Class:**
- Arduino sketch generation
- Embedded system deployment

### Code Generation Process

1. **Template Loading:** Loads target-specific template
2. **Equation Conversion:** Converts SymPy expressions to target syntax
3. **Parameter Substitution:** Embeds parameter values
4. **Initial Conditions:** Generates initial condition setup
5. **File Writing:** Writes generated code to file
6. **Optional Compilation:** Compiles to binary (if requested)

---

## Visualization System

### Components (`visualization/`)

#### 1. Animator (`animator.py`)
**Animator Class:**
- Mechanical system animations
- Particle trajectory animations
- Fluid particle visualizations

**Key Methods:**
- `animate_pendulum()`: Pendulum animation
- `animate_particles()`: Particle system animation
- `animate_fluid()`: SPH fluid animation
- `setup_figure()`: Figure configuration
- `save_animation_to_file()`: Export animations

**Features:**
- Trail rendering
- Customizable FPS
- Matplotlib integration

#### 2. Plotter (`plotter.py`)
**Plotter Class:**
- Time series plots
- Trajectory plots
- Energy plots
- Multi-panel figures

**Key Methods:**
- `plot_time_series()`: Variables vs time
- `plot_trajectory_2d()`: 2D trajectory
- `plot_trajectory_3d()`: 3D trajectory
- `plot_energy()`: Energy vs time
- `plot_multi_panel()`: Multiple subplots

#### 3. Phase Space Visualizer (`phase_space.py`)
**PhaseSpaceVisualizer Class:**
- Phase portraits
- Poincaré sections
- 3D phase space

**Key Methods:**
- `plot_phase_portrait()`: 2D phase portrait (q vs q_dot)
- `plot_phase_portrait_3d()`: 3D phase space
- `plot_poincare_section()`: Poincaré section

#### 4. Legacy Visualizer (`visualization.py`)
**MechanicsVisualizer Class:**
- Backward-compatible wrapper
- Redirects to new modular components

---

## Utilities & Infrastructure

### Configuration (`utils/config.py`)

**Config Class:**
Centralized configuration with 50+ constants and settings.

**Physical Constants:**
- `STANDARD_GRAVITY = 9.80665` m/s²
- `SPEED_OF_LIGHT = 299792458.0` m/s
- `PLANCK_CONSTANT = 6.62607015e-34` J·s
- `HBAR = 1.054571817e-34` J·s

**Numerical Constants:**
- `DEFAULT_RTOL = 1e-6`: Relative tolerance
- `DEFAULT_ATOL = 1e-8`: Absolute tolerance
- `ENERGY_TOLERANCE = 0.01`: Energy conservation tolerance
- `DEFAULT_NUM_POINTS = 1000`: Default simulation points
- `SIMPLIFICATION_TIMEOUT = 5.0`: Symbolic simplification timeout
- `MAX_PARSER_ERRORS = 10`: Maximum parser errors
- `SINGULARITY_THRESHOLD = 1e-12`: Singularity detection threshold

**Visualization Constants:**
- `DEFAULT_TRAIL_LENGTH = 150`: Animation trail length
- `DEFAULT_FPS = 30`: Animation FPS
- `DEFAULT_DPI = 100`: Figure DPI
- Colors: PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, etc.

**Cache Settings:**
- `DEFAULT_CACHE_SIZE = 128`: LRU cache size
- `DEFAULT_CACHE_MEMORY_MB = 100.0`: Cache memory limit

**Config Properties:**
- `enable_profiling`: Performance profiling
- `enable_performance_monitoring`: Performance monitoring
- `enable_memory_monitoring`: Memory monitoring
- `cache_symbolic_results`: Symbolic result caching
- `enable_adaptive_solver`: Adaptive solver selection
- And 20+ more configuration options

### Logging (`utils/logging.py`)

- Centralized logging configuration
- Logger instance: `logger`
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `setup_logging()`: Configure logging

### Caching (`utils/caching.py`)

**LRUCache Class:**
- LRU (Least Recently Used) cache
- Memory limit support
- Size limit support
- Used for symbolic expression caching

### Profiling (`utils/profiling.py`)

**Performance Monitoring:**
- `_perf_monitor`: Global performance monitor
- Timer functions: `start_timer()`, `stop_timer()`, `get_stats()`
- Memory snapshots: `snapshot_memory()`
- `@profile_function` decorator

**Timeout Support:**
- `timeout()` context manager
- Cross-platform timeout (Windows/Unix)
- Used for symbolic simplification timeouts

### Validation (`utils/validation.py`)

**Input Validation Functions:**
- `validate_file_path()`: File path validation (security)
  - Null byte prevention
  - Path traversal prevention (`..`)
  - Special character filtering
  - Path length limits
- `validate_array_safe()`: Array validation
- `safe_float_conversion()`: Safe float conversion
- `safe_array_access()`: Safe array access
- `validate_finite()`: NaN/Infinity checks
- `validate_time_span()`: Time span validation

### Registry (`utils/registry.py`)

**Variable Type Classification:**
- `VariableCategory` enum: COORDINATE, VELOCITY, CONSTANT, PARAMETER, EXTERNAL_FIELD
- `COORDINATE_TYPES`: Frozen set of coordinate types ('Angle', 'Position', 'Coordinate', etc.)
- `CONSTANT_TYPES`: Frozen set of constant types ('Mass', 'Length', 'Spring Constant', etc.)
- `COMMON_COORDINATE_NAMES`: Common coordinate variable names
- `COMMON_CONSTANT_NAMES`: Common constant variable names

**Functions:**
- `is_coordinate_type()`: Check if type is coordinate
- `is_constant_type()`: Check if type is constant
- `is_likely_coordinate()`: Determine if variable is coordinate (uses type + name)

### Units (`utils/units.py`)

**UnitSystem Class:**
- Unit conversion
- Dimensional analysis
- Unit validation

---

## Testing Infrastructure

### Test Organization

#### Unit Tests (`tests/unit/`)
32 test files covering individual modules:
- `test_compiler.py`: PhysicsCompiler tests
- `test_parser.py`: Parser tests
- `test_symbolic.py`: SymbolicEngine tests
- `test_solver.py`: NumericalSimulator tests
- `test_codegen.py`: Code generation tests
- `test_visualization.py`: Visualization tests
- `test_utils.py`: Utilities tests
- Coverage tests, extended tests

#### Physics Tests (`tests/physics/`)
36 test files validating physics correctness:
- Classical mechanics: `test_hamiltonian_formulation.py`, `test_constrained_systems.py`, `test_rigidbody_quaternions.py`, `test_chaotic_systems.py`
- Quantum: `test_quantum.py`, `test_tunneling.py`, `test_quantum_advanced.py`
- Relativity: `test_relativistic.py`, `test_relativistic_advanced.py`, `test_general_relativity.py`
- Electromagnetic: `test_electromagnetic.py`, `test_em_advanced.py`
- Fluids: `test_fluids.py`
- And 25+ more specialized tests

#### Property Tests (`tests/property/`)
3 files using Hypothesis for property-based testing:
- `test_physics_invariants.py`: Conservation laws
- `test_physics_properties.py`: Physics properties

#### Performance Tests (`tests/performance/`)
2 files for performance benchmarking:
- `test_benchmarks.py`: Performance benchmarks
- `test_performance_stress.py`: Stress tests

#### Integration Tests (`tests/integration/`)
1 file for end-to-end testing:
- `test_integration_pipeline.py`: Full pipeline tests

#### Backend Tests (`tests/test_backends/`)
2 files for code generation:
- `test_all_backends.py`: All code generation backends
- Additional backend-specific tests

### Test Statistics
- **Total Test Functions:** 1569+
- **Test Files:** 74+
- **Coverage:** Comprehensive coverage across all modules
- **Test Markers:** @pytest.mark.slow, @pytest.mark.gpu_required, @pytest.mark.physics_validation, @pytest.mark.performance, @pytest.mark.integration, @pytest.mark.numba_required

### Test Fixtures (`conftest.py`)
- `compiler`: PhysicsCompiler fixture
- `simple_dsl`: Simple oscillator DSL fixture
- `pendulum_dsl`: Pendulum DSL fixture

---

## Examples & Demos

### Python Examples (`examples/`)

30+ example scripts organized by difficulty:

#### Beginner (7 files)
- `01_getting_started.py`: First simulation
- `02_harmonic_oscillator.py`: Simple harmonic motion
- `03_simple_pendulum.py`: Basic pendulum
- `04_plotting_basics.py`: Visualization basics
- `marble_from_balcony.py`: Projectile motion
- `marble_launcher_all_settings.py`: Projectile with settings
- `projectile_basics.py`: Basic projectile

#### Intermediate (5 files)
- `05_double_pendulum.py`: Chaotic double pendulum
- `06_coupled_oscillators.py`: Coupled springs
- `07_2d_motion.py`: 2D motion
- `08_damped_systems.py`: Damping
- `09_forced_oscillators.py`: Driven systems

#### Advanced (8 files)
- `10_3d_gyroscope.py`: 3D rigid body
- `11_spherical_pendulum.py`: 3D pendulum
- `12_constrained_systems.py`: Constraints
- `13_hamiltonian_formulation.py`: Hamiltonian mechanics
- `14_chaotic_systems.py`: Chaos
- `15_energy_analysis.py`: Energy conservation
- `16_phase_space.py`: Phase space
- `20_quaternion_rigid_body.py`: Quaternions

#### Tools (4 files)
- `17_custom_visualizations.py`: Advanced plotting
- `18_export_import.py`: Serialization
- `19_performance_tuning.py`: Optimization
- `20_units_and_dimensions.py`: Units

#### Code Generation (2 files)
- `21_c++_code_export.py`: C++ generation
- `22_advanced_targets.py`: Multiple targets

#### Fluids (4 files)
- `23_sph_introduction.py`: SPH basics
- `24_sph_wave_tank.py`: Waves
- `25_sph_droplet.py`: Droplet
- `26_sph_sloshing.py`: Sloshing

#### Celestial (4 files)
- `27_elastic_pendulum.py`: Spring pendulum
- `28_n_body_gravity.py`: N-body
- `29_orbital_mechanics.py`: Orbits
- `30_figure8_orbit.py`: Figure-8 orbit

### Jupyter Notebooks (`examples/notebooks/`)
30+ interactive notebooks mirroring Python examples, organized by category

### Language-Specific Demos (`demos/`)
Demonstrations for code generation targets:
- `cpp_pendulum/`: C++ demo
- `cuda_pendulum/`: CUDA demo
- `fortran_pendulum/`: Fortran demo
- `javascript_pendulum/`: JavaScript demo
- `julia_pendulum/`: Julia demo
- `matlab_pendulum/`: MATLAB demo
- `openmp_pendulum/`: OpenMP demo
- `python_pendulum/`: Python demo
- `rust_pendulum/`: Rust demo
- `wasm_pendulum/`: WebAssembly demo

Each demo includes:
- README with instructions
- Compilation scripts (`compile.sh`)
- Generated code examples

---

## Documentation System

### Documentation Structure (`docs/`)

60+ reStructuredText files organized into comprehensive Sphinx documentation.

#### Getting Started
- Installation instructions
- Quick start guide
- Tutorials

#### User Guide
- Complete user guide
- DSL syntax reference
- Code generation guide
- CUDA guide
- Performance optimization
- Physics background

#### API Reference
Comprehensive API documentation for:
- Core modules
- Code generation
- Physics domains
- Analysis tools
- Visualization
- Utilities
- I/O

#### Physics Documentation
Detailed physics documentation for all domains:
- Lagrangian mechanics
- Hamiltonian mechanics
- Constraints
- Rigid body dynamics
- Oscillations
- Central forces
- Collisions
- Scattering
- Dissipation
- Stability
- Symmetry
- Canonical transformations
- Perturbation theory
- Non-holonomic constraints
- Variable mass
- Continuum mechanics
- Fluid dynamics
- Quantum mechanics
- Electromagnetism
- Relativity
- Statistical mechanics
- Thermodynamics
- Kinematics

#### Advanced Topics
- Architecture details
- Advanced features
- Extending the framework
- Performance tuning

#### Project Documentation
- Changelog
- Contributing guide
- License

### Documentation Features
- Sphinx-generated HTML
- Read the Docs hosting
- Code examples
- Theory explanations
- API references
- Cross-references

---

## Build System & Configuration

### pyproject.toml

Modern Python build configuration (PEP 517/518).

**Project Metadata:**
- Name: `mechanicsdsl-core`
- Version: `1.5.1`
- Author: Noah Parsons
- License: MIT
- Python: `>=3.8`
- Description: "A Domain-Specific Language and Transpiler for Classical Mechanics"

**Dependencies:**
- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `sympy>=1.8`
- `matplotlib>=3.4.0`

**Optional Dependencies:**
- `test`: pytest, pytest-cov, hypothesis>=6.0
- `codegen`: pybind11
- `jit`: numba>=0.56.0
- `typing`: mypy, types-setuptools
- `dev`: Full development dependencies (pytest, black, flake8, isort, bandit, mypy)
- `all`: All optional dependencies

**Build System:**
- `setuptools>=61.0`
- `setuptools.build_meta`

**Tool Configuration:**
- **pytest:** Test configuration, markers, warnings
- **black:** Code formatting (line length 100)
- **isort:** Import sorting (black profile)
- **mypy:** Type checking (ignore_missing_imports=true)

### requirements.txt

Core runtime dependencies:
- sympy
- numpy
- scipy
- matplotlib
- numba (optional)
- pytest (development)
- black (development)
- flake8 (development)

### Setup

**Installation:**
```bash
pip install mechanicsdsl-core
```

**Development Installation:**
```bash
pip install -e ".[dev]"
```

**Build Distribution:**
```bash
python -m build
```

---

## Dependencies & Requirements

### Core Dependencies

1. **NumPy (>=1.20.0)**
   - Numerical arrays and operations
   - Used in numerical integration
   - Array manipulation

2. **SciPy (>=1.7.0)**
   - `scipy.integrate.solve_ivp`: ODE solvers
   - Numerical integration methods
   - Optimization (if needed)

3. **SymPy (>=1.8)**
   - Symbolic mathematics
   - Expression manipulation
   - Differentiation and integration
   - Equation solving

4. **Matplotlib (>=3.4.0)**
   - Plotting and visualization
   - Animations
   - Phase space plots

### Optional Dependencies

1. **Numba (>=0.56.0)**
   - JIT compilation
   - Performance acceleration
   - Optional solver backend

2. **Pybind11**
   - C++ code generation
   - Python extension generation

3. **Pytest (development)**
   - Testing framework
   - Test discovery and execution

4. **Black, Flake8, isort (development)**
   - Code formatting
   - Linting
   - Import sorting

5. **Hypothesis (development)**
   - Property-based testing

6. **MyPy (development)**
   - Static type checking

### System Requirements

- **Python:** 3.8, 3.9, 3.10, 3.11, 3.12
- **Operating System:** Windows, Linux, macOS
- **Memory:** Varies by system complexity
- **Disk Space:** ~50MB for installation

---

## Development Tools

### VS Code Extension (`vscode-extension/`)

Syntax highlighting and snippets for MechanicsDSL:
- `language-configuration.json`: Language configuration
- `syntaxes/*.json`: Syntax highlighting rules
- `snippets/*.json`: Code snippets (15+ snippets for common systems)
- `package.json`: Extension metadata

### Pre-commit Hooks

Enhanced code quality automation:
- `isort`: Import sorting
- `bandit`: Security vulnerability scanning
- `docformatter`: Docstring formatting
- Additional hooks: check-ast, check-docstring-first

### CI/CD Pipeline

Automated testing and quality checks:
- Python CI (multiple versions)
- MyPy type checking (all Python versions)
- Benchmark tests (pytest-benchmark)
- Lint job (black, isort, flake8)
- Security scanning (bandit)
- Code coverage (codecov)

### Binder Support (`binder/`)

Jupyter Binder environment:
- `environment.yml`: Conda environment
- JupyterLab support
- Flexible Python version (3.9-3.12)
- Additional packages (ipywidgets, hypothesis)

### Conda Recipe (`conda-recipe/`)

Conda package distribution:
- `meta.yaml`: Package metadata
- Version 1.5.0
- Expanded test suite
- Feature description

---

## Version History & Changelog

### Current Version: 1.5.1 (2026-01-11)

**Major Additions:**
- Variable Registry Module (`utils/registry.py`)
- Solver Package Migration (modular `solver/` package)
- Protocol-based Typing (`protocols.py`)
- Actionable Exception Classes (14 new exception types)
- Expanded Constants (50+ magic numbers moved to config)

**Security Enhancements:**
- Pickle security warnings
- Enhanced file path validation
- Null byte injection prevention
- Path traversal detection

**Developer Experience:**
- VS Code Extension enhancements (15+ snippets)
- Pre-commit hooks (isort, bandit, docformatter)
- CI/CD pipeline improvements
- Pytest markers for test categorization

### Version 1.4.0 (2026-01-04)

**Major Additions:**
- 9 New Physics Domains (quantum, electromagnetic, relativistic, statistical, thermodynamics)
- 1,465+ unit tests
- VS Code Extension
- Conda Recipe
- Binder support
- MyPy CI

### Version 1.2.3 (2024-12-06)

**Added:**
- `HamiltonJacobi.solve()` method

### Version 1.2.2 (2024-12-06)

**Major Additions:**
- 13 New Classical Mechanics Modules
- 5 New Code Generators (Julia, Rust, MATLAB, Fortran, JavaScript)
- 30+ Interactive Notebooks
- Comprehensive documentation

### Version 1.2.0 (2025-12-31)

**Major Restructure:**
- Modular package structure (`utils/`, `domains/`, `visualization/`, `analysis/`, `codegen/`, `io/`)
- Complete documentation rewrite
- Enhanced backward compatibility

### Version 1.1.0 (2024-09-01)

**Added:**
- SPH fluid dynamics
- WebAssembly code generation
- Hamiltonian formulation
- Non-holonomic constraints
- Performance monitoring

### Version 1.0.0 (2024-06-15)

**Initial Release:**
- LaTeX-inspired DSL
- Lagrangian mechanics
- Automatic Euler-Lagrange derivation
- Numerical simulation
- 3D visualization
- C++ code generation
- Energy analysis
- Phase space visualization

See `CHANGELOG.md` for complete version history.

---

## Performance Characteristics

### Symbolic Computation
- **Caching:** LRU cache for expression conversion (default: 256 entries, 200MB)
- **Simplification:** Timeout-protected symbolic simplification (default: 5s)
- **Common Subexpression Elimination:** Automatic CSE in SymPy

### Numerical Simulation
- **Adaptive Solvers:** Automatic selection of RK45 (non-stiff) or LSODA (stiff)
- **Stiffness Detection:** Automatic detection and solver switching
- **Time Step Adaptation:** Adaptive time stepping for accuracy and efficiency

### Memory Management
- **Memory Monitoring:** Optional memory snapshot tracking
- **Garbage Collection:** Configurable GC thresholds
- **Cache Limits:** Memory and size limits on caches

### Code Generation
- **Template-Based:** Efficient template-based code generation
- **Optimization:** Target-specific optimizations (e.g., CUDA kernels)

### Benchmarks

Benchmark files in `benchmarks/`:
- `comparison_matrix.py`: Performance comparison
- `cuda_performance.py`: CUDA benchmarks
- `numba_performance.py`: Numba benchmarks

---

## Security Features

### Input Validation
- **Type Checking:** Comprehensive type validation
- **Size Limits:** 1MB limit on DSL source code
- **File Path Validation:** Null byte prevention, path traversal detection, special character filtering
- **Safe Conversions:** Safe float and array conversions

### Code Execution Safety
- **No eval():** AST-based parsing prevents code injection
- **No exec():** No dynamic code execution
- **Pattern Detection:** Warnings for dangerous patterns (`__import__`, `eval()`, `exec()`, `compile()`)

### Serialization Security
- **Pickle Warnings:** Explicit warnings about pickle security risks
- **JSON Preference:** Recommends JSON over pickle for untrusted data
- **File Validation:** File size and path validation

### Error Handling
- **Actionable Errors:** Exception classes with suggestions and documentation links
- **Error Recovery:** Parser error recovery
- **Validation Failures:** Clear validation error messages

---

## API Reference Summary

### Core API (`mechanics_dsl/__init__.py`)

**Primary Classes:**
- `PhysicsCompiler`: Main compiler class
- `MechanicsParser`: Parser class
- `SymbolicEngine`: Symbolic computation engine
- `NumericalSimulator`: Numerical solver

**Utility Functions:**
- `tokenize()`: Tokenization function
- `setup_logging()`: Logging configuration

**Analysis:**
- `PotentialEnergyCalculator`: Energy analysis

**Exceptions:**
- `MechanicsDSLError`: Base exception
- `ParseError`: Parsing errors
- `TokenizationError`: Tokenization errors
- `SemanticError`: Semantic errors
- `NoLagrangianError`: Missing Lagrangian
- `NoCoordinatesError`: Missing coordinates
- `SimulationError`: Simulation errors
- `IntegrationError`: Integration errors
- `InitialConditionError`: Initial condition errors
- `ParameterError`: Parameter errors

### Parser API (`parser/`)

**Classes:**
- `MechanicsParser`: Main parser
- `Token`: Token class
- AST node classes (33+ classes)

**Functions:**
- `tokenize()`: Tokenization

### Solver API (`solver/`)

**Classes:**
- `NumericalSimulator`: Numerical simulation engine

### Code Generation API (`codegen/`)

**Base Class:**
- `CodeGenerator`: Abstract base class

**Generators:**
- `CppGenerator`: C++ code generation
- `CudaGenerator`: CUDA code generation
- `WasmGenerator`: WebAssembly generation
- `PythonGenerator`: Python code generation
- `JuliaGenerator`: Julia generation
- `RustGenerator`: Rust generation
- `MatlabGenerator`: MATLAB generation
- `FortranGenerator`: Fortran generation
- `JavaScriptGenerator`: JavaScript generation
- `OpenMPGenerator`: OpenMP generation
- `ArduinoGenerator`: Arduino generation

### Visualization API (`visualization/`)

**Classes:**
- `Animator`: Animation handler
- `Plotter`: Plotting utilities
- `PhaseSpaceVisualizer`: Phase space visualization
- `MechanicsVisualizer`: Legacy visualizer (backward compatibility)

### Physics Domains API (`domains/`)

**Base Class:**
- `PhysicsDomain`: Abstract base class

**Classical Mechanics (`domains/classical/`):**
- `LagrangianMechanics`
- `HamiltonianMechanics`
- `ConstraintHandler`
- `RigidBodyDynamics`
- `StabilityAnalyzer`
- `NoetherAnalyzer`
- `CentralForceAnalyzer`
- `CanonicalTransformation`
- `NormalModeAnalyzer`
- And 8+ more classes

**Other Domains:**
- `SPHFluid` (fluids)
- `ChargedParticle` (electromagnetic)
- `RelativisticParticle` (relativistic)
- `QuantumState` (quantum)
- And more domain-specific classes

### Utilities API (`utils/`)

**Classes:**
- `Config`: Configuration management
- `LRUCache`: LRU cache implementation
- `UnitSystem`: Unit handling

**Functions:**
- `setup_logging()`: Logging setup
- `profile_function()`: Profiling decorator
- `timeout()`: Timeout context manager
- Validation functions: `validate_file_path()`, `validate_array_safe()`, etc.
- Registry functions: `is_likely_coordinate()`, `is_coordinate_type()`, etc.

---

## Future Extensibility

### Architecture Extensibility

1. **New Physics Domains:**
   - Inherit from `PhysicsDomain`
   - Implement required methods
   - Add to `domains/` package

2. **New Code Generation Targets:**
   - Inherit from `CodeGenerator`
   - Implement `generate()` and `generate_equations()`
   - Add templates if needed

3. **New Visualization Types:**
   - Add to `visualization/` package
   - Follow existing patterns

4. **New AST Node Types:**
   - Add to `parser/ast_nodes.py`
   - Update parser in `parser/core.py`
   - Update symbolic engine in `symbolic.py`

### Configuration Extensibility

- Add new constants to `utils/config.py`
- Add new configuration properties to `Config` class

### Parser Extensibility

- Add new token types to `parser/tokens.py`
- Add new command parsing to `parser/core.py`

### Testing Extensibility

- Add new test categories
- Add new fixtures to `conftest.py`
- Add new test markers

---

## Conclusion

MechanicsDSL is a comprehensive, well-architected framework for computational physics that successfully bridges symbolic mathematics and numerical simulation. The codebase demonstrates:

1. **Enterprise-Grade Design:** Modular architecture, comprehensive error handling, security hardening
2. **Extensive Physics Coverage:** Classical mechanics, quantum mechanics, relativity, fluids, and more
3. **Multiple Code Generation Targets:** 11+ target languages and platforms
4. **Comprehensive Testing:** 1569+ tests across 74+ test files
5. **Rich Documentation:** 60+ documentation files with theory and examples
6. **Performance Optimization:** Caching, adaptive solvers, profiling
7. **Developer Experience:** VS Code extension, pre-commit hooks, CI/CD
8. **Educational Focus:** Designed for physics education with intuitive syntax

The framework is production-ready, well-tested, and actively maintained, making it suitable for both educational use and research applications.

---

**Document Generated:** 2025-01-XX  
**Codebase Version:** 1.5.1  
**Analysis Completeness:** Comprehensive - All major components documented
