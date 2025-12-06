Changelog
=========

All notable changes to MechanicsDSL are documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[1.2.1] - 2024-12-05
--------------------

Fixed
~~~~~

- Fixed ``AttributeError: 'Config' object has no attribute 'default_trail_length'``
  in animator module
- Fixed Unicode encoding issues on Windows console (emojis replaced with ASCII)
- Fixed DSL parser handling of negative parameter values

Changed
~~~~~~~

- Examples now use ASCII characters for Windows compatibility
- Updated examples README with 30 tutorials in organized categories

[1.2.0] - 2024-XX-XX
--------------------

Added
~~~~~

**New Package Structure**

- **mechanics_dsl.utils/** - Modular utilities package
  
  - ``logging.py`` - Centralized logging configuration
  - ``config.py`` - Configuration management and constants
  - ``caching.py`` - LRU cache implementation
  - ``profiling.py`` - Performance monitoring and timeouts
  - ``validation.py`` - Input validation functions
  - ``units.py`` - Physical units system

- **mechanics_dsl.core/** - Core compiler infrastructure
  
  - Reorganized ``compiler.py``, ``parser.py``, ``symbolic.py``, ``solver.py``
  - Updated imports for new package structure

- **mechanics_dsl.domains/** - Physics domain implementations
  
  - ``base.py`` - Abstract ``PhysicsDomain`` class
  - ``classical/`` - Lagrangian, Hamiltonian, constraints, rigid body
  - ``fluids/`` - SPH simulation, boundary conditions

- **mechanics_dsl.visualization/** - Modular visualization
  
  - ``animator.py`` - Animation classes
  - ``plotter.py`` - General plotting utilities
  - ``phase_space.py`` - Phase space and Poincaré sections

- **mechanics_dsl.analysis/** - Analysis tools
  
  - ``energy.py`` - Energy conservation analysis
  - ``stability.py`` - Fixed point and eigenvalue analysis

- **mechanics_dsl.codegen/** - Code generation backends
  
  - ``base.py`` - Abstract ``CodeGenerator`` class
  - ``python.py`` - Python/NumPy code generation
  - Improved C++ templates

- **mechanics_dsl.io/** - I/O utilities
  
  - ``serialization.py`` - JSON/pickle serialization
  - ``export.py`` - CSV and JSON exporters

**Documentation**

- Complete rewrite with extensive API reference
- 30 detailed tutorials covering various physics scenarios
- DSL syntax reference with all commands
- Advanced topics guide (performance, extending, architecture)
- Code generation documentation (C++, Python, WASM, CUDA)

Changed
~~~~~~~

- Reorganized package structure for better modularity
- Improved backward compatibility with legacy imports
- Enhanced error messages throughout

Fixed
~~~~~

- Various import path issues
- Profiling conflicts with nested decorators

Deprecated
~~~~~~~~~~

- Direct imports from ``mechanics_dsl.utils`` (monolithic) - use submodules
- ``MechanicsVisualizer`` - use modular ``Animator``, ``Plotter``, ``PhaseSpaceVisualizer``

[1.1.0] - 2024-XX-XX
--------------------

Added
~~~~~

- SPH fluid dynamics support
- WebAssembly code generation
- Hamiltonian formulation support
- Non-holonomic constraints
- Performance monitoring utilities

Changed
~~~~~~~

- Improved symbolic simplification
- Better stiffness detection
- Enhanced visualization

[1.0.0] - 2024-XX-XX
--------------------

Added
~~~~~

- Initial release
- LaTeX-inspired DSL for classical mechanics
- Lagrangian mechanics support
- Automatic Euler-Lagrange equation derivation
- Numerical simulation with SciPy
- 3D visualization with Matplotlib
- C++ code generation
- Energy conservation analysis
- Phase space visualization

Features
~~~~~~~~

- Simple pendulum, double pendulum examples
- Harmonic oscillator support
- Basic constraint handling
- CSV export
