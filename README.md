MechanicsDSL is an open-source domain-specific language (DSL) and compiler for classical mechanics. It converts LaTeX-inspired model descriptions into symbolic equations (Lagrangian & Hamiltonian), numerical solvers, and animated visualizations — with production-focused safety, units, and testing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-prototype%20%2F%20research-orange)](https://github.com/MechanicsDSL/mechanics-dsl)

Quick highlights
- DSL → AST → SymPy → NumPy pipeline
- Automatic Euler–Lagrange & Hamiltonian derivation
- Safe AST-based unit parsing (no eval)
- SciPy numerical integration with diagnostic checks
- 2D/3D visualization and animation export (MP4/GIF)
- Built-in validation: energy conservation & analytic checks
- Designed for research, teaching, and reproducible simulation

Getting started (local)
1. Clone the repo:
   ```bash
   git clone https://github.com/<org>/mechanicsdsl.git
   cd mechanicsdsl
   ```
2. Create a virtualenv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install -r requirements-dev.txt
   ```
3. Run tests:
   ```bash
   pytest
   ```

Quick example (Python)
```python
from mechanicsdsl import run_example
out = run_example('simple_pendulum', t_span=(0, 10), show_animation=False)
print(out['result'])
```

Contributing

We welcome contributions. See CONTRIBUTING.md and CODE_OF_CONDUCT.md for guidance.

License

This project is MIT licensed. See LICENSE.
