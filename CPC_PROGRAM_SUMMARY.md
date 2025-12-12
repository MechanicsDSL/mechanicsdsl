# CPC Program Summary for MechanicsDSL

## PROGRAM SUMMARY

**Program Title:** MechanicsDSL

**CPC Library link to program files:** (to be assigned)

**Developer's repository link:** https://github.com/MechanicsDSL/mechanicsdsl

**Licensing provisions:** MIT License

**Programming language:** Python 3.8+

**Nature of problem:**
Classical mechanical systems require deriving equations of motion from Lagrangian or Hamiltonian principles followed by numerical integration. This process is traditionally manual and error-prone, requiring expertise in both analytical mechanics and numerical methods.

**Solution method:**
MechanicsDSL provides a domain-specific language with LaTeX-inspired syntax for defining mechanical systems. A multi-stage compiler automatically derives equations of motion via symbolic computation (SymPy) and generates optimized numerical solvers (SciPy). The framework implements Euler-Lagrange and Hamilton's equations with support for constraints, dissipation, and 17 specialized physics domains.

**Additional comments including restrictions and unusual features:**
- Supports code generation to 11 target platforms (C++, CUDA, WebAssembly, Julia, Rust, etc.)
- Includes SPH fluid dynamics module
- Automatic adaptive solver selection based on system characteristics
- No programming knowledge required for basic usage

## Running Time

- Simple pendulum compilation: <0.5s
- Double pendulum 10s simulation: ~0.8s
- Figure-8 three-body 100 periods: ~15s

## Dependencies

- Python 3.8+
- NumPy ≥1.20
- SciPy ≥1.7
- SymPy ≥1.9
- Matplotlib ≥3.4 (visualization)

## Installation

```bash
pip install mechanicsdsl-core
```

## Quick Start

```python
from mechanics_dsl import PhysicsCompiler

code = r"""
\system{pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1.0}{kg}
\parameter{l}{1.0}{m}
\parameter{g}{9.81}{m/s^2}
\lagrangian{0.5*m*l^2*\dot{theta}^2 + m*g*l*cos(theta)}
"""

compiler = PhysicsCompiler()
compiler.compile_dsl(code)
compiler.simulator.set_initial_conditions({'theta': 0.5, 'theta_dot': 0})
solution = compiler.simulate(t_span=(0, 10))
```
