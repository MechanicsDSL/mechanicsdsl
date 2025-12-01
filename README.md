# MechanicsDSL

[![Python CI](https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml/badge.svg)](https://github.com/MechanicsDSL/mechanicsdsl/actions/workflows/python-app.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17771040.svg)](https://doi.org/10.5281/zenodo.17771040)
[![Documentation Status](https://readthedocs.org/projects/mechanicsdsl/badge/?version=latest)](https://mechanicsdsl.readthedocs.io/en/latest/?badge=latest)

A Domain-Specific Language for Classical Mechanics - A comprehensive framework for symbolic and numerical analysis of classical mechanical systems using LaTeX-inspired notation.

## Features

- **Symbolic Computation**: Automatic derivation of equations of motion from Lagrangians and Hamiltonians
- **Numerical Simulation**: Advanced ODE solvers with adaptive step sizing
- **Visualization**: Interactive animations and phase space plots
- **Unit System**: Comprehensive dimensional analysis and unit checking
- **Constraint Handling**: Support for holonomic and non-holonomic constraints
- **Performance Monitoring**: Built-in profiling and optimization tools

## Installation
 
```bash
pip install mechanicsdsl-core
```

## Quick Start (In GitHub Codespaces)

```python
pip install mechanicsdsl-core

# Save as a .py file (e.g., demo.py)
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode for Codespaces
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# 1. Define Figure-8 System
# ============================================================================
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

\initial{
    x1=0, y1=0, x1_dot=0, y1_dot=0,
    x2=0, y2=0, x2_dot=0, y2_dot=0,
    x3=0, y3=0, x3_dot=0, y3_dot=0
}
"""

print("Initializing compiler...")
compiler = PhysicsCompiler()

print("Compiling DSL...")
result = compiler.compile_dsl(figure8_code)

if not result['success']:
    print(f"Compilation failed: {result.get('error')}")
    exit(1)

# ============================================================================
# 2. Initial Conditions
# ============================================================================
print("Chenciner & Montgomery initial conditions...")
compiler.simulator.set_initial_conditions({
    'x1': 0.97000436,  'y1': -0.24308753, 'x1_dot': 0.4662036850, 'y1_dot': 0.4323657300,
    'x2': -0.97000436, 'y2': 0.24308753,  'x2_dot': 0.4662036850, 'y2_dot': 0.4323657300,
    'x3': 0.0,         'y3': 0.0,         'x3_dot': -0.93240737,  'y3_dot': -0.86473146
})

# ============================================================================
# 3. Simulate and Check Periodicity
# ============================================================================
# Exact period T for figure-8
T_period = 6.32591398
print(f"Simulating for exactly one period T={T_period:.6f}...")

# Letting the solver adapt (likely selects LSODA)
solution = compiler.simulate(t_span=(0, T_period), num_points=2000)

if not solution['success']:
    print("Simulation failed")
    exit(1)

# --- PERIODICITY CHECK ---
y = solution['y']
state_initial = y[:, 0]
state_final = y[:, -1]

# Calculate Euclidean distance between start and end state
periodicity_error = np.linalg.norm(state_final - state_initial)

print("\n" + "="*50)
print("ANALYSIS RESULTS")
print("="*50)
print(f"Solver Used:       {solution.get('method_used', 'Adaptive')}")
print(f"Periodicity Error: {periodicity_error:.6e}")
print(f"Status:            {'CLOSED ORBIT' if periodicity_error < 1e-2 else 'DRIFT DETECTED'}")
print("="*50 + "\n")

# ============================================================================
# 4. Visualization
# ============================================================================
print("Plotting trajectories...")
coords = solution['coordinates']

def get_traj(name):
    idx = coords.index(name)
    return y[2*idx]

x1, y1 = get_traj('x1'), get_traj('y1')
x2, y2 = get_traj('x2'), get_traj('y2')
x3, y3 = get_traj('x3'), get_traj('y3')

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label='Body 1', color='#E63946', linewidth=2)
plt.plot(x2, y2, label='Body 2', color='#457B9D', linewidth=2, linestyle='--')
plt.plot(x3, y3, label='Body 3', color='#1D3557', linewidth=2, linestyle=':')

# Annotate the error on the plot
plt.title(f'Three-Body Figure-8 Orbit (Error: {periodicity_error:.2e})')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.savefig('figure8_periodicity.png', dpi=150)
print("Saved plot to 'figure8_periodicity.png'")

# Then run python name_of_file.py (e.g., python demo.py)
```

## Validation Gallery

MechanicsDSL has been rigorously tested against analytical solutions, chaotic systems, and conservation laws.

| **Coupled Modes** | **3D Dynamics** | **Complex Motion** |
|:---:|:---:|:---:|
| ![Wilberforce](docs/images/gallery_chaos.png)<br>_Wilberforce Pendulum Beats_ | ![Gyroscope](docs/images/gallery_gyroscope.png)<br>_Gyroscope Precession_ | ![Elastic](docs/images/gallery_constraint.png)<br>_Elastic Pendulum Trajectory_ |

| **Strange Attractors** | **Phase Space** | **Conservation** |
|:---:|:---:|:---:|
| ![Duffing](docs/images/gallery_duffing.png)<br>_Duffing Chaotic Attractor_ | ![Phase](docs/images/gallery_phase.png)<br>_Harmonic Portraits_ | ![Energy](docs/images/gallery_energy.png)<br>_Monotonic Energy Dissipation_ |

## Documentation

Full documentation is available at [https://mechanicsdsl.readthedocs.io/en/latest/index.html](https://mechanicsdsl.readthedocs.io/en/latest/index.html)

## License

MIT License - see LICENSE file for details.
