"""
Tutorial 24: SPH Wave Tank

Simulating waves in a tank using Smoothed Particle Hydrodynamics.

Physics:
- Water in a rectangular tank
- Wave generation via moving boundary or initial conditions
- Wave propagation and reflection
- SPH kernel interpolation
"""

import os
import subprocess
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define wave tank system
# ============================================================================

print("="*60)
print("SPH WAVE TANK SIMULATION")
print("="*60)

dsl_code = r"""
\system{wave_tank}

% Simulation resolution
\parameter{h}{0.03}{m}
\parameter{g}{9.81}{m/s^2}

% Initial water (calm pool)
\fluid{water}
\region{rectangle}{x=0.0 .. 2.0, y=0.0 .. 0.3}
\particle_mass{0.015}
\equation_of_state{tait}

% Tank boundaries
\boundary{tank}
\region{line}{x=-0.05, y=0.0 .. 0.6}     % Left wall
\region{line}{x=2.05, y=0.0 .. 0.6}      % Right wall
\region{line}{x=-0.05 .. 2.05, y=-0.05}  % Floor
"""

print("1. Initializing compiler...")
compiler = PhysicsCompiler()

print("2. Generating particles...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    print("\n[WARN]  Note: SPH requires full setup. See Tutorial 23 first.")
    exit(1)

print("[OK] Particle system generated!")

# ============================================================================
# Generate C++ code (optional - requires compiler)
# ============================================================================

try:
    print("3. Generating C++ SPH engine...")
    compiler.compile_to_cpp("wave_tank.cpp", target="standard", compile_binary=True)
    
    print("4. Running simulation...")
    if os.name == 'nt':
        subprocess.call(["wave_tank.exe"])
    else:
        subprocess.call(["./wave_tank"])
    
    print("5. Visualizing results...")
    compiler.visualizer.animate_fluid_from_csv("wave_tank_sph.csv")
    plt.show()
    
except Exception as e:
    print(f"\n[WARN]  C++ compilation skipped: {e}")
    print("   This is normal if you don't have a C++ compiler installed.")
    print("   The DSL parsing and particle generation still work!")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. SPH discretizes fluid into particles")
print("2. Each particle carries mass, velocity, density")
print("3. Smoothing length h controls resolution")
print("4. Tait equation of state: pressure from density")
print("5. Boundary particles prevent fluid escape")
print("6. Wave propagation emerges from particle interactions")
print("="*60)


