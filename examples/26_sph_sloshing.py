"""
Tutorial 26: SPH Sloshing Tank

Simulating liquid sloshing in a moving container.

Physics:
- Liquid in a tank
- Tank undergoes periodic motion
- Free surface waves develop
- Resonance when forcing matches natural frequency
"""

import os
import subprocess
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define sloshing tank system
# ============================================================================

print("="*60)
print("SPH SLOSHING TANK SIMULATION")
print("="*60)

dsl_code = r"""
\system{sloshing_tank}

% Simulation parameters
\parameter{h}{0.025}{m}
\parameter{g}{9.81}{m/s^2}

% Liquid in tank (partially filled)
\fluid{liquid}
\region{rectangle}{x=0.0 .. 0.8, y=0.0 .. 0.3}
\particle_mass{0.012}
\equation_of_state{tait}

% Tank walls (will oscillate in full simulation)
\boundary{tank}
\region{line}{x=-0.05, y=0.0 .. 0.6}    % Left wall
\region{line}{x=0.85, y=0.0 .. 0.6}     % Right wall
\region{line}{x=-0.05 .. 0.85, y=-0.05} % Floor
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
# Calculate natural frequency
# ============================================================================

import numpy as np

L = 0.8  # Tank length
h_water = 0.3  # Water depth
g = 9.81

# First mode natural frequency: omega = sqrt(g*pi/L * tanh(pi*h/L))
omega_n = np.sqrt(g * np.pi / L * np.tanh(np.pi * h_water / L))
f_n = omega_n / (2 * np.pi)
T_n = 1 / f_n

print(f"\nNatural sloshing frequency:")
print(f"   omega_n = {omega_n:.3f} rad/s")
print(f"   f_n = {f_n:.3f} Hz")
print(f"   T_n = {T_n:.3f} s")

# ============================================================================
# Generate and run simulation
# ============================================================================

try:
    print("\n3. Generating C++ SPH engine...")
    compiler.compile_to_cpp("sloshing_tank.cpp", target="standard", compile_binary=True)
    
    print("4. Running simulation...")
    if os.name == 'nt':
        subprocess.call(["sloshing_tank.exe"])
    else:
        subprocess.call(["./sloshing_tank"])
    
    print("5. Visualizing sloshing motion...")
    compiler.visualizer.animate_fluid_from_csv("sloshing_tank_sph.csv")
    plt.show()
    
except Exception as e:
    print(f"\n[WARN]  C++ compilation skipped: {e}")
    print("   This is normal if you don't have a C++ compiler installed.")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Sloshing is critical for ship/truck tank design")
print("2. Natural frequency depends on tank geometry & fill level")
print("3. Resonance causes large amplitude waves")
print("4. Baffles reduce sloshing (not shown here)")
print("5. SPH captures free-surface breaking waves")
print("6. Important for rocket propellant tanks!")
print("="*60)


