"""
Tutorial 25: SPH Droplet Impact

Simulating a water droplet falling and splashing on a surface.

Physics:
- Spherical droplet falls under gravity
- Impact creates splash dynamics
- Surface tension effects (simplified)
- Particle fragmentation during splash
"""

import os
import subprocess
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define droplet impact system
# ============================================================================

print("="*60)
print("SPH DROPLET IMPACT SIMULATION")
print("="*60)

dsl_code = r"""
\system{droplet_impact}

% High resolution for droplet details
\parameter{h}{0.02}{m}
\parameter{g}{9.81}{m/s^2}

% Falling droplet (circular region)
\fluid{droplet}
\region{circle}{center=(0.5, 0.8), radius=0.1}
\particle_mass{0.01}
\equation_of_state{tait}

% Pool of water at bottom
\fluid{pool}
\region{rectangle}{x=0.0 .. 1.0, y=0.0 .. 0.15}
\particle_mass{0.01}
\equation_of_state{tait}

% Container boundaries
\boundary{container}
\region{line}{x=-0.05, y=0.0 .. 1.2}    % Left wall
\region{line}{x=1.05, y=0.0 .. 1.2}     % Right wall
\region{line}{x=-0.05 .. 1.05, y=-0.05} % Floor
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
# Generate and run simulation
# ============================================================================

try:
    print("3. Generating C++ SPH engine...")
    compiler.compile_to_cpp("droplet_impact.cpp", target="standard", compile_binary=True)
    
    print("4. Running simulation (may take a moment)...")
    if os.name == 'nt':
        subprocess.call(["droplet_impact.exe"])
    else:
        subprocess.call(["./droplet_impact"])
    
    print("5. Visualizing splash dynamics...")
    compiler.visualizer.animate_fluid_from_csv("droplet_impact_sph.csv")
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
print("1. Droplet splash is a complex free-surface flow")
print("2. SPH naturally handles topology changes (splash, merge)")
print("3. Higher resolution (smaller h) captures more detail")
print("4. Surface tension can be added via cohesion forces")
print("5. Crown splash forms from momentum transfer")
print("6. Great test case for multiphase SPH")
print("="*60)


