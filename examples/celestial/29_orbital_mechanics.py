"""
Tutorial 29: Orbital Mechanics

Classical Kepler problem and orbital transfers.

Physics:
- Central force: F = -GMm/r²
- Conic section orbits (ellipse, parabola, hyperbola)
- Conservation of energy and angular momentum
- Kepler's laws
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define Kepler problem in Cartesian coordinates
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{kepler_orbit}

\defvar{x}{Horizontal position}{m}
\defvar{y}{Vertical position}{m}

\parameter{m}{1.0}{kg}
\parameter{M}{1000.0}{kg}
\parameter{G}{1.0}{N*m^2/kg^2}

\lagrangian{
    \frac{1}{2} * m * (\dot{x}^2 + \dot{y}^2)
    + G * M * m / \sqrt{x^2 + y^2}
}

\initial{x=10.0, y=0.0, x_dot=0.0, y_dot=8.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling Kepler orbit...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    exit(1)

print("[OK] Compilation successful!")

solution = compiler.simulate(t_span=(0, 100), num_points=5000)

if not solution['success']:
    print("[FAIL] Simulation failed!")
    exit(1)

print("[OK] Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
x = solution['y'][0]
x_dot = solution['y'][1]
y = solution['y'][2]
y_dot = solution['y'][3]

r = np.sqrt(x**2 + y**2)

# ============================================================================
# Analyze orbital elements
# ============================================================================

m, M, G = 1.0, 1000.0, 1.0
mu = G * M  # Standard gravitational parameter

# Energy and angular momentum
E = 0.5 * m * (x_dot**2 + y_dot**2) - G * M * m / r
L = m * (x * y_dot - y * x_dot)  # Angular momentum (z-component)

# Semi-major axis: a = -mu*m / (2*E)
a = -G * M * m / (2 * E[0])

# Eccentricity from energy and angular momentum
e = np.sqrt(1 + 2 * E[0] * L[0]**2 / (m * (G * M * m)**2))

# Orbital period (Kepler's 3rd law): T = 2pi sqrt(a³/μ)
T = 2 * np.pi * np.sqrt(a**3 / mu)

print(f"\nOrbital Elements:")
print(f"   Semi-major axis: a = {a:.4f} m")
print(f"   Eccentricity:    e = {e:.4f}")
print(f"   Orbital period:  T = {T:.4f} s")
print(f"   Energy:          E = {E[0]:.4f} J")
print(f"   Angular momentum: L = {L[0]:.4f} kg·m²/s")

# ============================================================================
# Plot orbit
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Orbit trajectory
axes[0, 0].plot(x, y, 'b-', linewidth=1.5)
axes[0, 0].plot(0, 0, 'yo', markersize=20, label='Central body')
axes[0, 0].plot(x[0], y[0], 'go', markersize=10, label='Start')
axes[0, 0].set_xlabel('x (m)', fontsize=12)
axes[0, 0].set_ylabel('y (m)', fontsize=12)
axes[0, 0].set_title(f'Kepler Orbit (e = {e:.3f})', fontsize=14)
axes[0, 0].set_aspect('equal')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Distance vs time
axes[0, 1].plot(t, r, 'b-', linewidth=1.5)
axes[0, 1].set_xlabel('Time (s)', fontsize=12)
axes[0, 1].set_ylabel('Distance r (m)', fontsize=12)
axes[0, 1].set_title('Orbital Distance vs Time', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# Energy conservation
axes[1, 0].plot(t, E, 'r-', linewidth=1.5)
axes[1, 0].set_xlabel('Time (s)', fontsize=12)
axes[1, 0].set_ylabel('Total Energy (J)', fontsize=12)
axes[1, 0].set_title('Energy Conservation', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# Angular momentum conservation
axes[1, 1].plot(t, L, 'g-', linewidth=1.5)
axes[1, 1].set_xlabel('Time (s)', fontsize=12)
axes[1, 1].set_ylabel('Angular Momentum (kg·m²/s)', fontsize=12)
axes[1, 1].set_title('Angular Momentum Conservation', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('29_orbital_mechanics.png', dpi=150)
print("\n[OK] Saved: 29_orbital_mechanics.png")

# ============================================================================
# Demonstrate different orbit types
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 12))

# Plot central body
ax.plot(0, 0, 'yo', markersize=25, label='Central body')

# Different initial velocities give different orbits
velocities = [6.0, 8.0, 10.0, 14.14]
colors = ['blue', 'green', 'orange', 'red']
labels = ['Ellipse (low e)', 'Ellipse (medium e)', 'Ellipse (high e)', 'Parabolic (escape)']

for v0, color, label in zip(velocities, colors, labels):
    comp = PhysicsCompiler()
    code = dsl_code.replace('y_dot=8.0', f'y_dot={v0}')
    comp.compile_dsl(code)
    sol = comp.simulate(t_span=(0, 150), num_points=3000)
    
    if sol['success']:
        ax.plot(sol['y'][0], sol['y'][2], '-', color=color, 
               linewidth=1.5, alpha=0.8, label=f'{label} (v_0={v0})')

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Different Orbit Types Based on Initial Velocity', fontsize=14)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)

plt.tight_layout()
plt.savefig('29_orbit_types.png', dpi=150)
print("[OK] Saved: 29_orbit_types.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Kepler orbits are conic sections (ellipse/parabola/hyperbola)")
print("2. E < 0: bound orbit (ellipse), E = 0: escape (parabola)")
print("3. Kepler's 3rd law: T² ~ a³")
print("4. Angular momentum determines orbital plane orientation")
print("5. Areal velocity is constant (Kepler's 2nd law)")
print("6. Foundation for space mission design!")
print("="*60)

plt.show()


