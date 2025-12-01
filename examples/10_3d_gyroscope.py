"""
Tutorial 10: 3D Gyroscope

A gyroscope is a spinning top that exhibits fascinating precession and nutation.
This is a 3D system requiring Euler angles.

Physics:
- Rigid body rotation in 3D
- Euler angles: θ (nutation), φ (precession), ψ (spin)
- Conservation of angular momentum
- Precession: slow rotation of spin axis
- Nutation: wobbling motion
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define gyroscope system
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = """
\\system{gyroscope}

\\defvar{theta}{Nutation angle}{rad}
\\defvar{phi}{Precession angle}{rad}
\\defvar{psi}{Spin angle}{rad}

\\parameter{I1}{1.0}{kg·m^2}
\\parameter{I3}{0.5}{kg·m^2}
\\parameter{m}{1.0}{kg}
\\parameter{g}{9.81}{m/s^2}
\\parameter{L}{0.5}{m}

\\lagrangian{
    \\frac{1}{2} * I1 * (\\dot{theta}^2 + \\dot{phi}^2 * \\sin{theta}^2) +
    \\frac{1}{2} * I3 * (\\dot{psi} + \\dot{phi} * \\cos{theta})^2 -
    m * g * L * \\cos{theta}
}

\\initial{theta=0.5, phi=0.0, psi=0.0, theta_dot=0.0, phi_dot=1.0, psi_dot=10.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling gyroscope system...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"❌ Compilation failed: {result.get('error')}")
    exit(1)

print("✅ Compilation successful!")

solution = compiler.simulate(t_span=(0, 5), num_points=1000)

if not solution['success']:
    print(f"❌ Simulation failed!")
    exit(1)

print("✅ Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
theta = solution['y'][0]
phi = solution['y'][2]
psi = solution['y'][4]

# ============================================================================
# Convert to 3D coordinates
# ============================================================================

L = 0.5  # Length of gyroscope axis

# Position of tip of gyroscope in space
x = L * np.sin(theta) * np.cos(phi)
y = L * np.sin(theta) * np.sin(phi)
z = L * np.cos(theta)

# ============================================================================
# Plot 1: Euler angles vs time
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t, theta, 'b-', linewidth=2)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('θ (rad)')
axes[0].set_title('Nutation Angle vs Time')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, phi, 'r-', linewidth=2)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('φ (rad)')
axes[1].set_title('Precession Angle vs Time')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, psi, 'g-', linewidth=2)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('ψ (rad)')
axes[2].set_title('Spin Angle vs Time')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('10_gyroscope_angles.png', dpi=150)
print("\n✅ Saved: 10_gyroscope_angles.png")

# ============================================================================
# Plot 2: 3D trajectory
# ============================================================================

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7, label='Tip trajectory')

# Plot several snapshots
indices = np.linspace(0, len(t)-1, 20, dtype=int)
for idx in indices:
    # Draw gyroscope axis
    ax.plot([0, x[idx]], [0, y[idx]], [0, z[idx]], 
            'r-', linewidth=2, alpha=0.5)

# Mark start and end
ax.plot([x[0]], [y[0]], [z[0]], 'go', markersize=10, label='Start')
ax.plot([x[-1]], [y[-1]], [z[-1]], 'ro', markersize=10, label='End')

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_zlabel('z (m)', fontsize=12)
ax.set_title('Gyroscope: 3D Trajectory', fontsize=14)
ax.legend()

plt.savefig('10_gyroscope_3d.png', dpi=150)
print("✅ Saved: 10_gyroscope_3d.png")

# ============================================================================
# Plot 3: Precession and nutation
# ============================================================================

# Project onto xy plane to see precession
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, label='Projection (xy plane)')
ax.plot(0, 0, 'ko', markersize=15, label='Pivot')

# Draw several positions
for idx in indices:
    ax.plot([0, x[idx]], [0, y[idx]], 'r-', linewidth=1, alpha=0.3)

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Gyroscope: Precession (top view)', fontsize=14)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('10_gyroscope_precession.png', dpi=150)
print("✅ Saved: 10_gyroscope_precession.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Gyroscopes precess: spin axis rotates slowly")
print("2. Nutation: wobbling motion of the spin axis")
print("3. Fast spin (ψ) stabilizes the gyroscope")
print("4. Precession frequency depends on spin rate")
print("5. 3D systems require Euler angles or quaternions")
print("6. Energy and angular momentum are conserved")
print("="*60)

plt.show()
