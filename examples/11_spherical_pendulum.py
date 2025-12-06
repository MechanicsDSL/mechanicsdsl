"""
Tutorial 11: Spherical Pendulum

A 3D pendulum that can swing in any direction, not just a plane.

Physics:
- Mass m suspended by rod of length L
- Two angles: theta (polar/tilt from vertical), φ (azimuthal/rotation around z-axis)
- Motion constrained to surface of a sphere
- Exhibits rich, quasi-periodic behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define the spherical pendulum
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{spherical_pendulum}

\defvar{theta}{Polar angle from vertical}{rad}
\defvar{phi}{Azimuthal angle}{rad}

\parameter{m}{1.0}{kg}
\parameter{L}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * m * L^2 * (\dot{theta}^2 + \sin{theta}^2 * \dot{phi}^2)
    + m * g * L * \cos{theta}
}

\initial{theta=0.5, phi=0.0, theta_dot=0.0, phi_dot=2.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling spherical pendulum...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    exit(1)

print("[OK] Compilation successful!")

solution = compiler.simulate(t_span=(0, 20), num_points=2000)

if not solution['success']:
    print("[FAIL] Simulation failed!")
    exit(1)

print("[OK] Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
theta = solution['y'][0]
theta_dot = solution['y'][1]
phi = solution['y'][2]
phi_dot = solution['y'][3]

# Convert to Cartesian coordinates
L = 1.0
x = L * np.sin(theta) * np.cos(phi)
y = L * np.sin(theta) * np.sin(phi)
z = -L * np.cos(theta)

# ============================================================================
# Plot 1: 3D trajectory
# ============================================================================

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(x, y, z, 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')

# Draw sphere surface (wireframe)
u = np.linspace(0, np.pi, 20)
v = np.linspace(0, 2 * np.pi, 20)
xs = L * np.outer(np.sin(u), np.cos(v))
ys = L * np.outer(np.sin(u), np.sin(v))
zs = -L * np.outer(np.cos(u), np.ones(np.size(v)))
ax.plot_wireframe(xs, ys, zs, alpha=0.1, color='gray')

# Mark start and end
ax.plot([x[0]], [y[0]], [z[0]], 'go', markersize=10, label='Start')
ax.plot([x[-1]], [y[-1]], [z[-1]], 'ro', markersize=10, label='End')

# Pivot point
ax.plot([0], [0], [0], 'ko', markersize=12, label='Pivot')

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_zlabel('z (m)', fontsize=12)
ax.set_title('Spherical Pendulum: 3D Trajectory', fontsize=14)
ax.legend()

plt.savefig('11_spherical_pendulum_3d.png', dpi=150)
print("\n[OK] Saved: 11_spherical_pendulum_3d.png")

# ============================================================================
# Plot 2: Angles vs time
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t, theta, 'b-', linewidth=1.5)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('theta (rad)')
axes[0, 0].set_title('Polar Angle (Tilt)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, phi, 'r-', linewidth=1.5)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('φ (rad)')
axes[0, 1].set_title('Azimuthal Angle (Rotation)')
axes[0, 1].grid(True, alpha=0.3)

# Phase space
axes[1, 0].plot(theta, theta_dot, 'b-', linewidth=1, alpha=0.7)
axes[1, 0].set_xlabel('theta (rad)')
axes[1, 0].set_ylabel('thetȧ (rad/s)')
axes[1, 0].set_title('Phase Space: theta')
axes[1, 0].grid(True, alpha=0.3)

# Top view (projection)
axes[1, 1].plot(x, y, 'purple', linewidth=1, alpha=0.7)
axes[1, 1].set_xlabel('x (m)')
axes[1, 1].set_ylabel('y (m)')
axes[1, 1].set_title('Top View (xy projection)')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('11_spherical_pendulum_analysis.png', dpi=150)
print("[OK] Saved: 11_spherical_pendulum_analysis.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Spherical pendulum has 2 degrees of freedom (theta, φ)")
print("2. Motion is generally quasi-periodic, not simple periodic")
print("3. Angular momentum about z-axis (Lz) is conserved")
print("4. The trajectory traces patterns on a sphere surface")
print("5. At small angles, reduces to 2D harmonic oscillator")
print("="*60)

plt.show()


