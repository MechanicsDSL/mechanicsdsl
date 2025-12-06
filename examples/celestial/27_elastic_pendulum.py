"""
Tutorial 27: Elastic Pendulum (Spring Pendulum)

A pendulum where the rod is replaced by a spring.

Physics:
- Mass m on spring with natural length L_0 and stiffness k
- Two degrees of freedom: r (radial stretch), theta (angle)
- Shows interesting energy transfer between modes
- Can exhibit chaotic behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define elastic pendulum
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{elastic_pendulum}

\defvar{r}{Radial distance from pivot}{m}
\defvar{theta}{Angle from vertical}{rad}

\parameter{m}{1.0}{kg}
\parameter{k}{50.0}{N/m}
\parameter{L0}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * m * (\dot{r}^2 + r^2 * \dot{theta}^2)
    + m * g * r * \cos{theta}
    - \frac{1}{2} * k * (r - L0)^2
}

\initial{r=1.2, theta=0.5, r_dot=0.0, theta_dot=0.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling elastic pendulum...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    exit(1)

print("[OK] Compilation successful!")

solution = compiler.simulate(t_span=(0, 30), num_points=3000)

if not solution['success']:
    print("[FAIL] Simulation failed!")
    exit(1)

print("[OK] Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
r = solution['y'][0]
r_dot = solution['y'][1]
theta = solution['y'][2]
theta_dot = solution['y'][3]

# Convert to Cartesian
x = r * np.sin(theta)
y = -r * np.cos(theta)

# Parameters
m, k, L0, g = 1.0, 50.0, 1.0, 9.81

# ============================================================================
# Calculate frequencies
# ============================================================================

# Radial (spring) frequency: omega_r = sqrt(k/m)
omega_r = np.sqrt(k / m)

# Pendulum frequency (at equilibrium r ≈ L0): omega_theta ≈ sqrt(g/L0)
omega_theta = np.sqrt(g / L0)

print(f"\nNatural frequencies:")
print(f"   Radial (spring):    omega_r = {omega_r:.3f} rad/s, T_r = {2*np.pi/omega_r:.3f} s")
print(f"   Angular (pendulum): omega_theta = {omega_theta:.3f} rad/s, T_theta = {2*np.pi/omega_theta:.3f} s")

# Check for resonance condition (2:1 ratio leads to interesting dynamics)
ratio = omega_r / omega_theta
print(f"   Frequency ratio: omega_r/omega_theta = {ratio:.3f}")
if abs(ratio - 2) < 0.1:
    print("   → Near 2:1 resonance! Expect strong energy transfer.")

# ============================================================================
# Plot results
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Trajectory
axes[0, 0].plot(x, y, 'b-', linewidth=0.8, alpha=0.7)
axes[0, 0].plot(0, 0, 'ko', markersize=10, label='Pivot')
axes[0, 0].set_xlabel('x (m)')
axes[0, 0].set_ylabel('y (m)')
axes[0, 0].set_title('Elastic Pendulum Trajectory')
axes[0, 0].set_aspect('equal')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# r vs time
axes[0, 1].plot(t, r, 'b-', linewidth=1)
axes[0, 1].axhline(L0, color='r', linestyle='--', label=f'L_0 = {L0} m')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('r (m)')
axes[0, 1].set_title('Radial Distance vs Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# theta vs time
axes[0, 2].plot(t, np.degrees(theta), 'r-', linewidth=1)
axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].set_ylabel('theta (degrees)')
axes[0, 2].set_title('Angle vs Time')
axes[0, 2].grid(True, alpha=0.3)

# Energy analysis
KE = 0.5 * m * (r_dot**2 + r**2 * theta_dot**2)
PE_gravity = -m * g * r * np.cos(theta)
PE_spring = 0.5 * k * (r - L0)**2
E_total = KE + PE_gravity + PE_spring

axes[1, 0].plot(t, KE, 'b-', label='Kinetic', alpha=0.7)
axes[1, 0].plot(t, PE_spring, 'r-', label='Spring PE', alpha=0.7)
axes[1, 0].plot(t, PE_gravity, 'g-', label='Gravity PE', alpha=0.7)
axes[1, 0].plot(t, E_total, 'k-', label='Total', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Energy (J)')
axes[1, 0].set_title('Energy Components')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Phase space r
axes[1, 1].plot(r, r_dot, 'b-', linewidth=0.5, alpha=0.7)
axes[1, 1].set_xlabel('r (m)')
axes[1, 1].set_ylabel('ṙ (m/s)')
axes[1, 1].set_title('Phase Space: Radial')
axes[1, 1].grid(True, alpha=0.3)

# Phase space theta
axes[1, 2].plot(theta, theta_dot, 'r-', linewidth=0.5, alpha=0.7)
axes[1, 2].set_xlabel('theta (rad)')
axes[1, 2].set_ylabel('thetȧ (rad/s)')
axes[1, 2].set_title('Phase Space: Angular')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('27_elastic_pendulum.png', dpi=150)
print("\n[OK] Saved: 27_elastic_pendulum.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Two coupled degrees of freedom: r and theta")
print("2. Energy transfers between radial and angular modes")
print("3. 2:1 frequency ratio causes parametric resonance")
print("4. Rich dynamics despite simple setup")
print("5. Used in studying autoparametric systems")
print("6. Shows quasi-periodic or chaotic behavior")
print("="*60)

plt.show()


