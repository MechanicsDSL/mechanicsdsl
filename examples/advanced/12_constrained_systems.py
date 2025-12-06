"""
Tutorial 12: Constrained Systems

A bead sliding on a rotating hoop - a classic example of constrained motion.

Physics:
- Bead of mass m constrained to move on a circular hoop of radius R
- Hoop rotates about vertical axis with angular velocity omega
- Only one degree of freedom: angle theta of bead from bottom
- Shows interesting stability behavior at high rotation speeds
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define bead on rotating hoop
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{bead_on_hoop}

\defvar{theta}{Angle of bead from bottom of hoop}{rad}

\parameter{m}{1.0}{kg}
\parameter{R}{1.0}{m}
\parameter{omega}{5.0}{rad/s}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * m * R^2 * \dot{theta}^2 
    + \frac{1}{2} * m * R^2 * omega^2 * \sin{theta}^2
    + m * g * R * \cos{theta}
}

\initial{theta=0.1, theta_dot=0.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling bead on rotating hoop...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    exit(1)

print("[OK] Compilation successful!")

solution = compiler.simulate(t_span=(0, 10), num_points=1000)

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

# Parameters
R = 1.0
omega = 5.0
g = 9.81

# Calculate equilibrium angle
# At equilibrium: omega²Rsinthetacostheta = gsintheta
# For theta ≠ 0: costheta = g/(omega²R)
cos_eq = g / (omega**2 * R)
if abs(cos_eq) <= 1:
    theta_eq = np.arccos(cos_eq)
    print(f"\nEquilibrium angle: theta_eq = {np.degrees(theta_eq):.1f} deg")
else:
    theta_eq = None
    print("\nNo off-center equilibrium (omega too slow)")

# ============================================================================
# Plot 1: Angle vs time
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t, np.degrees(theta), 'b-', linewidth=2)
if theta_eq:
    axes[0, 0].axhline(np.degrees(theta_eq), color='r', linestyle='--', 
                       label=f'Equilibrium ({np.degrees(theta_eq):.1f} deg)')
    axes[0, 0].legend()
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('theta (degrees)')
axes[0, 0].set_title('Bead Position vs Time')
axes[0, 0].grid(True, alpha=0.3)

# Phase portrait
axes[0, 1].plot(np.degrees(theta), theta_dot, 'b-', linewidth=1.5, alpha=0.7)
axes[0, 1].set_xlabel('theta (degrees)')
axes[0, 1].set_ylabel('thetȧ (rad/s)')
axes[0, 1].set_title('Phase Portrait')
axes[0, 1].grid(True, alpha=0.3)

# Visualize the hoop and bead positions
ax = axes[1, 0]
hoop_angles = np.linspace(0, 2*np.pi, 100)
hoop_x = R * np.sin(hoop_angles)
hoop_y = -R * np.cos(hoop_angles)
ax.plot(hoop_x, hoop_y, 'k-', linewidth=2, label='Hoop')

# Plot bead at several times
sample_indices = np.linspace(0, len(t)-1, 10, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))
for i, idx in enumerate(sample_indices):
    bead_x = R * np.sin(theta[idx])
    bead_y = -R * np.cos(theta[idx])
    ax.plot(bead_x, bead_y, 'o', markersize=12, color=colors[i], alpha=0.7)

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Bead Positions on Hoop (color = time)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Compare different rotation speeds
axes[1, 1].set_title('Effect of Rotation Speed on Equilibrium')
omegas = np.linspace(0.1, 10, 100)
theta_eqs = []
for w in omegas:
    cos_val = g / (w**2 * R)
    if abs(cos_val) <= 1:
        theta_eqs.append(np.degrees(np.arccos(cos_val)))
    else:
        theta_eqs.append(0)

axes[1, 1].plot(omegas, theta_eqs, 'b-', linewidth=2)
axes[1, 1].axvline(np.sqrt(g/R), color='r', linestyle='--', 
                   label=f'omega_crit = sqrt(g/R) = {np.sqrt(g/R):.2f}')
axes[1, 1].set_xlabel('omega (rad/s)')
axes[1, 1].set_ylabel('Equilibrium theta (degrees)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('12_constrained_systems.png', dpi=150)
print("\n[OK] Saved: 12_constrained_systems.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Constraint reduces 3D problem to 1D (single angle theta)")
print("2. Fast rotation creates new stable equilibrium off-center")
print("3. Critical angular velocity: omega_crit = sqrt(g/R)")
print("4. Below omega_crit: only theta=0 is stable (bottom)")
print("5. Above omega_crit: theta=0 unstable, new equilibrium appears")
print("6. This is a 'bifurcation' - qualitative change in behavior")
print("="*60)

plt.show()


