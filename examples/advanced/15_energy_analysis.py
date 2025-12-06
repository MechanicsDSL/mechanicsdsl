"""
Tutorial 15: Energy Analysis

Energy conservation is a fundamental principle in physics.
This tutorial shows how to:
1. Calculate kinetic, potential, and total energy
2. Check energy conservation
3. Analyze energy transfer
4. Visualize energy flow

We'll use the energy calculation utilities from MechanicsDSL.
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler, PotentialEnergyCalculator

# ============================================================================
# Example: Pendulum with energy analysis
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = """
\\system{pendulum_energy}

\\defvar{theta}{Angle}{rad}

\\parameter{m}{1.0}{kg}
\\parameter{L}{1.0}{m}
\\parameter{g}{9.81}{m/s^2}

\\lagrangian{
    \\frac{1}{2} * m * L^2 * \\dot{theta}^2 - m * g * L * (1 - \\cos{theta})
}

\\initial{theta=1.0, theta_dot=0.0}
"""

result = compiler.compile_dsl(dsl_code)
if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    exit(1)

solution = compiler.simulate(t_span=(0, 10), num_points=1000)

# ============================================================================
# Extract state variables
# ============================================================================

t = solution['t']
theta = solution['y'][0]
theta_dot = solution['y'][1]

# Convert to Cartesian
L = 1.0
x = L * np.sin(theta)
y = -L * np.cos(theta)

# ============================================================================
# Calculate energies manually
# ============================================================================

m, g = 1.0, 9.81

# Kinetic energy: T = (1/2)mL²thetȧ²
T = 0.5 * m * L**2 * theta_dot**2

# Potential energy: V = mgL(1 - cos theta)
# (Reference at bottom: V = 0 when theta = 0)
V = m * g * L * (1 - np.cos(theta))

# Total energy
E_total = T + V

# ============================================================================
# Plot energy analysis
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Energy vs time
axes[0].plot(t, T, 'b-', linewidth=2, label='Kinetic Energy', alpha=0.8)
axes[0].plot(t, V, 'r-', linewidth=2, label='Potential Energy', alpha=0.8)
axes[0].plot(t, E_total, 'g--', linewidth=2, label='Total Energy', alpha=0.9)
axes[0].set_xlabel('Time (s)', fontsize=12)
axes[0].set_ylabel('Energy (J)', fontsize=12)
axes[0].set_title('Energy vs Time', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Energy conservation error
E_error = E_total - E_total[0]
axes[1].plot(t, E_error, 'purple', linewidth=2)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Energy Error (J)', fontsize=12)
axes[1].set_title('Energy Conservation Error', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Energy vs position
axes[2].plot(theta, T, 'b-', linewidth=2, label='Kinetic', alpha=0.8)
axes[2].plot(theta, V, 'r-', linewidth=2, label='Potential', alpha=0.8)
axes[2].plot(theta, E_total, 'g--', linewidth=2, label='Total', alpha=0.9)
axes[2].set_xlabel('Angle theta (rad)', fontsize=12)
axes[2].set_ylabel('Energy (J)', fontsize=12)
axes[2].set_title('Energy vs Position', fontsize=14)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('15_energy_analysis.png', dpi=150)
print("[OK] Saved: 15_energy_analysis.png")

# ============================================================================
# Energy statistics
# ============================================================================

print("\n" + "="*60)
print("ENERGY STATISTICS:")
print("="*60)
print(f"Initial total energy: {E_total[0]:.6f} J")
print(f"Final total energy: {E_total[-1]:.6f} J")
print(f"Energy change: {E_total[-1] - E_total[0]:.2e} J")
print(f"Relative error: {abs(E_total[-1] - E_total[0]) / E_total[0] * 100:.2e}%")
print(f"Max kinetic energy: {np.max(T):.6f} J")
print(f"Max potential energy: {np.max(V):.6f} J")
print("="*60)

# ============================================================================
# Energy flow visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Create filled area plot
ax.fill_between(t, 0, T, alpha=0.5, color='blue', label='Kinetic Energy')
ax.fill_between(t, T, T+V, alpha=0.5, color='red', label='Potential Energy')
ax.plot(t, E_total, 'g-', linewidth=2, label='Total Energy')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Energy (J)', fontsize=12)
ax.set_title('Energy Flow Visualization', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('15_energy_flow.png', dpi=150)
print("[OK] Saved: 15_energy_flow.png")

# ============================================================================
# Phase space with energy contours
# ============================================================================

# Create energy contour plot
theta_range = np.linspace(-np.pi, np.pi, 100)
theta_dot_range = np.linspace(-3, 3, 100)
THETA, THETA_DOT = np.meshgrid(theta_range, theta_dot_range)

# Calculate energy at each point
E_contour = 0.5 * m * L**2 * THETA_DOT**2 + m * g * L * (1 - np.cos(THETA))

fig, ax = plt.subplots(figsize=(10, 8))

# Plot contours
contour = ax.contour(THETA, THETA_DOT, E_contour, levels=20, alpha=0.5)
ax.clabel(contour, inline=True, fontsize=8)

# Plot trajectory
ax.plot(theta, theta_dot, 'b-', linewidth=2, label='Trajectory')

ax.set_xlabel('Angle theta (rad)', fontsize=12)
ax.set_ylabel('Angular Velocity thetȧ (rad/s)', fontsize=12)
ax.set_title('Phase Space with Energy Contours', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('15_phase_space_energy.png', dpi=150)
print("[OK] Saved: 15_phase_space_energy.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Total energy should be conserved (constant)")
print("2. Energy oscillates between kinetic and potential")
print("3. Maximum kinetic when potential is minimum (and vice versa)")
print("4. Energy conservation error shows numerical accuracy")
print("5. Phase space trajectories follow energy contours")
print("6. Energy analysis helps validate simulations")
print("="*60)

plt.show()


