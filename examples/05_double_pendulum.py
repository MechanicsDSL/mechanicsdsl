"""
Tutorial 05: Double Pendulum (Chaotic System)

Two pendulums connected together create unpredictable motion.

Physics:
- Two masses m₁, m₂ connected by rods of length L₁, L₂
- Angles θ₁, θ₂ from vertical
- Lagrangian is complex but MechanicsDSL handles it automatically!
- Small differences in initial conditions lead to completely different trajectories

This is a great example of sensitive dependence on initial conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define the double pendulum system
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = """
\\system{double_pendulum}

\\defvar{theta1}{Angle of first pendulum}{rad}
\\defvar{theta2}{Angle of second pendulum}{rad}

\\parameter{m1}{1.0}{kg}
\\parameter{m2}{1.0}{kg}
\\parameter{L1}{1.0}{m}
\\parameter{L2}{1.0}{m}
\\parameter{g}{9.81}{m/s^2}

\\lagrangian{
    \\frac{1}{2} * (m1 + m2) * L1^2 * \\dot{theta1}^2 +
    \\frac{1}{2} * m2 * L2^2 * \\dot{theta2}^2 +
    m2 * L1 * L2 * \\dot{theta1} * \\dot{theta2} * \\cos{theta1 - theta2} +
    (m1 + m2) * g * L1 * \\cos{theta1} +
    m2 * g * L2 * \\cos{theta2}
}

\\initial{theta1=0.1, theta2=0.1, theta1_dot=0.0, theta2_dot=0.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling double pendulum...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"❌ Compilation failed: {result.get('error')}")
    exit(1)

print("✅ Compilation successful!")
print(f"   Compilation time: {result.get('compilation_time', 0):.4f} s")

# Simulate for a longer time to see chaos
print("\nSimulating...")
solution = compiler.simulate(t_span=(0, 20), num_points=2000)

if not solution['success']:
    print(f"❌ Simulation failed!")
    exit(1)

print("✅ Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
theta1 = solution['y'][0]
theta1_dot = solution['y'][1]
theta2 = solution['y'][2]
theta2_dot = solution['y'][3]

# Convert to Cartesian coordinates
L1, L2 = 1.0, 1.0

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)

x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# ============================================================================
# Plot 1: Angles vs time
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t, theta1, 'b-', linewidth=1.5, label='θ₁')
axes[0, 0].plot(t, theta2, 'r-', linewidth=1.5, label='θ₂')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Angle (rad)')
axes[0, 0].set_title('Angles vs Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Phase space for first pendulum
axes[0, 1].plot(theta1, theta1_dot, 'b-', linewidth=1, alpha=0.7)
axes[0, 1].set_xlabel('θ₁ (rad)')
axes[0, 1].set_ylabel('θ̇₁ (rad/s)')
axes[0, 1].set_title('Phase Space: First Pendulum')
axes[0, 1].grid(True, alpha=0.3)

# Phase space for second pendulum
axes[1, 0].plot(theta2, theta2_dot, 'r-', linewidth=1, alpha=0.7)
axes[1, 0].set_xlabel('θ₂ (rad)')
axes[1, 0].set_ylabel('θ̇₂ (rad/s)')
axes[1, 0].set_title('Phase Space: Second Pendulum')
axes[1, 0].grid(True, alpha=0.3)

# Trajectory of second mass (the "bob")
axes[1, 1].plot(x2, y2, 'purple', linewidth=1, alpha=0.7)
axes[1, 1].set_xlabel('x (m)')
axes[1, 1].set_ylabel('y (m)')
axes[1, 1].set_title('Trajectory of Second Mass')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_double_pendulum.png', dpi=150)
print("\n✅ Saved: 05_double_pendulum.png")

# ============================================================================
# Plot 2: Animation-style visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 10))

# Plot the trajectory
ax.plot(x2, y2, 'purple', linewidth=0.5, alpha=0.3, label='Trajectory')

# Plot pendulum at several time points
time_indices = np.linspace(0, len(t)-1, 20, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

for i, idx in enumerate(time_indices):
    # Draw pendulum rods
    ax.plot([0, x1[idx]], [0, y1[idx]], 'k-', linewidth=2, alpha=0.5)
    ax.plot([x1[idx], x2[idx]], [y1[idx], y2[idx]], 'k-', linewidth=2, alpha=0.5)
    
    # Draw masses
    ax.plot(x1[idx], y1[idx], 'bo', markersize=8, alpha=0.7)
    ax.plot(x2[idx], y2[idx], 'ro', markersize=10, alpha=0.7, color=colors[i])

# Mark pivot
ax.plot(0, 0, 'ko', markersize=15, label='Pivot')

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Double Pendulum: Trajectory and Snapshots', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('05_double_pendulum_trajectory.png', dpi=150)
print("✅ Saved: 05_double_pendulum_trajectory.png")

# ============================================================================
# Demonstrate chaos: Small change in initial conditions
# ============================================================================

print("\n" + "="*60)
print("DEMONSTRATING CHAOS:")
print("="*60)
print("Running simulation with slightly different initial conditions...")

compiler2 = PhysicsCompiler()

dsl_code2 = dsl_code.replace(
    '\\initial{theta1=0.1, theta2=0.1, theta1_dot=0.0, theta2_dot=0.0}',
    '\\initial{theta1=0.1001, theta2=0.1, theta1_dot=0.0, theta2_dot=0.0}'
)

result2 = compiler2.compile_dsl(dsl_code2)
solution2 = compiler2.simulate(t_span=(0, 20), num_points=2000)

theta1_2 = solution2['y'][0]
theta2_2 = solution2['y'][2]

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(t, theta2, 'b-', linewidth=1.5, label='θ₂ (original)', alpha=0.7)
ax.plot(t, theta2_2, 'r-', linewidth=1.5, label='θ₂ (θ₁₀ = 0.1001)', alpha=0.7)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Angle (rad)', fontsize=12)
ax.set_title('Chaos: Small Change in Initial Conditions (0.1 → 0.1001)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Highlight divergence
divergence_point = None
for i in range(len(t)):
    if abs(theta2[i] - theta2_2[i]) > 0.5:
        divergence_point = i
        break

if divergence_point:
    ax.axvline(t[divergence_point], color='green', linestyle='--', 
               label=f'Divergence at t={t[divergence_point]:.1f}s')
    ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('05_chaos_demonstration.png', dpi=150)
print("✅ Saved: 05_chaos_demonstration.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Double pendulum is CHAOTIC: sensitive to initial conditions")
print("2. Motion is unpredictable but deterministic")
print("3. Trajectory of second mass creates beautiful patterns")
print("4. Phase space shows complex, non-repeating orbits")
print("5. Energy is still conserved (if no damping)")
print("6. Small numerical errors can cause large differences")
print("="*60)

plt.show()
