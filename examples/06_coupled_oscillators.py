"""
Tutorial 06: Coupled Oscillators

Two masses connected by springs create interesting dynamics:
- Normal modes: Both masses move together or in opposite directions
- Energy transfer: Energy oscillates between the two masses
- Beat frequencies: When frequencies are close, you see "beats"

Physics:
- Two masses m₁, m₂ connected by spring k₁₂
- Each mass also connected to wall by springs k₁, k₂
- Lagrangian includes coupling terms
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define coupled oscillator system
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = """
\\system{coupled_oscillators}

\\defvar{x1}{Position of first mass}{m}
\\defvar{x2}{Position of second mass}{m}

\\parameter{m1}{1.0}{kg}
\\parameter{m2}{1.0}{kg}
\\parameter{k1}{10.0}{N/m}
\\parameter{k2}{10.0}{N/m}
\\parameter{k12}{5.0}{N/m}

\\lagrangian{
    \\frac{1}{2} * m1 * \\dot{x1}^2 +
    \\frac{1}{2} * m2 * \\dot{x2}^2 -
    \\frac{1}{2} * k1 * x1^2 -
    \\frac{1}{2} * k2 * x2^2 -
    \\frac{1}{2} * k12 * (x2 - x1)^2
}

\\initial{x1=1.0, x2=0.0, x1_dot=0.0, x2_dot=0.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling coupled oscillators...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"❌ Compilation failed: {result.get('error')}")
    exit(1)

print("✅ Compilation successful!")

# Calculate normal mode frequencies (analytical)
m1, m2 = 1.0, 1.0
k1, k2, k12 = 10.0, 10.0, 5.0

# For symmetric case (m1=m2, k1=k2), normal modes are:
# ω₁² = (k1 + k12)/m (symmetric mode)
# ω₂² = k1/m (antisymmetric mode)
omega1 = np.sqrt((k1 + k12) / m1)
omega2 = np.sqrt(k1 / m1)

print(f"\nNormal mode frequencies:")
print(f"   ω₁ (symmetric): {omega1:.3f} rad/s")
print(f"   ω₂ (antisymmetric): {omega2:.3f} rad/s")

solution = compiler.simulate(t_span=(0, 20), num_points=1000)

if not solution['success']:
    print(f"❌ Simulation failed!")
    exit(1)

print("✅ Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
x1 = solution['y'][0]
x1_dot = solution['y'][1]
x2 = solution['y'][2]
x2_dot = solution['y'][3]

# ============================================================================
# Plot 1: Positions vs time
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Positions
axes[0].plot(t, x1, 'b-', linewidth=2, label='x₁ (mass 1)')
axes[0].plot(t, x2, 'r-', linewidth=2, label='x₂ (mass 2)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Position (m)')
axes[0].set_title('Coupled Oscillators: Positions vs Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Velocities
axes[1].plot(t, x1_dot, 'b-', linewidth=2, label='ẋ₁')
axes[1].plot(t, x2_dot, 'r-', linewidth=2, label='ẋ₂')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_title('Velocities vs Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Energy transfer
T1 = 0.5 * m1 * x1_dot**2
T2 = 0.5 * m2 * x2_dot**2
V1 = 0.5 * k1 * x1**2
V2 = 0.5 * k2 * x2**2
V12 = 0.5 * k12 * (x2 - x1)**2

E1 = T1 + V1
E2 = T2 + V2
E_total = E1 + E2 + V12

axes[2].plot(t, E1, 'b-', linewidth=2, label='Energy of mass 1', alpha=0.7)
axes[2].plot(t, E2, 'r-', linewidth=2, label='Energy of mass 2', alpha=0.7)
axes[2].plot(t, E_total, 'g--', linewidth=2, label='Total energy')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Energy (J)')
axes[2].set_title('Energy Transfer Between Masses')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_coupled_oscillators.png', dpi=150)
print("\n✅ Saved: 06_coupled_oscillators.png")

# ============================================================================
# Plot 2: Normal mode analysis
# ============================================================================

# Normal mode coordinates
# Q₁ = (x₁ + x₂)/√2 (symmetric mode)
# Q₂ = (x₁ - x₂)/√2 (antisymmetric mode)
Q1 = (x1 + x2) / np.sqrt(2)
Q2 = (x1 - x2) / np.sqrt(2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normal mode 1 (symmetric)
axes[0, 0].plot(t, Q1, 'purple', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Q₁ = (x₁ + x₂)/√2')
axes[0, 0].set_title('Normal Mode 1: Symmetric (in-phase)')
axes[0, 0].grid(True, alpha=0.3)

# Normal mode 2 (antisymmetric)
axes[0, 1].plot(t, Q2, 'orange', linewidth=2)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Q₂ = (x₁ - x₂)/√2')
axes[0, 1].set_title('Normal Mode 2: Antisymmetric (out-of-phase)')
axes[0, 1].grid(True, alpha=0.3)

# Phase space for mode 1
Q1_dot = (x1_dot + x2_dot) / np.sqrt(2)
axes[1, 0].plot(Q1, Q1_dot, 'purple', linewidth=1.5)
axes[1, 0].set_xlabel('Q₁')
axes[1, 0].set_ylabel('Q̇₁')
axes[1, 0].set_title('Phase Space: Mode 1')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Phase space for mode 2
Q2_dot = (x1_dot - x2_dot) / np.sqrt(2)
axes[1, 1].plot(Q2, Q2_dot, 'orange', linewidth=1.5)
axes[1, 1].set_xlabel('Q₂')
axes[1, 1].set_ylabel('Q̇₂')
axes[1, 1].set_title('Phase Space: Mode 2')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axis('equal')

plt.tight_layout()
plt.savefig('06_normal_modes.png', dpi=150)
print("✅ Saved: 06_normal_modes.png")

# ============================================================================
# Plot 3: Visualization of masses
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

# Draw springs and masses at several time points
time_indices = np.linspace(0, len(t)-1, 30, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

wall_x = -2.0
x1_pos = wall_x + 1.0 + x1  # Relative to wall
x2_pos = x1_pos + 1.0 + (x2 - x1)  # Relative to first mass

for i, idx in enumerate(time_indices):
    # Draw springs (simplified as lines)
    # Spring 1: wall to mass 1
    ax.plot([wall_x, x1_pos[idx]], [0, 0], 'k-', linewidth=1, alpha=0.3)
    # Spring 12: mass 1 to mass 2
    ax.plot([x1_pos[idx], x2_pos[idx]], [0, 0], 'k-', linewidth=1, alpha=0.3)
    # Spring 2: mass 2 to wall
    ax.plot([x2_pos[idx], 2.0], [0, 0], 'k-', linewidth=1, alpha=0.3)
    
    # Draw masses
    ax.plot(x1_pos[idx], 0, 'bo', markersize=10, alpha=0.5, color=colors[i])
    ax.plot(x2_pos[idx], 0, 'ro', markersize=10, alpha=0.5, color=colors[i])

# Draw walls
ax.plot([wall_x, wall_x], [-0.5, 0.5], 'k-', linewidth=4, label='Wall')
ax.plot([2.0, 2.0], [-0.5, 0.5], 'k-', linewidth=4)

# Draw trajectory
ax.plot(x1_pos, np.zeros_like(x1_pos), 'b-', linewidth=1, alpha=0.3, label='Mass 1 path')
ax.plot(x2_pos, np.zeros_like(x2_pos), 'r-', linewidth=1, alpha=0.3, label='Mass 2 path')

ax.set_xlabel('Position (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Coupled Oscillators: System Visualization', fontsize=14)
ax.set_ylim(-0.6, 0.6)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_system_visualization.png', dpi=150)
print("✅ Saved: 06_system_visualization.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Coupled systems have NORMAL MODES (independent oscillations)")
print("2. Energy transfers between masses")
print("3. Normal modes are linear combinations of coordinates")
print("4. Each mode has its own frequency")
print("5. General motion is superposition of normal modes")
print("6. Total energy is conserved")
print("="*60)

plt.show()
