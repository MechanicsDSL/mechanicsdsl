"""
Tutorial 04: Plotting and Visualization Basics

MechanicsDSL has built-in visualization tools! This tutorial shows:
1. Basic plotting with matplotlib
2. Using the built-in animate() function
3. Energy plots
4. Phase space plots
5. Custom visualizations

We'll use a simple harmonic oscillator as our example.
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Setup: Create a harmonic oscillator
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{harmonic_oscillator}

\defvar{x}{Position}{m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}

\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}

\initial{x=1.0, x_dot=0.0}
"""

result = compiler.compile_dsl(dsl_code)
if not result['success']:
    print(f"❌ Compilation failed: {result.get('error')}")
    exit(1)

solution = compiler.simulate(t_span=(0, 10), num_points=500)

# ============================================================================
# Method 1: Manual plotting with matplotlib
# ============================================================================

print("Method 1: Manual plotting")
t = solution['t']
x = solution['y'][0]
x_dot = solution['y'][1]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Position vs time
axes[0, 0].plot(t, x, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Position (m)')
axes[0, 0].set_title('Position vs Time')
axes[0, 0].grid(True, alpha=0.3)

# Velocity vs time
axes[0, 1].plot(t, x_dot, 'r-', linewidth=2)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Velocity (m/s)')
axes[0, 1].set_title('Velocity vs Time')
axes[0, 1].grid(True, alpha=0.3)

# Phase space
axes[1, 0].plot(x, x_dot, 'purple', linewidth=1.5)
axes[1, 0].set_xlabel('Position (m)')
axes[1, 0].set_ylabel('Velocity (m/s)')
axes[1, 0].set_title('Phase Space')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Energy
m, k = 1.0, 10.0
T = 0.5 * m * x_dot**2
V = 0.5 * k * x**2
E = T + V

axes[1, 1].plot(t, T, 'b-', label='Kinetic', linewidth=2)
axes[1, 1].plot(t, V, 'r-', label='Potential', linewidth=2)
axes[1, 1].plot(t, E, 'g--', label='Total', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Energy (J)')
axes[1, 1].set_title('Energy vs Time')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_manual_plotting.png', dpi=150)
print("✅ Saved: 04_manual_plotting.png")

# ============================================================================
# Method 2: Using built-in visualization functions
# ============================================================================

print("\nMethod 2: Built-in visualization functions")

# Note: animate() creates an animation object
# In Jupyter, it will display automatically
# In scripts, you may need to save it

try:
    # Create animation (this may not display in all environments)
    anim = compiler.animate(solution, save_path='04_animation.gif', fps=30)
    print("✅ Animation created (may need to be saved manually)")
except Exception as e:
    print(f"⚠ Animation creation: {e}")
    print("   (This is normal in some environments)")

# ============================================================================
# Method 3: Energy plotting
# ============================================================================

print("\nMethod 3: Energy analysis")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual energies
ax.plot(t, T, 'b-', linewidth=2, label='Kinetic Energy', alpha=0.7)
ax.plot(t, V, 'r-', linewidth=2, label='Potential Energy', alpha=0.7)
ax.plot(t, E, 'g-', linewidth=2, label='Total Energy', alpha=0.9)

# Add fill between curves for visual appeal
ax.fill_between(t, 0, T, alpha=0.3, color='blue')
ax.fill_between(t, 0, V, alpha=0.3, color='red')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Energy (J)', fontsize=12)
ax.set_title('Energy Conservation in Harmonic Oscillator', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add text annotation
energy_error = abs(E[-1] - E[0])
ax.text(0.02, 0.98, f'Energy conservation error: {energy_error:.2e} J',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('04_energy_plot.png', dpi=150)
print("✅ Saved: 04_energy_plot.png")

# ============================================================================
# Method 4: Phase space with trajectory
# ============================================================================

print("\nMethod 4: Phase space visualization")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot trajectory
ax.plot(x, x_dot, 'b-', linewidth=2, alpha=0.7, label='Trajectory')

# Mark start and end
ax.plot(x[0], x_dot[0], 'go', markersize=12, label='Start', zorder=5)
ax.plot(x[-1], x_dot[-1], 'ro', markersize=12, label='End', zorder=5)

# Add velocity vectors at selected points
skip = len(x) // 10
for i in range(0, len(x), skip):
    # Calculate velocity vector (scaled for visibility)
    scale = 0.1
    dx = x_dot[i] * scale
    dy = -k/m * x[i] * scale  # Acceleration direction
    ax.arrow(x[i], x_dot[i], dx, dy,
             head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.5)

ax.set_xlabel('Position (m)', fontsize=12)
ax.set_ylabel('Velocity (m/s)', fontsize=12)
ax.set_title('Phase Space Trajectory with Velocity Vectors', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('04_phase_space.png', dpi=150)
print("✅ Saved: 04_phase_space.png")

# ============================================================================
# Method 5: Custom multi-panel visualization
# ============================================================================

print("\nMethod 5: Custom multi-panel layout")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main trajectory plot (large)
ax_main = fig.add_subplot(gs[0:2, 0:2])
ax_main.plot(x, x_dot, 'b-', linewidth=2)
ax_main.set_xlabel('Position (m)')
ax_main.set_ylabel('Velocity (m/s)')
ax_main.set_title('Phase Space Trajectory')
ax_main.grid(True, alpha=0.3)
ax_main.axis('equal')

# Position time series
ax_pos = fig.add_subplot(gs[0, 2])
ax_pos.plot(t, x, 'b-', linewidth=1.5)
ax_pos.set_xlabel('Time (s)')
ax_pos.set_ylabel('x (m)')
ax_pos.set_title('Position')
ax_pos.grid(True, alpha=0.3)

# Velocity time series
ax_vel = fig.add_subplot(gs[1, 2])
ax_vel.plot(t, x_dot, 'r-', linewidth=1.5)
ax_vel.set_xlabel('Time (s)')
ax_vel.set_ylabel('ẋ (m/s)')
ax_vel.set_title('Velocity')
ax_vel.grid(True, alpha=0.3)

# Energy plot
ax_energy = fig.add_subplot(gs[2, :])
ax_energy.plot(t, T, 'b-', label='Kinetic', linewidth=1.5)
ax_energy.plot(t, V, 'r-', label='Potential', linewidth=1.5)
ax_energy.plot(t, E, 'g--', label='Total', linewidth=2)
ax_energy.set_xlabel('Time (s)')
ax_energy.set_ylabel('Energy (J)')
ax_energy.set_title('Energy Conservation')
ax_energy.legend()
ax_energy.grid(True, alpha=0.3)

plt.savefig('04_custom_layout.png', dpi=150)
print("✅ Saved: 04_custom_layout.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*60)
print("VISUALIZATION METHODS:")
print("="*60)
print("1. Manual matplotlib: Full control, customizable")
print("2. Built-in animate(): Quick animations")
print("3. Energy plots: Check conservation")
print("4. Phase space: See system dynamics")
print("5. Custom layouts: Combine multiple views")
print("="*60)

plt.show()
