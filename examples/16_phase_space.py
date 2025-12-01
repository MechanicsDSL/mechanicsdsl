"""
Tutorial 16: Phase Space Visualization

Phase space is a powerful tool for understanding dynamics:
- Each point represents a complete state (position, momentum)
- Trajectories show how the system evolves
- Closed orbits indicate periodic motion
- Fixed points show equilibria
- Energy contours help visualize conservation

This tutorial shows various phase space visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Example 1: Harmonic Oscillator (Elliptical orbits)
# ============================================================================

print("="*60)
print("EXAMPLE 1: HARMONIC OSCILLATOR")
print("="*60)

compiler1 = PhysicsCompiler()

dsl1 = """
\\system{harmonic_oscillator_phase}

\\defvar{x}{Position}{m}

\\parameter{m}{1.0}{kg}
\\parameter{k}{10.0}{N/m}

\\lagrangian{\\frac{1}{2} * m * \\dot{x}^2 - \\frac{1}{2} * k * x^2}

\\initial{x=1.0, x_dot=0.0}
"""

result1 = compiler1.compile_dsl(dsl1)
solution1 = compiler1.simulate(t_span=(0, 10), num_points=1000)

x1 = solution1['y'][0]
x_dot1 = solution1['y'][1]

# ============================================================================
# Example 2: Pendulum (Nonlinear, complex orbits)
# ============================================================================

print("\n" + "="*60)
print("EXAMPLE 2: PENDULUM")
print("="*60)

compiler2 = PhysicsCompiler()

dsl2 = """
\\system{pendulum_phase}

\\defvar{theta}{Angle}{rad}

\\parameter{m}{1.0}{kg}
\\parameter{L}{1.0}{m}
\\parameter{g}{9.81}{m/s^2}

\\lagrangian{
    \\frac{1}{2} * m * L^2 * \\dot{theta}^2 - m * g * L * (1 - \\cos{theta})
}

\\initial{theta=0.5, theta_dot=0.0}
"""

result2 = compiler2.compile_dsl(dsl2)
solution2 = compiler2.simulate(t_span=(0, 10), num_points=1000)

theta2 = solution2['y'][0]
theta_dot2 = solution2['y'][1]

# ============================================================================
# Example 3: Multiple initial conditions (phase portrait)
# ============================================================================

print("\n" + "="*60)
print("EXAMPLE 3: PHASE PORTRAIT (Multiple Trajectories)")
print("="*60)

# Run multiple simulations with different initial conditions
trajectories = []
initial_conditions = [
    (0.5, 0.0), (1.0, 0.0), (1.5, 0.0),
    (0.0, 1.0), (0.0, 2.0), (0.0, 3.0),
    (0.5, 1.0), (1.0, 1.0), (1.5, 1.0),
]

for x0, x_dot0 in initial_conditions:
    compiler = PhysicsCompiler()
    dsl = f"""
\\system{{harmonic_phase_portrait}}

\\defvar{{x}}{{Position}}{{m}}

\\parameter{{m}}{{1.0}}{{kg}}
\\parameter{{k}}{{10.0}}{{N/m}}

\\lagrangian{{\\frac{{1}}{{2}} * m * \\dot{{x}}^2 - \\frac{{1}}{{2}} * k * x^2}}

\\initial{{x={x0}, x_dot={x_dot0}}}
"""
    result = compiler.compile_dsl(dsl)
    solution = compiler.simulate(t_span=(0, 5), num_points=500)
    trajectories.append((solution['y'][0], solution['y'][1]))

# ============================================================================
# Plot 1: Basic phase space
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Harmonic oscillator
axes[0, 0].plot(x1, x_dot1, 'b-', linewidth=2)
axes[0, 0].plot(x1[0], x_dot1[0], 'go', markersize=10, label='Start')
axes[0, 0].plot(x1[-1], x_dot1[-1], 'ro', markersize=10, label='End')
axes[0, 0].set_xlabel('Position (m)', fontsize=12)
axes[0, 0].set_ylabel('Velocity (m/s)', fontsize=12)
axes[0, 0].set_title('Harmonic Oscillator: Elliptical Orbit', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axis('equal')

# Pendulum
axes[0, 1].plot(theta2, theta_dot2, 'purple', linewidth=2)
axes[0, 1].plot(theta2[0], theta_dot2[0], 'go', markersize=10, label='Start')
axes[0, 1].plot(theta2[-1], theta_dot2[-1], 'ro', markersize=10, label='End')
axes[0, 1].set_xlabel('Angle (rad)', fontsize=12)
axes[0, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
axes[0, 1].set_title('Pendulum: Complex Orbit', fontsize=13)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase portrait
colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
for i, (x_traj, x_dot_traj) in enumerate(trajectories):
    axes[1, 0].plot(x_traj, x_dot_traj, color=colors[i], linewidth=1.5, alpha=0.7)
    axes[1, 0].plot(x_traj[0], x_dot_traj[0], 'o', color=colors[i], markersize=5)

axes[1, 0].set_xlabel('Position (m)', fontsize=12)
axes[1, 0].set_ylabel('Velocity (m/s)', fontsize=12)
axes[1, 0].set_title('Phase Portrait: Multiple Trajectories', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Energy contours
x_range = np.linspace(-2, 2, 100)
x_dot_range = np.linspace(-5, 5, 100)
X, X_DOT = np.meshgrid(x_range, x_dot_range)

m, k = 1.0, 10.0
E_contour = 0.5 * m * X_DOT**2 + 0.5 * k * X**2

contour = axes[1, 1].contour(X, X_DOT, E_contour, levels=15, alpha=0.6)
axes[1, 1].clabel(contour, inline=True, fontsize=8)
axes[1, 1].plot(x1, x_dot1, 'b-', linewidth=2, label='Trajectory')
axes[1, 1].set_xlabel('Position (m)', fontsize=12)
axes[1, 1].set_ylabel('Velocity (m/s)', fontsize=12)
axes[1, 1].set_title('Phase Space with Energy Contours', fontsize=13)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('16_phase_space.png', dpi=150)
print("\n✅ Saved: 16_phase_space.png")

# ============================================================================
# Plot 2: 3D phase space (for 2D systems)
# ============================================================================

# For coupled oscillators
compiler3 = PhysicsCompiler()

dsl3 = """
\\system{coupled_phase}

\\defvar{x1}{Position 1}{m}
\\defvar{x2}{Position 2}{m}

\\parameter{m}{1.0}{kg}
\\parameter{k}{10.0}{N/m}
\\parameter{k12}{5.0}{N/m}

\\lagrangian{
    \\frac{1}{2} * m * (\\dot{x1}^2 + \\dot{x2}^2) -
    \\frac{1}{2} * k * (x1^2 + x2^2) -
    \\frac{1}{2} * k12 * (x2 - x1)^2
}

\\initial{x1=1.0, x2=0.0, x1_dot=0.0, x2_dot=0.0}
"""

result3 = compiler3.compile_dsl(dsl3)
solution3 = compiler3.simulate(t_span=(0, 10), num_points=1000)

x1_3 = solution3['y'][0]
x2_3 = solution3['y'][2]
x1_dot3 = solution3['y'][1]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x1_3, x2_3, x1_dot3, 'b-', linewidth=2, alpha=0.7)
ax.set_xlabel('x₁ (m)', fontsize=12)
ax.set_ylabel('x₂ (m)', fontsize=12)
ax.set_zlabel('ẋ₁ (m/s)', fontsize=12)
ax.set_title('3D Phase Space: Coupled Oscillators', fontsize=14)

plt.savefig('16_phase_space_3d.png', dpi=150)
print("✅ Saved: 16_phase_space_3d.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Phase space: (position, momentum) or (q, p)")
print("2. Each point = complete state of system")
print("3. Trajectories show evolution in time")
print("4. Closed orbits = periodic motion")
print("5. Energy contours = constant energy surfaces")
print("6. Phase portraits show multiple trajectories")
print("7. Fixed points = equilibria (where velocity = 0)")
print("="*60)

plt.show()
