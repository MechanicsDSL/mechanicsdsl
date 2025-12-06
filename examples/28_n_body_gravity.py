"""
Tutorial 28: N-Body Gravity

Simulating gravitational interactions between multiple bodies.

Physics:
- N masses interacting via Newtonian gravity
- F = G*m_1*m_2/r² for each pair
- Conservation of energy, momentum, and angular momentum
- Can show stable orbits, chaos, or ejections
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define 4-body gravitational system
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{four_body_gravity}

\defvar{x1}{Position of body 1}{m} \defvar{y1}{Position of body 1}{m}
\defvar{x2}{Position of body 2}{m} \defvar{y2}{Position of body 2}{m}
\defvar{x3}{Position of body 3}{m} \defvar{y3}{Position of body 3}{m}
\defvar{x4}{Position of body 4}{m} \defvar{y4}{Position of body 4}{m}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{m3}{1.0}{kg}
\parameter{m4}{0.5}{kg}
\parameter{G}{1.0}{N*m^2/kg^2}

\lagrangian{
    \frac{1}{2} * m1 * (\dot{x1}^2 + \dot{y1}^2) +
    \frac{1}{2} * m2 * (\dot{x2}^2 + \dot{y2}^2) +
    \frac{1}{2} * m3 * (\dot{x3}^2 + \dot{y3}^2) +
    \frac{1}{2} * m4 * (\dot{x4}^2 + \dot{y4}^2) +
    G * m1 * m2 / \sqrt{(x1-x2)^2 + (y1-y2)^2 + 0.01} +
    G * m1 * m3 / \sqrt{(x1-x3)^2 + (y1-y3)^2 + 0.01} +
    G * m1 * m4 / \sqrt{(x1-x4)^2 + (y1-y4)^2 + 0.01} +
    G * m2 * m3 / \sqrt{(x2-x3)^2 + (y2-y3)^2 + 0.01} +
    G * m2 * m4 / \sqrt{(x2-x4)^2 + (y2-y4)^2 + 0.01} +
    G * m3 * m4 / \sqrt{(x3-x4)^2 + (y3-y4)^2 + 0.01}
}

\initial{
    x1=1.0, y1=0.0, x1_dot=0.0, y1_dot=0.5,
    x2=-1.0, y2=0.0, x2_dot=0.0, y2_dot=-0.5,
    x3=0.0, y3=1.0, x3_dot=-0.5, y3_dot=0.0,
    x4=0.0, y4=-1.0, x4_dot=0.5, y4_dot=0.0
}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling 4-body gravitational system...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    exit(1)

print("[OK] Compilation successful!")

solution = compiler.simulate(t_span=(0, 50), num_points=5000)

if not solution['success']:
    print("[FAIL] Simulation failed!")
    exit(1)

print("[OK] Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
y = solution['y']

# Extract positions (every other state is position, velocities in between)
x1, y1 = y[0], y[2]
x2, y2 = y[4], y[6]
x3, y3 = y[8], y[10]
x4, y4 = y[12], y[14]

# ============================================================================
# Plot trajectories
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
labels = ['Body 1', 'Body 2', 'Body 3', 'Body 4']

# Trajectories
for x, y_coord, color, label in zip([x1, x2, x3, x4], [y1, y2, y3, y4], colors, labels):
    axes[0].plot(x, y_coord, '-', linewidth=1, alpha=0.7, color=color, label=label)
    axes[0].plot(x[0], y_coord[0], 'o', markersize=10, color=color)
    axes[0].plot(x[-1], y_coord[-1], 's', markersize=8, color=color)

axes[0].set_xlabel('x (m)', fontsize=12)
axes[0].set_ylabel('y (m)', fontsize=12)
axes[0].set_title('4-Body Gravitational Trajectories', fontsize=14)
axes[0].set_aspect('equal')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Center of mass
m1, m2, m3, m4 = 1.0, 1.0, 1.0, 0.5
M_total = m1 + m2 + m3 + m4
x_cm = (m1*x1 + m2*x2 + m3*x3 + m4*x4) / M_total
y_cm = (m1*y1 + m2*y2 + m3*y3 + m4*y4) / M_total

axes[1].plot(t, x_cm, 'b-', label='x_cm', linewidth=1.5)
axes[1].plot(t, y_cm, 'r-', label='y_cm', linewidth=1.5)
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Position (m)', fontsize=12)
axes[1].set_title('Center of Mass (should be constant)', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('28_n_body_gravity.png', dpi=150)
print("\n[OK] Saved: 28_n_body_gravity.png")

# ============================================================================
# Check approximate conservation laws
# ============================================================================

print("\n" + "="*60)
print("CONSERVATION CHECK:")
print("="*60)

# Momentum
vx1, vy1 = y[1], y[3]
vx2, vy2 = y[5], y[7]
vx3, vy3 = y[9], y[11]
vx4, vy4 = y[13], y[15]

px = m1*vx1 + m2*vx2 + m3*vx3 + m4*vx4
py = m1*vy1 + m2*vy2 + m3*vy3 + m4*vy4

print(f"   x-momentum: initial = {px[0]:.6f}, final = {px[-1]:.6f}")
print(f"   y-momentum: initial = {py[0]:.6f}, final = {py[-1]:.6f}")
print(f"   Center of mass drift: Δx = {x_cm[-1]-x_cm[0]:.6f}, Δy = {y_cm[-1]-y_cm[0]:.6f}")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. N-body problem has no general analytical solution (N>2)")
print("2. Adding softening (0.01) prevents singularities at r=0")
print("3. Total momentum and center of mass are conserved")
print("4. Energy is conserved (symplectic integrator helps)")
print("5. 3+ bodies often leads to chaotic dynamics")
print("6. Important for: galaxies, star systems, molecule dynamics")
print("="*60)

plt.show()


