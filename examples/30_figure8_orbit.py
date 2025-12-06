"""
Tutorial 30: Figure-8 Three-Body Orbit

The famous Chenciner-Montgomery periodic solution to the three-body problem.

Physics:
- Three equal masses move in a figure-8 pattern
- All three bodies follow the SAME path, just phase-shifted
- Discovered in 2000 by Chenciner & Montgomery
- One of the most beautiful periodic solutions in celestial mechanics
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define Figure-8 system
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{figure8_orbit}

\defvar{x1}{Position of body 1}{m} \defvar{y1}{Position of body 1}{m}
\defvar{x2}{Position of body 2}{m} \defvar{y2}{Position of body 2}{m}
\defvar{x3}{Position of body 3}{m} \defvar{y3}{Position of body 3}{m}

\parameter{m}{1.0}{kg}
\parameter{G}{1.0}{N*m^2/kg^2}

\lagrangian{
    \frac{1}{2} * m * (\dot{x1}^2 + \dot{y1}^2 + \dot{x2}^2 + \dot{y2}^2 + \dot{x3}^2 + \dot{y3}^2)
    + G*m^2/\sqrt{(x1-x2)^2 + (y1-y2)^2}
    + G*m^2/\sqrt{(x2-x3)^2 + (y2-y3)^2}
    + G*m^2/\sqrt{(x1-x3)^2 + (y1-y3)^2}
}

\initial{x1=0, y1=0, x1_dot=0, y1_dot=0, x2=0, y2=0, x2_dot=0, y2_dot=0, x3=0, y3=0, x3_dot=0, y3_dot=0}
"""

# ============================================================================
# Compile the system
# ============================================================================

print("Compiling Figure-8 three-body system...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"[FAIL] Compilation failed: {result.get('error')}")
    exit(1)

print("[OK] Compilation successful!")

# ============================================================================
# Set Chenciner-Montgomery initial conditions (EXACT values from literature)
# ============================================================================

# These are the precise initial conditions for the figure-8 orbit
# Found by Chenciner & Montgomery (2000)
compiler.simulator.set_initial_conditions({
    'x1':  0.97000436,   'y1': -0.24308753,
    'x2': -0.97000436,   'y2':  0.24308753,
    'x3':  0.0,          'y3':  0.0,
    'x1_dot':  0.4662036850, 'y1_dot':  0.4323657300,
    'x2_dot':  0.4662036850, 'y2_dot':  0.4323657300,
    'x3_dot': -0.93240737,   'y3_dot': -0.86473146
})

# ============================================================================
# Simulate for exactly one period
# ============================================================================

T_period = 6.32591398  # Exact period of the figure-8 orbit

print(f"\nSimulating for one period: T = {T_period:.6f}")
solution = compiler.simulate(t_span=(0, T_period), num_points=2000)

if not solution['success']:
    print("[FAIL] Simulation failed!")
    exit(1)

print("[OK] Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
y = solution['y']

x1, y1 = y[0], y[2]
x2, y2 = y[4], y[6]
x3, y3 = y[8], y[10]

# ============================================================================
# Check periodicity
# ============================================================================

state_initial = y[:, 0]
state_final = y[:, -1]
periodicity_error = np.linalg.norm(state_final - state_initial)

print("\n" + "="*60)
print("PERIODICITY CHECK:")
print("="*60)
print(f"   Error after one period: {periodicity_error:.6e}")
print(f"   Status: {'[OK] CLOSED ORBIT' if periodicity_error < 0.01 else '[WARN] DRIFT DETECTED'}")
print("="*60)

# ============================================================================
# Plot the figure-8 orbit
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# All three trajectories
ax = axes[0]
ax.plot(x1, y1, 'r-', linewidth=2, label='Body 1', alpha=0.8)
ax.plot(x2, y2, 'b-', linewidth=2, label='Body 2', alpha=0.8)
ax.plot(x3, y3, 'g-', linewidth=2, label='Body 3', alpha=0.8)

# Mark positions at t=0
ax.plot(x1[0], y1[0], 'ro', markersize=15)
ax.plot(x2[0], y2[0], 'bo', markersize=15)
ax.plot(x3[0], y3[0], 'go', markersize=15)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Figure-8 Three-Body Orbit', fontsize=14)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Animation-style: show positions at different times
ax2 = axes[1]

# Draw the path (all bodies follow same curve)
ax2.plot(x1, y1, 'k-', linewidth=1, alpha=0.3)

# Show positions at several time steps
n_frames = 20
indices = np.linspace(0, len(t)-1, n_frames, dtype=int)

for i, idx in enumerate(indices):
    alpha = 0.3 + 0.7 * (i / n_frames)
    size = 8 + 7 * (i / n_frames)
    ax2.plot(x1[idx], y1[idx], 'ro', markersize=size, alpha=alpha)
    ax2.plot(x2[idx], y2[idx], 'bo', markersize=size, alpha=alpha)
    ax2.plot(x3[idx], y3[idx], 'go', markersize=size, alpha=alpha)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Body Positions Over One Period', fontsize=14)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('30_figure8_orbit.png', dpi=150)
print("\n[OK] Saved: 30_figure8_orbit.png")

# ============================================================================
# Energy and angular momentum conservation
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calculate energy
m, G = 1.0, 1.0
vx1, vy1 = y[1], y[3]
vx2, vy2 = y[5], y[7]
vx3, vy3 = y[9], y[11]

KE = 0.5 * m * (vx1**2 + vy1**2 + vx2**2 + vy2**2 + vx3**2 + vy3**2)

r12 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
r23 = np.sqrt((x2-x3)**2 + (y2-y3)**2)
r13 = np.sqrt((x1-x3)**2 + (y1-y3)**2)

PE = -G * m**2 * (1/r12 + 1/r23 + 1/r13)
E_total = KE + PE

axes[0].plot(t, E_total, 'k-', linewidth=1.5)
axes[0].set_xlabel('Time', fontsize=12)
axes[0].set_ylabel('Total Energy', fontsize=12)
axes[0].set_title('Energy Conservation', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Angular momentum
Lz = m * (x1*vy1 - y1*vx1 + x2*vy2 - y2*vx2 + x3*vy3 - y3*vx3)

axes[1].plot(t, Lz, 'b-', linewidth=1.5)
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Angular Momentum (z)', fontsize=12)
axes[1].set_title('Angular Momentum Conservation', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('30_figure8_conservation.png', dpi=150)
print("[OK] Saved: 30_figure8_conservation.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. All three bodies follow the SAME figure-8 curve!")
print("2. They are phase-shifted by T/3 along the path")
print("3. Discovered by Chenciner & Montgomery in 2000")
print("4. Requires very precise initial conditions")
print("5. Total angular momentum is zero (by symmetry)")
print("6. One of the most beautiful solutions in celestial mechanics")
print("="*60)

plt.show()


