"""
Tutorial 14: Chaotic Systems - The Duffing Oscillator

The Duffing oscillator is a classic example of a chaotic system.

Physics:
- Nonlinear spring: F = -kx - betax³ (hardening or softening)
- With damping and periodic forcing
- Shows period doubling route to chaos
- Strange attractors in phase space
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Define Duffing oscillator
# ============================================================================

compiler = PhysicsCompiler()

# Duffing equation: ẍ + δẋ + alphax + betax³ = γcos(omegat)
# We'll model the conservative part with Lagrangian
# The driven-damped version needs external forcing

dsl_code = r"""
\system{duffing_oscillator}

\defvar{x}{Displacement}{m}

\parameter{m}{1.0}{kg}
\parameter{alpha}{1.0}{N/m}
\parameter{beta}{1.0}{N/m^3}

\lagrangian{
    \frac{1}{2} * m * \dot{x}^2 
    + \frac{1}{2} * alpha * x^2 
    - \frac{1}{4} * beta * x^4
}

\initial{x=0.5, x_dot=0.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling Duffing oscillator...")
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
x = solution['y'][0]
x_dot = solution['y'][1]

# ============================================================================
# Plot 1: Time series and phase space
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series
axes[0, 0].plot(t, x, 'b-', linewidth=1)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('x (m)')
axes[0, 0].set_title('Duffing Oscillator: Displacement vs Time')
axes[0, 0].grid(True, alpha=0.3)

# Phase portrait
axes[0, 1].plot(x, x_dot, 'b-', linewidth=0.5, alpha=0.7)
axes[0, 1].set_xlabel('x (m)')
axes[0, 1].set_ylabel('ẋ (m/s)')
axes[0, 1].set_title('Phase Portrait (Double-Well Potential)')
axes[0, 1].grid(True, alpha=0.3)

# Potential energy landscape
x_range = np.linspace(-2, 2, 200)
alpha, beta = -1.0, 1.0
V = 0.5 * alpha * x_range**2 + 0.25 * beta * x_range**4

axes[1, 0].plot(x_range, V, 'r-', linewidth=2)
axes[1, 0].set_xlabel('x (m)')
axes[1, 0].set_ylabel('V(x) (J)')
axes[1, 0].set_title('Double-Well Potential')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(0, color='k', linestyle='-', linewidth=0.5)

# Mark minima
x_min = np.sqrt(-alpha/beta)
axes[1, 0].axvline(x_min, color='g', linestyle='--', alpha=0.5)
axes[1, 0].axvline(-x_min, color='g', linestyle='--', alpha=0.5)

# Sensitivity to initial conditions
axes[1, 1].set_title('Sensitivity to Initial Conditions')

# Run with slightly different initial conditions
colors = ['blue', 'red', 'green']
ics = [0.5, 0.501, 0.502]

for ic, color in zip(ics, colors):
    compiler2 = PhysicsCompiler()
    dsl_code2 = dsl_code.replace('x=0.5', f'x={ic}')
    compiler2.compile_dsl(dsl_code2)
    sol2 = compiler2.simulate(t_span=(0, 50), num_points=5000)
    if sol2['success']:
        axes[1, 1].plot(sol2['t'], sol2['y'][0], color=color, 
                       linewidth=1, alpha=0.7, label=f'x_0={ic}')

axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('x (m)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('14_chaotic_systems.png', dpi=150)
print("\n[OK] Saved: 14_chaotic_systems.png")

# ============================================================================
# Plot 2: Energy analysis
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate energies
m = 1.0
KE = 0.5 * m * x_dot**2
PE = 0.5 * alpha * x**2 + 0.25 * beta * x**4
E_total = KE + PE

ax.plot(t, KE, 'b-', linewidth=1.5, alpha=0.7, label='Kinetic')
ax.plot(t, PE, 'r-', linewidth=1.5, alpha=0.7, label='Potential')
ax.plot(t, E_total, 'k-', linewidth=2, label='Total')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Energy (J)', fontsize=12)
ax.set_title('Energy Conservation in Duffing Oscillator', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('14_duffing_energy.png', dpi=150)
print("[OK] Saved: 14_duffing_energy.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Duffing oscillator has a double-well potential")
print("2. alpha < 0: unstable equilibrium at origin")
print("3. Two stable equilibria at x = ±sqrt(-alpha/beta)")
print("4. Trajectories can hop between wells at high energy")
print("5. With damping + forcing → chaotic strange attractors")
print("6. Classic example of nonlinear dynamics")
print("="*60)

plt.show()


