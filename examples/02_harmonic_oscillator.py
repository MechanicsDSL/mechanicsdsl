"""
Tutorial 02: Harmonic Oscillator

The harmonic oscillator is one of the most important systems in physics.
It appears everywhere: springs, pendulums (small angles), molecular vibrations, etc.

Physics:
- A mass m attached to a spring with spring constant k
- Restoring force: F = -kx
- Lagrangian: L = (1/2)m·ẋ² - (1/2)kx²
- Natural frequency: ω = √(k/m)

We'll simulate this and see the characteristic sinusoidal motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Create compiler
# ============================================================================

compiler = PhysicsCompiler()

# ============================================================================
# Define the harmonic oscillator system
# ============================================================================

dsl_code = r"""
\system{harmonic_oscillator}

\defvar{x}{Position}{m}

\parameter{m}{1.0}{kg}
\parameter{k}{10.0}{N/m}

\lagrangian{\frac{1}{2} * m * \dot{x}^2 - \frac{1}{2} * k * x^2}

\initial{x=1.0, x_dot=0.0}
"""

# ============================================================================
# Compile and simulate
# ============================================================================

print("Compiling harmonic oscillator...")
result = compiler.compile_dsl(dsl_code)

if not result['success']:
    print(f"❌ Compilation failed: {result.get('error')}")
    exit(1)

print("✅ Compilation successful!")

# Calculate natural frequency for reference
m = 1.0
k = 10.0
omega = np.sqrt(k / m)
period = 2 * np.pi / omega

print(f"\nSystem parameters:")
print(f"   Mass: {m} kg")
print(f"   Spring constant: {k} N/m")
print(f"   Natural frequency: ω = {omega:.3f} rad/s")
print(f"   Period: T = {period:.3f} s")

# Simulate for 3 periods
t_span = (0, 3 * period)
print(f"\nSimulating for {t_span[1]:.2f} seconds ({3} periods)...")

solution = compiler.simulate(t_span=t_span, num_points=500)

if not solution['success']:
    print(f"❌ Simulation failed!")
    exit(1)

print("✅ Simulation successful!")

# ============================================================================
# Extract results
# ============================================================================

t = solution['t']
x = solution['y'][0]
x_dot = solution['y'][1]

# Analytical solution for comparison
# x(t) = A·cos(ωt + φ) where A = initial position, φ = 0
A = 1.0  # Initial amplitude
x_analytical = A * np.cos(omega * t)

# ============================================================================
# Plot results
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Position vs time
axes[0].plot(t, x, 'b-', linewidth=2, label='Numerical')
axes[0].plot(t, x_analytical, 'r--', linewidth=1.5, alpha=0.7, label='Analytical')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Position (m)')
axes[0].set_title('Harmonic Oscillator: Position vs Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Velocity vs time
axes[1].plot(t, x_dot, 'g-', linewidth=2)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_title('Harmonic Oscillator: Velocity vs Time')
axes[1].grid(True, alpha=0.3)

# Phase space (position vs velocity)
axes[2].plot(x, x_dot, 'purple', linewidth=1.5)
axes[2].set_xlabel('Position (m)')
axes[2].set_ylabel('Velocity (m/s)')
axes[2].set_title('Phase Space Trajectory')
axes[2].grid(True, alpha=0.3)
axes[2].axis('equal')

plt.tight_layout()
plt.savefig('02_harmonic_oscillator.png', dpi=150)
print("\n✅ Plot saved as '02_harmonic_oscillator.png'")

# ============================================================================
# Energy analysis
# ============================================================================

# Calculate kinetic and potential energy
T = 0.5 * m * x_dot**2  # Kinetic energy
V = 0.5 * k * x**2      # Potential energy
E_total = T + V         # Total energy (should be conserved)

print("\nEnergy Analysis:")
print(f"   Initial total energy: {E_total[0]:.6f} J")
print(f"   Final total energy: {E_total[-1]:.6f} J")
print(f"   Energy conservation error: {abs(E_total[-1] - E_total[0]):.2e} J")

# Plot energy
plt.figure(figsize=(10, 6))
plt.plot(t, T, 'b-', label='Kinetic Energy', linewidth=2)
plt.plot(t, V, 'r-', label='Potential Energy', linewidth=2)
plt.plot(t, E_total, 'g--', label='Total Energy', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy Conservation in Harmonic Oscillator')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('02_harmonic_oscillator_energy.png', dpi=150)
print("✅ Energy plot saved as '02_harmonic_oscillator_energy.png'")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Motion is sinusoidal: x(t) = A·cos(ωt)")
print("2. Energy oscillates between kinetic and potential")
print("3. Total energy is conserved (constant)")
print("4. Phase space trajectory is an ellipse")
print("5. Period depends only on m and k: T = 2π√(m/k)")
print("="*60)

plt.show()
