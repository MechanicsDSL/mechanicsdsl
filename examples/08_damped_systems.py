"""
Tutorial 08: Damped Systems

Real systems lose energy! This tutorial covers:
1. Damped harmonic oscillator
2. Overdamped, underdamped, and critically damped motion
3. Energy dissipation

Physics:
- Damping force: F_damp = -b·ẋ (proportional to velocity)
- Can be added via \\damping{} command
- Three regimes based on damping coefficient b
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Underdamped System (b < 2sqrt(mk))
# ============================================================================

print("="*60)
print("UNDERDAMPED SYSTEM (oscillatory decay)")
print("="*60)

compiler1 = PhysicsCompiler()

dsl_underdamped = """
\\system{underdamped_oscillator}

\\var{x}{Position}{m}

\\parameter{m}{1.0}{kg}
\\parameter{k}{10.0}{N/m}
\\parameter{b}{1.0}{N·s/m}

\\lagrangian{\\frac{1}{2} * m * \\dot{x}^2 - \\frac{1}{2} * k * x^2}

\\damping{-b * \\dot{x}}

\\initial{x=1.0, x_dot=0.0}
"""

result1 = compiler1.compile_dsl(dsl_underdamped)
if not result1['success']:
    print(f"[FAIL] Compilation failed: {result1.get('error')}")
    exit(1)

solution1 = compiler1.simulate(t_span=(0, 10), num_points=1000)

# ============================================================================
# Critically Damped System (b = 2sqrt(mk))
# ============================================================================

print("\n" + "="*60)
print("CRITICALLY DAMPED SYSTEM (fastest decay)")
print("="*60)

compiler2 = PhysicsCompiler()

m, k = 1.0, 10.0
b_critical = 2 * np.sqrt(m * k)

dsl_critical = f"""
\\system{critically_damped_oscillator}

\\var{{x}}{{Position}}{{m}}

\\parameter{{m}}{{1.0}}{{kg}}
\\parameter{{k}}{{10.0}}{{N/m}}
\\parameter{{b}}{{b_critical:.3f}}{{N·s/m}}

\\lagrangian{{\\frac{{1}}{{2}} * m * \\dot{{x}}^2 - \\frac{{1}}{{2}} * k * x^2}}

\\damping{{-b * \\dot{{x}}}}

\\initial{{x=1.0, x_dot=0.0}}
"""

result2 = compiler2.compile_dsl(dsl_critical)
if not result2['success']:
    print(f"[FAIL] Compilation failed: {result2.get('error')}")
    exit(1)

print(f"Critical damping coefficient: b = {b_critical:.3f} N·s/m")

solution2 = compiler2.simulate(t_span=(0, 10), num_points=1000)

# ============================================================================
# Overdamped System (b > 2sqrt(mk))
# ============================================================================

print("\n" + "="*60)
print("OVERDAMPED SYSTEM (slow decay, no oscillation)")
print("="*60)

compiler3 = PhysicsCompiler()

b_over = 3 * np.sqrt(m * k)

dsl_overdamped = f"""
\\system{overdamped_oscillator}

\\var{{x}}{{Position}}{{m}}

\\parameter{{m}}{{1.0}}{{kg}}
\\parameter{{k}}{{10.0}}{{N/m}}
\\parameter{{b}}{{b_over:.3f}}{{N·s/m}}

\\lagrangian{{\\frac{{1}}{{2}} * m * \\dot{{x}}^2 - \\frac{{1}}{{2}} * k * x^2}}

\\damping{{-b * \\dot{{x}}}}

\\initial{{x=1.0, x_dot=0.0}}
"""

result3 = compiler3.compile_dsl(dsl_overdamped)
if not result3['success']:
    print(f"[FAIL] Compilation failed: {result3.get('error')}")
    exit(1)

print(f"Overdamping coefficient: b = {b_over:.3f} N·s/m")

solution3 = compiler3.simulate(t_span=(0, 10), num_points=1000)

# ============================================================================
# Extract results
# ============================================================================

t1 = solution1['t']
x1 = solution1['y'][0]
x_dot1 = solution1['y'][1]

t2 = solution2['t']
x2 = solution2['y'][0]
x_dot2 = solution2['y'][1]

t3 = solution3['t']
x3 = solution3['y'][0]
x_dot3 = solution3['y'][1]

# ============================================================================
# Plot comparison
# ============================================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Underdamped
axes[0, 0].plot(t1, x1, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Position (m)')
axes[0, 0].set_title('Underdamped: Position vs Time')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x1, x_dot1, 'b-', linewidth=1.5)
axes[0, 1].set_xlabel('Position (m)')
axes[0, 1].set_ylabel('Velocity (m/s)')
axes[0, 1].set_title('Underdamped: Phase Space')
axes[0, 1].grid(True, alpha=0.3)

# Critically damped
axes[1, 0].plot(t2, x2, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Position (m)')
axes[1, 0].set_title('Critically Damped: Position vs Time')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x2, x_dot2, 'r-', linewidth=1.5)
axes[1, 1].set_xlabel('Position (m)')
axes[1, 1].set_ylabel('Velocity (m/s)')
axes[1, 1].set_title('Critically Damped: Phase Space')
axes[1, 1].grid(True, alpha=0.3)

# Overdamped
axes[2, 0].plot(t3, x3, 'g-', linewidth=2)
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Position (m)')
axes[2, 0].set_title('Overdamped: Position vs Time')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(x3, x_dot3, 'g-', linewidth=1.5)
axes[2, 1].set_xlabel('Position (m)')
axes[2, 1].set_ylabel('Velocity (m/s)')
axes[2, 1].set_title('Overdamped: Phase Space')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_damping_comparison.png', dpi=150)
print("\n[OK] Saved: 08_damping_comparison.png")

# ============================================================================
# Energy dissipation
# ============================================================================

m, k = 1.0, 10.0
b1, b2, b3 = 1.0, b_critical, b_over

T1 = 0.5 * m * x_dot1**2
V1 = 0.5 * k * x1**2
E1 = T1 + V1

T2 = 0.5 * m * x_dot2**2
V2 = 0.5 * k * x2**2
E2 = T2 + V2

T3 = 0.5 * m * x_dot3**2
V3 = 0.5 * k * x3**2
E3 = T3 + V3

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(t1, E1, 'b-', linewidth=2, label='Underdamped', alpha=0.8)
ax.plot(t2, E2, 'r-', linewidth=2, label='Critically Damped', alpha=0.8)
ax.plot(t3, E3, 'g-', linewidth=2, label='Overdamped', alpha=0.8)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Total Energy (J)', fontsize=12)
ax.set_title('Energy Dissipation in Damped Systems', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')  # Log scale to see decay better

plt.tight_layout()
plt.savefig('08_energy_dissipation.png', dpi=150)
print("[OK] Saved: 08_energy_dissipation.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. UNDERDAMPED: Oscillates while decaying (b < 2sqrt(mk))")
print("2. CRITICALLY DAMPED: Fastest return to equilibrium (b = 2sqrt(mk))")
print("3. OVERDAMPED: Slow decay, no oscillation (b > 2sqrt(mk))")
print("4. Energy always decreases (dissipated by damping)")
print("5. Phase space spirals toward origin")
print("6. Use \\damping{} command to add damping forces")
print("="*60)

plt.show()



