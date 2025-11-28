"""
Tutorial 03: Simple Pendulum

A pendulum is a mass suspended from a pivot point, swinging under gravity.
For small angles, it's a harmonic oscillator. For large angles, it's nonlinear!

Physics:
- Mass m on a string of length L
- Angle θ from vertical
- Lagrangian: L = (1/2)mL²θ̇² - mgL(1 - cos θ)
- Small angle approximation: sin θ ≈ θ (harmonic oscillator)
- Large angles: nonlinear, period depends on amplitude

We'll compare small and large angle behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Small Angle Pendulum (≈ Harmonic Oscillator)
# ============================================================================

print("="*60)
print("SMALL ANGLE PENDULUM (θ₀ = 0.1 rad ≈ 5.7°)")
print("="*60)

compiler1 = PhysicsCompiler()

dsl_small = """
\\system{small_angle_pendulum}

\\var{theta}{Angle}{rad}

\\parameter{m}{1.0}{kg}
\\parameter{L}{1.0}{m}
\\parameter{g}{9.81}{m/s^2}

\\lagrangian{\\frac{1}{2} * m * L^2 * \\dot{theta}^2 - m * g * L * (1 - \\cos{theta})}

\\initial{theta=0.1, theta_dot=0.0}
"""

result1 = compiler1.compile_dsl(dsl_small)
if not result1['success']:
    print(f"❌ Compilation failed: {result1.get('error')}")
    exit(1)

# Calculate period for small angles: T = 2π√(L/g)
L = 1.0
g = 9.81
T_small = 2 * np.pi * np.sqrt(L / g)
print(f"Expected period (small angle): {T_small:.3f} s")

solution1 = compiler1.simulate(t_span=(0, 3 * T_small), num_points=500)

# ============================================================================
# Large Angle Pendulum (Nonlinear)
# ============================================================================

print("\n" + "="*60)
print("LARGE ANGLE PENDULUM (θ₀ = 1.5 rad ≈ 86°)")
print("="*60)

compiler2 = PhysicsCompiler()

dsl_large = """
\\system{large_angle_pendulum}

\\var{theta}{Angle}{rad}

\\parameter{m}{1.0}{kg}
\\parameter{L}{1.0}{m}
\\parameter{g}{9.81}{m/s^2}

\\lagrangian{\\frac{1}{2} * m * L^2 * \\dot{theta}^2 - m * g * L * (1 - \\cos{theta})}

\\initial{theta=1.5, theta_dot=0.0}
"""

result2 = compiler2.compile_dsl(dsl_large)
if not result2['success']:
    print(f"❌ Compilation failed: {result2.get('error')}")
    exit(1)

solution2 = compiler2.simulate(t_span=(0, 3 * T_small), num_points=500)

# ============================================================================
# Extract and compare results
# ============================================================================

t1 = solution1['t']
theta1 = solution1['y'][0]
theta_dot1 = solution1['y'][1]

t2 = solution2['t']
theta2 = solution2['y'][0]
theta_dot2 = solution2['y'][1]

# Convert to Cartesian coordinates for visualization
def to_cartesian(theta, L):
    x = L * np.sin(theta)
    y = -L * np.cos(theta)  # Negative because y increases downward
    return x, y

x1, y1 = to_cartesian(theta1, L)
x2, y2 = to_cartesian(theta2, L)

# ============================================================================
# Plot comparison
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Time series comparison
ax1 = plt.subplot(3, 2, 1)
ax1.plot(t1, theta1, 'b-', linewidth=2, label='Small angle (0.1 rad)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (rad)')
ax1.set_title('Small Angle: θ vs Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = plt.subplot(3, 2, 2)
ax2.plot(t2, theta2, 'r-', linewidth=2, label='Large angle (1.5 rad)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (rad)')
ax2.set_title('Large Angle: θ vs Time')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Phase space
ax3 = plt.subplot(3, 2, 3)
ax3.plot(theta1, theta_dot1, 'b-', linewidth=1.5)
ax3.set_xlabel('Angle (rad)')
ax3.set_ylabel('Angular Velocity (rad/s)')
ax3.set_title('Small Angle: Phase Space')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 2, 4)
ax4.plot(theta2, theta_dot2, 'r-', linewidth=1.5)
ax4.set_xlabel('Angle (rad)')
ax4.set_ylabel('Angular Velocity (rad/s)')
ax4.set_title('Large Angle: Phase Space')
ax4.grid(True, alpha=0.3)

# Trajectory visualization
ax5 = plt.subplot(3, 2, 5)
# Plot pendulum rod
for i in range(0, len(x1), 10):
    ax5.plot([0, x1[i]], [0, y1[i]], 'b-', alpha=0.1, linewidth=0.5)
ax5.plot(x1, y1, 'b-', linewidth=2, label='Trajectory')
ax5.plot(0, 0, 'ko', markersize=10, label='Pivot')
ax5.set_xlabel('x (m)')
ax5.set_ylabel('y (m)')
ax5.set_title('Small Angle: Trajectory')
ax5.set_aspect('equal')
ax5.grid(True, alpha=0.3)
ax5.legend()

ax6 = plt.subplot(3, 2, 6)
for i in range(0, len(x2), 10):
    ax6.plot([0, x2[i]], [0, y2[i]], 'r-', alpha=0.1, linewidth=0.5)
ax6.plot(x2, y2, 'r-', linewidth=2, label='Trajectory')
ax6.plot(0, 0, 'ko', markersize=10, label='Pivot')
ax6.set_xlabel('x (m)')
ax6.set_ylabel('y (m)')
ax6.set_title('Large Angle: Trajectory')
ax6.set_aspect('equal')
ax6.grid(True, alpha=0.3)
ax6.legend()

plt.tight_layout()
plt.savefig('03_pendulum_comparison.png', dpi=150)
print("\n✅ Plot saved as '03_pendulum_comparison.png'")

# ============================================================================
# Period analysis
# ============================================================================

# Find periods by detecting zero crossings
def find_period(t, theta):
    """Find period by detecting when theta returns to initial value"""
    # Find first zero crossing after initial
    initial = theta[0]
    crossings = []
    for i in range(1, len(theta)):
        if (theta[i-1] >= initial and theta[i] < initial) or \
           (theta[i-1] <= initial and theta[i] > initial):
            crossings.append(t[i])
    if len(crossings) >= 2:
        return crossings[1] - crossings[0]
    return None

period1 = find_period(t1, theta1)
period2 = find_period(t2, theta2)

print("\nPeriod Analysis:")
print(f"   Small angle period: {period1:.3f} s (expected: {T_small:.3f} s)")
if period2:
    print(f"   Large angle period: {period2:.3f} s")
    print(f"   Period increase: {(period2/period1 - 1)*100:.1f}%")
    print("\n   💡 Large angles have LONGER periods!")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("1. Small angles: motion is sinusoidal (harmonic)")
print("2. Large angles: motion is non-sinusoidal (nonlinear)")
print("3. Period increases with amplitude for large angles")
print("4. Phase space for small angles: ellipse")
print("5. Phase space for large angles: more complex shape")
print("6. Energy is conserved in both cases")
print("="*60)

plt.show()

