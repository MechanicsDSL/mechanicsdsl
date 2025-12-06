"""
Tutorial 07: 2D Motion (Projectile and Orbits)

This tutorial covers motion in 2D:
1. Projectile motion (under gravity)
2. Circular orbits (Kepler problem)
3. Elliptical orbits

Physics:
- Projectile: x and y coordinates, gravity in y-direction
- Orbits: Central force (gravity), polar coordinates (r, φ)
"""

import numpy as np
import matplotlib.pyplot as plt
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Example 1: Projectile Motion
# ============================================================================

print("="*60)
print("EXAMPLE 1: PROJECTILE MOTION")
print("="*60)

compiler1 = PhysicsCompiler()

dsl_projectile = """
\\system{projectile}

\\var{x}{Horizontal position}{m}
\\var{y}{Vertical position}{m}

\\parameter{m}{1.0}{kg}
\\parameter{g}{9.81}{m/s^2}

\\lagrangian{
    \\frac{1}{2} * m * (\\dot{x}^2 + \\dot{y}^2) - m * g * y
}

\\initial{x=0.0, y=0.0, x_dot=10.0, y_dot=10.0}
"""

result1 = compiler1.compile_dsl(dsl_projectile)
if not result1['success']:
    print(f"[FAIL] Compilation failed: {result1.get('error')}")
    exit(1)

# Calculate flight time analytically
v0x, v0y = 10.0, 10.0
g = 9.81
t_flight = 2 * v0y / g
range_max = v0x * t_flight

print(f"Analytical predictions:")
print(f"   Flight time: {t_flight:.3f} s")
print(f"   Maximum range: {range_max:.3f} m")
print(f"   Maximum height: {v0y**2 / (2*g):.3f} m")

solution1 = compiler1.simulate(t_span=(0, t_flight * 1.2), num_points=500)

x = solution1['y'][0]
y = solution1['y'][1]
x_dot = solution1['y'][2]
y_dot = solution1['y'][3]
t = solution1['t']

# Analytical solution for comparison
x_analytical = v0x * t
y_analytical = v0y * t - 0.5 * g * t**2

# ============================================================================
# Example 2: Circular Orbit (Kepler Problem)
# ============================================================================

print("\n" + "="*60)
print("EXAMPLE 2: CIRCULAR ORBIT")
print("="*60)

compiler2 = PhysicsCompiler()

dsl_orbit = """
\\system{circular_orbit}

\\var{r}{Radial distance}{m}
\\var{phi}{Azimuthal angle}{rad}

\\parameter{m}{1.0}{kg}
\\parameter{M}{1000.0}{kg}
\\parameter{G}{6.674e-11}{m^3/(kg·s^2)}

\\lagrangian{
    \\frac{1}{2} * m * (\\dot{r}^2 + r^2 * \\dot{phi}^2) +
    G * m * M / r
}

\\initial{r=1.0, phi=0.0, r_dot=0.0, phi_dot=1.0}
"""

result2 = compiler2.compile_dsl(dsl_orbit)
if not result2['success']:
    print(f"[FAIL] Compilation failed: {result2.get('error')}")
    exit(1)

# For circular orbit: v = sqrt(GM/r)
G, M, r0 = 6.674e-11, 1000.0, 1.0
v_circular = np.sqrt(G * M / r0)
period = 2 * np.pi * r0 / v_circular

print(f"Orbital parameters:")
print(f"   Orbital velocity: {v_circular:.6f} m/s")
print(f"   Period: {period:.3f} s")

solution2 = compiler2.simulate(t_span=(0, 2 * period), num_points=1000)

r = solution2['y'][0]
phi = solution2['y'][2]
t_orbit = solution2['t']

# Convert to Cartesian
x_orbit = r * np.cos(phi)
y_orbit = r * np.sin(phi)

# ============================================================================
# Plot 1: Projectile Motion
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trajectory
axes[0, 0].plot(x, y, 'b-', linewidth=2, label='Numerical')
axes[0, 0].plot(x_analytical, y_analytical, 'r--', linewidth=1.5, 
                alpha=0.7, label='Analytical')
axes[0, 0].set_xlabel('x (m)')
axes[0, 0].set_ylabel('y (m)')
axes[0, 0].set_title('Projectile Trajectory')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_aspect('equal')

# Position vs time
axes[0, 1].plot(t, x, 'b-', linewidth=2, label='x')
axes[0, 1].plot(t, y, 'r-', linewidth=2, label='y')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Position (m)')
axes[0, 1].set_title('Position vs Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Velocity vs time
axes[1, 0].plot(t, x_dot, 'b-', linewidth=2, label='vₓ')
axes[1, 0].plot(t, y_dot, 'r-', linewidth=2, label='vᵧ')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Velocity (m/s)')
axes[1, 0].set_title('Velocity vs Time')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Energy
m = 1.0
T = 0.5 * m * (x_dot**2 + y_dot**2)
V = m * g * y
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
plt.savefig('07_projectile.png', dpi=150)
print("\n[OK] Saved: 07_projectile.png")

# ============================================================================
# Plot 2: Orbital Motion
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Orbit trajectory
axes[0, 0].plot(x_orbit, y_orbit, 'b-', linewidth=1.5)
axes[0, 0].plot(0, 0, 'ro', markersize=15, label='Central body')
axes[0, 0].set_xlabel('x (m)')
axes[0, 0].set_ylabel('y (m)')
axes[0, 0].set_title('Circular Orbit Trajectory')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_aspect('equal')

# Radial distance vs time
axes[0, 1].plot(t_orbit, r, 'b-', linewidth=2)
axes[0, 1].axhline(r0, color='r', linestyle='--', label=f'Initial r = {r0} m')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Radial Distance (m)')
axes[0, 1].set_title('Radial Distance vs Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Angular position
axes[1, 0].plot(t_orbit, phi, 'b-', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Angle (rad)')
axes[1, 0].set_title('Angular Position vs Time')
axes[1, 0].grid(True, alpha=0.3)

# Energy
r_dot = solution2['y'][1]
phi_dot = solution2['y'][3]
T_orbit = 0.5 * m * (r_dot**2 + r**2 * phi_dot**2)
V_orbit = -G * m * M / r
E_orbit = T_orbit + V_orbit

axes[1, 1].plot(t_orbit, T_orbit, 'b-', label='Kinetic', linewidth=2)
axes[1, 1].plot(t_orbit, V_orbit, 'r-', label='Potential', linewidth=2)
axes[1, 1].plot(t_orbit, E_orbit, 'g--', label='Total', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Energy (J)')
axes[1, 1].set_title('Orbital Energy vs Time')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_orbit.png', dpi=150)
print("[OK] Saved: 07_orbit.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("PROJECTILE:")
print("1. Horizontal motion is uniform (constant velocity)")
print("2. Vertical motion is accelerated (gravity)")
print("3. Trajectory is a parabola")
print("4. Maximum range at 45 deg launch angle")
print("\nORBITS:")
print("1. Circular orbits have constant radius")
print("2. Angular velocity is constant")
print("3. Energy is negative (bound orbit)")
print("4. Period depends on orbital radius")
print("="*60)

plt.show()



