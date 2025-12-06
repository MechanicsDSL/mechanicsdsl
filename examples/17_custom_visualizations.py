"""
Tutorial 17: Custom Visualizations

Advanced matplotlib techniques for physics visualization.

Topics:
- Colormaps based on time or energy
- Animated trajectories
- Multi-panel layouts
- 3D visualization with trails
- Publication-quality figures
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mechanics_dsl import PhysicsCompiler

# ============================================================================
# Generate data: Double pendulum for interesting trajectories
# ============================================================================

compiler = PhysicsCompiler()

dsl_code = r"""
\system{double_pendulum}

\defvar{theta1}{Angle of first pendulum}{rad}
\defvar{theta2}{Angle of second pendulum}{rad}

\parameter{m1}{1.0}{kg}
\parameter{m2}{1.0}{kg}
\parameter{L1}{1.0}{m}
\parameter{L2}{1.0}{m}
\parameter{g}{9.81}{m/s^2}

\lagrangian{
    \frac{1}{2} * (m1 + m2) * L1^2 * \dot{theta1}^2 +
    \frac{1}{2} * m2 * L2^2 * \dot{theta2}^2 +
    m2 * L1 * L2 * \dot{theta1} * \dot{theta2} * \cos{theta1 - theta2} +
    (m1 + m2) * g * L1 * \cos{theta1} +
    m2 * g * L2 * \cos{theta2}
}

\initial{theta1=2.0, theta2=2.0, theta1_dot=0.0, theta2_dot=0.0}
"""

print("Generating trajectory data...")
result = compiler.compile_dsl(dsl_code)
solution = compiler.simulate(t_span=(0, 15), num_points=1500)

t = solution['t']
theta1 = solution['y'][0]
theta2 = solution['y'][2]

L1, L2 = 1.0, 1.0
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

print("[OK] Data generated!")

# ============================================================================
# Technique 1: Color-coded trajectory by time
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 10))

# Create line segments
points = np.array([x2, y2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Color by time
norm = plt.Normalize(t.min(), t.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(t[:-1])
lc.set_linewidth(2)

ax.add_collection(lc)
ax.autoscale()
ax.set_aspect('equal')

cbar = plt.colorbar(lc, ax=ax)
cbar.set_label('Time (s)', fontsize=12)

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Trajectory Colored by Time', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('17_viz_time_colored.png', dpi=150)
print("\n[OK] Saved: 17_viz_time_colored.png")

# ============================================================================
# Technique 2: Multi-panel professional layout
# ============================================================================

fig = plt.figure(figsize=(16, 10))

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Main trajectory
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(x2, y2, 'b-', linewidth=0.8, alpha=0.7)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title('Double Pendulum Trajectory', fontsize=14)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Angles vs time
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(t, np.degrees(theta1), 'b-', label='theta_1', linewidth=1.5)
ax2.plot(t, np.degrees(theta2), 'r-', label='theta_2', linewidth=1.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (deg)')
ax2.set_title('Angles vs Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Phase space theta1
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(theta1, solution['y'][1], 'b-', linewidth=0.5, alpha=0.7)
ax3.set_xlabel('theta_1 (rad)')
ax3.set_ylabel('thetȧ_1 (rad/s)')
ax3.set_title('Phase Space: theta_1')
ax3.grid(True, alpha=0.3)

# Phase space theta2
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(theta2, solution['y'][3], 'r-', linewidth=0.5, alpha=0.7)
ax4.set_xlabel('theta_2 (rad)')
ax4.set_ylabel('thetȧ_2 (rad/s)')
ax4.set_title('Phase Space: theta_2')
ax4.grid(True, alpha=0.3)

# Energy (if we compute it)
ax5 = fig.add_subplot(gs[1, 2])
m1, m2, g = 1.0, 1.0, 9.81
KE = (0.5 * (m1 + m2) * L1**2 * solution['y'][1]**2 + 
      0.5 * m2 * L2**2 * solution['y'][3]**2 +
      m2 * L1 * L2 * solution['y'][1] * solution['y'][3] * np.cos(theta1 - theta2))
PE = -((m1 + m2) * g * L1 * np.cos(theta1) + m2 * g * L2 * np.cos(theta2))
E_total = KE + PE

ax5.plot(t, E_total, 'k-', linewidth=1.5)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Total Energy (J)')
ax5.set_title('Energy Conservation')
ax5.grid(True, alpha=0.3)

plt.savefig('17_viz_multipanel.png', dpi=150)
print("[OK] Saved: 17_viz_multipanel.png")

# ============================================================================
# Technique 3: Density/heatmap plot
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 10))

# Create 2D histogram
h, xedges, yedges = np.histogram2d(x2, y2, bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

im = ax.imshow(h.T, extent=extent, origin='lower', cmap='hot', 
               aspect='auto', interpolation='gaussian')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Visit Density', fontsize=12)

ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Trajectory Density Heatmap', fontsize=14)

plt.tight_layout()
plt.savefig('17_viz_heatmap.png', dpi=150)
print("[OK] Saved: 17_viz_heatmap.png")

# ============================================================================
# Key insights
# ============================================================================

print("\n" + "="*60)
print("VISUALIZATION TECHNIQUES:")
print("="*60)
print("1. LineCollection + colormap for time/energy coloring")
print("2. GridSpec for professional multi-panel layouts")
print("3. 2D histograms for trajectory density")
print("4. Animation for dynamic visualization")
print("5. Always label axes and include colorbars")
print("6. Use alpha for overlapping trajectories")
print("="*60)

plt.show()


