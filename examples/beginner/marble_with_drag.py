#!/usr/bin/env python3
"""
Marble with Air Drag - Full Projectile Motion Solution

Solves the complete kinematics problem for a marble launched from a balcony
including quadratic air drag: F_drag = -½ρCdA|v|v

Problem:
    A marble is launched from a 12ft (3.66m) balcony at 85° with 5.8 m/s initial velocity.
    Calculate where it lands, time of flight, and compare with/without drag.

The Drag Equation:
    F_drag = ½ρCdAv²  (magnitude)
    
    Where:
        ρ = air density (1.225 kg/m³ at sea level)
        Cd = drag coefficient (0.47 for a sphere)
        A = cross-sectional area (πr² for sphere)
        v = velocity magnitude

Equations of Motion with Drag:
    m*ax = -k*|v|*vx
    m*ay = -mg - k*|v|*vy
    
    where k = ½ρCdA

Author: MechanicsDSL Examples (Extended for Drag)
"""

import math
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard constants
G = 9.81  # m/s² gravitational acceleration
RHO_AIR = 1.225  # kg/m³ air density at sea level, 15°C

# Marble properties (standard glass marble)
MARBLE_DIAMETER_MM = 16  # mm (standard marble)
MARBLE_DENSITY = 2500  # kg/m³ (glass)
CD_SPHERE = 0.47  # Drag coefficient for a smooth sphere


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarbleProperties:
    """Physical properties of the marble."""
    diameter: float  # meters
    mass: float  # kg
    cross_section_area: float  # m²
    drag_coefficient: float
    
    @classmethod
    def standard_marble(cls, diameter_mm: float = 16.0, material_density: float = 2500.0):
        """Create a standard glass marble."""
        d = diameter_mm / 1000  # convert to meters
        r = d / 2
        volume = (4/3) * math.pi * r**3
        mass = material_density * volume
        area = math.pi * r**2
        return cls(
            diameter=d,
            mass=mass,
            cross_section_area=area,
            drag_coefficient=CD_SPHERE
        )
    
    @property
    def drag_constant(self) -> float:
        """k = ½ρCdA (used in F_drag = k*v²)"""
        return 0.5 * RHO_AIR * self.drag_coefficient * self.cross_section_area
    
    @property
    def terminal_velocity(self) -> float:
        """Terminal velocity: v_t = sqrt(2mg / (ρCdA))"""
        return math.sqrt(2 * self.mass * G / (RHO_AIR * self.drag_coefficient * self.cross_section_area))


@dataclass
class LaunchConditions:
    """Initial launch conditions."""
    height: float  # meters
    angle_deg: float  # degrees above horizontal
    speed: float  # m/s
    x0: float = 0.0  # initial x position
    
    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle_deg)
    
    @property
    def v0x(self) -> float:
        return self.speed * math.cos(self.angle_rad)
    
    @property
    def v0y(self) -> float:
        return self.speed * math.sin(self.angle_rad)


@dataclass
class TrajectoryResult:
    """Results from trajectory calculation."""
    time_of_flight: float
    range: float  # horizontal distance
    max_height: float
    impact_speed: float
    impact_angle_deg: float  # below horizontal
    landing_x: float
    landing_y: float
    
    # Full trajectory data
    times: np.ndarray
    x_positions: np.ndarray
    y_positions: np.ndarray
    x_velocities: np.ndarray
    y_velocities: np.ndarray
    
    def summary(self) -> str:
        return f"""
+-----------------------------------------------------------+
|              TRAJECTORY ANALYSIS RESULTS                  |
+-----------------------------------------------------------+
|  Time of Flight:     {self.time_of_flight:8.4f} s                    |
|  Horizontal Range:   {self.range:8.4f} m                    |
|  Maximum Height:     {self.max_height:8.4f} m                    |
|  Landing Position:   ({self.landing_x:.3f}, {self.landing_y:.3f}) m           |
|  Impact Speed:       {self.impact_speed:8.4f} m/s                  |
|  Impact Angle:       {self.impact_angle_deg:8.2f} deg below horizontal  |
+-----------------------------------------------------------+
"""


# =============================================================================
# SOLVERS
# =============================================================================

def solve_without_drag(launch: LaunchConditions, y_final: float = 0.0) -> TrajectoryResult:
    """
    Analytical solution for projectile motion WITHOUT drag.
    
    Uses closed-form kinematic equations:
        x(t) = x0 + v0x*t
        y(t) = y0 + v0y*t - ½gt²
    """
    # Time of flight: solve y_final = y0 + v0y*t - ½gt²
    # Quadratic: -½gt² + v0y*t + (y0 - y_final) = 0
    a = -0.5 * G
    b = launch.v0y
    c = launch.height - y_final
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("Projectile never reaches target height")
    
    t1 = (-b + math.sqrt(discriminant)) / (2*a)
    t2 = (-b - math.sqrt(discriminant)) / (2*a)
    t_flight = max(t for t in [t1, t2] if t > 0)
    
    # Max height
    t_apex = launch.v0y / G if launch.v0y > 0 else 0
    max_height = launch.height + launch.v0y * t_apex - 0.5 * G * t_apex**2
    
    # Landing
    landing_x = launch.x0 + launch.v0x * t_flight
    landing_y = y_final
    range_val = landing_x - launch.x0
    
    # Impact velocity
    vx_impact = launch.v0x
    vy_impact = launch.v0y - G * t_flight
    impact_speed = math.sqrt(vx_impact**2 + vy_impact**2)
    impact_angle = math.degrees(math.atan2(-vy_impact, vx_impact))
    
    # Generate trajectory points
    n_points = 500
    times = np.linspace(0, t_flight, n_points)
    x_pos = launch.x0 + launch.v0x * times
    y_pos = launch.height + launch.v0y * times - 0.5 * G * times**2
    vx = np.full_like(times, launch.v0x)
    vy = launch.v0y - G * times
    
    return TrajectoryResult(
        time_of_flight=t_flight,
        range=range_val,
        max_height=max_height,
        impact_speed=impact_speed,
        impact_angle_deg=impact_angle,
        landing_x=landing_x,
        landing_y=landing_y,
        times=times,
        x_positions=x_pos,
        y_positions=y_pos,
        x_velocities=vx,
        y_velocities=vy
    )


def solve_with_drag(
    launch: LaunchConditions, 
    marble: MarbleProperties,
    y_final: float = 0.0
) -> TrajectoryResult:
    """
    Numerical solution for projectile motion WITH quadratic drag.
    
    Equations of motion:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = -(k/m) * |v| * vx
        dvy/dt = -g - (k/m) * |v| * vy
    
    where k = ½ρCdA and |v| = sqrt(vx² + vy²)
    """
    k = marble.drag_constant
    m = marble.mass
    
    def equations(t, state):
        x, y, vx, vy = state
        speed = math.sqrt(vx**2 + vy**2)
        
        # Derivatives
        dx_dt = vx
        dy_dt = vy
        dvx_dt = -(k/m) * speed * vx
        dvy_dt = -G - (k/m) * speed * vy
        
        return [dx_dt, dy_dt, dvx_dt, dvy_dt]
    
    def hit_ground(t, state):
        """Event: y crosses y_final"""
        return state[1] - y_final
    hit_ground.terminal = True
    hit_ground.direction = -1  # Only when going down
    
    # Initial state: [x, y, vx, vy]
    y0_state = [launch.x0, launch.height, launch.v0x, launch.v0y]
    
    # Solve ODE
    max_time = 20.0  # Maximum simulation time
    sol = solve_ivp(
        equations,
        (0, max_time),
        y0_state,
        method='RK45',
        events=hit_ground,
        dense_output=True,
        max_step=0.01
    )
    
    # Extract results
    if sol.t_events[0].size > 0:
        t_flight = sol.t_events[0][0]
    else:
        t_flight = sol.t[-1]
    
    # Generate trajectory at uniform time points
    n_points = 500
    times = np.linspace(0, t_flight, n_points)
    trajectory = sol.sol(times)
    
    x_pos = trajectory[0]
    y_pos = trajectory[1]
    vx = trajectory[2]
    vy = trajectory[3]
    
    # Final state
    final_state = sol.sol(t_flight)
    landing_x = final_state[0]
    landing_y = final_state[1]
    vx_impact = final_state[2]
    vy_impact = final_state[3]
    
    impact_speed = math.sqrt(vx_impact**2 + vy_impact**2)
    impact_angle = math.degrees(math.atan2(-vy_impact, vx_impact))
    
    # Max height
    max_height = np.max(y_pos)
    range_val = landing_x - launch.x0
    
    return TrajectoryResult(
        time_of_flight=t_flight,
        range=range_val,
        max_height=max_height,
        impact_speed=impact_speed,
        impact_angle_deg=impact_angle,
        landing_x=landing_x,
        landing_y=landing_y,
        times=times,
        x_positions=x_pos,
        y_positions=y_pos,
        x_velocities=vx,
        y_velocities=vy
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(
    no_drag: TrajectoryResult,
    with_drag: TrajectoryResult,
    title: str = "Marble Trajectory: With vs Without Air Drag"
) -> None:
    """Plot both trajectories for comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Trajectory plot
    ax1 = axes[0, 0]
    ax1.plot(no_drag.x_positions, no_drag.y_positions, 'b-', linewidth=2, label='No Drag (Ideal)')
    ax1.plot(with_drag.x_positions, with_drag.y_positions, 'r--', linewidth=2, label='With Air Drag')
    ax1.axhline(y=0, color='brown', linestyle='-', linewidth=3, label='Ground')
    ax1.scatter([no_drag.landing_x], [0], color='blue', s=100, zorder=5, marker='x')
    ax1.scatter([with_drag.landing_x], [0], color='red', s=100, zorder=5, marker='x')
    ax1.set_xlabel('Horizontal Distance (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=-0.1)
    ax1.set_ylim(bottom=-0.5)
    
    # Speed vs time
    ax2 = axes[0, 1]
    speed_no_drag = np.sqrt(no_drag.x_velocities**2 + no_drag.y_velocities**2)
    speed_with_drag = np.sqrt(with_drag.x_velocities**2 + with_drag.y_velocities**2)
    ax2.plot(no_drag.times, speed_no_drag, 'b-', linewidth=2, label='No Drag')
    ax2.plot(with_drag.times, speed_with_drag, 'r--', linewidth=2, label='With Drag')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('Speed vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Height vs time
    ax3 = axes[1, 0]
    ax3.plot(no_drag.times, no_drag.y_positions, 'b-', linewidth=2, label='No Drag')
    ax3.plot(with_drag.times, with_drag.y_positions, 'r--', linewidth=2, label='With Drag')
    ax3.axhline(y=0, color='brown', linestyle='-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Height vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Comparison bar chart
    ax4 = axes[1, 1]
    metrics = ['Range (m)', 'Max Height (m)', 'Time of Flight (s)', 'Impact Speed (m/s)']
    no_drag_vals = [no_drag.range, no_drag.max_height, no_drag.time_of_flight, no_drag.impact_speed]
    with_drag_vals = [with_drag.range, with_drag.max_height, with_drag.time_of_flight, with_drag.impact_speed]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, no_drag_vals, width, label='No Drag', color='blue', alpha=0.7)
    ax4.bar(x + width/2, with_drag_vals, width, label='With Drag', color='red', alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Key Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('marble_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN SOLUTION
# =============================================================================

def solve_marble_problem():
    """
    Complete solution for the marble from balcony problem.
    
    Given:
        - Height: 12 ft = 3.6576 m
        - Angle: 85° above horizontal
        - Initial velocity: 5.8 m/s
        - Object: Standard 16mm glass marble
    """
    print("=" * 70)
    print("   MARBLE FROM BALCONY - COMPLETE SOLUTION WITH AIR DRAG")
    print("=" * 70)
    print()
    
    # Convert units
    height_ft = 12
    height_m = height_ft * 0.3048  # 3.6576 m
    
    # Problem setup
    launch = LaunchConditions(
        height=height_m,
        angle_deg=85.0,
        speed=5.8
    )
    
    marble = MarbleProperties.standard_marble(diameter_mm=16.0)
    
    # Print problem description
    print("GIVEN:")
    print(f"  Height:          {height_ft} ft = {height_m:.4f} m")
    print(f"  Launch angle:    {launch.angle_deg} deg above horizontal")
    print(f"  Initial speed:   {launch.speed} m/s")
    print()
    print("MARBLE PROPERTIES:")
    print(f"  Diameter:        {marble.diameter*1000:.1f} mm")
    print(f"  Mass:            {marble.mass*1000:.3f} g")
    print(f"  Cross-section:   {marble.cross_section_area*1e4:.4f} cm^2")
    print(f"  Drag coefficient (Cd): {marble.drag_coefficient}")
    print(f"  Terminal velocity:     {marble.terminal_velocity:.2f} m/s")
    print()
    print("INITIAL VELOCITY COMPONENTS:")
    print(f"  v0x = {launch.speed} x cos({launch.angle_deg} deg) = {launch.v0x:.4f} m/s")
    print(f"  v0y = {launch.speed} x sin({launch.angle_deg} deg) = {launch.v0y:.4f} m/s")
    print()
    
    # Solve both cases
    print("-" * 70)
    print("SOLVING...")
    print("-" * 70)
    
    result_no_drag = solve_without_drag(launch)
    result_with_drag = solve_with_drag(launch, marble)
    
    # Display results
    print()
    print("=" * 70)
    print("                    WITHOUT AIR DRAG (IDEAL)")
    print("=" * 70)
    print(result_no_drag.summary())
    
    print()
    print("=" * 70)
    print("                    WITH AIR DRAG (REALISTIC)")
    print("=" * 70)
    print(result_with_drag.summary())
    
    # Comparison
    print()
    print("=" * 70)
    print("                      EFFECT OF AIR DRAG")
    print("=" * 70)
    print()
    
    range_diff = result_no_drag.range - result_with_drag.range
    range_pct = (range_diff / result_no_drag.range) * 100
    
    height_diff = result_no_drag.max_height - result_with_drag.max_height
    height_pct = (height_diff / result_no_drag.max_height) * 100
    
    time_diff = result_no_drag.time_of_flight - result_with_drag.time_of_flight
    time_pct = (time_diff / result_no_drag.time_of_flight) * 100
    
    speed_diff = result_no_drag.impact_speed - result_with_drag.impact_speed
    speed_pct = (speed_diff / result_no_drag.impact_speed) * 100
    
    print(f"  Range reduction:        {range_diff:.4f} m ({range_pct:.2f}% shorter)")
    print(f"  Max height reduction:   {height_diff:.4f} m ({height_pct:.2f}% lower)")
    print(f"  Flight time reduction:  {time_diff:.4f} s ({time_pct:.2f}% shorter)")
    print(f"  Impact speed reduction: {speed_diff:.4f} m/s ({speed_pct:.2f}% slower)")
    print()
    
    print("CONCLUSION:")
    if abs(range_pct) < 5:
        print("  Air drag has a SMALL effect on this trajectory.")
        print(f"  At {launch.speed} m/s, which is only {launch.speed/marble.terminal_velocity*100:.1f}% of terminal velocity,")
        print("  drag forces are relatively minor.")
    else:
        print("  Air drag has a SIGNIFICANT effect on this trajectory.")
    
    print()
    print("-" * 70)
    print("PLATE LANDING ANALYSIS:")
    print("-" * 70)
    print()
    print(f"  If you place a plate at ground level (y=0):")
    print(f"  * Without drag: Place plate {result_no_drag.landing_x:.3f} m from building")
    print(f"  * With drag:    Place plate {result_with_drag.landing_x:.3f} m from building")
    print(f"  * Difference:   {abs(range_diff):.4f} m = {abs(range_diff)*100:.2f} cm")
    print()
    
    # Plot
    print("Generating trajectory plot...")
    plot_comparison(result_no_drag, result_with_drag, 
                   f"Marble from {height_ft}ft Balcony @ {launch.angle_deg} deg, {launch.speed} m/s")
    
    return result_no_drag, result_with_drag


if __name__ == "__main__":
    solve_marble_problem()
