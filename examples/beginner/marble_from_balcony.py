#!/usr/bin/env python3
"""
Marble from Balcony - Physics Problem Example

A classic physics problem demonstrating projectile motion with initial height.

Problem Statement:
==================
A marble is launched from a second-floor balcony (4 meters above ground)
with an initial speed of 5 m/s at an angle of 30° above horizontal.

Find:
1. How long is the marble in the air?
2. How far from the building does it land?
3. What is its velocity when it hits the ground?
4. What is the maximum height reached?

This example shows how to use MechanicsDSL's kinematics module to solve
this problem analytically, showing all work step-by-step.

Author: MechanicsDSL Examples
"""

from mechanics_dsl.domains.kinematics import (
    ProjectileMotion,
    analyze_projectile,
    KinematicEquations,
)


def solve_marble_problem():
    """
    Complete solution to the marble from balcony problem.
    """
    print("=" * 70)
    print("MARBLE FROM BALCONY - Complete Solution")
    print("=" * 70)
    print()
    
    # Given values
    v0 = 5.0          # Initial speed: 5 m/s
    theta_deg = 30.0  # Launch angle: 30 degrees above horizontal
    y0 = 4.0          # Initial height: 4 m (second floor balcony)
    g = 9.81          # Gravitational acceleration
    
    print("GIVEN:")
    print(f"  Initial speed:    v0 = {v0} m/s")
    print(f"  Launch angle:     theta = {theta_deg} degrees")
    print(f"  Initial height:   y0 = {y0} m")
    print(f"  Gravity:          g = {g} m/s^2")
    print()
    
    # Create the projectile motion analyzer
    marble = ProjectileMotion(
        v0=v0,
        theta_deg=theta_deg,
        y0=y0,
        g=g,
    )
    
    # Get complete analysis
    result = marble.analyze(y_final=0)  # Landing at ground level
    
    print("-" * 70)
    print("SOLUTION:")
    print("-" * 70)
    print()
    
    # Step 1: Decompose initial velocity
    print("STEP 1: Decompose initial velocity into components")
    print()
    print("  v0x = v0 * cos(theta)")
    print(f"      = {v0} * cos({theta_deg} deg)")
    print(f"      = {result.v0x:.4f} m/s")
    print()
    print("  v0y = v0 * sin(theta)")
    print(f"      = {v0} * sin({theta_deg} deg)")
    print(f"      = {result.v0y:.4f} m/s")
    print()
    
    # Step 2: Time of flight
    print("STEP 2: Calculate time of flight")
    print()
    print("  Using the kinematic equation for vertical motion:")
    print("    y = y0 + v0y*t - (1/2)*g*t^2")
    print()
    print("  At landing, y = 0:")
    print("    0 = 4 + 2.5t - 4.905t^2")
    print()
    print("  Using the quadratic formula:")
    print(f"    t = {result.time_of_flight:.4f} s")
    print()
    
    # Step 3: Horizontal range
    print("STEP 3: Calculate horizontal range")
    print()
    print("  Range = v0x * t")
    print(f"       = {result.v0x:.4f} * {result.time_of_flight:.4f}")
    print(f"       = {result.range:.4f} m")
    print()
    
    # Step 4: Maximum height
    print("STEP 4: Calculate maximum height")
    print()
    print("  At maximum height, vy = 0")
    print("  Time to max height: t_apex = v0y/g")
    print(f"                           = {result.v0y:.4f}/{g}")
    print(f"                           = {result.time_to_max_height:.4f} s")
    print()
    print("  Maximum height: y_max = y0 + v0y^2/(2g)")
    print(f"                       = {y0} + {result.v0y:.4f}^2/(2 * {g})")
    print(f"                       = {result.max_height:.4f} m")
    print()
    
    # Step 5: Impact velocity
    print("STEP 5: Calculate impact velocity")
    print()
    print("  Horizontal component (constant):")
    print(f"    vx = {result.impact_vx:.4f} m/s")
    print()
    print("  Vertical component using v^2 = v0^2 + 2a(y - y0):")
    print(f"    vy = {result.impact_vy:.4f} m/s (downward)")
    print()
    print("  Impact speed: v = sqrt(vx^2 + vy^2)")
    print(f"              = sqrt({result.impact_vx:.4f}^2 + {result.impact_vy:.4f}^2)")
    print(f"              = {result.impact_velocity:.4f} m/s")
    print()
    print("  Impact angle below horizontal:")
    print(f"              = {result.impact_angle_deg:.2f} degrees")
    print()
    
    # Final answers
    print("=" * 70)
    print("FINAL ANSWERS:")
    print("=" * 70)
    print()
    print(f"  1. Time in air:     {result.time_of_flight:.3f} seconds")
    print(f"  2. Horizontal range:{result.range:.3f} meters")
    print(f"  3. Maximum height:  {result.max_height:.3f} meters")
    print(f"  4. Impact velocity: {result.impact_velocity:.3f} m/s at "
          f"{result.impact_angle_deg:.1f} deg below horizontal")
    print()
    
    return result


def show_trajectory():
    """
    Display the trajectory path at key points.
    """
    print()
    print("=" * 70)
    print("TRAJECTORY PATH")
    print("=" * 70)
    print()
    
    marble = ProjectileMotion(v0=5, theta_deg=30, y0=4, g=9.81)
    t_flight = marble.time_of_flight()
    t_apex = marble.time_to_max_height()
    
    # Key times
    times = [0, t_apex/2, t_apex, (t_apex + t_flight)/2, t_flight]
    
    print(f"{'Time (s)':>10} {'x (m)':>10} {'y (m)':>10} {'Description':>20}")
    print("-" * 55)
    
    descriptions = ["Launch", "Rising", "Apex", "Falling", "Landing"]
    
    for t, desc in zip(times, descriptions):
        x, y = marble.position_at_time(t)
        print(f"{t:10.3f} {x:10.3f} {y:10.3f} {desc:>20}")
    
    print()


def show_kinematic_equations_used():
    """
    Show which kinematic equations were used in the solution.
    """
    print()
    print("=" * 70)
    print("KINEMATIC EQUATIONS USED")
    print("=" * 70)
    print()
    
    print("This problem uses the following kinematic equations:")
    print()
    
    equations = KinematicEquations.all_equations()
    
    # Equations used in this problem
    used = [
        (1, "For finding vertical velocity component changes"),
        (2, "For vertical position as a function of time"),
        (3, "For impact velocity (time-independent)"),
    ]
    
    for num, purpose in used:
        eq = KinematicEquations.get_equation(num)
        print(f"  Equation {num}: {eq.latex}")
        print(f"    Purpose: {purpose}")
        print()


def full_step_by_step():
    """
    Use the built-in show_work() method.
    """
    print()
    print("=" * 70)
    print("FULL STEP-BY-STEP SOLUTION (using show_work)")
    print("=" * 70)
    
    marble = ProjectileMotion(v0=5, theta_deg=30, y0=4, g=9.81)
    work = marble.show_work()
    print(work)


if __name__ == '__main__':
    # Solve the problem with explanation
    result = solve_marble_problem()
    
    # Show trajectory
    show_trajectory()
    
    # Show equations used
    show_kinematic_equations_used()
    
    # Uncomment for full work output:
    # full_step_by_step()
