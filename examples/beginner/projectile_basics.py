#!/usr/bin/env python3
"""
Projectile Motion Basics Example

This example demonstrates the fundamental concepts of projectile motion
using MechanicsDSL's kinematics module.

Topics covered:
1. Decomposing initial velocity into components
2. Calculating range, max height, and time of flight
3. Finding velocity at any time
4. Plotting trajectories

Author: MechanicsDSL Examples
"""

from mechanics_dsl.domains.kinematics import (
    ProjectileMotion,
    analyze_projectile,
    max_range_angle,
)


def example_basic_projectile():
    """
    Basic projectile motion example.
    
    A ball is thrown with initial speed 20 m/s at 45° above horizontal
    from ground level.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Projectile Motion")
    print("=" * 60)
    print()
    
    # Create the projectile
    proj = ProjectileMotion(
        v0=20.0,        # Initial speed: 20 m/s
        theta_deg=45.0, # Launch angle: 45°
        y0=0.0,         # Starting from ground level
        g=9.81,         # Standard gravity
    )
    
    # Analyze the motion
    result = proj.analyze()
    
    # Print results
    print(f"Initial velocity components:")
    print(f"  Horizontal (v₀ₓ): {result.v0x:.2f} m/s")
    print(f"  Vertical (v₀ᵧ):   {result.v0y:.2f} m/s")
    print()
    
    print(f"Key results:")
    print(f"  Time of flight:   {result.time_of_flight:.2f} s")
    print(f"  Horizontal range: {result.range:.2f} m")
    print(f"  Maximum height:   {result.max_height:.2f} m")
    print()
    
    print(f"At impact:")
    print(f"  Speed:            {result.impact_velocity:.2f} m/s")
    print(f"  Angle below horizontal: {result.impact_angle_deg:.1f}°")
    print()


def example_max_range():
    """
    Demonstrate that 45° gives maximum range on flat ground.
    """
    print("=" * 60)
    print("EXAMPLE 2: Maximum Range Angle")
    print("=" * 60)
    print()
    
    v0 = 20.0  # Same initial speed for all
    
    print(f"Testing different launch angles with v₀ = {v0} m/s:")
    print("-" * 40)
    
    angles = [15, 30, 45, 60, 75]
    ranges = []
    
    for angle in angles:
        proj = ProjectileMotion(v0=v0, theta_deg=angle)
        r = proj.range()
        ranges.append(r)
        print(f"  θ = {angle:2d}°: Range = {r:.2f} m")
    
    max_idx = ranges.index(max(ranges))
    print()
    print(f"Maximum range occurs at θ = {angles[max_idx]}°")
    print(f"Theoretical optimum: θ = {max_range_angle()}°")
    print()


def example_position_over_time():
    """
    Show position at different times during flight.
    """
    print("=" * 60)
    print("EXAMPLE 3: Position Over Time")
    print("=" * 60)
    print()
    
    proj = ProjectileMotion(v0=20, theta_deg=45)
    t_flight = proj.time_of_flight()
    
    print(f"Trajectory of ball launched at 20 m/s, 45°:")
    print("-" * 50)
    print(f"{'Time (s)':>10} {'x (m)':>10} {'y (m)':>10} {'Speed (m/s)':>12}")
    print("-" * 50)
    
    for t in [0, 0.5, 1.0, 1.5, 2.0, 2.5, t_flight]:
        if t <= t_flight:
            x, y = proj.position_at_time(t)
            speed = proj.speed_at_time(t)
            print(f"{t:10.2f} {x:10.2f} {y:10.2f} {speed:12.2f}")
    
    print()


def example_velocity_analysis():
    """
    Analyze velocity components during flight.
    """
    print("=" * 60)
    print("EXAMPLE 4: Velocity Analysis")
    print("=" * 60)
    print()
    
    proj = ProjectileMotion(v0=20, theta_deg=60)
    t_apex = proj.time_to_max_height()
    t_flight = proj.time_of_flight()
    
    print(f"Velocity analysis (v₀=20 m/s, θ=60°):")
    print()
    
    # At launch
    vx, vy = proj.velocity_at_time(0)
    print(f"At launch (t=0):")
    print(f"  vₓ = {vx:.2f} m/s, vᵧ = {vy:.2f} m/s")
    
    # At apex
    vx, vy = proj.velocity_at_time(t_apex)
    print(f"At apex (t={t_apex:.2f}s):")
    print(f"  vₓ = {vx:.2f} m/s, vᵧ = {vy:.2f} m/s")
    
    # At landing
    vx, vy = proj.velocity_at_time(t_flight)
    print(f"At landing (t={t_flight:.2f}s):")
    print(f"  vₓ = {vx:.2f} m/s, vᵧ = {vy:.2f} m/s")
    
    print()
    print("Note: vₓ remains constant, while vᵧ changes linearly with time.")
    print()


def example_show_work():
    """
    Demonstrate the show_work() feature for educational use.
    """
    print("=" * 60)
    print("EXAMPLE 5: Show Your Work")
    print("=" * 60)
    print()
    
    # Use the convenience function with show_work
    work = analyze_projectile(
        v0=15,
        theta_deg=30,
        y0=2,  # Starting 2m above ground
        show_work=True
    )
    
    print(work)


if __name__ == '__main__':
    example_basic_projectile()
    example_max_range()
    example_position_over_time()
    example_velocity_analysis()
    
    # Uncomment to see full step-by-step solution:
    # example_show_work()
