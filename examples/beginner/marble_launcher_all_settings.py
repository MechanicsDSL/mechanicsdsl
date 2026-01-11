#!/usr/bin/env python3
"""
Marble Launcher Comprehensive Simulation

Simulates ALL settings of the marble launcher:
- Velocity: 3 to 9 m/s (every integer value)
- Angle: 0 to 90 degrees (every integer value)

That's 7 velocities × 91 angles = 637 total simulations!

Results include range, max height, time of flight, and impact velocity
for each combination.

Author: MechanicsDSL Examples
"""

import csv
import sys
from typing import List, Tuple
from mechanics_dsl.domains.kinematics import ProjectileMotion


# Marble launcher parameters
Y0 = 4.0  # Balcony height in meters
G = 9.81  # Gravity

# Launcher settings ranges
VELOCITY_MIN = 3  # m/s
VELOCITY_MAX = 9  # m/s
ANGLE_MIN = 0     # degrees
ANGLE_MAX = 90    # degrees


def simulate_all_settings():
    """
    Simulate every combination of velocity and angle.
    
    Returns:
        List of result dictionaries
    """
    results = []
    total_simulations = (VELOCITY_MAX - VELOCITY_MIN + 1) * (ANGLE_MAX - ANGLE_MIN + 1)
    
    print("=" * 70)
    print("MARBLE LAUNCHER - COMPREHENSIVE SIMULATION")
    print("=" * 70)
    print()
    print(f"Initial height: {Y0} m")
    print(f"Velocity range: {VELOCITY_MIN} to {VELOCITY_MAX} m/s")
    print(f"Angle range:    {ANGLE_MIN} to {ANGLE_MAX} degrees")
    print(f"Total simulations: {total_simulations}")
    print()
    print("Running simulations...")
    print()
    
    count = 0
    for v0 in range(VELOCITY_MIN, VELOCITY_MAX + 1):
        for theta in range(ANGLE_MIN, ANGLE_MAX + 1):
            count += 1
            
            # Create projectile and analyze
            proj = ProjectileMotion(v0=float(v0), theta_deg=float(theta), y0=Y0, g=G)
            result = proj.analyze(y_final=0)
            
            # Store results
            results.append({
                'velocity_mps': v0,
                'angle_deg': theta,
                'range_m': result.range,
                'max_height_m': result.max_height,
                'time_of_flight_s': result.time_of_flight,
                'impact_velocity_mps': result.impact_velocity,
                'impact_angle_deg': result.impact_angle_deg,
                'v0x_mps': result.v0x,
                'v0y_mps': result.v0y,
            })
            
            # Progress indicator every 100 simulations
            if count % 100 == 0:
                print(f"  Completed {count}/{total_simulations} simulations...")
    
    print(f"  Completed {count}/{total_simulations} simulations!")
    print()
    
    return results


def find_optimal_settings(results: List[dict]):
    """
    Find optimal settings for different objectives.
    """
    print("=" * 70)
    print("OPTIMAL SETTINGS")
    print("=" * 70)
    print()
    
    # Maximum range
    max_range = max(results, key=lambda r: r['range_m'])
    print("MAXIMUM RANGE:")
    print(f"  Velocity: {max_range['velocity_mps']} m/s")
    print(f"  Angle:    {max_range['angle_deg']} degrees")
    print(f"  Range:    {max_range['range_m']:.3f} m")
    print()
    
    # Maximum height
    max_height = max(results, key=lambda r: r['max_height_m'])
    print("MAXIMUM HEIGHT:")
    print(f"  Velocity: {max_height['velocity_mps']} m/s")
    print(f"  Angle:    {max_height['angle_deg']} degrees")
    print(f"  Height:   {max_height['max_height_m']:.3f} m")
    print()
    
    # Maximum time of flight
    max_time = max(results, key=lambda r: r['time_of_flight_s'])
    print("MAXIMUM TIME OF FLIGHT:")
    print(f"  Velocity: {max_time['velocity_mps']} m/s")
    print(f"  Angle:    {max_time['angle_deg']} degrees")
    print(f"  Time:     {max_time['time_of_flight_s']:.3f} s")
    print()
    
    # Maximum impact velocity
    max_impact = max(results, key=lambda r: r['impact_velocity_mps'])
    print("MAXIMUM IMPACT VELOCITY:")
    print(f"  Velocity: {max_impact['velocity_mps']} m/s")
    print(f"  Angle:    {max_impact['angle_deg']} degrees")
    print(f"  Impact:   {max_impact['impact_velocity_mps']:.3f} m/s")
    print()
    
    # Minimum impact velocity (gentlest landing)
    min_impact = min(results, key=lambda r: r['impact_velocity_mps'])
    print("MINIMUM IMPACT VELOCITY (gentlest landing):")
    print(f"  Velocity: {min_impact['velocity_mps']} m/s")
    print(f"  Angle:    {min_impact['angle_deg']} degrees")
    print(f"  Impact:   {min_impact['impact_velocity_mps']:.3f} m/s")
    print()


def print_summary_table(results: List[dict]):
    """
    Print a summary table for each velocity.
    """
    print("=" * 70)
    print("SUMMARY BY VELOCITY (best angle for max range)")
    print("=" * 70)
    print()
    print(f"{'Velocity':>10} {'Best Angle':>12} {'Max Range':>12} {'Max Height':>12} {'Time':>10}")
    print(f"{'(m/s)':>10} {'(deg)':>12} {'(m)':>12} {'(m)':>12} {'(s)':>10}")
    print("-" * 60)
    
    for v0 in range(VELOCITY_MIN, VELOCITY_MAX + 1):
        # Get all results for this velocity
        v_results = [r for r in results if r['velocity_mps'] == v0]
        
        # Find best angle for range
        best = max(v_results, key=lambda r: r['range_m'])
        
        print(f"{v0:>10} {best['angle_deg']:>12} {best['range_m']:>12.3f} "
              f"{best['max_height_m']:>12.3f} {best['time_of_flight_s']:>10.3f}")
    
    print()


def print_all_results(results: List[dict]):
    """
    Print all 637 simulation results.
    """
    print("=" * 70)
    print("ALL SIMULATION RESULTS")
    print("=" * 70)
    print()
    print(f"{'V (m/s)':>8} {'Angle':>7} {'Range (m)':>11} {'Height (m)':>12} "
          f"{'Time (s)':>10} {'Impact (m/s)':>13}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['velocity_mps']:>8} {r['angle_deg']:>7} {r['range_m']:>11.4f} "
              f"{r['max_height_m']:>12.4f} {r['time_of_flight_s']:>10.4f} "
              f"{r['impact_velocity_mps']:>13.4f}")
    
    print()
    print(f"Total: {len(results)} simulations")


def save_to_csv(results: List[dict], filename: str = "marble_launcher_all_settings.csv"):
    """
    Save all results to a CSV file.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to: {filename}")


def main():
    # Run all simulations
    results = simulate_all_settings()
    
    # Find optimal settings
    find_optimal_settings(results)
    
    # Summary table
    print_summary_table(results)
    
    # Ask if user wants to see all results
    print("Would you like to see all 637 results? (Automatically showing for completeness)")
    print()
    print_all_results(results)
    
    # Save to CSV
    print()
    save_to_csv(results)
    
    return results


if __name__ == '__main__':
    main()
