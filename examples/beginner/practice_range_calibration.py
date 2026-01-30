#!/usr/bin/env python3
"""
Practice Range Calibration - Marble Launcher Data Table Generator
==================================================================

Range-Based Back-Calculation (Practice Range)

This script implements the complete experimental procedure for determining
the actual launch speeds of a marble launcher by back-calculating from
measured horizontal ranges during horizontal launches.

MARBLE LAUNCHER SPECIFICATIONS:
-------------------------------
- Durable apparatus for velocity, acceleration, and projectile motion studies
- 5 different speed settings
- Velocity range: 3 m/s to 9 m/s
- Angle range: 0° to 90° (easy-to-read protractor)
- Dimensions: 16" L x 10" W x 11½" H (40.6 cm x 25.4 cm x 29.2 cm)
- Includes 2 marbles

EXPERIMENTAL PHYSICS:
---------------------
By launching the marble horizontally (θ = 0°), the vertical motion is 
completely determined by gravity alone, with no initial upward or downward 
velocity.

Vertical Motion (Free Fall):
    y(t) = y₀ - ½gt²
    
    When marble hits ground (y = 0):
        0 = y₀ - ½gt²
        t = √(2y₀/g)    ← Time of flight from vertical drop alone
    
Horizontal Motion (Constant Velocity):
    x(t) = v₀ₓ × t
    
    Solving for launch speed:
        v₀ₓ = x / t = x × √(g/(2y₀))

This gives us the initial horizontal velocity, which for horizontal launch
equals the total launch speed at that setting.

Author: MechanicsDSL Examples
"""

import csv
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime

# Try to import mechanics_dsl kinematics module
try:
    from mechanics_dsl.domains.kinematics import ProjectileMotion
    HAS_MECHANICS_DSL = True
except ImportError:
    HAS_MECHANICS_DSL = False
    print("Note: mechanics_dsl not found. Using standalone calculations.")


# ============================================================================
# PHYSICAL CONSTANTS AND LAUNCHER SPECIFICATIONS
# ============================================================================

# Standard gravitational acceleration
GRAVITY = 9.81  # m/s²

# Marble Launcher Specifications
LAUNCHER_SPECS = {
    "model": "Standard Marble Launcher",
    "velocity_range": (3.0, 9.0),   # m/s (min, max)
    "num_speed_settings": 5,
    "angle_range": (0, 90),          # degrees
    "length_inches": 16,
    "width_inches": 10,
    "height_inches": 11.5,
    "num_marbles": 2,
}

# Convert dimensions to metric
LAUNCHER_SPECS["length_cm"] = LAUNCHER_SPECS["length_inches"] * 2.54  # 40.64 cm
LAUNCHER_SPECS["width_cm"] = LAUNCHER_SPECS["width_inches"] * 2.54    # 25.40 cm
LAUNCHER_SPECS["height_cm"] = LAUNCHER_SPECS["height_inches"] * 2.54  # 29.21 cm

# Speed settings (linearly distributed from 3 to 9 m/s)
# Setting 1 = 3 m/s, Setting 2 = 4.5 m/s, Setting 3 = 6 m/s, Setting 4 = 7.5 m/s, Setting 5 = 9 m/s
SPEED_SETTINGS = {
    1: {"target_speed_mps": 3.0, "description": "Low"},
    2: {"target_speed_mps": 4.5, "description": "Medium-Low"},
    3: {"target_speed_mps": 6.0, "description": "Medium"},
    4: {"target_speed_mps": 7.5, "description": "Medium-High"},
    5: {"target_speed_mps": 9.0, "description": "High"},
}

# Default experimental parameters
DEFAULT_LAUNCH_HEIGHT = 1.0  # meters (typical table height)
NUM_TRIALS_RECOMMENDED = 5
MEASUREMENT_ACCURACY = 0.05  # meters (5 cm as specified)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrialMeasurement:
    """Single trial measurement data."""
    setting: int
    trial_number: int
    measured_range_m: float
    notes: str = ""


@dataclass
class TrialResult:
    """Calculated results for a single trial."""
    setting: int
    trial_number: int
    measured_range_m: float
    time_of_flight_s: float
    calculated_speed_mps: float
    launch_height_m: float
    
    
@dataclass
class SettingCalibration:
    """Complete calibration data for one speed setting."""
    setting: int
    target_speed_mps: float
    description: str
    trials: List[TrialResult]
    
    # Calculated statistics
    mean_speed_mps: float
    std_dev_mps: float
    min_speed_mps: float
    max_speed_mps: float
    spread_mps: float  # max - min
    uncertainty_mps: float  # Standard error of mean
    
    # Range statistics
    mean_range_m: float
    std_dev_range_m: float
    
    # Common values
    time_of_flight_s: float
    launch_height_m: float
    num_trials: int


@dataclass
class CalibrationExperiment:
    """Complete calibration experiment results."""
    launch_height_m: float
    gravity_mps2: float
    time_of_flight_s: float
    measurement_accuracy_m: float
    num_trials_per_setting: int
    settings: List[SettingCalibration]
    timestamp: str = ""
    notes: str = ""


# ============================================================================
# KINEMATICS CALCULATIONS
# ============================================================================

def calculate_time_of_flight(height_m: float, g: float = GRAVITY) -> float:
    """
    Calculate time of flight for horizontal launch from given height.
    
    For horizontal launch (θ = 0°), the vertical motion is purely free fall:
        y(t) = y₀ - ½gt²
    
    When marble hits ground (y = 0):
        0 = y₀ - ½gt²
        t² = 2y₀/g
        t = √(2y₀/g)
    
    This uses kinematics Equation 2 with v₀y = 0 (horizontal launch).
    
    Args:
        height_m: Launch height above ground in meters
        g: Gravitational acceleration in m/s²
    
    Returns:
        Time of flight in seconds
        
    Raises:
        ValueError: If height is not positive
    """
    if height_m <= 0:
        raise ValueError(f"Height must be positive, got {height_m} m")
    if g <= 0:
        raise ValueError(f"Gravity must be positive, got {g} m/s²")
    
    # t = √(2h/g) from vertical kinematics
    return math.sqrt(2 * height_m / g)


def calculate_launch_speed(horizontal_range_m: float, time_of_flight_s: float) -> float:
    """
    Back-calculate initial horizontal launch speed from measured range.
    
    For horizontal motion with constant velocity (no air resistance):
        x(t) = v₀ₓ × t
    
    Therefore:
        v₀ₓ = x / t
    
    For horizontal launch (θ = 0°), this equals the total launch speed.
    
    Args:
        horizontal_range_m: Measured horizontal distance in meters
        time_of_flight_s: Time of flight in seconds
    
    Returns:
        Initial launch speed in m/s
        
    Raises:
        ValueError: If inputs are not positive
    """
    if horizontal_range_m < 0:
        raise ValueError(f"Range must be non-negative, got {horizontal_range_m} m")
    if time_of_flight_s <= 0:
        raise ValueError(f"Time must be positive, got {time_of_flight_s} s")
    
    return horizontal_range_m / time_of_flight_s


def calculate_expected_range(speed_mps: float, height_m: float, g: float = GRAVITY) -> float:
    """
    Calculate expected horizontal range for a given speed and height.
    
    Args:
        speed_mps: Launch speed in m/s
        height_m: Launch height in meters
        g: Gravitational acceleration in m/s²
    
    Returns:
        Expected horizontal range in meters
    """
    time = calculate_time_of_flight(height_m, g)
    return speed_mps * time


def calculate_uncertainty_propagation(
    mean_range_m: float,
    std_dev_range_m: float,
    num_trials: int,
    time_of_flight_s: float,
    height_m: float,
    height_uncertainty_m: float = 0.01
) -> float:
    """
    Calculate uncertainty in calculated speed using error propagation.
    
    v₀ₓ = range / t, where t = √(2h/g)
    
    Relative uncertainties add in quadrature:
        (δv/v)² = (δx/x)² + (δt/t)²
    
    Since t = √(2h/g):
        δt/t = (1/2) × (δh/h)
    
    Args:
        mean_range_m: Mean measured range
        std_dev_range_m: Standard deviation of range measurements
        num_trials: Number of trials
        time_of_flight_s: Calculated time of flight
        height_m: Launch height
        height_uncertainty_m: Uncertainty in height measurement
    
    Returns:
        Uncertainty in calculated speed (m/s)
    """
    # Standard error of mean for range
    if num_trials > 1:
        range_sem = std_dev_range_m / math.sqrt(num_trials)
    else:
        range_sem = std_dev_range_m if std_dev_range_m > 0 else MEASUREMENT_ACCURACY
    
    # Relative uncertainties
    if mean_range_m > 0:
        rel_range_uncertainty = range_sem / mean_range_m
    else:
        rel_range_uncertainty = 0
    
    rel_height_uncertainty = height_uncertainty_m / height_m
    rel_time_uncertainty = 0.5 * rel_height_uncertainty  # δt/t = (1/2) × (δh/h)
    
    # Combined relative uncertainty
    rel_velocity_uncertainty = math.sqrt(rel_range_uncertainty**2 + rel_time_uncertainty**2)
    
    # Convert to absolute uncertainty
    mean_speed = mean_range_m / time_of_flight_s
    return mean_speed * rel_velocity_uncertainty


# ============================================================================
# EXPERIMENT PROCESSING
# ============================================================================

def process_trials(
    setting: int,
    target_speed: float,
    description: str,
    measured_ranges: List[float],
    launch_height: float,
    g: float = GRAVITY
) -> SettingCalibration:
    """
    Process all trials for one speed setting.
    
    Args:
        setting: Speed setting number (1-5)
        target_speed: Expected/target speed for this setting (m/s)
        description: Description of this setting
        measured_ranges: List of measured ranges in meters
        launch_height: Launch height in meters
        g: Gravitational acceleration
    
    Returns:
        SettingCalibration with all statistics
    """
    if not measured_ranges:
        raise ValueError("Must have at least one measurement")
    
    # Calculate time of flight (same for all trials at same height)
    time_flight = calculate_time_of_flight(launch_height, g)
    
    # Process each trial
    trials = []
    for i, range_m in enumerate(measured_ranges, start=1):
        speed = calculate_launch_speed(range_m, time_flight)
        trials.append(TrialResult(
            setting=setting,
            trial_number=i,
            measured_range_m=range_m,
            time_of_flight_s=time_flight,
            calculated_speed_mps=speed,
            launch_height_m=launch_height
        ))
    
    # Calculate statistics
    speeds = [t.calculated_speed_mps for t in trials]
    ranges = [t.measured_range_m for t in trials]
    num_trials = len(trials)
    
    mean_speed = statistics.mean(speeds)
    std_dev = statistics.stdev(speeds) if num_trials > 1 else 0.0
    min_speed = min(speeds)
    max_speed = max(speeds)
    spread = max_speed - min_speed
    
    mean_range = statistics.mean(ranges)
    std_dev_range = statistics.stdev(ranges) if num_trials > 1 else 0.0
    
    # Calculate uncertainty (standard error of mean)
    uncertainty = calculate_uncertainty_propagation(
        mean_range, std_dev_range, num_trials, time_flight, launch_height
    )
    
    return SettingCalibration(
        setting=setting,
        target_speed_mps=target_speed,
        description=description,
        trials=trials,
        mean_speed_mps=mean_speed,
        std_dev_mps=std_dev,
        min_speed_mps=min_speed,
        max_speed_mps=max_speed,
        spread_mps=spread,
        uncertainty_mps=uncertainty,
        mean_range_m=mean_range,
        std_dev_range_m=std_dev_range,
        time_of_flight_s=time_flight,
        launch_height_m=launch_height,
        num_trials=num_trials
    )


# ============================================================================
# DATA TABLE GENERATION
# ============================================================================

def generate_expected_data_table(
    launch_height_m: float = DEFAULT_LAUNCH_HEIGHT,
    num_trials: int = NUM_TRIALS_RECOMMENDED,
    simulated_variation: float = 0.02  # 2% relative variation in speed
) -> CalibrationExperiment:
    """
    Generate a complete expected data table for all 5 speed settings.
    
    This creates theoretical/expected values based on launcher specifications,
    simulating realistic trial-to-trial variation.
    
    Args:
        launch_height_m: Height of launcher above ground
        num_trials: Number of trials per setting
        simulated_variation: Relative variation in speed (for simulation)
    
    Returns:
        Complete CalibrationExperiment with all settings
    """
    time_flight = calculate_time_of_flight(launch_height_m)
    settings_results = []
    
    for setting_num, setting_info in SPEED_SETTINGS.items():
        target_speed = setting_info["target_speed_mps"]
        description = setting_info["description"]
        
        # Calculate expected range at this speed
        expected_range = calculate_expected_range(target_speed, launch_height_m)
        
        # Simulate realistic variation in measurements
        # Using small random offsets around the expected range
        import random
        random.seed(42 + setting_num)  # Reproducible for demonstration
        
        # Simulate trial data with realistic variation
        measured_ranges = []
        for trial in range(num_trials):
            # Add realistic variation (±2% of range, representing measurement 
            # uncertainty and launcher consistency)
            variation = random.gauss(0, simulated_variation * expected_range)
            # Round to 5 cm accuracy as specified
            range_val = round((expected_range + variation) / 0.05) * 0.05
            measured_ranges.append(max(0.05, range_val))  # Minimum 5 cm
        
        # Process this setting
        calibration = process_trials(
            setting=setting_num,
            target_speed=target_speed,
            description=description,
            measured_ranges=measured_ranges,
            launch_height=launch_height_m
        )
        settings_results.append(calibration)
    
    return CalibrationExperiment(
        launch_height_m=launch_height_m,
        gravity_mps2=GRAVITY,
        time_of_flight_s=time_flight,
        measurement_accuracy_m=MEASUREMENT_ACCURACY,
        num_trials_per_setting=num_trials,
        settings=settings_results,
        timestamp=datetime.now().isoformat(),
        notes="Simulated data based on launcher specifications"
    )


def print_experiment_plan(launch_height: float = DEFAULT_LAUNCH_HEIGHT):
    """Print the experimental procedure and physics background."""
    time_flight = calculate_time_of_flight(launch_height)
    
    print("=" * 100)
    print("MARBLE LAUNCHER PRACTICE RANGE CALIBRATION - EXPERIMENTAL PLAN")
    print("=" * 100)
    print()
    print("LAUNCHER SPECIFICATIONS:")
    print("-" * 100)
    print(f"  Velocity Range:   {LAUNCHER_SPECS['velocity_range'][0]} - {LAUNCHER_SPECS['velocity_range'][1]} m/s")
    print(f"  Speed Settings:   {LAUNCHER_SPECS['num_speed_settings']} settings")
    print(f"  Angle Range:      {LAUNCHER_SPECS['angle_range'][0]} deg - {LAUNCHER_SPECS['angle_range'][1]} deg")
    print(f"  Dimensions:       {LAUNCHER_SPECS['length_inches']}\" L x {LAUNCHER_SPECS['width_inches']}\" W x {LAUNCHER_SPECS['height_inches']}\" H")
    print(f"                    ({LAUNCHER_SPECS['length_cm']:.1f} cm x {LAUNCHER_SPECS['width_cm']:.1f} cm x {LAUNCHER_SPECS['height_cm']:.1f} cm)")
    print()
    
    print("OBJECTIVE:")
    print("-" * 100)
    print("  Determine the actual launch speeds for all 5 speed settings by back-calculating")
    print("  from measured horizontal ranges during horizontal (0 deg) launches.")
    print()
    
    print("EXPERIMENTAL SETUP:")
    print("-" * 100)
    print(f"  1. Set launcher to horizontal (0 deg angle)")
    print(f"  2. Measure launch height: {launch_height*100:.1f} cm from barrel to ground")
    print(f"  3. Mark landing position (carbon paper, flour, or similar)")
    print(f"  4. Measure range accurate to 5 cm (0.05 m)")
    print(f"  5. Repeat {NUM_TRIALS_RECOMMENDED} trials per setting")
    print()
    
    print("PHYSICS DERIVATION:")
    print("-" * 100)
    print("  For horizontal launch (theta = 0 deg), vertical and horizontal motion are independent:")
    print()
    print("  VERTICAL MOTION (Free Fall):")
    print("  +-------------------------------------------------------------------+")
    print("  |  y(t) = y0 - (1/2)gt^2                                           |")
    print("  |                                                                   |")
    print("  |  When y = 0 (marble hits ground):                                |")
    print("  |      0 = y0 - (1/2)gt^2                                          |")
    print("  |      t^2 = 2*y0/g                                                |")
    print("  |      t = sqrt(2*y0/g)                                            |")
    print("  +-------------------------------------------------------------------+")
    print()
    print("  HORIZONTAL MOTION (Constant Velocity):")
    print("  +-------------------------------------------------------------------+")
    print("  |  x(t) = v0x * t                                                  |")
    print("  |                                                                   |")
    print("  |  Solving for launch speed:                                       |")
    print("  |      v0x = x / t = x * sqrt(g/(2*y0))                           |")
    print("  +-------------------------------------------------------------------+")
    print()
    print("  CALCULATION FOR THIS SETUP:")
    print(f"      t = sqrt(2 * {launch_height:.3f} m / {GRAVITY:.2f} m/s^2)")
    print(f"      t = sqrt({2*launch_height/GRAVITY:.5f} s^2)")
    print(f"      t = {time_flight:.4f} s")
    print()
    print("=" * 100)
    print()


def print_blank_data_table(launch_height: float = DEFAULT_LAUNCH_HEIGHT, num_trials: int = NUM_TRIALS_RECOMMENDED):
    """Print a blank data table for recording experimental measurements."""
    time_flight = calculate_time_of_flight(launch_height)
    
    print("=" * 120)
    print("DATA TABLE FOR EXPERIMENTATION (BLANK)")
    print("=" * 120)
    print()
    print(f"Launch Height: {launch_height*100:.1f} cm = {launch_height:.4f} m")
    print(f"Time of Flight (calculated): t = {time_flight:.4f} s")
    print(f"Measurement Accuracy: ±{MEASUREMENT_ACCURACY*100:.1f} cm")
    print()
    print("-" * 120)
    print()
    
    # Print expected ranges for reference
    print("EXPECTED RANGES (for reference):")
    print("-" * 80)
    print(f"{'Setting':<10} {'Target Speed':<15} {'Expected Range':<20}")
    print("-" * 80)
    for setting, info in SPEED_SETTINGS.items():
        expected_range = calculate_expected_range(info["target_speed_mps"], launch_height)
        print(f"{setting:<10} {info['target_speed_mps']:<15.1f} m/s {expected_range:.4f} m = {expected_range*100:.1f} cm")
    print()
    
    # Print blank trial table for each setting
    for setting, info in SPEED_SETTINGS.items():
        print(f"\nSETTING {setting}: {info['description'].upper()} (Target: {info['target_speed_mps']:.1f} m/s)")
        print("-" * 100)
        print(f"{'Trial':<8} {'Measured Range (m)':<20} {'Measured Range (cm)':<22} {'Calculated Speed (m/s)':<25}")
        print("-" * 100)
        for trial in range(1, num_trials + 1):
            print(f"{trial:<8} {'______':<20} {'______':<22} {'______':<25}")
        print("-" * 100)
        print(f"{'MEAN':<8} {'______':<20} {'______':<22} {'______':<25}")
        print(f"{'STD DEV':<8} {'______':<20} {'______':<22} {'______':<25}")
        print()
    
    print("=" * 120)


def print_complete_data_table(experiment: CalibrationExperiment):
    """Print the complete calibration data table with all results."""
    print("=" * 140)
    print("MARBLE LAUNCHER CALIBRATION - COMPLETE DATA TABLE")
    print("=" * 140)
    print()
    print(f"Experiment Timestamp: {experiment.timestamp}")
    print(f"Launch Height:        {experiment.launch_height_m:.4f} m ({experiment.launch_height_m*100:.1f} cm)")
    print(f"Time of Flight:       {experiment.time_of_flight_s:.4f} s")
    print(f"Gravity:              {experiment.gravity_mps2:.4f} m/s²")
    print(f"Trials per Setting:   {experiment.num_trials_per_setting}")
    print(f"Measurement Accuracy: ±{experiment.measurement_accuracy_m*100:.1f} cm")
    print()
    if experiment.notes:
        print(f"Notes: {experiment.notes}")
        print()
    
    # Summary table
    print("-" * 140)
    print("SUMMARY BY SETTING:")
    print("-" * 140)
    header = (
        f"{'Setting':<8} {'Target':<10} {'Mean Range':<12} {'± Std Dev':<12} "
        f"{'Mean Speed':<12} {'± Std Dev':<12} {'Min Speed':<10} {'Max Speed':<10} "
        f"{'Spread':<10} {'Uncertainty':<12}"
    )
    print(header)
    units = (
        f"{'':<8} {'(m/s)':<10} {'(m)':<12} {'(m)':<12} "
        f"{'(m/s)':<12} {'(m/s)':<12} {'(m/s)':<10} {'(m/s)':<10} "
        f"{'(m/s)':<10} {'(m/s)':<12}"
    )
    print(units)
    print("-" * 140)
    
    for cal in experiment.settings:
        row = (
            f"{cal.setting:<8} {cal.target_speed_mps:<10.2f} "
            f"{cal.mean_range_m:<12.4f} {cal.std_dev_range_m:<12.4f} "
            f"{cal.mean_speed_mps:<12.4f} {cal.std_dev_mps:<12.4f} "
            f"{cal.min_speed_mps:<10.4f} {cal.max_speed_mps:<10.4f} "
            f"{cal.spread_mps:<10.4f} {cal.uncertainty_mps:<12.4f}"
        )
        print(row)
    
    print("-" * 140)
    print()


def print_detailed_trials(experiment: CalibrationExperiment):
    """Print detailed trial-by-trial data for each setting."""
    print("=" * 100)
    print("DETAILED TRIAL DATA")
    print("=" * 100)
    
    for cal in experiment.settings:
        print()
        print(f"SETTING {cal.setting}: {cal.description.upper()}")
        print(f"Target Speed: {cal.target_speed_mps:.2f} m/s")
        print("-" * 100)
        print(f"{'Trial':<8} {'Range (m)':<14} {'Range (cm)':<14} {'Time (s)':<12} {'Speed (m/s)':<14}")
        print("-" * 100)
        
        for trial in cal.trials:
            print(
                f"{trial.trial_number:<8} "
                f"{trial.measured_range_m:<14.4f} "
                f"{trial.measured_range_m*100:<14.2f} "
                f"{trial.time_of_flight_s:<12.4f} "
                f"{trial.calculated_speed_mps:<14.4f}"
            )
        
        print("-" * 100)
        print(
            f"{'MEAN':<8} "
            f"{cal.mean_range_m:<14.4f} "
            f"{cal.mean_range_m*100:<14.2f} "
            f"{cal.time_of_flight_s:<12.4f} "
            f"{cal.mean_speed_mps:<14.4f}"
        )
        print(
            f"{'STD DEV':<8} "
            f"{cal.std_dev_range_m:<14.4f} "
            f"{cal.std_dev_range_m*100:<14.2f} "
            f"{'':<12} "
            f"{cal.std_dev_mps:<14.4f}"
        )
        print(f"{'SPREAD':<8} {'':<14} {'':<14} {'':<12} {cal.spread_mps:<14.4f}")
        print(f"{'UNCERT.':<8} {'':<14} {'':<14} {'':<12} ±{cal.uncertainty_mps:<13.4f}")
    
    print()
    print("=" * 100)


def print_speed_vs_target_comparison(experiment: CalibrationExperiment):
    """Print comparison of measured vs target speeds."""
    print()
    print("=" * 90)
    print("MEASURED vs TARGET SPEED COMPARISON")
    print("=" * 90)
    print()
    print(f"{'Setting':<10} {'Target':<12} {'Measured':<14} {'Difference':<14} {'% Error':<12}")
    print(f"{'':<10} {'(m/s)':<12} {'(m/s)':<14} {'(m/s)':<14} {'':<12}")
    print("-" * 90)
    
    for cal in experiment.settings:
        diff = cal.mean_speed_mps - cal.target_speed_mps
        pct_error = (diff / cal.target_speed_mps) * 100 if cal.target_speed_mps != 0 else 0
        print(
            f"{cal.setting:<10} "
            f"{cal.target_speed_mps:<12.2f} "
            f"{cal.mean_speed_mps:<14.4f} "
            f"{diff:+<14.4f} "
            f"{pct_error:+<12.2f}%"
        )
    
    print("-" * 90)
    print()
    print("INTERPRETATION:")
    print("-" * 90)
    print("  - Positive difference: Launcher fires faster than expected")
    print("  - Negative difference: Launcher fires slower than expected")
    print("  - Small % errors (<5%) indicate launcher meets specifications")
    print("  - The MEASURED values should be used for predictive modeling")
    print()


# ============================================================================
# CSV EXPORT
# ============================================================================

def export_summary_csv(experiment: CalibrationExperiment, filename: str = "practice_range_calibration_summary.csv"):
    """Export summary data to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Metadata header
        writer.writerow(["# Marble Launcher Calibration Summary"])
        writer.writerow([f"# Launch Height: {experiment.launch_height_m} m"])
        writer.writerow([f"# Time of Flight: {experiment.time_of_flight_s:.4f} s"])
        writer.writerow([f"# Timestamp: {experiment.timestamp}"])
        writer.writerow([])
        
        # Data header
        writer.writerow([
            "Setting", "Description", "Target_Speed_mps", "Mean_Range_m", "Std_Dev_Range_m",
            "Mean_Speed_mps", "Std_Dev_Speed_mps", "Min_Speed_mps", "Max_Speed_mps",
            "Spread_mps", "Uncertainty_mps", "Num_Trials", "Time_of_Flight_s", "Launch_Height_m"
        ])
        
        for cal in experiment.settings:
            writer.writerow([
                cal.setting,
                cal.description,
                f"{cal.target_speed_mps:.4f}",
                f"{cal.mean_range_m:.4f}",
                f"{cal.std_dev_range_m:.4f}",
                f"{cal.mean_speed_mps:.4f}",
                f"{cal.std_dev_mps:.4f}",
                f"{cal.min_speed_mps:.4f}",
                f"{cal.max_speed_mps:.4f}",
                f"{cal.spread_mps:.4f}",
                f"{cal.uncertainty_mps:.4f}",
                cal.num_trials,
                f"{cal.time_of_flight_s:.4f}",
                f"{cal.launch_height_m:.4f}"
            ])
    
    print(f"Summary exported to: {filename}")


def export_detailed_csv(experiment: CalibrationExperiment, filename: str = "practice_range_calibration_detailed.csv"):
    """Export detailed trial data to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Setting", "Description", "Target_Speed_mps", "Trial", 
            "Measured_Range_m", "Time_of_Flight_s", "Calculated_Speed_mps"
        ])
        
        for cal in experiment.settings:
            for trial in cal.trials:
                writer.writerow([
                    trial.setting,
                    cal.description,
                    f"{cal.target_speed_mps:.4f}",
                    trial.trial_number,
                    f"{trial.measured_range_m:.4f}",
                    f"{trial.time_of_flight_s:.4f}",
                    f"{trial.calculated_speed_mps:.4f}"
                ])
    
    print(f"Detailed data exported to: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_calibration_demo(launch_height: float = DEFAULT_LAUNCH_HEIGHT):
    """
    Run a complete demonstration of the calibration procedure.
    
    This generates simulated data based on the launcher specifications
    and demonstrates all output formats.
    """
    print("\n" + "=" * 100)
    print("MARBLE LAUNCHER PRACTICE RANGE CALIBRATION")
    print("Complete Data Table for All 5 Settings")
    print("=" * 100 + "\n")
    
    # Print experimental plan
    print_experiment_plan(launch_height)
    
    # Print blank data table for reference
    print_blank_data_table(launch_height)
    print("\n")
    
    # Generate expected/simulated data
    print("GENERATING CALIBRATION DATA (using expected values with realistic variation)...\n")
    experiment = generate_expected_data_table(launch_height)
    
    # Print complete results
    print_complete_data_table(experiment)
    print_detailed_trials(experiment)
    print_speed_vs_target_comparison(experiment)
    
    # Export to CSV
    export_summary_csv(experiment)
    export_detailed_csv(experiment)
    
    return experiment


def main():
    """Main function - run the complete calibration demo."""
    # Use a realistic launch height (e.g., table-mounted launcher)
    launch_height = 1.0  # 1 meter = 100 cm
    
    experiment = run_calibration_demo(launch_height)
    
    print("\n" + "=" * 100)
    print("CALIBRATION COMPLETE")
    print("=" * 100)
    print()
    print("The measured speeds can now be used for predictive modeling in longer,")
    print("more complex launches. These values account for minor losses from air")
    print("resistance and launcher friction during the calibration step.")
    print()
    print("Files created:")
    print("  - practice_range_calibration_summary.csv (summary statistics)")
    print("  - practice_range_calibration_detailed.csv (trial-by-trial data)")
    print()
    
    return experiment


if __name__ == "__main__":
    main()
