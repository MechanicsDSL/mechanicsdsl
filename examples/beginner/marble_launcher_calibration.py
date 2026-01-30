#!/usr/bin/env python3
"""
Marble Launcher Calibration Experiment
======================================

Range-Based Back-Calculation (Practice Range)

This script implements the experimental procedure to determine the actual
launch speeds of a marble launcher by back-calculating from measured ranges.

Experimental Setup:
-------------------
1. Launch marble horizontally (0°) from a known height
2. Measure horizontal range (accurate to 5 cm)
3. Repeat multiple trials and average
4. Calculate time of flight from vertical drop using kinematics
5. Back-calculate initial horizontal velocity from range and time
6. Repeat for all 5 speed settings

Why Horizontal Launch?
-----------------------
By launching horizontally, the vertical motion is completely determined by
gravity alone, with no initial upward or downward velocity. This allows us
to determine the time of flight independently from the vertical motion,
then use that time to analyze the horizontal motion separately.

Physics:
--------
For horizontal launch (θ = 0°):
  - Vertical motion: y(t) = y₀ - ½gt²
  - Time of flight: t = √(2y₀/g)  [when y_final = 0]
  - Horizontal motion: x(t) = v₀ₓt
  - Initial horizontal speed: v₀ₓ = x_range / t

Launcher Specifications:
------------------------
- 5 different speed settings
- Velocity range: 3 m/s to 9 m/s
- Can launch at any angle 0-90°
- Dimensions: 16" L x 10" W x 11½" H

Author: MechanicsDSL Examples
"""

import csv
import math
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Using kinematics calculations directly
# These match the mechanics_dsl.domains.kinematics module


# ============================================================================
# EXPERIMENTAL PARAMETERS
# ============================================================================

# Launcher height (barrel to ground) - MEASURE THIS!
# Typical value: ~0.30 m (30 cm) for table-top setup
# Adjust based on your actual setup
LAUNCH_HEIGHT = 0.30  # meters (30 cm)

# Gravitational acceleration
GRAVITY = 9.81  # m/s²

# Number of trials per setting (recommended: 5-10)
NUM_TRIALS = 5

# Measurement uncertainty (5 cm = 0.05 m as specified)
MEASUREMENT_UNCERTAINTY = 0.05  # meters

# Speed settings (5 settings from 3 to 9 m/s)
# These are the TARGET speeds - we'll measure the ACTUAL speeds
SPEED_SETTINGS = [1, 2, 3, 4, 5]  # Setting numbers (1-5)
TARGET_SPEEDS = [3.0, 4.5, 6.0, 7.5, 9.0]  # Expected speeds in m/s


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class TrialData:
    """Data for a single trial."""

    setting: int
    trial_number: int
    measured_range: float  # meters
    calculated_time: float  # seconds (from vertical drop)
    calculated_speed: float  # m/s (back-calculated)


@dataclass
class SettingSummary:
    """Summary statistics for one speed setting."""

    setting: int
    target_speed: float  # Expected speed (m/s)
    trials: List[TrialData]
    mean_speed: float  # m/s
    std_dev: float  # m/s
    min_speed: float  # m/s
    max_speed: float  # m/s
    mean_range: float  # m
    std_dev_range: float  # m
    uncertainty_speed: float  # m/s (propagated uncertainty)


# ============================================================================
# KINEMATICS CALCULATIONS
# ============================================================================


def calculate_time_of_flight(height: float, g: float = GRAVITY) -> float:
    """
    Calculate time of flight for horizontal launch from given height.

    For horizontal launch (θ = 0°), the vertical motion is:
        y(t) = y₀ - ½gt²

    When the marble hits the ground (y = 0):
        0 = y₀ - ½gt²
        t = √(2y₀/g)

    This uses the kinematics equation for free fall, matching the
    mechanics_dsl.domains.kinematics.motion_1d.free_fall_time function.

    Args:
        height: Launch height in meters (y₀)
        g: Gravitational acceleration in m/s²

    Returns:
        Time of flight in seconds
    """
    if height <= 0:
        raise ValueError(f"Height must be positive, got {height} m")

    # Free fall time: t = √(2h/g)
    # This matches mechanics_dsl.domains.kinematics.motion_1d.free_fall_time
    time_flight = math.sqrt(2 * height / g)

    return time_flight


def back_calculate_launch_speed(
    measured_range: float, time_of_flight: float
) -> float:
    """
    Back-calculate initial horizontal launch speed from measured range.

    For horizontal motion (constant velocity):
        x(t) = v₀ₓt
        v₀ₓ = x(t) / t

    Args:
        measured_range: Horizontal distance traveled in meters
        time_of_flight: Time in air in seconds

    Returns:
        Initial horizontal velocity in m/s
    """
    if time_of_flight <= 0:
        raise ValueError(f"Time of flight must be positive, got {time_of_flight} s")

    return measured_range / time_of_flight


def calculate_uncertainty(
    mean_range: float,
    std_dev_range: float,
    time_of_flight: float,
    height_uncertainty: float = 0.01,
) -> float:
    """
    Calculate uncertainty in calculated speed using error propagation.

    v₀ₓ = range / t
    where t = √(2h/g)

    Uncertainty in v₀ₓ comes from:
    1. Uncertainty in range measurement (σ_range)
    2. Uncertainty in height measurement (affects t)

    Using error propagation:
        σ_v = v₀ₓ * √((σ_range/range)² + (σ_t/t)²)
        where σ_t/t ≈ (1/2) * (σ_h/h)

    Args:
        mean_range: Mean measured range in meters
        std_dev_range: Standard deviation of range measurements in meters
        time_of_flight: Calculated time of flight in seconds
        height_uncertainty: Uncertainty in height measurement in meters

    Returns:
        Uncertainty in calculated speed in m/s
    """
    # Uncertainty from range measurements
    range_uncertainty = std_dev_range / math.sqrt(NUM_TRIALS) if NUM_TRIALS > 1 else std_dev_range

    # Uncertainty in time from height uncertainty
    # t = √(2h/g), so σ_t/t = (1/2) * (σ_h/h)
    relative_height_uncertainty = height_uncertainty / LAUNCH_HEIGHT
    relative_time_uncertainty = 0.5 * relative_height_uncertainty

    # Combined uncertainty in speed
    relative_range_uncertainty = range_uncertainty / mean_range if mean_range > 0 else 0
    relative_speed_uncertainty = math.sqrt(
        relative_range_uncertainty**2 + relative_time_uncertainty**2
    )

    mean_speed = mean_range / time_of_flight
    speed_uncertainty = mean_speed * relative_speed_uncertainty

    return speed_uncertainty


# ============================================================================
# EXPERIMENTAL PROCEDURE
# ============================================================================


def run_single_trial(
    setting: int, trial_number: int, measured_range: float
) -> TrialData:
    """
    Process a single experimental trial.

    Args:
        setting: Speed setting number (1-5)
        trial_number: Trial number for this setting
        measured_range: Measured horizontal range in meters

    Returns:
        TrialData with calculated values
    """
    # Calculate time of flight from vertical drop
    time_flight = calculate_time_of_flight(LAUNCH_HEIGHT, GRAVITY)

    # Back-calculate launch speed
    launch_speed = back_calculate_launch_speed(measured_range, time_flight)

    return TrialData(
        setting=setting,
        trial_number=trial_number,
        measured_range=measured_range,
        calculated_time=time_flight,
        calculated_speed=launch_speed,
    )


def process_setting(
    setting: int, target_speed: float, measured_ranges: List[float]
) -> SettingSummary:
    """
    Process all trials for one speed setting.

    Args:
        setting: Speed setting number (1-5)
        target_speed: Expected speed for this setting (m/s)
        measured_ranges: List of measured ranges for each trial (meters)

    Returns:
        SettingSummary with statistics
    """
    if len(measured_ranges) != NUM_TRIALS:
        raise ValueError(
            f"Expected {NUM_TRIALS} trials, got {len(measured_ranges)}"
        )

    # Process each trial
    trials = []
    for i, range_val in enumerate(measured_ranges, start=1):
        trial = run_single_trial(setting, i, range_val)
        trials.append(trial)

    # Calculate statistics
    speeds = [t.calculated_speed for t in trials]
    ranges = [t.measured_range for t in trials]

    mean_speed = statistics.mean(speeds)
    std_dev_speed = statistics.stdev(speeds) if len(speeds) > 1 else 0.0

    mean_range = statistics.mean(ranges)
    std_dev_range = statistics.stdev(ranges) if len(ranges) > 1 else 0.0

    # Calculate uncertainty
    time_flight = calculate_time_of_flight(LAUNCH_HEIGHT, GRAVITY)
    uncertainty = calculate_uncertainty(mean_range, std_dev_range, time_flight)

    return SettingSummary(
        setting=setting,
        target_speed=target_speed,
        trials=trials,
        mean_speed=mean_speed,
        std_dev=std_dev_speed,
        min_speed=min(speeds),
        max_speed=max(speeds),
        mean_range=mean_range,
        std_dev_range=std_dev_range,
        uncertainty_speed=uncertainty,
    )


# ============================================================================
# DATA TABLE GENERATION
# ============================================================================


def print_experiment_plan():
    """Print the experimental procedure."""
    print("=" * 80)
    print("MARBLE LAUNCHER CALIBRATION - EXPERIMENTAL PLAN")
    print("=" * 80)
    print()
    print("OBJECTIVE:")
    print("  Determine the actual launch speeds for all 5 speed settings")
    print("  by back-calculating from measured horizontal ranges.")
    print()
    print("SETUP:")
    print(f"  1. Set launcher to horizontal (0° angle)")
    print(f"  2. Measure launch height: {LAUNCH_HEIGHT*100:.1f} cm from barrel to ground")
    print(f"  3. Mark landing position on floor (use carbon paper or similar)")
    print(f"  4. Measure range accurate to 5 cm (0.05 m)")
    print()
    print("PROCEDURE:")
    print(f"  For each of the 5 speed settings:")
    print(f"    1. Set launcher to setting (1-5)")
    print(f"    2. Launch marble horizontally")
    print(f"    3. Measure horizontal range (accurate to 5 cm)")
    print(f"    4. Repeat {NUM_TRIALS} times")
    print(f"    5. Average results and calculate launch speed")
    print()
    print("PHYSICS:")
    print("  Time of flight (from vertical drop):")
    print(f"    t = sqrt(2h/g) = sqrt(2 * {LAUNCH_HEIGHT:.3f} / {GRAVITY:.2f})")
    time_flight = calculate_time_of_flight(LAUNCH_HEIGHT, GRAVITY)
    print(f"    t = {time_flight:.4f} s")
    print()
    print("  Initial horizontal speed (from measured range):")
    print("    v0x = range / t")
    print()
    print("=" * 80)
    print()


def print_data_table(summaries: List[SettingSummary]):
    """
    Print comprehensive data table for all settings.

    Args:
        summaries: List of SettingSummary objects for each setting
    """
    print("=" * 120)
    print("MARBLE LAUNCHER CALIBRATION DATA TABLE")
    print("=" * 120)
    print()
    print(f"Launch Height: {LAUNCH_HEIGHT*100:.1f} cm")
    print(f"Time of Flight (calculated): {calculate_time_of_flight(LAUNCH_HEIGHT, GRAVITY):.4f} s")
    print(f"Number of Trials per Setting: {NUM_TRIALS}")
    print(f"Measurement Uncertainty: ±{MEASUREMENT_UNCERTAINTY*100:.1f} cm")
    print()
    print("-" * 120)
    print(
        f"{'Setting':>8} {'Target':>8} {'Mean Range':>12} {'Std Dev':>10} "
        f"{'Mean Speed':>12} {'Std Dev':>10} {'Min':>8} {'Max':>8} {'Uncertainty':>12}"
    )
    print(
        f"{'':>8} {'(m/s)':>8} {'(m)':>12} {'(m)':>10} "
        f"{'(m/s)':>12} {'(m/s)':>10} {'(m/s)':>8} {'(m/s)':>8} {'(m/s)':>12}"
    )
    print("-" * 120)

    for summary in summaries:
        print(
            f"{summary.setting:>8} {summary.target_speed:>8.2f} "
            f"{summary.mean_range:>12.4f} {summary.std_dev_range:>10.4f} "
            f"{summary.mean_speed:>12.4f} {summary.std_dev:>10.4f} "
            f"{summary.min_speed:>8.4f} {summary.max_speed:>8.4f} "
            f"{summary.uncertainty_speed:>12.4f}"
        )

    print("-" * 120)
    print()


def print_detailed_trials(summaries: List[SettingSummary]):
    """Print detailed trial-by-trial data."""
    print("=" * 100)
    print("DETAILED TRIAL DATA")
    print("=" * 100)
    print()

    for summary in summaries:
        print(f"Setting {summary.setting} (Target: {summary.target_speed:.2f} m/s)")
        print("-" * 100)
        print(
            f"{'Trial':>6} {'Range (m)':>12} {'Time (s)':>12} {'Speed (m/s)':>14}"
        )
        print("-" * 100)

        for trial in summary.trials:
            print(
                f"{trial.trial_number:>6} {trial.measured_range:>12.4f} "
                f"{trial.calculated_time:>12.4f} {trial.calculated_speed:>14.4f}"
            )

        print(
            f"{'MEAN':>6} {summary.mean_range:>12.4f} "
            f"{trial.calculated_time:>12.4f} {summary.mean_speed:>14.4f}"
        )
        print(
            f"{'STD':>6} {summary.std_dev_range:>12.4f} "
            f"{'':>12} {summary.std_dev:>14.4f}"
        )
        print()


# ============================================================================
# EXAMPLE USAGE WITH SAMPLE DATA
# ============================================================================


def example_with_sample_data():
    """
    Example using simulated/sample data.

    In a real experiment, you would replace the sample data with your
    actual measured ranges.
    """
    print_experiment_plan()

    # SAMPLE DATA - Replace with your actual measurements!
    # Format: {setting: [list of measured ranges in meters]}
    sample_data = {
        1: [0.42, 0.43, 0.41, 0.42, 0.43],  # Setting 1 (target ~3 m/s)
        2: [0.63, 0.64, 0.62, 0.63, 0.64],  # Setting 2 (target ~4.5 m/s)
        3: [0.84, 0.85, 0.83, 0.84, 0.85],  # Setting 3 (target ~6 m/s)
        4: [1.05, 1.06, 1.04, 1.05, 1.06],  # Setting 4 (target ~7.5 m/s)
        5: [1.26, 1.27, 1.25, 1.26, 1.27],  # Setting 5 (target ~9 m/s)
    }

    print("NOTE: Using SAMPLE DATA for demonstration.")
    print("Replace with your actual measured ranges!")
    print()

    # Process all settings
    summaries = []
    for setting, target_speed in zip(SPEED_SETTINGS, TARGET_SPEEDS):
        if setting in sample_data:
            summary = process_setting(setting, target_speed, sample_data[setting])
            summaries.append(summary)

    # Print results
    print_data_table(summaries)
    print_detailed_trials(summaries)

    return summaries


def interactive_data_entry():
    """
    Interactive function to enter experimental data.

    This function prompts the user to enter measured ranges for each setting.
    """
    print_experiment_plan()
    print("INTERACTIVE DATA ENTRY")
    print("=" * 80)
    print()
    print("Enter measured ranges for each setting.")
    print(f"Launch height: {LAUNCH_HEIGHT*100:.1f} cm")
    print(f"Number of trials per setting: {NUM_TRIALS}")
    print()

    summaries = []

    for setting, target_speed in zip(SPEED_SETTINGS, TARGET_SPEEDS):
        print(f"\nSetting {setting} (Target: {target_speed:.2f} m/s)")
        print("-" * 80)

        ranges = []
        for trial_num in range(1, NUM_TRIALS + 1):
            while True:
                try:
                    range_input = input(
                        f"  Trial {trial_num} - Measured range (meters): "
                    )
                    range_val = float(range_input)
                    if range_val > 0:
                        ranges.append(range_val)
                        break
                    else:
                        print("    Range must be positive!")
                except ValueError:
                    print("    Please enter a valid number!")

        summary = process_setting(setting, target_speed, ranges)
        summaries.append(summary)

    # Print results
    print("\n" + "=" * 80)
    print_data_table(summaries)
    print_detailed_trials(summaries)

    return summaries


# ============================================================================
# CSV EXPORT
# ============================================================================


def export_to_csv(summaries: List[SettingSummary], filename: str = "marble_launcher_calibration.csv"):
    """
    Export data table to CSV file.

    Args:
        summaries: List of SettingSummary objects
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'Setting',
            'Target Speed (m/s)',
            'Mean Range (m)',
            'Std Dev Range (m)',
            'Mean Speed (m/s)',
            'Std Dev Speed (m/s)',
            'Min Speed (m/s)',
            'Max Speed (m/s)',
            'Uncertainty Speed (m/s)',
            'Time of Flight (s)',
            'Launch Height (m)',
            'Number of Trials'
        ])

        # Write data rows
        for summary in summaries:
            writer.writerow([
                summary.setting,
                f"{summary.target_speed:.4f}",
                f"{summary.mean_range:.4f}",
                f"{summary.std_dev_range:.4f}",
                f"{summary.mean_speed:.4f}",
                f"{summary.std_dev:.4f}",
                f"{summary.min_speed:.4f}",
                f"{summary.max_speed:.4f}",
                f"{summary.uncertainty_speed:.4f}",
                f"{summary.trials[0].calculated_time:.4f}",
                f"{LAUNCH_HEIGHT:.4f}",
                NUM_TRIALS
            ])

    print(f"\nData table exported to: {filename}")


def export_detailed_trials_to_csv(summaries: List[SettingSummary], filename: str = "marble_launcher_detailed_trials.csv"):
    """
    Export detailed trial-by-trial data to CSV.

    Args:
        summaries: List of SettingSummary objects
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'Setting',
            'Target Speed (m/s)',
            'Trial Number',
            'Measured Range (m)',
            'Time of Flight (s)',
            'Calculated Speed (m/s)'
        ])

        # Write data rows
        for summary in summaries:
            for trial in summary.trials:
                writer.writerow([
                    trial.setting,
                    f"{summary.target_speed:.4f}",
                    trial.trial_number,
                    f"{trial.measured_range:.4f}",
                    f"{trial.calculated_time:.4f}",
                    f"{trial.calculated_speed:.4f}"
                ])

            # Add summary row
            writer.writerow([
                trial.setting,
                f"{summary.target_speed:.4f}",
                'MEAN',
                f"{summary.mean_range:.4f}",
                f"{trial.calculated_time:.4f}",
                f"{summary.mean_speed:.4f}"
            ])
            writer.writerow([
                trial.setting,
                f"{summary.target_speed:.4f}",
                'STD',
                f"{summary.std_dev_range:.4f}",
                '',
                f"{summary.std_dev:.4f}"
            ])

    print(f"Detailed trial data exported to: {filename}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main function - run example with sample data and export to CSV."""
    summaries = example_with_sample_data()

    # Export to CSV
    export_to_csv(summaries)
    export_detailed_trials_to_csv(summaries)

    print("\n" + "=" * 80)
    print("To enter your own data, use: interactive_data_entry()")
    print("=" * 80)


if __name__ == "__main__":
    main()
