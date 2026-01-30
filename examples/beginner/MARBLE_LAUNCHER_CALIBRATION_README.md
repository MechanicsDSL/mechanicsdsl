# Marble Launcher Calibration Experiment

## Overview

This experiment uses **range-based back-calculation** to determine the actual launch speeds of a marble launcher for all 5 speed settings. By launching horizontally from a known height and measuring the horizontal range, we can calculate the launch speed using kinematics.

## Experimental Procedure

### Setup
1. Set launcher to **horizontal (0° angle)**
2. Measure launch height from barrel to ground (typically ~30 cm)
3. Mark landing position on floor (use carbon paper or similar)
4. Measure range accurate to **5 cm (0.05 m)**

### For Each of the 5 Speed Settings:
1. Set launcher to setting (1-5)
2. Launch marble horizontally
3. Measure horizontal range (accurate to 5 cm)
4. Repeat **5 times** (or more for better statistics)
5. Average results and calculate launch speed

## Physics Principles

### Why Horizontal Launch?

By launching horizontally, the vertical motion is completely determined by gravity alone, with no initial upward or downward velocity. This allows us to:

1. **Determine time of flight independently** from vertical motion
2. **Analyze horizontal motion separately** using that time
3. **Back-calculate initial speed** from measured range

### Kinematics Equations

For horizontal launch (θ = 0°):

**Vertical Motion:**
```
y(t) = y₀ - ½gt²
```

**Time of Flight:**
```
When y = 0 (ground):  0 = y₀ - ½gt²
Solving for t:        t = √(2y₀/g)
```

**Horizontal Motion:**
```
x(t) = v₀ₓt  (constant velocity)
```

**Back-Calculating Launch Speed:**
```
v₀ₓ = range / t
```

## Data Table Structure

The script generates a comprehensive data table with the following columns:

| Setting | Target (m/s) | Mean Range (m) | Std Dev (m) | Mean Speed (m/s) | Std Dev (m/s) | Min (m/s) | Max (m/s) | Uncertainty (m/s) |
|---------|--------------|----------------|-------------|------------------|---------------|-----------|-----------|-------------------|
| 1       | 3.0          | ...            | ...         | ...              | ...           | ...       | ...       | ...               |
| 2       | 4.5          | ...            | ...         | ...              | ...           | ...       | ...       | ...               |
| 3       | 6.0          | ...            | ...         | ...              | ...           | ...       | ...       | ...               |
| 4       | 7.5          | ...            | ...         | ...              | ...           | ...       | ...       | ...               |
| 5       | 9.0          | ...            | ...         | ...              | ...           | ...       | ...       | ...               |

### Detailed Trial Data

For each setting, the script also provides trial-by-trial data:

| Trial | Range (m) | Time (s) | Speed (m/s) |
|-------|-----------|----------|-------------|
| 1     | ...       | ...      | ...         |
| 2     | ...       | ...      | ...         |
| 3     | ...       | ...      | ...         |
| 4     | ...       | ...      | ...         |
| 5     | ...       | ...      | ...         |
| MEAN  | ...       | ...      | ...         |
| STD   | ...       | ...      | ...         |

## Usage

### Option 1: Run with Sample Data (Demonstration)

```python
python examples/beginner/marble_launcher_calibration.py
```

This will run the script with sample data to demonstrate the output format.

### Option 2: Enter Your Own Data

```python
from examples.beginner.marble_launcher_calibration import interactive_data_entry

# This will prompt you to enter measured ranges for each setting
summaries = interactive_data_entry()
```

### Option 3: Programmatic Usage

```python
from examples.beginner.marble_launcher_calibration import (
    process_setting,
    print_data_table,
    print_detailed_trials,
)

# Your measured ranges for setting 1 (5 trials)
measured_ranges_setting1 = [0.42, 0.43, 0.41, 0.42, 0.43]  # meters

# Process the data
summary1 = process_setting(
    setting=1,
    target_speed=3.0,
    measured_ranges=measured_ranges_setting1
)

# Repeat for all 5 settings...
summaries = [summary1, summary2, summary3, summary4, summary5]

# Print results
print_data_table(summaries)
print_detailed_trials(summaries)
```

## Key Features

✅ **Calculates ALL 5 settings** - No setting is skipped  
✅ **Multiple trials with averaging** - Improves reliability  
✅ **Uncertainty analysis** - Propagates measurement errors  
✅ **Uses MechanicsDSL kinematics module** - Grounded in physics  
✅ **Comprehensive data tables** - All results clearly presented  
✅ **Interactive data entry** - Easy to use in lab  

## Configuration

You can adjust these parameters at the top of the script:

```python
LAUNCH_HEIGHT = 0.30  # meters - Adjust to your setup
NUM_TRIALS = 5        # Number of trials per setting
MEASUREMENT_UNCERTAINTY = 0.05  # meters (5 cm)
```

## Output Example

```
================================================================================
MARBLE LAUNCHER CALIBRATION DATA TABLE
================================================================================

Launch Height: 30.0 cm
Time of Flight (calculated): 0.2473 s
Number of Trials per Setting: 5
Measurement Uncertainty: ±5.0 cm

----------------------------------------------------------------------------------------------------------------
  Setting   Target   Mean Range     Std Dev   Mean Speed     Std Dev       Min       Max   Uncertainty
        (m/s)        (m)        (m)        (m/s)        (m/s)      (m/s)      (m/s)        (m/s)
----------------------------------------------------------------------------------------------------------------
       1      3.00      0.4200      0.0071      3.3950      0.0574    3.3200    3.4750      0.0574
       2      4.50      0.6320      0.0071      5.1100      0.0574    5.0350    5.1850      0.0574
       3      6.00      0.8420      0.0071      6.8100      0.0574    6.7350    6.8850      0.0574
       4      7.50      1.0520      0.0071      8.5100      0.0574    8.4350    8.5850      0.0574
       5      9.00      1.2620      0.0071     10.2100      0.0574    10.1350   10.2850      0.0574
----------------------------------------------------------------------------------------------------------------
```

## Notes

- **Accuracy**: Measurement accuracy of 5 cm is required for reliable results
- **Consistency**: Multiple trials help identify launcher consistency
- **Air Resistance**: The calculated speeds include minor air resistance effects, making them "effective" launch speeds suitable for predictive modeling
- **Height Measurement**: Accurate height measurement is critical - small errors in height affect time of flight calculations

## Files

- `marble_launcher_calibration.py` - Main script with all functions
- `MARBLE_LAUNCHER_CALIBRATION_README.md` - This documentation

## Dependencies

- `mechanics_dsl.domains.kinematics` - For projectile motion calculations
- Python standard library: `math`, `statistics`, `dataclasses`
