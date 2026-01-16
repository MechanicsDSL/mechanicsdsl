# MechanicsDSL Embedded Examples

This directory contains examples for running MechanicsDSL on embedded and ARM platforms.

## Examples

### raspberry_pi_pendulum.py

Complete demonstration of MechanicsDSL on Raspberry Pi:
- Physics simulation in Python
- C++ code export for native performance
- CMake project generation

```bash
# Run on Raspberry Pi
python raspberry_pi_pendulum.py --simulate

# Export C++ for native execution
python raspberry_pi_pendulum.py --export-cpp
```

### raspberry_pi_imu.py

Real-time IMU sensor integration with MPU6050:
- I2C communication with MPU6050
- Accelerometer and gyroscope fusion
- Complementary filter for angle estimation
- Real-time data logging

```bash
# Hardware required: MPU6050 connected via I2C
python raspberry_pi_imu.py
```

## Hardware Setup

### Raspberry Pi GPIO Pinout (MPU6050)

```
MPU6050 Pin  → Pi Pin
-----------    ------
VCC          → 3.3V (Pin 1)
GND          → GND (Pin 6)
SDA          → GPIO 2 (Pin 3)
SCL          → GPIO 3 (Pin 5)
```

### Enable I2C

```bash
sudo raspi-config
# Navigate: Interface Options → I2C → Enable
```

## Installation on Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv libopenblas-dev

# Create environment
python3 -m venv ~/mdsl-env
source ~/mdsl-env/bin/activate

# Install MechanicsDSL
pip install mechanicsdsl-core[embedded]
```

## Performance Tips

1. **Use C++ export** for compute-heavy simulations
2. **Enable OpenBLAS** for optimized linear algebra
3. **Run headless** (no GUI) for maximum CPU availability
4. **Use `nice -10`** for higher process priority

## See Also

- [ARM Optimization Guide](../../docs/arm_optimization.md)
- [C++ CMake Example](../codegen/cpp_cmake_example/)
- [Rust Cargo Example](../codegen/rust_cargo_example/)
