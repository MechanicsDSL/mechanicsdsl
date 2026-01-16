# ARM Optimization Guide for MechanicsDSL

This guide covers optimizing MechanicsDSL simulations for ARM-based systems like Raspberry Pi.

## Supported ARM Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Raspberry Pi 3 | ARMv8 (aarch64) | ✅ Fully supported |
| Raspberry Pi 4/5 | ARMv8 (aarch64) | ✅ Fully supported |
| Raspberry Pi Zero 2 | ARMv8 (aarch64) | ✅ Supported |
| Jetson Nano | ARMv8 + CUDA | ✅ GPU acceleration available |
| Apple M1/M2/M3 | ARM64 | ✅ Excellent performance |

## Installation on Raspberry Pi

### Basic Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3-pip python3-venv libopenblas-dev

# Create virtual environment
python3 -m venv ~/mechanicsdsl-env
source ~/mechanicsdsl-env/bin/activate

# Install MechanicsDSL
pip install mechanicsdsl-core
```

### With Hardware Support

```bash
# For GPIO/sensor integration
pip install "mechanicsdsl-core[embedded]"

# This installs:
# - RPi.GPIO for hardware control
# - smbus2 for I2C sensors
```

## Performance Optimization

### 1. Use NumPy's OpenBLAS Backend

Raspberry Pi can leverage OpenBLAS for faster linear algebra:

```bash
# Install optimized NumPy
sudo apt install libopenblas-dev
pip install --no-binary numpy numpy
```

Verify BLAS is active:
```python
import numpy as np
np.__config__.show()  # Should show 'openblas'
```

### 2. Enable Numba JIT (Optional)

For compute-heavy simulations:

```bash
pip install numba llvmlite
```

Use JIT-compiled solver:
```python
from mechanics_dsl import PhysicsCompiler

compiler = PhysicsCompiler()
compiler.compile_dsl(code)
compiler.enable_jit()  # Enables Numba acceleration
```

### 3. Generate Native C++ Code

For maximum performance, export to C++:

```python
from mechanics_dsl.codegen.cpp import CppGenerator

# Generate project with ARM optimizations
generator = CppGenerator(...)
generator.generate_project("./my_simulation")
```

Build on Raspberry Pi:
```bash
cd my_simulation
mkdir build && cd build
cmake .. && make -j4
./my_simulation  # Runs at native speed
```

### 4. Memory Optimization

Raspberry Pi has limited RAM. Optimize with:

```python
from mechanics_dsl import PhysicsCompiler

compiler = PhysicsCompiler(
    low_memory_mode=True,  # Uses float32 instead of float64
    chunk_size=100         # Smaller output chunks
)
```

## Benchmark Results

Tested on Raspberry Pi 4 (4GB RAM, 64-bit OS):

| Simulation | Python | C++ (ARM) | Speedup |
|------------|--------|-----------|---------|
| Simple Pendulum (10s) | 45 ms | 3 ms | 15x |
| Double Pendulum (10s) | 120 ms | 8 ms | 15x |
| N-Body (3 bodies, 10s) | 850 ms | 45 ms | 19x |
| SPH Fluid (100 particles) | 12 s | 0.8 s | 15x |

## Real-Time Control

For robotics applications requiring real-time control:

### Using systemd Service

Create `/etc/systemd/system/physics-sim.service`:

```ini
[Unit]
Description=MechanicsDSL Physics Controller
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/simulation
ExecStart=/home/pi/simulation/controller
Restart=always
Nice=-10
CPUAffinity=2 3

[Install]
WantedBy=multi-user.target
```

### Setting Process Priority

```python
import os
os.nice(-10)  # Higher priority (requires root)
```

## Troubleshooting

### ImportError: libopenblas.so

```bash
sudo apt install libopenblas-base
```

### Out of Memory

- Use `low_memory_mode=True`
- Reduce simulation points
- Export to C++ for large simulations

### Slow Startup

First-run JIT compilation is slow. Pre-compile with:

```bash
python -c "from mechanics_dsl import PhysicsCompiler; PhysicsCompiler()"
```

## Example: Real-Time Pendulum Controller

```python
import time
from mechanics_dsl import PhysicsCompiler

code = r"""
\system{pendulum}
\defvar{theta}{Angle}{rad}
\parameter{m}{1}{kg}
\parameter{l}{0.5}{m}
\lagrangian{0.5*m*l^2*\dot{theta}^2 - m*9.81*l*(1-\cos{theta})}
"""

compiler = PhysicsCompiler()
compiler.compile_dsl(code)

# Real-time loop at 100 Hz
dt = 0.01
while True:
    start = time.perf_counter()
    
    # Step simulation
    compiler.step(dt)
    angle = compiler.get_state('theta')
    
    # Use angle for motor control here
    print(f"θ = {angle:.3f} rad")
    
    # Maintain 100 Hz
    elapsed = time.perf_counter() - start
    time.sleep(max(0, dt - elapsed))
```

## Further Resources

- [Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)
- [MechanicsDSL Examples](https://github.com/MechanicsDSL/mechanicsdsl/tree/main/examples/embedded)
- [OpenBLAS ARM](https://github.com/xianyi/OpenBLAS/wiki/User-Manual)
