# C++ CMake Example

This example demonstrates generating a complete C++ simulation project with CMake build support.

## Quick Start

```bash
# Generate the project
python generate_project.py

# Build and run
cd output
mkdir build && cd build
cmake ..
make -j$(nproc)
./double_pendulum
```

## Generated Files

When you run `generate_project.py`, it creates:

```
output/
├── double_pendulum.cpp    # C++ simulation code
├── CMakeLists.txt         # CMake build configuration
└── README.md              # Build instructions
```

## Cross-Compilation for Raspberry Pi

```bash
# Install ARM toolchain
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Cross-compile
cd output/build
cmake -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ ..
make -j4

# Copy to Pi
scp double_pendulum pi@raspberrypi:~/
```

## Features

- **ARM/NEON Detection**: CMake automatically detects ARM and enables NEON optimizations
- **OpenMP Support**: Parallel loops if OpenMP is available
- **C++17**: Modern C++ standard
- **CSV Output**: Results saved to CSV for analysis

## Customization

Edit `generate_project.py` to change:

- System definition (DSL code)
- Output directory
- Simulation parameters
