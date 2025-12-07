# CUDA Pendulum Demo

GPU-accelerated pendulum simulation using CUDA.

## Files

- `simple_pendulum.cu` - CUDA kernel code
- `simple_pendulum.h` - Header file
- `simple_pendulum_cpu.cpp` - CPU fallback
- `CMakeLists.txt` - Build configuration

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Run

```bash
# CUDA version (requires NVIDIA GPU)
./simple_pendulum_cuda

# CPU fallback (any system)
./simple_pendulum_cpu
```

## Requirements

- CUDA Toolkit 11.0+ (for GPU version)
- CMake 3.18+
- C++17 compiler
