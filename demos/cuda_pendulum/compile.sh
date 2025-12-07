#!/bin/bash
# Compile script for CUDA pendulum demo

echo "Compiling CUDA pendulum simulation..."

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found. Trying CPU fallback..."
    
    if ! command -v g++ &> /dev/null; then
        echo "Error: g++ not found. Please install a C++ compiler."
        exit 1
    fi
    
    g++ -O3 -std=c++17 -o simple_pendulum_cpu simple_pendulum_cpu.cpp
    echo "Built CPU fallback: ./simple_pendulum_cpu"
    exit 0
fi

# Build with CMake
mkdir -p build
cd build
cmake ..
make

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run CUDA version: ./simple_pendulum_cuda"
    echo "Run CPU version:  ./simple_pendulum_cpu"
else
    echo "Build failed."
    exit 1
fi
