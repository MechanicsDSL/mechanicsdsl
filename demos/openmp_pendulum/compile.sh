#!/bin/bash
# Compile script for OpenMP pendulum demo

echo "Compiling OpenMP pendulum simulation..."

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ not found. Please install a C++ compiler."
    exit 1
fi

g++ -fopenmp -O3 -std=c++17 -march=native -o pendulum_openmp pendulum_openmp.cpp

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./pendulum_openmp"
else
    echo "Build failed."
    exit 1
fi
